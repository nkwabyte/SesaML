import math
import os
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import PipelineConfig
from ..data.text_transform import TextTransform
from ..utils.run_logger import RunManager, save_model_meta
from .evaluator import Evaluator

# Consecutive degenerate-loss steps tolerated before aborting. Generous enough
# that a transient non-finite batch does not kill an otherwise healthy run.
DEGENERATE_LOSS_PATIENCE = 100


class DegenerateLossError(RuntimeError):
    """Raised when the CTC loss is pinned at 0.0 or non-finite, i.e. nothing is being learned."""


class Trainer:
    """
    Manages the training loop for the Akan DeepSpeech2 CTC model.

    Every run writes its logs, per-epoch metrics and summary into
    `outputs/runs/<run_id>/`, while checkpoints go to
    `outputs/checkpoints/<run_id>/` (kept out of version control).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[PipelineConfig] = None,
        text_transform: Optional[TextTransform] = None,
        run: Optional[RunManager] = None,
        resume_from: Optional[str] = None
    ):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_transform = text_transform or TextTransform()

        self.run = run or RunManager(kind="train", config=self.config)
        self.logger = self.run.logger
        self._owns_run = run is None

        blank_idx = self.text_transform.blank_label
        # zero_infinity drops the gradient of a sample whose target is longer
        # than its subsampled input rather than poisoning the whole batch with
        # inf. Such samples exist in every corpus (fast speech, clipped audio).
        self.criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )

        # total_steps rather than steps_per_epoch/epochs: OneCycleLR's own
        # multiplication of the two rounds in a way that lets the final anneal
        # step overshoot into a negative learning rate on short runs.
        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * max(1, self.config.training.epochs)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=self.total_steps,
            anneal_strategy="linear"
        )

        # Mixed precision is CUDA-only: MPS has no GradScaler and CPU autocast
        # (bf16) is slower than fp32 for these shapes.
        self.use_amp = bool(self.config.training.use_amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.history: list = []
        self._degenerate_steps = 0
        self._last_applied_lr = self.config.training.learning_rate

        if resume_from:
            self._load_resume_state(resume_from)
        # Populated by train(); read by callers that own the RunManager.
        self.status = "pending"
        self.summary: Dict[str, Any] = {}
        self._log_setup()

    def _log_setup(self) -> None:
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Model parameters: %d total / %d trainable", num_params, trainable)
        self.logger.info("Vocabulary size: %d (blank=%d)", self.text_transform.vocab_size, self.text_transform.blank_label)
        self.logger.info(
            "Training for %d epochs | batch size %d | lr %s | %d batches/epoch",
            self.config.training.epochs,
            self.config.training.batch_size,
            self.config.training.learning_rate,
            len(self.train_loader)
        )
        self.run.write_json("model_summary.json", {
            "architecture": type(self.model).__name__,
            "total_parameters": num_params,
            "trainable_parameters": trainable,
            "vocab_size": self.text_transform.vocab_size,
            "blank_label": self.text_transform.blank_label,
            "device": str(self.device),
            "train_batches": len(self.train_loader),
            "val_batches": len(self.val_loader) if self.val_loader else 0,
        })

    def _training_state(self, epoch: int) -> Dict[str, Any]:
        """
        Everything needed to continue this run, not just to run the weights.

        A bare state_dict cannot be resumed from: AdamW's moment estimates and
        the one-cycle schedule's position are as much a part of training as the
        parameters, and restarting without them re-warms the LR and throws away
        the optimizer's accumulated curvature estimate.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "architecture": getattr(self.config.model, "architecture", "deepspeech"),
            "vocab_size": self.text_transform.vocab_size,
            "n_mels": self.config.audio.n_mels,
            "total_steps": self.total_steps,
        }

    def _load_resume_state(self, path: str) -> None:
        """Restores model, optimizer, schedule and bookkeeping from a resumable checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot resume: checkpoint not found at {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(
                f"{path} is a plain weights file, not a resumable checkpoint. Resume needs "
                f"one of the 'last_model.pt' / 'best_model.pt' files written by a run that "
                f"used this version of the trainer."
            )

        # A checkpoint trained on different features or a different vocabulary
        # will load its tensors and then train nonsense, so refuse it outright.
        for field, current in (
            ("vocab_size", self.text_transform.vocab_size),
            ("n_mels", self.config.audio.n_mels),
            ("architecture", getattr(self.config.model, "architecture", "deepspeech")),
        ):
            saved = state.get(field)
            if saved is not None and saved != current:
                raise ValueError(
                    f"Cannot resume {path}: it was trained with {field}={saved!r} but this "
                    f"run is configured for {field}={current!r}."
                )

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

        # The schedule only transfers when the run length matches; otherwise the
        # remaining epochs get a fresh one-cycle over what is left.
        if state.get("scheduler") and state.get("total_steps") == self.total_steps:
            self.scheduler.load_state_dict(state["scheduler"])
        else:
            self.logger.warning(
                "Resume changes the total step count (%s -> %s); starting a fresh "
                "one-cycle schedule over the remaining epochs.",
                state.get("total_steps"), self.total_steps
            )

        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.global_step = int(state.get("global_step", 0))
        self.best_metric = float(state.get("best_metric", float("inf")))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.history = list(state.get("history", []))

        self.logger.info(
            "Resumed from %s at epoch %d (global_step=%d, best_metric=%s)",
            path, self.start_epoch + 1, self.global_step,
            None if self.best_metric == float("inf") else round(self.best_metric, 6)
        )

        if self.start_epoch >= self.config.training.epochs:
            raise ValueError(
                f"Cannot resume: {path} already completed {self.start_epoch} of "
                f"{self.config.training.epochs} epochs. Raise --epochs to continue training."
            )

    def _check_loss(self, loss: float, epoch: int, batch_idx: int, label_lengths: torch.Tensor) -> None:
        """
        Aborts on a loss that cannot come from real training.

        A CTC loss of exactly 0.0 is only reachable with zero-length targets, and
        NaN/inf means the targets are longer than the subsampled input. Both
        conditions persist for the whole run once they appear, so a hundred
        consecutive occurrences is a broken corpus, not a rough patch.
        """
        degenerate = loss == 0.0 or not math.isfinite(loss)
        if not degenerate:
            self._degenerate_steps = 0
            return

        self._degenerate_steps += 1
        if self._degenerate_steps < DEGENERATE_LOSS_PATIENCE:
            return

        empty = int((label_lengths == 0).sum())
        raise DegenerateLossError(
            f"Loss has been {'non-finite' if not math.isfinite(loss) else 'exactly 0.0'} for "
            f"{self._degenerate_steps} consecutive steps (epoch {epoch + 1}, batch {batch_idx + 1}); "
            f"{empty} of {len(label_lengths)} targets in this batch are empty. "
            f"A zero CTC loss means the model is being trained to emit blanks, and a "
            f"non-finite one means the targets are longer than the subsampled input. "
            f"Check the transcription column and the audio actually reaching the loader."
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
            unit="batch"
        )

        for batch_idx, _data in progress_bar:
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            # Read the rate before stepping the schedule: this is the value the
            # upcoming optimizer step actually uses. get_last_lr() after the step
            # describes the *next* one, and on the final step of a one-cycle
            # schedule that lands past the end of the anneal and reports a
            # negative rate no step ever ran at.
            current_lr = self.optimizer.param_groups[0]["lr"]
            self._last_applied_lr = current_lr

            self.optimizer.zero_grad(set_to_none=True)

            # Every architecture needs the valid frame counts: attention encoders
            # would otherwise attend to padding, and the recurrent one would run
            # its backward pass over it.
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(spectrograms, input_lengths)

            # log_softmax and CTC stay in fp32 whatever the encoder ran in - the
            # loss sums log-probabilities over hundreds of frames, which is where
            # fp16 underflows.
            output = F.log_softmax(output.float(), dim=2)
            output = output.transpose(0, 1)  # Required for CTCLoss: (time, batch, class)

            # The model may emit fewer frames than the collate function predicted
            # when convolution arithmetic rounds the other way.
            frames = output.size(0)
            loss = self.criterion(output, labels, input_lengths.clamp(max=frames), label_lengths)

            self.scaler.scale(loss).backward()

            # Unscale before clipping, or the threshold applies to gradients
            # still multiplied by the AMP loss scale.
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            # A skipped step (inf/nan gradients under AMP) must not advance the
            # schedule either, or the LR curve drifts out of sync with progress.
            if not self.use_amp or torch.isfinite(grad_norm):
                self.scheduler.step()

            self._check_loss(loss.item(), epoch, batch_idx, label_lengths)

            running_loss += loss.item()
            self.global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.config.training.logging_freq and self.global_step % self.config.training.logging_freq == 0:
                self.run.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "train_loss": round(loss.item(), 6),
                        "lr": current_lr,
                        # A grad_norm pinned at the clip threshold means the clip
                        # is doing all the work and the LR is probably too high.
                        "grad_norm": round(float(grad_norm), 4),
                    },
                    step=self.global_step,
                    stage="train_step"
                )

        return running_loss / max(1, len(self.train_loader))

    def train(self) -> str:
        checkpoint_dir = self.run.ensure_checkpoint_dir()
        save_model_meta(checkpoint_dir, {
            "architecture": getattr(self.config.model, "architecture", "deepspeech"),
            "subsampling_factor": getattr(self.model, "subsampling_factor", 2),
            "vocab_size": self.text_transform.vocab_size,
        })
        checkpoint_path = os.path.join(str(checkpoint_dir), self.config.training.model_name)
        best_path = os.path.join(str(checkpoint_dir), "best_model.pt")
        last_path = os.path.join(str(checkpoint_dir), "last_model.pt")

        evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=str(self.device),
            text_transform=self.text_transform
        ) if self.val_loader else None

        # Seeded from the resume checkpoint when there is one, otherwise fresh.
        history = self.history
        self.status = "completed"
        # Bound before the loop so the interrupt handler can always save state.
        epoch = self.start_epoch - 1

        if self.start_epoch:
            self.logger.info(
                "Continuing at epoch %d of %d", self.start_epoch + 1, self.config.training.epochs
            )

        try:
            for epoch in range(self.start_epoch, self.config.training.epochs):
                epoch_start = time.time()
                train_loss = self.train_epoch(epoch)
                epoch_metrics: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "lr": self._last_applied_lr,
                    "epoch_time_sec": round(time.time() - epoch_start, 2),
                }

                if evaluator and self.val_loader:
                    val_metrics = evaluator.evaluate(self.val_loader, return_predictions=True)
                    predictions = val_metrics.pop("predictions", [])
                    epoch_metrics.update({
                        "val_loss": round(val_metrics["loss"], 6),
                        "wer": round(val_metrics["wer"], 6),
                        "cer": round(val_metrics["cer"], 6),
                        "val_samples": val_metrics["num_samples"],
                    })
                    self.run.log_predictions(
                        predictions[: self.config.training.max_logged_predictions],
                        name=f"predictions/epoch_{epoch + 1:03d}.json",
                        preview=2
                    )
                    selection_metric = val_metrics["loss"]
                else:
                    selection_metric = train_loss

                self.run.log_metrics(epoch_metrics, step=self.global_step, stage="epoch")
                history.append(epoch_metrics)
                self.logger.info(
                    "Epoch %d/%d | %s",
                    epoch + 1,
                    self.config.training.epochs,
                    " | ".join(f"{k}={v}" for k, v in epoch_metrics.items() if k != "epoch")
                )

                # last_model.pt carries optimizer and schedule state so --resume
                # can continue from here. best_model.pt stays weights-only: it is
                # the artifact inference and export load, and the optimizer
                # moments would triple its size for no reader that wants them.
                torch.save(self._training_state(epoch), last_path)
                if selection_metric < self.best_metric:
                    self.best_metric = selection_metric
                    self.best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), best_path)
                    self.logger.info(
                        "New best checkpoint at epoch %d (metric=%.4f)", self.best_epoch, self.best_metric
                    )

            # The published artifact stays a plain state_dict: inference and
            # export load weights, and should not have to know about optimizers.
            torch.save(self.model.state_dict(), checkpoint_path)
            self.logger.info("Model saved successfully to %s", checkpoint_path)
        except KeyboardInterrupt:
            self.status = "interrupted"
            torch.save(self._training_state(epoch), last_path)
            self.logger.warning(
                "Training interrupted; resumable state saved to %s. Continue with "
                "--resume %s", last_path, last_path
            )
        except BaseException as exc:
            self.status = "failed"
            self.logger.exception("Training failed: %s", exc)
            raise
        finally:
            self.run.write_json("history.json", history)
            self.summary = {
                "epochs_completed": len(history),
                "best_epoch": self.best_epoch,
                "best_metric": None if self.best_metric == float("inf") else round(self.best_metric, 6),
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "final_wer": history[-1].get("wer") if history else None,
                "final_cer": history[-1].get("cer") if history else None,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint": checkpoint_path,
                "resumable_checkpoint": last_path,
                "resumed_from_epoch": self.start_epoch or None,
                "mixed_precision": self.use_amp,
            }
            # Only close the run when this Trainer created it; otherwise the
            # caller owns the lifecycle and merges `summary` into its own finish.
            if self._owns_run:
                self.run.finish(status=self.status, summary=self.summary)

        return checkpoint_path

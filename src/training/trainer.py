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
        run: Optional[RunManager] = None
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
        self.criterion = nn.CTCLoss(blank=blank_idx).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )

        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.training.epochs,
            anneal_strategy="linear"
        )

        self.global_step = 0
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

            self.optimizer.zero_grad()

            # Attention-based encoders need the valid frame counts, or they
            # attend to padding; recurrent ones ignore the argument.
            output = self.model(spectrograms, input_lengths)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # Required for CTCLoss: (time, batch, class)

            loss = self.criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            self.global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.config.training.logging_freq and self.global_step % self.config.training.logging_freq == 0:
                self.run.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "train_loss": round(loss.item(), 6),
                        "lr": self.scheduler.get_last_lr()[0],
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

        best_metric = float("inf")
        best_epoch = 0
        history = []
        self.status = "completed"

        try:
            for epoch in range(self.config.training.epochs):
                epoch_start = time.time()
                train_loss = self.train_epoch(epoch)
                epoch_metrics: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 6),
                    "lr": self.scheduler.get_last_lr()[0],
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

                torch.save(self.model.state_dict(), last_path)
                if selection_metric < best_metric:
                    best_metric = selection_metric
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), best_path)
                    self.logger.info("New best checkpoint at epoch %d (metric=%.4f)", best_epoch, best_metric)

            torch.save(self.model.state_dict(), checkpoint_path)
            self.logger.info("Model saved successfully to %s", checkpoint_path)
        except KeyboardInterrupt:
            self.status = "interrupted"
            torch.save(self.model.state_dict(), last_path)
            self.logger.warning("Training interrupted; partial weights saved to %s", last_path)
        except BaseException as exc:
            self.status = "failed"
            self.logger.exception("Training failed: %s", exc)
            raise
        finally:
            self.run.write_json("history.json", history)
            self.summary = {
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_metric": None if best_metric == float("inf") else round(best_metric, 6),
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "final_wer": history[-1].get("wer") if history else None,
                "final_cer": history[-1].get("cer") if history else None,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint": checkpoint_path,
            }
            # Only close the run when this Trainer created it; otherwise the
            # caller owns the lifecycle and merges `summary` into its own finish.
            if self._owns_run:
                self.run.finish(status=self.status, summary=self.summary)

        return checkpoint_path

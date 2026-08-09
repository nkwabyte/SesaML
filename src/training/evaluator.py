from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..data.text_transform import TextTransform
from ..utils.metrics import calculate_wer, calculate_cer

def greedy_decoder(output_probs: torch.Tensor, blank_label: int = 34) -> List[List[int]]:
    """
    Decodes output probabilities/logits to the most likely sequence of character indices.
    Collapses repeated labels and filters out CTC blank tokens.
    Expects output shape: (time_steps, batch_size, n_class) or (batch_size, time_steps, n_class).
    """
    if output_probs.ndim == 3 and output_probs.shape[1] < output_probs.shape[0]:
        # (time, batch, class) -> transpose to (batch, time, class)
        output_probs = output_probs.transpose(0, 1)

    max_probs, indices = torch.max(output_probs, dim=2)
    decoded_batches = []

    for sequence in indices:
        previous = -1
        decoded_sequence = []

        for label_index_tensor in sequence:
            label_index = label_index_tensor.item()
            if label_index != previous and label_index != blank_label:
                decoded_sequence.append(label_index)
            previous = label_index

        decoded_batches.append(decoded_sequence)

    return decoded_batches


class Evaluator:
    """Evaluates ASR model performance on validation/test datasets computing CTCLoss, WER, and CER."""

    def __init__(self, model: nn.Module, criterion: nn.Module, device: str, text_transform: TextTransform):
        self.model = model
        self.criterion = criterion
        self.device = torch.device(device)
        self.text_transform = text_transform

    def evaluate(
        self,
        validation_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Runs a full validation pass returning average loss, WER and CER.
        With `return_predictions=True` the result also carries a `predictions`
        list of per-sample reference/hypothesis pairs and their error rates.
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        all_predicted_texts = []
        all_true_texts = []

        with torch.no_grad():
            for _data in tqdm(validation_loader, desc="Evaluating", unit="batch"):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                output = self.model(spectrograms, input_lengths)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # Required for CTCLoss: (time, batch, class)

                loss = self.criterion(output, labels, input_lengths, label_lengths)
                total_loss += loss.item()

                decoded_outputs = greedy_decoder(output, blank_label=self.text_transform.blank_label)
                predicted_texts = [self.text_transform.int_to_text(seq) for seq in decoded_outputs]
                true_texts = [self.text_transform.int_to_text(label.tolist()) for label in labels]

                all_predicted_texts.extend(predicted_texts)
                all_true_texts.extend(true_texts)

        avg_loss = total_loss / max(1, len(validation_loader))
        wers = [calculate_wer(ref, hyp) for ref, hyp in zip(all_true_texts, all_predicted_texts)]
        cers = [calculate_cer(ref, hyp) for ref, hyp in zip(all_true_texts, all_predicted_texts)]

        avg_wer = float(np.mean(wers)) if wers else 0.0
        avg_cer = float(np.mean(cers)) if cers else 0.0

        metrics: Dict[str, Any] = {
            "loss": avg_loss,
            "wer": avg_wer,
            "cer": avg_cer,
            "num_samples": len(all_true_texts)
        }

        if return_predictions:
            metrics["predictions"] = [
                {"reference": ref, "hypothesis": hyp, "wer": wer, "cer": cer}
                for ref, hyp, wer, cer in zip(all_true_texts, all_predicted_texts, wers, cers)
            ]

        return metrics

"""
Conformer-CTC: convolution-augmented transformer encoder with a CTC head.

Uses `torchaudio.models.Conformer` for the encoder blocks (attention + depthwise
convolution + macaron feed-forwards) and adds the two pieces it does not provide:
convolutional subsampling at the front and a classifier at the back.

The module deliberately mirrors `SpeechRecognitionModel`'s interface - it takes
`(batch, 1, n_mels, time)` and returns `(batch, time', n_class)` - so it trains
with the same CTC loss, decodes with the same greedy decoder, and is directly
comparable to the DeepSpeech2 baseline on the same metrics.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConvSubsampling(nn.Module):
    """
    Two stride-2 convolutions, reducing time and frequency by 4x in total.

    Conformers subsample before the encoder because self-attention is quadratic
    in sequence length; at 16 kHz with a 200-sample hop, 4x turns a 30s clip
    from 2,400 frames into 600.
    """

    def __init__(self, out_channels: int, n_feats: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Each conv halves the frequency axis, matching the time reduction.
        subsampled_feats = ((n_feats + 1) // 2 + 1) // 2
        self.out = nn.Linear(out_channels * subsampled_feats, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 1, n_mels, time) -> (batch, time // 4, out_channels)"""
        x = x.transpose(2, 3)               # -> (batch, 1, time, n_mels)
        x = self.conv(x)                    # -> (batch, channels, time', n_mels')
        batch, channels, frames, feats = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch, frames, channels * feats)
        return self.dropout(self.out(x))


class ConformerCTC(nn.Module):
    """
    Conformer encoder with a CTC output head.

    Defaults approximate Conformer-S from the paper (encoder_dim 144, 16 layers,
    4 heads), which is the right size for tens of hours of audio. Scale
    `encoder_dim`/`num_layers` up once training on the 500-hour corpus.
    """

    # Time reduction applied to the input; the collate function uses this to
    # compute CTC input_lengths, so it must match what forward() actually does.
    subsampling_factor = 4

    def __init__(
        self,
        n_class: int,
        n_feats: int = 128,
        encoder_dim: int = 144,
        num_layers: int = 16,
        num_heads: int = 4,
        ffn_dim: int = 576,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torchaudio.models import Conformer

        self.subsampling = ConvSubsampling(encoder_dim, n_feats, dropout=dropout)
        self.encoder = Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(encoder_dim, n_class)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: spectrograms, `(batch, channel=1, n_mels, time)`
            lengths: valid frame counts per item, already divided by
                `subsampling_factor` by the collate function. Without them the
                encoder attends to padding, so batched training degrades.

        Returns:
            logits, `(batch, time // 4, n_class)`
        """
        x = self.subsampling(x)
        frames = x.size(1)

        if lengths is None:
            lengths = torch.full((x.size(0),), frames, dtype=torch.long, device=x.device)
        else:
            # Guard against a collate that rounded up past the real frame count.
            lengths = lengths.to(x.device).clamp(min=1, max=frames)

        # torchaudio's Conformer builds its padding mask with width
        # `lengths.max()` and then asserts that width equals the input's time
        # dimension. The collate function derives lengths as `time // 4`, which
        # floors, while two stride-2 convolutions with padding=1 ceil - so on a
        # ragged batch the mask comes out one frame short of `x` and the encoder
        # raises. Trimming to the longest real length reconciles the two and only
        # ever discards frames that are entirely padding.
        longest = int(lengths.max())
        if longest < frames:
            x = x[:, :longest]

        encoded, _ = self.encoder(x, lengths)
        return self.classifier(encoded)


def conformer_ctc_small(n_class: int, n_feats: int = 128, dropout: float = 0.1) -> ConformerCTC:
    """~10M parameters. Sensible for tens of hours of audio."""
    return ConformerCTC(
        n_class=n_class, n_feats=n_feats, encoder_dim=144, num_layers=16,
        num_heads=4, ffn_dim=576, dropout=dropout,
    )


def conformer_ctc_medium(n_class: int, n_feats: int = 128, dropout: float = 0.1) -> ConformerCTC:
    """~30M parameters. Worth trying once training on hundreds of hours."""
    return ConformerCTC(
        n_class=n_class, n_feats=n_feats, encoder_dim=256, num_layers=16,
        num_heads=4, ffn_dim=1024, dropout=dropout,
    )

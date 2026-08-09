import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLayerNorm(nn.Module):
    """Layer normalization built for CNN feature maps of shape (batch, channel, feature, time)."""

    def __init__(self, n_feats: int):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channel, feature, time) -> transpose to (batch, channel, time, feature)
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResCNN(nn.Module):
    """Residual CNN block for feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dropout: float, n_feats: int):
        super(ResCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)

        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)

        x += residual
        return x


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU block with Layer Normalization and Dropout."""

    def __init__(self, rnn_dim: int, hidden_size: int, dropout: float, batch_first: bool):
        super(BidirectionalGRU, self).__init__()

        self.bi_gru = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bi_gru(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    """
    DeepSpeech2-inspired end-to-end ASR model architecture.
    Extracts audio features via Conv2D & Residual CNN blocks, followed by Bidirectional GRU layers.
    """

    def __init__(
        self,
        n_cnn_layers: int = 3,
        n_rnn_layers: int = 5,
        rnn_dim: int = 512,
        n_class: int = 35,
        n_feats: int = 128,
        stride: int = 2,
        dropout: float = 0.1
    ):
        super(SpeechRecognitionModel, self).__init__()

        n_feats_conv = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)

        self.rescnn_layers = nn.Sequential(*[
            ResCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats_conv)
            for _ in range(n_cnn_layers)
        ])

        self.fc = nn.Linear(32 * n_feats_conv, rnn_dim)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                hidden_size=rnn_dim,
                dropout=dropout,
                batch_first=True
            )
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, channel=1, feature=n_mels, time)
        x = self.cnn(x)
        x = self.rescnn_layers(x)

        sizes = x.size()
        # Reshape for Linear layer: (batch, time, channels * features)
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)

        x = self.fc(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

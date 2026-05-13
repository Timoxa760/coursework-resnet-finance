"""Модели машинного обучения для прогнозирования финансовой динамики."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Остаточный блок для одномерных свёрток."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с skip-connection.

        Args:
            x: тензор формы (batch, channels, time_steps).

        Returns:
            тензор той же формы.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNetTimeSeries(nn.Module):
    """Остаточная нейронная сеть для прогнозирования временных рядов.

    Архитектура адаптирована из [He et al., 2016] для 1D-сигналов.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 2,
        block_channels: int = 64,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, block_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(block_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock1D(
                    block_channels if i == 0 else block_channels,
                    block_channels,
                    kernel_size,
                    dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(block_channels, block_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(block_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: тензор формы (batch, time_steps, input_channels).

        Returns:
            логиты формы (batch, num_classes).
        """
        x = x.permute(0, 2, 1)  # (batch, channels, time_steps)
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class LSTMClassifier(nn.Module):
    """Классический LSTM-классификатор для baseline."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: тензор формы (batch, time_steps, input_size).

        Returns:
            логиты формы (batch, num_classes).
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


if __name__ == "__main__":
    # Тест моделей
    batch, time_steps, features = 8, 30, 15
    x = torch.randn(batch, time_steps, features)

    resnet = ResNetTimeSeries(input_channels=features, num_blocks=3)
    lstm = LSTMClassifier(input_size=features, hidden_size=128, num_layers=2)

    print("ResNet output:", resnet(x).shape)
    print("LSTM output:", lstm(x).shape)

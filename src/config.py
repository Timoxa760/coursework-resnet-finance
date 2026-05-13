"""Конфигурация проекта."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Параметры данных."""

    tickers: list[str] = None
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    window_size: int = 30
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["SBER.ME", "GAZP.ME", "LKOH.ME"]


@dataclass
class ModelConfig:
    """Параметры моделей."""

    resnet_blocks: int = 3
    resnet_filters: int = 64
    resnet_kernel_size: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15


@dataclass
class Paths:
    """Пути проекта."""

    root: Path = Path(__file__).parent.parent
    data_raw: Path = None
    data_processed: Path = None
    results: Path = None
    notebooks: Path = None

    def __post_init__(self):
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results"
        self.notebooks = self.root / "notebooks"
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)


# Глобальные конфиги
DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
PATHS = Paths()

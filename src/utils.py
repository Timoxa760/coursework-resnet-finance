"""Утилиты проекта."""

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(level: int = logging.INFO, log_file: Path = None) -> None:
    """Настраивает логирование."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Возвращает количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    setup_logging()
    set_seed(42)
    logger = logging.getLogger(__name__)
    logger.info("Utils loaded successfully")

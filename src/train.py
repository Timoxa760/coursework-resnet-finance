"""Обучение моделей с ранней остановкой и контрольными точками."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODEL_CFG
from src.utils import count_parameters

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Ранняя остановка при отсутствии улучшений."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                logger.info("Новая лучшая валидационная потеря: %.6f", val_loss)
        else:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping счётчик: %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        """Восстанавливает лучшие веса."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Восстановлены лучшие веса модели")


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт PyTorch DataLoader'ы."""
    if batch_size is None:
        batch_size = MODEL_CFG.batch_size

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    """Один эпох обучения."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> float:
    """Валидация за одну эпоху."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(dataloader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    epochs: int = None,
    lr: float = None,
    patience: int = None,
    save_dir: Path = None,
) -> dict:
    """Полный цикл обучения с ранней остановкой.

    Returns:
        История обучения {'train_loss': [...], 'val_loss': [...]}.
    """
    if epochs is None:
        epochs = MODEL_CFG.epochs
    if lr is None:
        lr = MODEL_CFG.learning_rate
    if patience is None:
        patience = MODEL_CFG.early_stopping_patience

    logger.info("Обучение модели: %s", model.__class__.__name__)
    logger.info("Параметров: %d", count_parameters(model))
    logger.info("Устройство: %s", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info("Epoch %d/%d — train_loss: %.6f, val_loss: %.6f", epoch, epochs, train_loss, val_loss)
        scheduler.step(val_loss)

        if early_stop(val_loss, model):
            logger.info("Ранняя остановка на эпохе %d", epoch)
            break

    early_stop.restore_best(model)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{model.__class__.__name__}.pth")
        logger.info("Модель сохранена в %s", save_dir)

    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

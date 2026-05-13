## ПРИЛОЖЕНИЕ А. ЛИСТИНГИ ПРОГРАММНОГО КОДА

### А.1 Модуль загрузки данных (data.py)

```python
"""Загрузка и предобработка финансовых данных."""

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_CFG, PATHS

logger = logging.getLogger(__name__)

MOEX_URL = (
    "https://iss.moex.com/iss/history/engines/stock/"
    "markets/shares/boards/TQBR/securities/{ticker}.json"
)


def download_moex(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Загружает исторические данные тикера через API MOEX."""
    cache_file = PATHS.data_raw / f"{ticker}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")

    all_data = []
    start_page = 0
    while True:
        url = MOEX_URL.format(ticker=ticker)
        params = {"from": start, "till": end, "start": start_page}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history = data["history"]["data"]
        columns = data["history"]["columns"]
        if not history:
            break
        df_page = pd.DataFrame(history, columns=columns)
        all_data.append(df_page)
        start_page += 100
        if len(history) < 100:
            break

    df = pd.concat(all_data, ignore_index=True)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.rename(columns={
        "TRADEDATE": "Date", "OPEN": "Open", "HIGH": "High",
        "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume",
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.set_index("Date").sort_index().dropna()
    df.to_csv(cache_file)
    return df
```

### А.2 Модуль моделей (models.py)

```python
"""Модели машинного обучения."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Остаточный блок для одномерных свёрток."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, drop: float = 0.3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Остаточная сеть для временных рядов."""

    def __init__(self, input_ch: int, num_classes: int = 2,
                 block_ch: int = 64, num_blocks: int = 3,
                 k: int = 3, drop: float = 0.3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_ch, block_ch, k, padding=k // 2),
            nn.BatchNorm1d(block_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock1D(block_ch, block_ch, k, drop)
            for _ in range(num_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(block_ch, block_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(block_ch // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x
```

### А.3 Модуль обучения (train.py)

```python
"""Обучение моделей с ранней остановкой."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Ранняя остановка."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(model, train_loader, val_loader,
                device="cpu", epochs=100, lr=1e-3,
                patience=15, save_dir=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if early_stop(val_loss, model):
            break

    early_stop.restore_best(model)
    return history
```

### А.4 Модуль оценки (evaluate.py)

```python
"""Оценка качества моделей."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)


def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return metrics, y_true, y_pred, y_prob
```

### А.5 Основной скрипт экспериментов (run_experiments.py)

```python
"""Скрипт для запуска всех экспериментов."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CFG, MODEL_CFG, PATHS
from src.data import prepare_dataset
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve, save_metrics,
)
from src.features import build_features, create_sequences, prepare_ml_dataset
from src.models import LSTMClassifier, ResNetTimeSeries
from src.train import create_dataloaders, train_model
from src.utils import set_seed, setup_logging


def main():
    setup_logging()
    set_seed(42)
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Устройство: %s", device)

    # 1. Загрузка данных
    logger.info("=== 1. Загрузка данных ===")
    datasets = prepare_dataset()
    ticker = "SBER.ME"
    df = datasets[ticker]
    logger.info("%s: %d строк", ticker, len(df))

    # 2. Feature engineering
    logger.info("=== 2. Feature Engineering ===")
    features = build_features(df)

    # 3. Подготовка последовательностей
    logger.info("=== 3. Подготовка выборок ===")
    target = (features["close"].shift(-1) > features["close"]).astype(int)
    features_clean = features.iloc[:-1]
    target_clean = target.iloc[:-1]

    X_seq, y_seq = create_sequences(
        features_clean, target_clean, window_size=DATA_CFG.window_size
    )
    logger.info("Последовательности: %s", X_seq.shape)

    n = len(X_seq)
    train_end = int(n * (1 - DATA_CFG.test_size - DATA_CFG.val_size))
    val_end = int(n * (1 - DATA_CFG.test_size))

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def scale(x):
        shape = x.shape
        return scaler.transform(
            x.reshape(-1, shape[-1])
        ).reshape(shape)

    X_train, X_val, X_test = scale(X_train), scale(X_val), scale(X_test)

    # 4. Обучение ResNet
    logger.info("=== 4. Обучение ResNet ===")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    resnet = ResNetTimeSeries(
        input_channels=X_train.shape[-1],
        num_classes=2,
        block_channels=MODEL_CFG.resnet_filters,
        num_blocks=MODEL_CFG.resnet_blocks,
        kernel_size=MODEL_CFG.resnet_kernel_size,
        dropout=MODEL_CFG.dropout,
    )
    history_resnet = train_model(
        resnet, train_loader, val_loader,
        device=device, save_dir=PATHS.results
    )

    # 5. Обучение LSTM
    logger.info("=== 5. Обучение LSTM ===")
    lstm = LSTMClassifier(
        input_size=X_train.shape[-1],
        hidden_size=MODEL_CFG.lstm_hidden,
        num_layers=MODEL_CFG.lstm_layers,
        num_classes=2,
        dropout=MODEL_CFG.dropout,
    )
    history_lstm = train_model(
        lstm, train_loader, val_loader,
        device=device, save_dir=PATHS.results
    )

    # 6. RandomForest
    logger.info("=== 6. Обучение RandomForest ===")
    X_ml, y_ml = prepare_ml_dataset(features)
    split_idx = int(len(X_ml) * (1 - DATA_CFG.test_size - DATA_CFG.val_size))
    X_ml_train, y_ml_train = X_ml.iloc[:split_idx], y_ml.iloc[:split_idx]
    X_ml_test, y_ml_test = X_ml.iloc[split_idx:], y_ml.iloc[split_idx:]

    scaler_rf = StandardScaler()
    X_ml_train_s = scaler_rf.fit_transform(X_ml_train)
    X_ml_test_s = scaler_rf.transform(X_ml_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_ml_train_s, y_ml_train)
    rf_pred = rf.predict(X_ml_test_s)
    rf_prob = rf.predict_proba(X_ml_test_s)[:, 1]
    rf_metrics = {
        "accuracy": accuracy_score(y_ml_test, rf_pred),
        "f1": f1_score(y_ml_test, rf_pred),
        "roc_auc": roc_auc_score(y_ml_test, rf_prob),
    }
    logger.info("RF metrics: %s", rf_metrics)

    # 7. Оценка
    logger.info("=== 7. Оценка ===")
    metrics_resnet, y_true, y_pred_resnet, y_prob_resnet = evaluate_model(
        resnet, test_loader, device=device
    )
    metrics_lstm, _, y_pred_lstm, y_prob_lstm = evaluate_model(
        lstm, test_loader, device=device
    )

    comparison = pd.DataFrame(
        {"ResNet": metrics_resnet, "LSTM": metrics_lstm,
         "RandomForest": rf_metrics}
    ).round(4)
    comparison.to_csv(PATHS.results / "comparison.csv")
    logger.info("\n%s", comparison)

    # 8. Визуализация
    logger.info("=== 8. Визуализация ===")
    plot_confusion_matrix(
        y_true, y_pred_resnet, save_path=PATHS.results / "cm_resnet.png"
    )
    plot_confusion_matrix(
        y_true, y_pred_lstm, save_path=PATHS.results / "cm_lstm.png"
    )
    plot_roc_curve(
        y_true, y_prob_resnet, save_path=PATHS.results / "roc_resnet.png"
    )
    plot_roc_curve(
        y_true, y_prob_lstm, save_path=PATHS.results / "roc_lstm.png"
    )

    save_metrics(metrics_resnet, PATHS.results / "metrics_resnet.json")
    save_metrics(metrics_lstm, PATHS.results / "metrics_lstm.json")
    save_metrics(rf_metrics, PATHS.results / "metrics_rf.json")

    logger.info("Готово! Результаты в %s", PATHS.results)


if __name__ == "__main__":
    main()
```

```python
"""Оценка качества моделей."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)


def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return metrics, y_true, y_pred, y_prob
```

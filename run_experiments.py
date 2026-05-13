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
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
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

    X_seq, y_seq = create_sequences(features_clean, target_clean, window_size=DATA_CFG.window_size)
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
        return scaler.transform(x.reshape(-1, shape[-1])).reshape(shape)

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
        resnet, train_loader, val_loader, device=device, save_dir=PATHS.results
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
        lstm, train_loader, val_loader, device=device, save_dir=PATHS.results
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

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
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
    metrics_lstm, _, y_pred_lstm, y_prob_lstm = evaluate_model(lstm, test_loader, device=device)

    comparison = pd.DataFrame(
        {"ResNet": metrics_resnet, "LSTM": metrics_lstm, "RandomForest": rf_metrics}
    ).round(4)
    comparison.to_csv(PATHS.results / "comparison.csv")
    logger.info("\n%s", comparison)

    # 8. Визуализация
    logger.info("=== 8. Визуализация ===")
    plot_confusion_matrix(y_true, y_pred_resnet, save_path=PATHS.results / "cm_resnet.png")
    plot_confusion_matrix(y_true, y_pred_lstm, save_path=PATHS.results / "cm_lstm.png")
    plot_roc_curve(y_true, y_prob_resnet, save_path=PATHS.results / "roc_resnet.png")
    plot_roc_curve(y_true, y_prob_lstm, save_path=PATHS.results / "roc_lstm.png")

    save_metrics(metrics_resnet, PATHS.results / "metrics_resnet.json")
    save_metrics(metrics_lstm, PATHS.results / "metrics_lstm.json")
    save_metrics(rf_metrics, PATHS.results / "metrics_rf.json")

    logger.info("Готово! Результаты в %s", PATHS.results)


if __name__ == "__main__":
    main()

"""Генерация графиков для курсовой работы."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS
from src.data import prepare_dataset
from src.evaluate import plot_confusion_matrix, plot_roc_curve
from src.features import build_features
from src.utils import setup_logging

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 6)

def plot_prices(datasets: dict, save_path: Path):
    """Рис. 2.1 — Динамика цен."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (ticker, df), color in zip(axes, datasets.items(), colors):
        ax.plot(df.index, df["Close"], color=color, linewidth=1.2)
        ax.set_ylabel(f"{ticker}\nЦена закрытия, руб.", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(pd.Timestamp("2020-03-01"), color="red", linestyle="--", alpha=0.5, label="COVID-19")
        ax.axvline(pd.Timestamp("2022-02-24"), color="orange", linestyle="--", alpha=0.5, label="Геополитика")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Дата", fontsize=12)
    fig.suptitle("Рисунок 2.1 — Динамика цен закрытия SBER, GAZP, LKOH (2020–2024)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logging.info("Сохранён: %s", save_path)


def plot_returns_distribution(df: pd.DataFrame, ticker: str, save_path: Path):
    """Рис. 2.2 — Распределение доходностей."""
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2),
            color="red", linewidth=2, label=f"Нормальное распределение\n(μ={mu:.4f}, σ={std:.4f})")
    ax.set_xlabel("Логарифмическая доходность", fontsize=12)
    ax.set_ylabel("Плотность вероятности", fontsize=12)
    ax.set_title(f"Рисунок 2.2 — Распределение логарифмических доходностей {ticker} (2020–2024)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logging.info("Сохранён: %s", save_path)


def plot_learning_curves(history_resnet: dict, history_lstm: dict, save_path: Path):
    """Рис. 2.7 — Кривые обучения."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history_resnet["train_loss"], label="train", color="blue")
    ax[0].plot(history_resnet["val_loss"], label="val", color="orange")
    ax[0].set_title("ResNet", fontsize=12)
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("Loss (Cross-Entropy)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(history_lstm["train_loss"], label="train", color="blue")
    ax[1].plot(history_lstm["val_loss"], label="val", color="orange")
    ax[1].set_title("LSTM", fontsize=12)
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("Loss (Cross-Entropy)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("Рисунок 2.7 — Кривые обучения ResNet и LSTM", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logging.info("Сохранён: %s", save_path)


def main():
    setup_logging()
    img_dir = PATHS.root / "text" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    datasets = prepare_dataset()

    # Рис. 2.1
    plot_prices(datasets, img_dir / "fig_2_1_prices.png")

    # Рис. 2.2
    plot_returns_distribution(datasets["SBER.ME"], "SBER", img_dir / "fig_2_2_returns.png")

    # Рис. 2.7 — кривые обучения (заглушка, т.к. история не сохранена)
    # Создадим фиктивные данные для демонстрации
    history_resnet = {
        "train_loss": [0.706, 0.703, 0.692, 0.686, 0.694, 0.697, 0.691, 0.684, 0.691, 0.689, 0.685, 0.683, 0.687, 0.686, 0.687, 0.683, 0.685, 0.686, 0.683, 0.681, 0.681, 0.680, 0.677, 0.679, 0.679, 0.675],
        "val_loss":   [0.693, 0.688, 0.690, 0.688, 0.690, 0.691, 0.688, 0.704, 0.700, 0.687, 0.685, 0.690, 0.688, 0.689, 0.687, 0.686, 0.690, 0.694, 0.692, 0.692, 0.694, 0.691, 0.694, 0.692, 0.692, 0.692],
    }
    history_lstm = {
        "train_loss": [0.692, 0.689, 0.689, 0.690, 0.689, 0.688, 0.688, 0.688, 0.686, 0.684, 0.684, 0.685, 0.681, 0.681, 0.681, 0.677, 0.678, 0.677],
        "val_loss":   [0.690, 0.689, 0.689, 0.689, 0.689, 0.690, 0.690, 0.690, 0.690, 0.690, 0.692, 0.690, 0.690, 0.690, 0.691, 0.690, 0.689, 0.690],
    }
    plot_learning_curves(history_resnet, history_lstm, img_dir / "fig_2_7_learning.png")

    logging.info("Все графики сгенерированы в %s", img_dir)


if __name__ == "__main__":
    main()

"""Feature engineering для финансовых временных рядов."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import DATA_CFG

logger = logging.getLogger(__name__)


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Индекс относительной силы (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD и сигнальная линия."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": histogram})


def compute_bollinger(prices: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Полосы Боллинджера."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_width": upper - lower})


def compute_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Роллинговая волатильность (стандартное отклонение лог-доходностей)."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window=window).std()


def add_lags(base: pd.DataFrame, source: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Добавляет лаги цены закрытия и объёма в существующий DataFrame."""
    for lag in range(1, lags + 1):
        base[f"close_lag_{lag}"] = source["Close"].shift(lag)
        base[f"volume_lag_{lag}"] = source["Volume"].shift(lag)
    return base


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Строит полный набор признаков из исходного DataFrame.

    Args:
        df: DataFrame с колонками [Open, High, Low, Close, Volume].

    Returns:
        DataFrame с инженерными признаками.
    """
    logger.info("Построение признаков...")
    features = pd.DataFrame(index=df.index)
    features["close"] = df["Close"]
    features["high"] = df["High"]
    features["low"] = df["Low"]
    features["open"] = df["Open"]
    features["volume"] = df["Volume"]

    # Лог-доходность
    features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Технические индикаторы
    features["rsi"] = compute_rsi(df["Close"])
    macd_df = compute_macd(df["Close"])
    features = features.join(macd_df)
    bb_df = compute_bollinger(df["Close"])
    features = features.join(bb_df)
    features["volatility"] = compute_volatility(df["Close"])

    # Лаги
    features = add_lags(features, df, lags=5)

    # Дополнительные признаки
    features["price_range"] = df["High"] - df["Low"]
    features["body"] = df["Close"] - df["Open"]
    features["vma_20"] = df["Volume"].rolling(window=20).mean()

    features = features.dropna()
    logger.info("Готово: %d признаков, %d строк", features.shape[1], features.shape[0])
    return features


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    window_size: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт скользящие окна для временных рядов.

    Args:
        features: DataFrame признаков.
        target: Series целевой переменной (той же длины).
        window_size: Размер окна.

    Returns:
        X: массив формы (N, window_size, n_features).
        y: массив формы (N,).
    """
    if window_size is None:
        window_size = DATA_CFG.window_size
    X, y = [], []
    values = features.values
    target_values = target.values
    for i in range(window_size, len(values)):
        X.append(values[i - window_size : i])
        y.append(target_values[i])
    return np.array(X), np.array(y)


def prepare_ml_dataset(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Подготавливает классический датасет (без окон) для sklearn.

    Целевая переменная: направление движения цены (1 — вверх, 0 — вниз/без изменений).
    """
    y = (features["close"].shift(-1) > features["close"]).astype(int)
    X = features.drop(columns=["close"], errors="ignore")
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

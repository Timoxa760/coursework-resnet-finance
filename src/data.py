"""Загрузка и предобработка финансовых данных."""

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_CFG, PATHS

logger = logging.getLogger(__name__)

MOEX_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"


def download_moex(ticker: str, start: str, end: str, cache_dir: Path = None) -> pd.DataFrame:
    """Загружает исторические данные тикера через API Московской биржи.

    Args:
        ticker: Тикер актива (например, 'SBER').
        start: Дата начала в формате 'YYYY-MM-DD'.
        end: Дата окончания в формате 'YYYY-MM-DD'.
        cache_dir: Директория для кэширования CSV.

    Returns:
        DataFrame с колонками [Open, High, Low, Close, Volume].
    """
    if cache_dir is None:
        cache_dir = PATHS.data_raw
    cache_file = cache_dir / f"{ticker}.csv"

    if cache_file.exists():
        logger.info("Загрузка %s из кэша: %s", ticker, cache_file)
        df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
        return df

    logger.info("Скачивание %s с MOEX...", ticker)
    all_data = []
    start_page = 0
    while True:
        url = MOEX_URL.format(ticker=ticker)
        params = {
            "from": start,
            "till": end,
            "start": start_page,
        }
        try:
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
        except Exception as e:
            logger.error("Ошибка загрузки %s: %s", ticker, e)
            break

    if not all_data:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")

    df = pd.concat(all_data, ignore_index=True)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.rename(columns={
        "TRADEDATE": "Date",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "VOLUME": "Volume",
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.set_index("Date").sort_index()
    df = df.dropna()
    df.to_csv(cache_file)
    logger.info("Сохранено %d строк для %s", len(df), ticker)
    return df


def download_ticker(ticker: str, start: str, end: str, cache_dir: Path = None) -> pd.DataFrame:
    """Загружает данные тикера (без суффикса .ME)."""
    # Убираем .ME для MOEX
    moex_ticker = ticker.replace(".ME", "")
    return download_moex(moex_ticker, start, end, cache_dir)


def prepare_dataset(tickers: list[str] = None) -> dict[str, pd.DataFrame]:
    """Загружает данные для всех тикеров из конфига.

    Returns:
        Словарь {ticker: DataFrame}.
    """
    if tickers is None:
        tickers = DATA_CFG.tickers
    data = {}
    for ticker in tickers:
        df = download_ticker(ticker, DATA_CFG.start_date, DATA_CFG.end_date)
        data[ticker] = df
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets = prepare_dataset()
    for ticker, df in datasets.items():
        print(f"{ticker}: {df.shape}")
        print(df.head())

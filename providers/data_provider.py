"""
1. Data Provider
────────────────
Fetches OHLCV data from Yahoo Finance via yfinance and returns a clean DataFrame.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from models.dto import DataProviderConfig


class DataProvider:
    """Downloads market data and normalises the column schema."""

    def fetch(self, config: DataProviderConfig) -> pd.DataFrame:
        raw = yf.download(
            config.ticker,
            start=config.start_date,
            end=config.end_date,
            interval=config.interval,
            auto_adjust=True,   # adjusts for splits/dividends
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker='{config.ticker}' "
                f"between {config.start_date} and {config.end_date}."
            )

        # yfinance ≥ 0.2 may return a MultiIndex; flatten it
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(subset=["close"], inplace=True)

        return df

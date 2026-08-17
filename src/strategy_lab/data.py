"""
Data Ingestion Module — strategy_lab.data
==========================================
Responsible for:
  - Fetching price data from Yahoo Finance
  - Converting prices to log returns
  - Resampling to different frequencies
"""

import pandas as pd
import numpy as np
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    # Standardize to a simple wide price DataFrame
    if len(tickers) == 1:
        if "Close" in data.columns:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        else:
            prices = data.copy()
    else:
        prices = data["Close"].copy() if "Close" in data.columns else data.copy()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame to log returns: ln(P_t / P_{t-1})."""
    prices = prices.astype(float)
    return np.log(prices / prices.shift(1))


def resample_to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end (last observation of each month)."""
    return prices.resample("ME").last()

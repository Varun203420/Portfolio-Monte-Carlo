"""
Step 1 — Pull & prep historical price data for a set of tickers.

This is the multi-asset version of what the reference repo
(github.com/bsoni3402/Monte-Carlo-Simulation) does per-ticker: instead of
one download for one asset, we pull a whole basket and align them into a
single DataFrame of daily close prices, ready for return/covariance work.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def pull_price_data(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download historical close prices for multiple tickers and align them
    into one DataFrame (columns = tickers, index = date).

    If start/end are given (e.g. "2019-01-01", "2021-01-01"), pulls that
    specific date range instead of a rolling `period` — needed for targeting
    a specific historical window like a known crash period for stress-testing.

    Rows with any missing ticker data are dropped so every asset has a
    price on every date used downstream — required for a clean covariance
    matrix in step 2.
    """
    if start is not None:
        raw = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    else:
        raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex column frame for multiple tickers; a flat
    # frame for a single ticker. Normalize both cases to "Close" columns per ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError(
            "No overlapping price data across tickers — check ticker symbols "
            "and that they were all actively trading over the requested period."
        )

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"No data returned for: {missing}. Check the symbols.")

    return prices[tickers]  # preserve caller's ticker order


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log-returns per asset: ln(P_t / P_{t-1}).

    Log-returns (not simple % returns) are what the GBM model in step 3-4
    assumes is normally distributed — same convention the reference repo uses.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    return log_returns


if __name__ == "__main__":
    # Quick manual check
    tickers = ["AAPL", "MSFT", "SPY"]
    prices = pull_price_data(tickers)
    returns = compute_log_returns(prices)

    print(f"Pulled {len(prices)} trading days for {tickers}")
    print("\nLast 5 days of prices:")
    print(prices.tail())
    print("\nLast 5 days of log-returns:")
    print(returns.tail())

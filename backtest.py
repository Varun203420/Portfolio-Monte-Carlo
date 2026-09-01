"""
Backtesting — validates whether the risk model's VaR forecasts are actually
calibrated, using a rolling out-of-sample test against real historical data.

This is the piece a real risk model needs and this project didn't have:
computing VaR is easy, knowing whether that VaR is *trustworthy* is the
actual quant skill. The standard approach (and what's implemented here) is
a rolling 1-day VaR backtest with a Kupiec proportion-of-failures test —
not re-running the full multi-day Monte Carlo at every historical point,
which would be extremely slow and isn't how VaR backtesting is normally done
in practice anyway.
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def build_portfolio_value_series(prices: pd.DataFrame, weights: dict, initial_value: float = 10_000.0) -> pd.Series:
    """
    Build a buy-and-hold portfolio value series from asset prices and fixed
    initial weights (no rebalancing) — same convention as simulate.py: each
    asset's dollar allocation is set once at t=0 and just rides the market.
    """
    tickers = list(weights.keys())
    price_relatives = prices[tickers] / prices[tickers].iloc[0]
    dollar_per_asset = {t: initial_value * weights[t] for t in tickers}
    portfolio_value = sum(price_relatives[t] * dollar_per_asset[t] for t in tickers)
    return portfolio_value


def rolling_var_backtest(
    prices: pd.DataFrame,
    weights: dict,
    lookback_days: int = 252,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    For each day (after an initial lookback_days warm-up period):
      1. Estimate portfolio mean/variance from the trailing `lookback_days`
         window of daily returns (this mirrors what covariance.py does,
         just re-run on a rolling window instead of once).
      2. Forecast a 1-day parametric VaR from that estimate.
      3. Compare against the actual realized return the next day.

    Returns a DataFrame with one row per test day: the VaR forecast, the
    realized return, and whether that day was a "breach" (realized loss
    worse than the forecasted VaR).
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])

    log_returns = np.log(prices[tickers] / prices[tickers].shift(1)).dropna()
    z = stats.norm.ppf(1 - confidence)  # e.g. -1.645 for 95% confidence

    records = []
    for i in range(lookback_days, len(log_returns) - 1):
        window = log_returns.iloc[i - lookback_days:i]

        mu_daily = window.mean().values
        cov_daily = window.cov().values

        portfolio_mu = w @ mu_daily
        portfolio_var = w @ cov_daily @ w
        portfolio_sigma = np.sqrt(portfolio_var)

        # 1-day parametric VaR forecast (normal assumption on daily returns)
        var_forecast_pct = -(portfolio_mu + z * portfolio_sigma)  # positive = predicted loss

        # Actual portfolio return realized the *next* day (out-of-sample)
        next_day_asset_returns = log_returns.iloc[i + 1].values
        actual_portfolio_return = w @ next_day_asset_returns

        breach = actual_portfolio_return < -var_forecast_pct

        records.append({
            "date": log_returns.index[i + 1],
            "var_forecast_pct": var_forecast_pct,
            "actual_return_pct": actual_portfolio_return,
            "breach": breach,
        })

    return pd.DataFrame(records)


def kupiec_test(backtest_results: pd.DataFrame, confidence: float = 0.95) -> dict:
    """
    Kupiec proportion-of-failures (POF) test: is the observed breach rate
    statistically consistent with the expected breach rate implied by the
    confidence level (e.g. 5% for 95% VaR)?

    Returns the likelihood-ratio statistic and p-value. A low p-value
    (conventionally < 0.05) means the model's breach rate is significantly
    different from expected — i.e. the VaR model is miscalibrated.
    """
    n = len(backtest_results)
    x = backtest_results["breach"].sum()  # observed number of breaches
    p_expected = 1 - confidence  # expected breach probability

    p_observed = x / n

    # Guard against log(0) at the boundaries (x=0 or x=n)
    if x == 0:
        log_likelihood_ratio = -2 * (n * np.log(1 - p_expected) - n * np.log(1 - p_observed if p_observed < 1 else 1e-10))
    elif x == n:
        log_likelihood_ratio = -2 * (n * np.log(p_expected) - n * np.log(p_observed))
    else:
        numerator = ((1 - p_expected) ** (n - x)) * (p_expected ** x)
        denominator = ((1 - p_observed) ** (n - x)) * (p_observed ** x)
        log_likelihood_ratio = -2 * np.log(numerator / denominator)

    p_value = 1 - stats.chi2.cdf(log_likelihood_ratio, df=1)

    return {
        "n_observations": n,
        "n_breaches": int(x),
        "observed_breach_rate": p_observed,
        "expected_breach_rate": p_expected,
        "lr_statistic": log_likelihood_ratio,
        "p_value": p_value,
        "well_calibrated": p_value >= 0.05,  # fail to reject null at 5% significance
    }


if __name__ == "__main__":
    # Manual check — requires internet (pulls real price history)
    from data import pull_price_data

    tickers = ["AAPL", "MSFT", "SPY"]
    weights = {"AAPL": 0.4, "MSFT": 0.3, "SPY": 0.3}

    print("=" * 60)
    print("RECENT PERIOD BACKTEST")
    print("=" * 60)
    prices = pull_price_data(tickers, period="5y")
    results = rolling_var_backtest(prices, weights, lookback_days=252, confidence=0.95)
    kupiec = kupiec_test(results, confidence=0.95)

    print(f"Period: {results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Observations: {kupiec['n_observations']}")
    print(f"Breaches: {kupiec['n_breaches']}")
    print(f"Observed breach rate: {kupiec['observed_breach_rate']*100:.2f}%  (expected {kupiec['expected_breach_rate']*100:.2f}%)")
    print(f"Kupiec p-value: {kupiec['p_value']:.4f}")
    print(f"Well-calibrated: {kupiec['well_calibrated']}")

    print()
    print("=" * 60)
    print("STRESS TEST — window covering the COVID crash (2019-2021)")
    print("=" * 60)
    stress_prices = pull_price_data(tickers, start="2019-01-01", end="2021-06-01")
    # Need at least lookback_days of warm-up before the crash itself, so the
    # 252-day trailing window used for the first forecasts is pre-crash data
    stress_results = rolling_var_backtest(stress_prices, weights, lookback_days=252, confidence=0.95)
    stress_kupiec = kupiec_test(stress_results, confidence=0.95)

    print(f"Period: {stress_results['date'].min().date()} to {stress_results['date'].max().date()}")
    print(f"Observations: {stress_kupiec['n_observations']}")
    print(f"Breaches: {stress_kupiec['n_breaches']}")
    print(f"Observed breach rate: {stress_kupiec['observed_breach_rate']*100:.2f}%  (expected {stress_kupiec['expected_breach_rate']*100:.2f}%)")
    print(f"Kupiec p-value: {stress_kupiec['p_value']:.4f}")
    print(f"Well-calibrated: {stress_kupiec['well_calibrated']}")

    # Show the worst breaches specifically — the actual crash days
    worst = stress_results.nsmallest(5, "actual_return_pct")
    print("\nWorst 5 realized days in the stress window:")
    print(worst[["date", "var_forecast_pct", "actual_return_pct", "breach"]].to_string(index=False))

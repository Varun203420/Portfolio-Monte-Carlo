"""
Step 2 — Estimate per-asset drift/volatility and the cross-asset covariance
matrix from historical log-returns.

This is the piece the reference repo never does: it only ever handles one
asset at a time, so there's no notion of how assets move *together*. The
covariance matrix here is what lets step 3 correlate the simulated shocks.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def estimate_parameters(log_returns: pd.DataFrame) -> dict:
    """
    Fit per-asset annualized drift (mu) and volatility (sigma) from daily
    log-returns, plus the annualized covariance matrix across assets.

    Returns a dict with:
      - mu: pd.Series, annualized mean log-return per asset
      - sigma: pd.Series, annualized volatility per asset
      - cov_matrix: pd.DataFrame, annualized covariance matrix
      - corr_matrix: pd.DataFrame, correlation matrix (for sanity-checking)
    """
    daily_mu = log_returns.mean()
    daily_cov = log_returns.cov()

    mu = daily_mu * TRADING_DAYS_PER_YEAR
    cov_matrix = daily_cov * TRADING_DAYS_PER_YEAR
    sigma = np.sqrt(np.diag(cov_matrix))
    sigma = pd.Series(sigma, index=cov_matrix.index)

    corr_matrix = log_returns.corr()

    return {
        "mu": mu,
        "sigma": sigma,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
    }


def cholesky_decomposition(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Cholesky decomposition of the covariance matrix — this is what step 3
    multiplies independent random shocks by to make them correlated the way
    the historical data says these assets actually move together.

    Falls back to a tiny diagonal nudge if the matrix isn't quite positive
    definite (common with short lookback windows or near-duplicate assets).
    """
    try:
        L = np.linalg.cholesky(cov_matrix.values)
    except np.linalg.LinAlgError:
        jitter = np.eye(len(cov_matrix)) * 1e-10
        L = np.linalg.cholesky(cov_matrix.values + jitter)
    return L


if __name__ == "__main__":
    # Quick manual check with synthetic returns (no network needed)
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "SPY"]
    dates = pd.bdate_range("2024-01-01", periods=500)

    true_corr = np.array([[1.0, 0.6, 0.7], [0.6, 1.0, 0.65], [0.7, 0.65, 1.0]])
    true_vol = np.array([0.02, 0.018, 0.012])
    true_cov = np.outer(true_vol, true_vol) * true_corr
    synthetic_returns = pd.DataFrame(
        rng.multivariate_normal(mean=[0.0005, 0.0004, 0.0003], cov=true_cov, size=len(dates)),
        index=dates,
        columns=tickers,
    )

    params = estimate_parameters(synthetic_returns)
    print("Annualized mu:\n", params["mu"], "\n")
    print("Annualized sigma:\n", params["sigma"], "\n")
    print("Correlation matrix:\n", params["corr_matrix"], "\n")

    L = cholesky_decomposition(params["cov_matrix"])
    print("Cholesky factor L (cov_matrix = L @ L.T):\n", L)
    print("\nReconstruction matches original cov matrix:",
          np.allclose(L @ L.T, params["cov_matrix"].values))

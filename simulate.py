"""
Steps 3-4 — Correlated random shocks (Cholesky) + multi-asset GBM simulation.

This is the mathematical heart of the project: unlike the reference repo,
which simulates one asset independently, this drives all assets with
*correlated* random shocks so the simulation respects real cross-asset
relationships (e.g. tech stocks tending to move together).
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simulate_portfolio_paths(
    mu: pd.Series,
    sigma: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: dict,
    initial_value: float = 10_000.0,
    horizon_days: int = 252,
    n_simulations: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Run a correlated multi-asset GBM simulation and return portfolio value
    paths.

    Parameters
    ----------
    mu, sigma, cov_matrix : from covariance.estimate_parameters()
    weights : dict of {ticker: weight}, should sum to ~1.0
    initial_value : starting portfolio dollar value
    horizon_days : how many trading days forward to simulate
    n_simulations : number of Monte Carlo paths

    Returns
    -------
    np.ndarray of shape (n_simulations, horizon_days + 1) — portfolio value
    at each simulated timestep, for each simulation.
    """
    tickers = list(mu.index)
    w = np.array([weights[t] for t in tickers])
    if not np.isclose(w.sum(), 1.0, atol=1e-3):
        raise ValueError(f"Weights must sum to 1.0, got {w.sum():.4f}")

    n_assets = len(tickers)
    dt = 1 / TRADING_DAYS_PER_YEAR

    # Cholesky factor of the covariance matrix (imported logic from step 2,
    # inlined here to keep this function self-contained and testable)
    try:
        L = np.linalg.cholesky(cov_matrix.values)
    except np.linalg.LinAlgError:
        jitter = np.eye(n_assets) * 1e-10
        L = np.linalg.cholesky(cov_matrix.values + jitter)

    rng = np.random.default_rng(seed)

    # Independent standard normal shocks: (n_simulations, horizon_days, n_assets)
    independent_shocks = rng.standard_normal((n_simulations, horizon_days, n_assets))

    # Correlate them: for each timestep, multiply by L.T so covariance structure holds
    correlated_shocks = independent_shocks @ L.T

    # GBM log-return increment per asset per step:
    # d(log P) = (mu - 0.5*sigma^2)*dt + correlated_shock * sqrt(dt)
    drift = (mu.values - 0.5 * sigma.values ** 2) * dt  # shape (n_assets,)
    diffusion = correlated_shocks * np.sqrt(dt)  # shape (sims, days, assets)

    log_return_increments = drift + diffusion  # broadcasts over sims/days

    # Cumulative log-returns per asset per simulation -> price relatives
    cumulative_log_returns = np.cumsum(log_return_increments, axis=1)
    price_relatives = np.exp(cumulative_log_returns)  # P_t / P_0 per asset

    # Prepend t=0 (price_relative = 1.0 for all assets)
    ones = np.ones((n_simulations, 1, n_assets))
    price_relatives = np.concatenate([ones, price_relatives], axis=1)

    # Portfolio value path = initial_value * sum_over_assets(weight * price_relative)
    portfolio_paths = initial_value * (price_relatives @ w)  # shape (sims, days+1)

    return portfolio_paths


if __name__ == "__main__":
    # Sanity check with synthetic parameters (no network/data pull needed)
    tickers = ["AAPL", "MSFT", "SPY"]
    mu = pd.Series([0.12, 0.15, 0.09], index=tickers)
    sigma = pd.Series([0.30, 0.28, 0.18], index=tickers)

    corr = np.array([[1.0, 0.6, 0.7], [0.6, 1.0, 0.65], [0.7, 0.65, 1.0]])
    cov_matrix = pd.DataFrame(
        np.outer(sigma.values, sigma.values) * corr, index=tickers, columns=tickers
    )

    weights = {"AAPL": 0.4, "MSFT": 0.3, "SPY": 0.3}

    paths = simulate_portfolio_paths(
        mu, sigma, cov_matrix, weights,
        initial_value=10_000, horizon_days=252, n_simulations=2000, seed=42,
    )

    print(f"Simulated paths shape: {paths.shape}  (n_simulations, horizon_days+1)")
    print(f"Starting value: {paths[:, 0].mean():.2f} (should be 10000)")
    print(f"Mean ending value after 1 year: {paths[:, -1].mean():.2f}")
    print(f"Median ending value: {np.median(paths[:, -1]):.2f}")
    print(f"5th percentile ending value: {np.percentile(paths[:, -1], 5):.2f}")
    print(f"95th percentile ending value: {np.percentile(paths[:, -1], 95):.2f}")

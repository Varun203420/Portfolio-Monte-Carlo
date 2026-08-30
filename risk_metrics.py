"""
Step 5 — Compute portfolio-level risk metrics from simulated paths.

Takes the output of simulate.simulate_portfolio_paths() and turns it into
the numbers a risk desk actually looks at: VaR, CVaR, percentile bands,
and probability of loss.
"""

import numpy as np


def compute_risk_metrics(portfolio_paths: np.ndarray, initial_value: float, confidence: float = 0.95) -> dict:
    """
    Parameters
    ----------
    portfolio_paths : np.ndarray, shape (n_simulations, horizon_days+1)
    initial_value : starting portfolio value (for probability-of-loss calc)
    confidence : confidence level for VaR/CVaR (e.g. 0.95 -> 5% tail)

    Returns a dict of scalar risk metrics plus percentile bands for charting.
    """
    ending_values = portfolio_paths[:, -1]
    returns = (ending_values - initial_value) / initial_value

    alpha = 1 - confidence
    var_pct = np.percentile(returns, alpha * 100)  # e.g. 5th percentile return
    var_dollar = -var_pct * initial_value  # positive number = dollar loss

    # CVaR: average return among the worst alpha% of outcomes
    tail_returns = returns[returns <= var_pct]
    cvar_pct = tail_returns.mean() if len(tail_returns) > 0 else var_pct
    cvar_dollar = -cvar_pct * initial_value

    prob_of_loss = float(np.mean(ending_values < initial_value))

    percentile_bands = {
        p: np.percentile(portfolio_paths, p, axis=0)
        for p in [5, 25, 50, 75, 95]
    }

    return {
        "confidence": confidence,
        "var_pct": var_pct,
        "var_dollar": var_dollar,
        "cvar_pct": cvar_pct,
        "cvar_dollar": cvar_dollar,
        "prob_of_loss": prob_of_loss,
        "mean_ending_value": float(ending_values.mean()),
        "median_ending_value": float(np.median(ending_values)),
        "percentile_bands": percentile_bands,  # each value is an array over time, for fan charts
    }


if __name__ == "__main__":
    # Sanity check using the same synthetic setup as simulate.py
    import pandas as pd
    from simulate import simulate_portfolio_paths

    tickers = ["AAPL", "MSFT", "SPY"]
    mu = pd.Series([0.12, 0.15, 0.09], index=tickers)
    sigma = pd.Series([0.30, 0.28, 0.18], index=tickers)
    corr = np.array([[1.0, 0.6, 0.7], [0.6, 1.0, 0.65], [0.7, 0.65, 1.0]])
    cov_matrix = pd.DataFrame(
        np.outer(sigma.values, sigma.values) * corr, index=tickers, columns=tickers
    )
    weights = {"AAPL": 0.4, "MSFT": 0.3, "SPY": 0.3}

    initial_value = 10_000
    paths = simulate_portfolio_paths(
        mu, sigma, cov_matrix, weights,
        initial_value=initial_value, horizon_days=252, n_simulations=5000, seed=42,
    )

    metrics = compute_risk_metrics(paths, initial_value=initial_value, confidence=0.95)

    print(f"95% VaR: ${metrics['var_dollar']:.2f} ({metrics['var_pct']*100:.2f}%)")
    print(f"95% CVaR: ${metrics['cvar_dollar']:.2f} ({metrics['cvar_pct']*100:.2f}%)")
    print(f"Probability of loss: {metrics['prob_of_loss']*100:.1f}%")
    print(f"Mean ending value: ${metrics['mean_ending_value']:.2f}")
    print(f"Median ending value: ${metrics['median_ending_value']:.2f}")

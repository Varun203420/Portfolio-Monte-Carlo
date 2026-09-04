"""
Step 7 — Turn computed risk metrics into a plain-English risk narrative
using the Claude API.

This is the differentiator vs. the reference repo: instead of stopping at
raw VaR/CVaR numbers and a chart, generate a readable explanation of what
those numbers actually mean for this specific portfolio.
"""

import os

import anthropic

MODEL = "claude-sonnet-4-5"

NARRATIVE_SYSTEM_PROMPT = """You are explaining investment risk to someone with no finance or
math background — think a curious parent, not a finance professional.

Guidelines:
- 3-5 sentences, no bullet points, no headers.
- Lead with the most important thing: in a bad year, roughly how much money could be lost,
  stated in dollars and everyday percent terms (e.g. "could lose around $2,400, a bit under
  a quarter of the starting amount").
- Never use the terms "VaR," "CVaR," "volatility," "annualized," "confidence level," or
  "tail risk" — these are jargon. Explain the same ideas in plain words instead:
    - Instead of "95% VaR": say something like "in a bad-but-not-worst-case year"
    - Instead of "CVaR": say something like "if things go really wrong, on average this is
      how bad it tends to get"
    - Instead of "volatility": say "how much [asset]'s price tends to swing around"
  Do not use these plain-language substitutes as rigid templates — vary the phrasing
  naturally each time so it doesn't read like a fill-in-the-blank form.
- Mention which specific holding contributes most to the ups and downs, and briefly say why
  (e.g. it's a bigger chunk of the portfolio, or it swings around more than the others).
- Explain in one plain sentence why "how often does this happen" and "how bad is it when it
  does happen" are two different questions worth knowing separately — without naming VaR/CVaR.
- Avoid any other financial jargon (e.g. "basis points," "drawdown," "Sharpe") entirely.
- Do not give investment advice or tell the reader what to do — describe the risk, don't
  prescribe action.
"""


def generate_risk_narrative(metrics: dict, weights: dict, sigma) -> str:
    """
    Parameters
    ----------
    metrics : dict returned by risk_metrics.compute_risk_metrics()
    weights : dict of {ticker: weight} used in the simulation
    sigma : pd.Series of annualized volatility per asset, from covariance.estimate_parameters()

    Returns
    -------
    str — a short plain-English risk narrative
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    # Identify the asset with the highest weight * volatility contribution,
    # a simple proxy for "biggest driver of portfolio risk"
    contribution = {t: weights[t] * sigma[t] for t in weights}
    top_contributor = max(contribution, key=contribution.get)

    prompt = f"""Portfolio weights: {weights}
Per-asset annualized volatility: {sigma.to_dict()}
Estimated largest risk contributor: {top_contributor}

Simulation results (95% confidence):
- VaR: ${metrics['var_dollar']:.2f} ({metrics['var_pct']*100:.2f}%)
- CVaR: ${metrics['cvar_dollar']:.2f} ({metrics['cvar_pct']*100:.2f}%)
- Probability of any loss: {metrics['prob_of_loss']*100:.1f}%
- Mean ending value: ${metrics['mean_ending_value']:.2f}
- Median ending value: ${metrics['median_ending_value']:.2f}

Write the risk narrative now."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=NARRATIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


if __name__ == "__main__":
    # Manual check — requires ANTHROPIC_API_KEY, reuses the same synthetic
    # setup as simulate.py / risk_metrics.py so no live data pull is needed
    import numpy as np
    import pandas as pd
    from simulate import simulate_portfolio_paths
    from risk_metrics import compute_risk_metrics

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

    narrative = generate_risk_narrative(metrics, weights, sigma)
    print("AI Risk Narrative:\n")
    print(narrative)

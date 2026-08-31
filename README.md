# Portfolio Monte Carlo Risk Simulator

A multi-asset portfolio risk calculator built on correlated Monte Carlo
simulation, with an AI layer that turns raw risk metrics into a plain-English
narrative.

Inspired by [bsoni3402/Monte-Carlo-Simulation](https://github.com/bsoni3402/Monte-Carlo-Simulation),
which simulates a single asset (AAPL or BTC) independently using Geometric
Brownian Motion. This project extends that into a genuine **portfolio** tool:
multiple assets, modeled with their real historical correlations, combined
into one risk profile — plus natural language input and an AI-generated
risk summary on top.

## What it does

1. Pulls historical price data for a set of tickers
2. Fits each asset's expected return, volatility, and how they move together (covariance matrix)
3. Runs thousands of correlated Monte Carlo simulations of the portfolio's future value
4. Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR)
5. Lets you describe a portfolio in plain English instead of a config file
6. Generates a plain-English risk summary via the Claude API

## Example output

```
Input: "60% AAPL, 30% bonds, 10% cash"
Parsed: {'AAPL': 0.6, 'BND': 0.3, 'SHV': 0.1}
```

```
95% VaR: $2,385.70 (-23.86%)
95% CVaR: $3,029.49 (-30.29%)
Probability of loss: 33.9%
```

> "At the 95% confidence level, your portfolio could lose as much as $2,386
> (23.86%) over the simulation period, with AAPL being the primary driver of
> this volatility given its 40% weight and highest volatility at 30%
> annualized. If losses do exceed that threshold, they tend to be more
> severe — the average loss in those worst 5% of scenarios is $3,029
> (30.29%)..."

## Architecture

| File | Responsibility |
|---|---|
| `data.py` | Pull multi-ticker price history (`yfinance`), align into one table, compute log-returns |
| `covariance.py` | Fit annualized drift/volatility per asset, build the covariance matrix, Cholesky decomposition |
| `simulate.py` | Generate correlated random shocks, run multi-asset GBM simulation, combine into portfolio value paths |
| `risk_metrics.py` | Compute VaR, CVaR, percentile bands, probability of loss |
| `nlp_parser.py` | Parse a free-text portfolio description into structured tickers/weights via the Claude API |
| `risk_narrative.py` | Turn risk metrics into a plain-English summary via the Claude API |

The core mathematical idea: rather than simulating each asset independently
(which ignores how assets actually move together), a Cholesky decomposition
of the historical covariance matrix is used to correlate the random shocks
driving each asset's simulated path — so the simulation respects real
cross-asset relationships like tech stocks tending to move together.

## Setup

```bash
pip install -r requirements.txt
```

The AI features (`nlp_parser.py`, `risk_narrative.py`) require an Anthropic
API key, set as an environment variable:

```bash
# macOS/Linux
export ANTHROPIC_API_KEY="your-key-here"

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-key-here"
```

## Usage

Each module can be run standalone to see it in action:

```bash
python data.py              # pull real price data
python covariance.py        # fit parameters on synthetic data
python simulate.py          # run a sample simulation
python risk_metrics.py      # compute risk metrics on a sample simulation
python nlp_parser.py        # parse sample portfolio descriptions
python risk_narrative.py    # generate a sample risk narrative
```

## Roadmap

- [ ] Streamlit frontend tying the full pipeline into one interactive app
- [ ] Stress-testing against specific historical scenarios
- [ ] Fat-tailed return distributions (real markets show more extreme moves than GBM's normal-distribution assumption predicts)

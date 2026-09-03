"""
Step 8 — Streamlit frontend.

Ties the whole pipeline into one interactive app:
  natural language input -> price data -> covariance/correlation ->
  Monte Carlo simulation -> risk metrics -> AI narrative -> fan chart
"""
import os
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data import pull_price_data, compute_log_returns
from covariance import estimate_parameters
from simulate import simulate_portfolio_paths
from risk_metrics import compute_risk_metrics
from nlp_parser import parse_portfolio_text
from risk_narrative import generate_risk_narrative

st.set_page_config(page_title="Portfolio Risk Simulator", layout="centered")

st.title("Portfolio Monte Carlo Risk Simulator")
st.caption(
    "Describe a portfolio in plain English. Behind the scenes: real historical "
    "data, a correlated multi-asset Monte Carlo simulation, and an AI-generated "
    "risk summary."
)

with st.form("portfolio_form"):
    description = st.text_input(
        "Describe your portfolio",
        placeholder="e.g. 60% AAPL, 30% bonds, 10% cash",
    )
    col1, col2 = st.columns(2)
    with col1:
        initial_value = st.number_input("Starting value ($)", value=10_000, step=1000)
    with col2:
        horizon_days = st.number_input("Horizon (trading days)", value=252, step=21)
    n_simulations = st.slider("Number of simulations", 500, 10_000, 2000, step=500)
    submitted = st.form_submit_button("Run simulation")

if submitted and description.strip():
    with st.spinner("Parsing portfolio description..."):
        try:
            weights = parse_portfolio_text(description)
        except Exception as e:
            st.error(f"Couldn't parse that portfolio description: {e}")
            st.stop()

    st.success(f"Parsed portfolio: {weights}")
    tickers = list(weights.keys())

    with st.spinner(f"Pulling historical data for {', '.join(tickers)}..."):
        try:
            prices = pull_price_data(tickers)
            log_returns = compute_log_returns(prices)
        except Exception as e:
            st.error(f"Couldn't pull price data: {e}")
            st.stop()

    with st.spinner("Fitting risk model and running simulation..."):
        params = estimate_parameters(log_returns)
        paths = simulate_portfolio_paths(
            params["mu"], params["sigma"], params["cov_matrix"], weights,
            initial_value=initial_value, horizon_days=int(horizon_days),
            n_simulations=int(n_simulations), seed=42,
        )
        metrics = compute_risk_metrics(paths, initial_value=initial_value, confidence=0.95)

    # --- Risk metrics ---
    st.subheader("Risk Metrics (95% confidence)")
    m1, m2, m3 = st.columns(3)
    m1.metric("VaR", f"${metrics['var_dollar']:,.2f}", f"-{metrics['var_pct']*100:.1f}%")
    m2.metric("CVaR", f"${metrics['cvar_dollar']:,.2f}", f"-{metrics['cvar_pct']*100:.1f}%")
    m3.metric("Prob. of loss", f"{metrics['prob_of_loss']*100:.1f}%")

    # --- Fan chart ---
    st.subheader("Simulated Portfolio Value Over Time")
    days = np.arange(paths.shape[1])
    bands = metrics["percentile_bands"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(days, bands[5], bands[95], alpha=0.2, label="5th-95th percentile")
    ax.fill_between(days, bands[25], bands[75], alpha=0.35, label="25th-75th percentile")
    ax.plot(days, bands[50], color="black", linewidth=1.5, label="Median")
    ax.axhline(initial_value, color="gray", linestyle="--", linewidth=1, label="Starting value")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

    # --- AI narrative ---
    st.subheader("Risk Narrative")
    with st.spinner("Generating risk narrative..."):
        try:
            narrative = generate_risk_narrative(metrics, weights, params["sigma"])
            st.write(narrative)
        except Exception as e:
            st.warning(f"Couldn't generate the AI narrative: {e}")

elif submitted:
    st.warning("Enter a portfolio description first.")

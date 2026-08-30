"""
Step 6 — Parse a free-text portfolio description into structured
tickers + weights using the Claude API.

Lets the user type something like "60% AAPL, 30% bonds, 10% cash" instead
of hardcoding a tickers list and weights dict — this is the first real
"AI" touchpoint in the project.
"""

import json
import os

import anthropic

MODEL = "claude-sonnet-4-5"  # fast + cheap enough for a structured parsing task

PARSE_SYSTEM_PROMPT = """You convert a plain-English portfolio description into structured JSON.

Rules:
- Map informal asset names to real, tradeable ticker symbols (e.g. "bonds" -> "BND", "cash" -> "SHV" or "BIL", "gold" -> "GLD", "the S&P" -> "SPY").
- Weights must be expressed as decimals that sum to 1.0 (e.g. 60% -> 0.6).
- If the user's weights don't sum to 100%, normalize them proportionally so they sum to 1.0.
- Respond with ONLY valid JSON, no preamble, no markdown code fences, in this exact shape:
  {"tickers": {"TICKER": weight, ...}}
"""


def parse_portfolio_text(description: str) -> dict:
    """
    Parameters
    ----------
    description : free text like "60% AAPL, 30% bonds, 10% cash"

    Returns
    -------
    dict of {ticker: weight}, weights summing to 1.0

    Raises
    ------
    ValueError if the model's response isn't valid JSON in the expected shape,
    or if the resulting weights don't sum to ~1.0.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}],
    )

    raw_text = response.content[0].text.strip()

    # Defensive: models sometimes wrap JSON in markdown code fences even when
    # told not to. Strip ```json / ``` fences if present before parsing.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw_text}") from e

    if "tickers" not in parsed or not isinstance(parsed["tickers"], dict):
        raise ValueError(f"Unexpected JSON shape from model:\n{parsed}")

    weights = parsed["tickers"]
    total = sum(weights.values())
    if not (0.98 <= total <= 1.02):  # allow small rounding slack
        raise ValueError(f"Parsed weights sum to {total:.4f}, expected ~1.0: {weights}")

    return weights


if __name__ == "__main__":
    # Manual check — requires ANTHROPIC_API_KEY to be set in your environment
    test_inputs = [
        "60% AAPL, 30% bonds, 10% cash",
        "half in the S&P, half in gold",
        "I want mostly tech: apple, microsoft, and nvidia equally weighted",
    ]

    for text in test_inputs:
        print(f"Input:  {text}")
        try:
            result = parse_portfolio_text(text)
            print(f"Parsed: {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")

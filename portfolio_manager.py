"""
portfolio_manager.py — Portfolio valuation and market data

Works both as a Streamlit module (with caching) and as a plain Python
module in the CLI runner (run_daily.py), where Streamlit is not required.
"""

import math
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

# Streamlit is optional — the CLI runner doesn't need it.
try:
    import streamlit as st
    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False

from database import (
    get_positions, get_sim_state, get_snapshots, update_total_value,
)

# Re-export get_snapshots so callers can do:
#   from portfolio_manager import get_snapshots
__all__ = [
    "fetch_prices", "get_market_snapshot", "get_portfolio_valuation",
    "compute_kpis", "get_snapshots", "format_pct", "format_dollar",
    "STARTING_CAPITAL",
]

STARTING_CAPITAL = 1_000_000.0


def _maybe_cache(fn):
    """Use st.cache_data (5 min TTL) when Streamlit is running, else lru_cache."""
    if _STREAMLIT:
        return st.cache_data(ttl=300)(fn)
    return lru_cache(maxsize=8)(fn)


# ─── Market data ──────────────────────────────────────────────────────────────

@_maybe_cache
def fetch_prices(tickers: tuple, lookback_days: int = 60) -> pd.DataFrame:
    """
    Download adjusted close prices for the last N calendar days.
    Returns a DataFrame with tickers as columns, dates as index.

    Note: accepts a tuple (not list) so lru_cache can hash the argument.
    Callers may pass either a list or tuple — we normalise inside.
    """
    tickers = list(tickers)  # yfinance expects a list
    start   = (datetime.today() - timedelta(days=lookback_days * 1.5)).strftime("%Y-%m-%d")
    raw     = yf.download(
        tickers   = tickers,
        start     = start,
        auto_adjust = True,
        progress  = False,
        threads   = True,
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return prices.ffill(limit=3).dropna(how="all")


def get_market_snapshot(prices_df: pd.DataFrame) -> dict:
    """
    For each ticker in prices_df, return:
        price, ret_1d, ret_5d, ret_20d, vol_20d (annualised)
    """
    snap = {}
    for col in prices_df.columns:
        s = prices_df[col].dropna()
        r = s.pct_change().dropna()
        snap[col] = {
            "price":   round(float(s.iloc[-1]), 2)                          if len(s) >= 1  else None,
            "ret_1d":  round(float(r.iloc[-1]), 4)                          if len(r) >= 1  else None,
            "ret_5d":  round(float(s.iloc[-1] / s.iloc[-6]  - 1), 4)        if len(s) >= 6  else None,
            "ret_20d": round(float(s.iloc[-1] / s.iloc[-21] - 1), 4)        if len(s) >= 21 else None,
            "vol_20d": round(float(r.iloc[-20:].std() * math.sqrt(252)), 4) if len(r) >= 20 else None,
        }
    return snap


# ─── Valuation ────────────────────────────────────────────────────────────────

def get_portfolio_valuation(prices: dict) -> tuple[float, float, float]:
    """
    Compute portfolio value from a {symbol: price} mapping.
    Returns (total_equity, holdings_value, cash_balance).
    """
    positions      = get_positions()
    holdings_value = sum(
        p["shares"] * prices.get(p["symbol"], p["avg_cost"])
        for p in positions
    )
    sim  = get_sim_state()
    cash = sim["cash_balance"] if sim else 0.0
    total = cash + holdings_value

    update_total_value(total)
    return total, holdings_value, cash


# ─── Performance metrics ──────────────────────────────────────────────────────

def compute_kpis(snapshots: list[dict]) -> dict:
    """
    Given daily_snapshots rows, compute summary performance KPIs.
    Returns: total_ret, max_dd, win_rate, vol_ann — all may be None.
    """
    if not snapshots or len(snapshots) < 2:
        return {"total_ret": None, "max_dd": None, "win_rate": None, "vol_ann": None}

    vals = pd.Series([s["total_equity"] for s in snapshots], dtype=float)
    rets = vals.pct_change().dropna()

    total_ret = float(vals.iloc[-1] / STARTING_CAPITAL - 1)
    peak      = vals.cummax()
    max_dd    = float((vals / peak - 1).min())
    win_rate  = float((rets > 0).mean())  if len(rets) > 0 else None
    vol_ann   = float(rets.std() * math.sqrt(252)) if len(rets) > 1 else None

    return {
        "total_ret": total_ret,
        "max_dd":    max_dd,
        "win_rate":  win_rate,
        "vol_ann":   vol_ann,
    }


# ─── Formatters ───────────────────────────────────────────────────────────────

def format_pct(v, sign: bool = True, decimals: int = 2) -> str:
    if v is None:
        return "—"
    prefix = "+" if (sign and v > 0) else ""
    return f"{prefix}{v * 100:.{decimals}f}%"


def format_dollar(v) -> str:
    return f"${v:,.0f}" if v is not None else "—"

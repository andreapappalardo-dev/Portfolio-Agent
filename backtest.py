#!/usr/bin/env python3
"""
backtest.py — Alpha Score Strategy · 10-Day Backtest
MiF Deep Learning Competition · Group B

Alpha Score (all components normalised to [-1, +1] before weighting):
  0.40 × 5-day momentum      — primary signal, normalised at ±20%
  0.25 × opening gap         — today open vs yesterday close, normalised at ±5%
  0.15 × trend vs SMA-20     — +1 above, -1 below
  0.10 × RSI sweet spot      — +1 if 40-65, +0.5 if 35-40, -1 if >70
  0.10 × relative volume     — above-avg volume confirms momentum

Execution model (mirrors live competition system):
  - Day 1: INITIALIZATION — $1M deployed into 10-20 top-alpha stocks
           with rank-based weights (best alpha → largest allocation)
  - Day 2+: DAILY REBALANCING — up to 2 trades/day to:
           (a) exit positions whose alpha dropped below SELL_THRESHOLD or stop hit
           (b) buy highest-alpha new candidates to replace exits
           (c) rebalance weight drift toward current rank-based targets
  - Entry / exit price = today's OPEN
  - Intraday stop = if today's LOW < stop price, exit AT stop price
  - End-of-day mark = today's CLOSE (for P&L reporting)
"""

import math
import sys
import warnings
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe: liquid, well-known Russell 3000 stocks ─────────────────────────
# Deliberately excludes micro-caps to ensure realistic fills at 40% sizing
# ── S&P 500 stocks (liquid, large-cap, momentum-tradeable) ───────────────────
SP500_STOCKS = [
    # Tech / AI / Semiconductors
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD",
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX",
    "NOW", "PANW", "CRWD", "FTNT", "PLTR", "NFLX",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "C", "WFC", "SCHW",
    # Health
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ISRG", "AMGN", "VRTX",
    # Consumer / Retail
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW",
    # Industrials / Defence
    "CAT", "GE", "HON", "RTX", "LMT", "NOC", "GD", "UPS",
    # Energy (S&P 500 energy stocks — live trading)
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY",
    # Communication
    "DIS", "CMCSA", "T", "VZ", "TMUS",
]

# ── ETFs & Index funds ────────────────────────────────────────────────────────
ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "VTI",   # Total market
    "XLK",   # Tech sector
    "XLF",   # Finance sector
    "XLV",   # Health sector
    "XLE",   # Energy sector
    "XLI",   # Industrials
    "GLD",   # Gold
    "TLT",   # 20yr Treasury bonds
    "HYG",   # High yield bonds
]

# ── Oil — BACKTEST ONLY (too volatile/illiquid for once-daily competition) ───
OIL_BACKTEST_ONLY = [
    "USO",   # WTI crude oil ETF
    "BNO",   # Brent crude oil ETF
    "UCO",   # 2x leveraged crude (high volatility)
    "DBO",   # DB oil fund
]

# Final universe: S&P stocks + ETFs + oil (oil included here for backtest)
UNIVERSE = list(dict.fromkeys(SP500_STOCKS + ETFS + OIL_BACKTEST_ONLY))

# ── Strategy parameters ───────────────────────────────────────────────────────
STARTING_CAPITAL    = 1_000_000.0
MAX_POSITION_PCT    = 0.40      # hard cap per single position
TC_BPS              = 10.0      # transaction cost in basis points
STOP_LOSS_PCT       = -0.06     # -6% hard stop from entry price
TARGET_GAIN_PCT     = 0.12      # +12% take-profit target (informational)
BUY_THRESHOLD       = 0.45      # alpha score to enter a new position
SELL_THRESHOLD      = 0.10      # alpha below this → exit at next open
LOOKBACK_DAYS       = 35        # history needed for indicators (SMA20 + buffer)
BACKTEST_DAYS       = 10        # number of trading days to simulate

# ── Portfolio construction ────────────────────────────────────────────────────
INIT_MIN_POSITIONS  = 10        # minimum stocks bought at initialization
INIT_MAX_POSITIONS  = 20        # maximum stocks bought at initialization
MAX_ACTIVE_POSITIONS= 20        # max simultaneous holdings during rebalancing
DAILY_TRADE_LIMIT   = 2         # max trades per day (mirrors competition)
DEPLOY_FRAC         = 0.90      # deploy 90% of capital at init; keep 10% buffer
REBAL_THRESHOLD     = 0.05      # trigger rebalance trade if weight drifts >5%

# ── Broad index ETFs excluded from stock selection (can be held as ETF picks) ─
INDEX_ONLY = {"SPY", "QQQ", "IWM", "DIA", "VTI"}

# ── Alpha score weights ───────────────────────────────────────────────────────
W_MOMENTUM = 0.40
W_GAP      = 0.25
W_TREND    = 0.15
W_RSI      = 0.10
W_VOLUME   = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta  = closes.diff().dropna()
    gains  = delta.clip(lower=0).tail(period)
    losses = (-delta.clip(upper=0)).tail(period)
    avg_g  = gains.mean()
    avg_l  = losses.mean()
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 1)


def _sma(closes: pd.Series, period: int) -> float:
    if len(closes) < period:
        return closes.iloc[-1]
    return closes.tail(period).mean()


def compute_alpha(
    closes:   pd.Series,   # historical closes, oldest → newest
    volumes:  pd.Series,   # historical volumes
    today_open: float,     # today's open price
) -> dict:
    """
    Compute the alpha score for one stock on one day.
    Returns a dict with individual component scores and the total alpha.
    """
    if len(closes) < 21 or today_open <= 0:
        return {"alpha": 0.0, "momentum": 0, "gap": 0, "trend": 0,
                "rsi": 0, "volume": 0}

    cur      = closes.iloc[-1]    # yesterday's close
    prev_5d  = closes.iloc[-6] if len(closes) >= 6 else closes.iloc[0]

    # 1. Momentum: 5-day return, normalised at ±20%
    ret_5d       = (cur / prev_5d - 1) if prev_5d > 0 else 0
    mom_score    = _clip(ret_5d / 0.20, -1.0, 1.0)

    # 2. Opening gap: today open vs yesterday close, normalised at ±5%
    gap          = (today_open / cur - 1) if cur > 0 else 0
    gap_score    = _clip(gap / 0.05, -1.0, 1.0)

    # 3. Trend vs SMA-20
    sma20        = _sma(closes, 20)
    trend_score  = 1.0 if cur > sma20 else -1.0

    # 4. RSI sweet spot
    rsi          = _rsi(closes, 14)
    if   40 <= rsi <= 65: rsi_score =  1.0
    elif 35 <= rsi <  40: rsi_score =  0.5   # approaching oversold — bounce potential
    elif rsi > 70:        rsi_score = -1.0   # overbought penalty
    else:                 rsi_score =  0.0

    # 5. Relative volume: today volume vs 20d average
    avg_vol    = volumes.tail(20).mean() if len(volumes) >= 20 else volumes.mean()
    today_vol  = volumes.iloc[-1]
    rel_vol    = (today_vol / avg_vol - 1) if avg_vol > 0 else 0
    vol_score  = _clip(rel_vol / 2.0, -1.0, 1.0)

    alpha = (W_MOMENTUM * mom_score
           + W_GAP      * gap_score
           + W_TREND    * trend_score
           + W_RSI      * rsi_score
           + W_VOLUME   * vol_score)

    return {
        "alpha":    round(alpha, 4),
        "momentum": round(mom_score, 3),
        "gap":      round(gap_score, 3),
        "trend":    round(trend_score, 3),
        "rsi":      round(rsi_score, 3),
        "volume":   round(vol_score, 3),
        "rsi_raw":  rsi,
        "ret_5d":   round(ret_5d * 100, 2),
        "gap_raw":  round(gap * 100, 2),
        "sma20":    round(sma20, 2),
        "price_yesterday_close": round(cur, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: list, lookback: int = 40) -> dict:
    """
    Download OHLCV data for each ticker.
    Returns {ticker: DataFrame with columns Open, High, Low, Close, Volume}.
    """
    print(f"\n  Downloading {len(tickers)} tickers ({lookback} days)...", end="", flush=True)
    raw = yf.download(
        tickers,
        period=f"{lookback}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    print(" done.")

    data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        # Multiple tickers
        for t in tickers:
            try:
                df = raw.xs(t, axis=1, level=1).dropna(how="all")
                if len(df) >= 22:
                    data[t] = df
            except KeyError:
                pass
    else:
        # Single ticker
        df = raw.dropna(how="all")
        if len(df) >= 22:
            data[tickers[0]] = df

    print(f"  {len(data)} tickers with sufficient history.\n")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, data: dict):
        self.data    = data
        self.tickers = list(data.keys())

        all_dates = sorted(set(
            d for df in data.values() for d in df.index
        ))
        self.all_dates = all_dates

        if len(all_dates) < BACKTEST_DAYS + 21:
            raise ValueError("Not enough historical data — need at least 31 trading days.")
        self.bt_dates   = all_dates[-BACKTEST_DAYS:]
        self.hist_dates = all_dates[:-BACKTEST_DAYS]

    # ── Per-day signal computation ────────────────────────────────────────────

    def signals_for_day(self, day_idx: int) -> pd.DataFrame:
        today    = self.bt_dates[day_idx]
        rows = []
        for t in self.tickers:
            df = self.data[t]
            if today not in df.index:
                continue
            hist = df[df.index < today]
            if len(hist) < 21:
                continue
            today_row   = df.loc[today]
            today_open  = float(today_row["Open"])
            today_high  = float(today_row["High"])
            today_low   = float(today_row["Low"])
            today_close = float(today_row["Close"])
            today_vol   = float(today_row["Volume"])

            sig = compute_alpha(hist["Close"], hist["Volume"], today_open)
            sig.update({
                "symbol":  t,
                "date":    today,
                "open":    today_open,
                "high":    today_high,
                "low":     today_low,
                "close":   today_close,
                "volume":  today_vol,
            })
            rows.append(sig)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("alpha", ascending=False).reset_index(drop=True)

    # ── Rank-based weight calculator ──────────────────────────────────────────

    def _rank_weights(self, n: int, deploy: float = DEPLOY_FRAC,
                      cap: float = MAX_POSITION_PCT) -> list:
        """
        Return rank-based weights for n stocks that sum to `deploy`.
        Stock #1 receives the most capital; last stock the least.
        Weights are capped at `cap` and surplus redistributed.
        """
        scores = [n - i for i in range(n)]          # n, n-1, ..., 1
        total  = sum(scores)
        w      = [s / total * deploy for s in scores]
        for _ in range(10):
            surplus  = sum(max(0, x - cap) for x in w)
            w        = [min(x, cap) for x in w]
            uncapped = [i for i, x in enumerate(w) if x < cap]
            if surplus < 1e-9 or not uncapped:
                break
            uc_sum = sum(w[i] for i in uncapped)
            for i in uncapped:
                w[i] += surplus * (w[i] / uc_sum) if uc_sum else surplus / len(uncapped)
        return w

    # ── Portfolio valuation helper ────────────────────────────────────────────

    def _total_value(self, positions: dict, signals: pd.DataFrame, cash: float) -> float:
        alpha_px = {r["symbol"]: r["close"] for _, r in signals.iterrows()}
        hold_val = sum(
            p["shares"] * alpha_px.get(sym, p["entry"])
            for sym, p in positions.items()
        )
        return cash + hold_val

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION PHASE  (Day 1 only)
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize(self, signals: pd.DataFrame, cash: float,
                    total: float, today) -> tuple:
        """
        Deploy ~90% of $1M into 10-20 top-alpha stocks using rank-based weights.
        Returns (positions dict, remaining cash, list of trade-log entries).
        """
        # Eligible: positive alpha, exclude pure broad-index ETFs
        eligible = signals[
            (signals["alpha"] > 0) &
            (~signals["symbol"].isin(INDEX_ONLY))
        ].head(INIT_MAX_POSITIONS)

        # Fallback: if fewer than INIT_MIN have positive alpha, take top N by alpha
        if len(eligible) < INIT_MIN_POSITIONS:
            eligible = signals[
                ~signals["symbol"].isin(INDEX_ONLY)
            ].head(INIT_MAX_POSITIONS)

        picks   = eligible.to_dict("records")
        n       = len(picks)
        weights = self._rank_weights(n)

        positions = {}
        trades    = []

        print(f"\n  ╔══ INITIALIZATION — deploying into {n} stocks ══╗")
        for rank, (pick, w) in enumerate(zip(picks, weights), 1):
            sym      = pick["symbol"]
            entry_px = pick["open"]
            if entry_px <= 0:
                continue
            alloc = w * total
            if alloc > cash:
                print(f"    #{rank:2d} {sym:<6}  SKIPPED (insufficient cash)")
                continue

            shares   = alloc / entry_px
            tc       = alloc * TC_BPS / 10_000
            cash    -= alloc + tc
            stop_px  = round(entry_px * (1 + STOP_LOSS_PCT), 4)
            tgt_px   = round(entry_px * (1 + TARGET_GAIN_PCT), 4)

            positions[sym] = {
                "shares":        shares,
                "entry":         entry_px,
                "stop":          stop_px,
                "target":        tgt_px,
                "entry_date":    today,
                "alpha_entry":   pick["alpha"],
                "target_weight": w,     # updated daily during rebalancing
            }

            mom_dir = "↑" if pick["momentum"] > 0 else "↓"
            print(f"    #{rank:2d} {sym:<6}  α={pick['alpha']:+.3f}  "
                  f"wt={w*100:.1f}%  px=${entry_px:.2f}  "
                  f"sh={shares:.1f}  mom{mom_dir}{pick['ret_5d']:+.1f}%")

            trades.append({
                "date":    today,
                "action":  "BUY (INIT)",
                "symbol":  sym,
                "price":   entry_px,
                "shares":  shares,
                "value":   alloc,
                "pnl_pct": None,
                "reason":  (
                    f"[INIT] Rank #{rank} of {n}. "
                    f"Alpha={pick['alpha']:.3f}, 5d-mom={pick['ret_5d']:+.1f}%, "
                    f"RSI={pick['rsi_raw']:.0f}. "
                    f"Target weight {w*100:.1f}%. "
                    f"Stop at ${stop_px:.2f} (-6%), target ${tgt_px:.2f} (+12%)."
                ),
            })

        deployed = total - cash
        print(f"  ╚══ Deployed ${deployed:,.0f} ({deployed/total*100:.1f}%)  "
              f"Cash remaining: ${cash:,.0f} ══╝")
        return positions, cash, trades

    # ─────────────────────────────────────────────────────────────────────────
    # DAILY REBALANCING  (Day 2 onwards)
    # ─────────────────────────────────────────────────────────────────────────

    def _rebalance(self, positions: dict, signals: pd.DataFrame,
                   cash: float, total: float, today,
                   trade_log: list) -> tuple:
        """
        Execute up to DAILY_TRADE_LIMIT rebalancing trades.

        Priority order:
          1. Exit positions where intraday low breached the stop price
          2. Exit positions whose alpha fell below SELL_THRESHOLD
          3. Buy the highest-alpha new candidate not yet held
             (replaces exits; fills empty slots up to MAX_ACTIVE_POSITIONS)
          4. Rebalance weight drift: trim overweight / add to underweight
             (only if trade slots remain after steps 1-3)
        """
        alpha_map  = {r["symbol"]: r         for _, r in signals.iterrows()}
        trades_left = DAILY_TRADE_LIMIT
        new_trades  = []

        # ── 1. Intraday stop-loss exits ───────────────────────────────────────
        stops_hit = []
        for sym, pos in list(positions.items()):
            row = alpha_map.get(sym)
            if row is None:
                continue
            if row["low"] <= pos["stop"]:
                exit_px  = pos["stop"]
                proceeds = pos["shares"] * exit_px
                tc       = proceeds * TC_BPS / 10_000
                cash    += proceeds - tc
                pnl_pct  = (exit_px / pos["entry"] - 1) * 100
                days_h   = (today - pos["entry_date"]).days
                new_trades.append({
                    "date":    today, "action": "SELL (STOP)",
                    "symbol":  sym,   "price":   exit_px,
                    "shares":  pos["shares"], "value": proceeds,
                    "pnl_pct": pnl_pct,
                    "reason":  (
                        f"Stop-loss hit after {days_h}d. "
                        f"Intraday low ${row['low']:.2f} ≤ stop ${pos['stop']:.2f} "
                        f"(-6% from entry ${pos['entry']:.2f}). "
                        f"Exit P&L: {pnl_pct:+.2f}%."
                    ),
                })
                stops_hit.append(sym)
        for sym in stops_hit:
            del positions[sym]
        # Stop-loss exits don't count against the 2-trade limit

        # ── 2. Alpha-threshold SELL (counts against trade limit) ──────────────
        to_sell = [
            (sym, alpha_map[sym]["alpha"])
            for sym in list(positions.keys())
            if sym in alpha_map and alpha_map[sym]["alpha"] < SELL_THRESHOLD
        ]
        # Sell weakest first
        to_sell.sort(key=lambda x: x[1])
        for sym, alpha_val in to_sell:
            if trades_left == 0:
                break
            row      = alpha_map[sym]
            exit_px  = row["open"]
            proceeds = positions[sym]["shares"] * exit_px
            tc       = proceeds * TC_BPS / 10_000
            cash    += proceeds - tc
            pnl_pct  = (exit_px / positions[sym]["entry"] - 1) * 100
            days_h   = (today - positions[sym]["entry_date"]).days
            new_trades.append({
                "date":    today, "action": "SELL (ALPHA)",
                "symbol":  sym,   "price":   exit_px,
                "shares":  positions[sym]["shares"], "value": proceeds,
                "pnl_pct": pnl_pct,
                "reason":  (
                    f"Alpha {alpha_val:.3f} below sell threshold {SELL_THRESHOLD}. "
                    f"Held {days_h}d. Exit P&L: {pnl_pct:+.2f}%."
                ),
            })
            del positions[sym]
            trades_left -= 1

        # ── 3. Re-rank and update target weights ──────────────────────────────
        # Universe = current holdings with good alpha + fresh top candidates
        held_syms = set(positions.keys())

        held_rows = [
            alpha_map[s] for s in held_syms
            if s in alpha_map and alpha_map[s]["alpha"] >= SELL_THRESHOLD
        ]
        new_candidates = [
            r for _, r in signals.iterrows()
            if r["symbol"] not in held_syms
            and r["symbol"] not in INDEX_ONLY
            and r["alpha"] > BUY_THRESHOLD
        ]

        # Combine: held (sorted by alpha desc) + candidates (sorted by alpha desc)
        held_rows.sort(key=lambda r: r["alpha"], reverse=True)
        combined = held_rows + new_candidates
        combined = combined[:MAX_ACTIVE_POSITIONS]

        # Assign new target weights
        n_combined = len(combined)
        if n_combined > 0:
            new_weights = self._rank_weights(n_combined)
            for row, w in zip(combined, new_weights):
                sym = row["symbol"]
                if sym in positions:
                    positions[sym]["target_weight"] = w

        # ── 4. BUY new candidates to fill vacated / empty slots ───────────────
        slots_available = MAX_ACTIVE_POSITIONS - len(positions)
        for row in new_candidates:
            if trades_left == 0 or slots_available <= 0:
                break
            sym      = row["symbol"]
            entry_px = row["open"]
            if entry_px <= 0:
                continue

            # Find target weight for this candidate in combined ranking
            try:
                rank_idx  = next(i for i, r in enumerate(combined) if r["symbol"] == sym)
                tgt_w     = new_weights[rank_idx]
            except (StopIteration, NameError):
                tgt_w = min(MAX_POSITION_PCT, DEPLOY_FRAC / max(len(positions) + 1, 1))

            alloc = tgt_w * total
            alloc = min(alloc, cash * 0.99)
            if alloc < entry_px:
                continue

            shares   = alloc / entry_px
            tc       = alloc * TC_BPS / 10_000
            cash    -= alloc + tc
            stop_px  = round(entry_px * (1 + STOP_LOSS_PCT), 4)
            tgt_px   = round(entry_px * (1 + TARGET_GAIN_PCT), 4)

            positions[sym] = {
                "shares":        shares,
                "entry":         entry_px,
                "stop":          stop_px,
                "target":        tgt_px,
                "entry_date":    today,
                "alpha_entry":   row["alpha"],
                "target_weight": tgt_w,
            }

            trend_w = "above" if row["trend"] > 0 else "below"
            gap_w   = f"gap-up {row['gap_raw']:+.1f}%" if row["gap_raw"] > 1 else (
                      f"gap-down {row['gap_raw']:+.1f}%" if row["gap_raw"] < -1 else "flat open")
            new_trades.append({
                "date":    today, "action": "BUY",
                "symbol":  sym,   "price":   entry_px,
                "shares":  shares, "value":  alloc,
                "pnl_pct": None,
                "reason":  (
                    f"Alpha {row['alpha']:.3f} > threshold {BUY_THRESHOLD}. "
                    f"5d-mom {row['ret_5d']:+.1f}%, {gap_w}, RSI {row['rsi_raw']:.0f} "
                    f"({'sweet spot' if 40 <= row['rsi_raw'] <= 65 else 'outside sweet spot'}), "
                    f"trading {trend_w} SMA-20. "
                    f"Target weight {tgt_w*100:.1f}%. Stop ${stop_px:.2f}."
                ),
            })
            trades_left     -= 1
            slots_available -= 1

        # ── 5. Weight-drift rebalancing (uses remaining trade slots) ──────────
        if trades_left > 0 and n_combined > 0:
            # Compute current weights
            cur_weights = {}
            for sym, pos in positions.items():
                px = alpha_map[sym]["close"] if sym in alpha_map else pos["entry"]
                cur_weights[sym] = (pos["shares"] * px) / total

            # Find biggest positive drift (overweight) and negative drift (underweight)
            drifts = {}
            for sym, pos in positions.items():
                target = pos.get("target_weight", 0)
                actual = cur_weights.get(sym, 0)
                drifts[sym] = actual - target   # positive = overweight

            # TRIM overweight positions (sell partial)
            overweight = [(s, d) for s, d in drifts.items() if d > REBAL_THRESHOLD]
            overweight.sort(key=lambda x: -x[1])   # biggest overweight first
            for sym, drift in overweight:
                if trades_left == 0:
                    break
                pos      = positions[sym]
                row      = alpha_map.get(sym)
                if row is None:
                    continue
                exit_px  = row["open"]
                # Sell enough to bring weight back to target
                trim_val = drift * total
                trim_sh  = trim_val / exit_px
                if trim_sh <= 0 or trim_sh >= pos["shares"]:
                    continue
                tc        = trim_val * TC_BPS / 10_000
                cash     += trim_val - tc
                pos["shares"] -= trim_sh
                pnl_pct   = (exit_px / pos["entry"] - 1) * 100
                new_trades.append({
                    "date":    today, "action": "TRIM",
                    "symbol":  sym,   "price":   exit_px,
                    "shares":  trim_sh, "value":  trim_val,
                    "pnl_pct": pnl_pct,
                    "reason":  (
                        f"Weight drift rebalance: overweight by {drift*100:.1f}%. "
                        f"Trimming {trim_sh:.1f} shares to restore target "
                        f"{pos['target_weight']*100:.1f}% weight."
                    ),
                })
                trades_left -= 1

            # ADD to underweight positions (buy partial)
            underweight = [(s, d) for s, d in drifts.items() if d < -REBAL_THRESHOLD]
            underweight.sort(key=lambda x: x[1])   # biggest underweight first
            for sym, drift in underweight:
                if trades_left == 0:
                    break
                pos      = positions[sym]
                row      = alpha_map.get(sym)
                if row is None:
                    continue
                entry_px = row["open"]
                add_val  = abs(drift) * total
                add_val  = min(add_val, cash * 0.99)
                if add_val < entry_px:
                    continue
                add_sh   = add_val / entry_px
                tc       = add_val * TC_BPS / 10_000
                cash    -= add_val + tc
                pos["shares"] += add_sh
                new_trades.append({
                    "date":    today, "action": "ADD",
                    "symbol":  sym,   "price":   entry_px,
                    "shares":  add_sh, "value":   add_val,
                    "pnl_pct": None,
                    "reason":  (
                        f"Weight drift rebalance: underweight by {abs(drift)*100:.1f}%. "
                        f"Adding {add_sh:.1f} shares to restore target "
                        f"{pos['target_weight']*100:.1f}% weight."
                    ),
                })
                trades_left -= 1

        trade_log.extend(new_trades)
        return positions, cash

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN BACKTEST LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cash      = STARTING_CAPITAL
        positions = {}           # {symbol: {shares, entry, stop, target, entry_date, ...}}
        daily_log = []
        trade_log = []

        print("=" * 72)
        print("  ALPHA SCORE BACKTEST — 10 Trading Days  (Group B)")
        print(f"  Universe: {len(self.tickers)} stocks")
        print(f"  Start capital: ${STARTING_CAPITAL:,.0f}")
        print(f"  Init: {INIT_MIN_POSITIONS}–{INIT_MAX_POSITIONS} stocks, rank-based weights")
        print(f"  Rebalancing: up to {DAILY_TRADE_LIMIT} trades/day")
        print(f"  Stop: {STOP_LOSS_PCT*100:.0f}%   TC: {TC_BPS} bps")
        print(f"  Buy α > {BUY_THRESHOLD}   Sell α < {SELL_THRESHOLD}")
        print("=" * 72)

        for day_idx, today in enumerate(self.bt_dates):
            date_str = today.strftime("%Y-%m-%d (%a)")
            signals  = self.signals_for_day(day_idx)

            if signals.empty:
                print(f"\n  {date_str}  — no signal data, skipping")
                continue

            total = self._total_value(positions, signals, cash)

            # ── Day 1: initialization ─────────────────────────────────────────
            if day_idx == 0:
                positions, cash, init_trades = self._initialize(
                    signals, cash, total, today
                )
                trade_log.extend(init_trades)

            # ── Day 2+: daily rebalancing ─────────────────────────────────────
            else:
                positions, cash = self._rebalance(
                    positions, signals, cash, total, today, trade_log
                )

            # ── Mark portfolio at today's close ───────────────────────────────
            alpha_lookup = {r["symbol"]: r["alpha"] for _, r in signals.iterrows()}
            close_px     = {r["symbol"]: r["close"] for _, r in signals.iterrows()}

            holdings_value = sum(
                pos["shares"] * close_px.get(sym, pos["entry"])
                for sym, pos in positions.items()
            )
            total_value  = cash + holdings_value
            ret_vs_start = (total_value / STARTING_CAPITAL - 1) * 100
            ret_vs_prev  = (
                (total_value / daily_log[-1]["total_value"] - 1) * 100
                if daily_log else 0.0
            )

            top5 = signals.head(5)[["symbol","alpha","momentum","gap_raw","rsi_raw","ret_5d"]]

            daily_log.append({
                "date":        today,
                "date_str":    date_str,
                "cash":        cash,
                "holdings":    holdings_value,
                "total_value": total_value,
                "ret_pct":     ret_vs_start,
                "daily_ret":   ret_vs_prev,
                "positions":   list(positions.keys()),
                "n_positions": len(positions),
                "n_trades":    len([t for t in trade_log if t["date"] == today]),
                "top5":        top5,
            })

            # ── Print day summary ─────────────────────────────────────────────
            direction = "▲" if ret_vs_prev >= 0 else "▼"
            label     = "INIT" if day_idx == 0 else f"Day {day_idx+1:2d}"
            print(f"\n  ┌─ {label} │ {date_str} {'─'*36}┐")
            print(f"  │  Portfolio: ${total_value:>12,.0f}  "
                  f"({ret_vs_start:+.2f}% vs start)  "
                  f"{direction} {abs(ret_vs_prev):.2f}% today       │")
            print(f"  │  Cash: ${cash:>12,.0f}   "
                  f"Holdings: ${holdings_value:>12,.0f}  "
                  f"Positions: {len(positions):2d}          │")

            # Show held positions and their current alpha
            held_alpha = sorted(
                [(s, alpha_lookup.get(s, 0), close_px.get(s, positions[s]["entry"]))
                 for s in positions],
                key=lambda x: -x[1]
            )
            chunks = [held_alpha[i:i+4] for i in range(0, len(held_alpha), 4)]
            for chunk in chunks:
                line = "  │  " + "  ".join(
                    f"{s:6s}α={a:+.2f}(${p:.0f})" for s, a, p in chunk
                )
                print(f"{line:<74}│")

            day_trades = [t for t in trade_log if t["date"] == today]
            if day_trades:
                print(f"  │  Trades:                                                           │")
                for t in day_trades:
                    pnl_s = f" P&L {t['pnl_pct']:+.1f}%" if t["pnl_pct"] is not None else ""
                    print(f"  │   {t['action']:12s} {t['symbol']:6s} "
                          f"${t['price']:8.2f}  {t['shares']:8.1f}sh  "
                          f"${t['value']:>10,.0f}{pnl_s:<12}  │")
            print(f"  └{'─'*70}┘")

        return {"daily_log": daily_log, "trade_log": trade_log}


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics & chart
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(daily_log: list) -> dict:
    rets = [d["daily_ret"] / 100 for d in daily_log]
    values = [d["total_value"] for d in daily_log]

    total_ret  = (values[-1] / STARTING_CAPITAL - 1) * 100 if values else 0
    peak       = STARTING_CAPITAL
    max_dd     = 0.0
    for v in values:
        peak   = max(peak, v)
        dd     = (v - peak) / peak * 100
        max_dd = min(max_dd, dd)

    win_days   = sum(1 for r in rets if r > 0)
    loss_days  = sum(1 for r in rets if r < 0)
    avg_ret    = np.mean(rets) * 100 if rets else 0
    std_ret    = np.std(rets) * 100  if len(rets) > 1 else 0
    sharpe     = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0

    return {
        "total_return":    total_ret,
        "max_drawdown":    max_dd,
        "win_days":        win_days,
        "loss_days":       loss_days,
        "avg_daily_ret":   avg_ret,
        "daily_std":       std_ret,
        "annualised_sharpe": sharpe,
        "final_value":     values[-1] if values else STARTING_CAPITAL,
    }


def print_summary(stats: dict, daily_log: list, trade_log: list):
    print("\n" + "=" * 72)
    print("  BACKTEST SUMMARY  (Init + Daily Rebalancing Strategy)")
    print("=" * 72)
    print(f"  Total Return:          {stats['total_return']:>+8.2f}%")
    print(f"  Final Portfolio Value: ${stats['final_value']:>12,.0f}")
    print(f"  Max Drawdown:          {stats['max_drawdown']:>+8.2f}%")
    print(f"  Win Days / Loss Days:  {stats['win_days']} / {stats['loss_days']}")
    print(f"  Avg Daily Return:      {stats['avg_daily_ret']:>+8.3f}%")
    print(f"  Daily Std Dev:         {stats['daily_std']:>8.3f}%")
    print(f"  Annualised Sharpe:     {stats['annualised_sharpe']:>8.2f}")
    print(f"  Total Trades:          {len(trade_log)}")

    buys_init  = [t for t in trade_log if t["action"] == "BUY (INIT)"]
    buys_daily = [t for t in trade_log if t["action"] == "BUY"]
    trims      = [t for t in trade_log if t["action"] == "TRIM"]
    adds       = [t for t in trade_log if t["action"] == "ADD"]
    sells_pnl  = [t for t in trade_log if t["pnl_pct"] is not None]

    print(f"  Init positions:        {len(buys_init)}")
    print(f"  Daily BUYs:            {len(buys_daily)}")
    print(f"  Rebalance TRIMs/ADDs:  {len(trims)} / {len(adds)}")

    if sells_pnl:
        wins   = [t for t in sells_pnl if t["pnl_pct"] > 0]
        losses = [t for t in sells_pnl if t["pnl_pct"] <= 0]
        avg_w  = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
        avg_l  = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        print(f"  Trade Win Rate:        {len(wins)}/{len(sells_pnl)} "
              f"({len(wins)/len(sells_pnl)*100:.0f}%)")
        print(f"  Avg Win / Avg Loss:    {avg_w:+.2f}% / {avg_l:+.2f}%")

    print("\n  ── Daily P&L Table ──────────────────────────────────────────────")
    print(f"  {'Day':<4} {'Date':<14} {'Portfolio':>12} {'Daily Ret':>10} "
          f"{'vs Start':>10} {'Positions'}")
    print(f"  {'─'*4} {'─'*14} {'─'*12} {'─'*10} {'─'*10} {'─'*20}")
    for i, d in enumerate(daily_log):
        arrow = "▲" if d["daily_ret"] >= 0 else "▼"
        print(f"  {i+1:<4} {d['date'].strftime('%Y-%m-%d %a'):<14} "
              f"${d['total_value']:>11,.0f} "
              f"{arrow}{abs(d['daily_ret']):>8.2f}% "
              f"{d['ret_pct']:>+9.2f}%  "
              f"{', '.join(d['positions']) or '—'}")

    print("\n  ── Trade Log ────────────────────────────────────────────────────")
    print(f"  {'Date':<12} {'Action':<14} {'Symbol':<8} {'Price':>8} "
          f"{'Shares':>8} {'Value':>10} {'P&L':>8}")
    print(f"  {'─'*12} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for t in trade_log:
        pnl_str = f"{t['pnl_pct']:+.1f}%" if t["pnl_pct"] is not None else ""
        print(f"  {t['date'].strftime('%Y-%m-%d'):<12} {t['action']:<14} "
              f"{t['symbol']:<8} ${t['price']:>7.2f} "
              f"{t['shares']:>8.1f} ${t['value']:>9,.0f} {pnl_str:>8}")
        print(f"    ↳ {t['reason']}")
    print("=" * 72)


def plot_results(daily_log: list, trade_log: list, output_path: str = "backtest_results.png"):
    dates  = [d["date"] for d in daily_log]
    values = [d["total_value"] for d in daily_log]
    rets   = [d["daily_ret"] for d in daily_log]

    # SPY benchmark (use first ticker's close as proxy if available)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             facecolor="#0D1117",
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle("Alpha Score Strategy — 10-Day Backtest (Group B)",
                 color="#E6EDF3", fontsize=14, fontweight="bold", y=0.98)

    DARK  = "#0D1117"
    PANEL = "#161B22"
    BLUE  = "#1F6FEB"
    GREEN = "#3FB950"
    RED   = "#F85149"
    GREY  = "#8B949E"
    TEXT  = "#E6EDF3"

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    # ── Chart 1: Portfolio value ──────────────────────────────────────────
    ax1 = axes[0]
    # Baseline
    baseline = [STARTING_CAPITAL] * len(dates)
    ax1.plot(dates, baseline, color=GREY, linewidth=1, linestyle="--", alpha=0.5)
    ax1.fill_between(dates, baseline, values,
                     where=[v >= STARTING_CAPITAL for v in values],
                     color=GREEN, alpha=0.15)
    ax1.fill_between(dates, baseline, values,
                     where=[v < STARTING_CAPITAL for v in values],
                     color=RED, alpha=0.15)
    ax1.plot(dates, values, color=BLUE, linewidth=2.5, marker="o",
             markersize=5, markerfacecolor=BLUE, zorder=5)

    # Mark trades
    buy_dates  = [t["date"] for t in trade_log if t["action"] == "BUY"]
    sell_dates = [t["date"] for t in trade_log if "SELL" in t["action"]]
    for d in buy_dates:
        if d in dates:
            idx = dates.index(d)
            ax1.axvline(d, color=GREEN, linewidth=1, alpha=0.4)
    for d in sell_dates:
        if d in dates:
            ax1.axvline(d, color=RED, linewidth=1, alpha=0.4)

    # Final value annotation
    if values:
        total_ret = (values[-1] / STARTING_CAPITAL - 1) * 100
        color = GREEN if total_ret >= 0 else RED
        ax1.annotate(f"${values[-1]:,.0f}\n{total_ret:+.2f}%",
                     xy=(dates[-1], values[-1]),
                     xytext=(10, 0), textcoords="offset points",
                     color=color, fontsize=9, fontweight="bold",
                     va="center")

    ax1.set_title("Portfolio Value", color=TEXT, fontsize=10, pad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Value ($)", color=GREY, fontsize=8)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Chart 2: Daily returns bar ────────────────────────────────────────
    ax2 = axes[1]
    colors = [GREEN if r >= 0 else RED for r in rets]
    ax2.bar(dates, rets, color=colors, alpha=0.8, width=0.6)
    ax2.axhline(0, color=GREY, linewidth=0.8)
    ax2.set_title("Daily Return (%)", color=TEXT, fontsize=10, pad=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.2f}%"))
    ax2.set_ylabel("Return", color=GREY, fontsize=8)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Chart 3: Alpha score of held positions ────────────────────────────
    ax3 = axes[2]
    # Show number of positions held each day
    n_pos = [d.get("n_positions", len(d["positions"])) for d in daily_log]
    ax3.fill_between(dates, n_pos, step="mid", color=BLUE, alpha=0.3)
    ax3.step(dates, n_pos, color=BLUE, linewidth=1.5, where="mid")
    ax3.set_ylim(0, MAX_ACTIVE_POSITIONS + 1)
    ax3.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax3.set_title("Positions Held", color=TEXT, fontsize=10, pad=8)
    ax3.set_ylabel("# Positions", color=GREY, fontsize=8)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Reference lines ───────────────────────────────────────────────────
    ax3.axhline(INIT_MAX_POSITIONS, color=GREEN, linewidth=0.8,
                linestyle=":", alpha=0.5, label=f"Max ({INIT_MAX_POSITIONS})")
    ax3.axhline(INIT_MIN_POSITIONS, color=GREY, linewidth=0.8,
                linestyle=":", alpha=0.4, label=f"Min ({INIT_MIN_POSITIONS})")

    # Legend / caption
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GREEN, linewidth=1.5, label="Buy signal"),
        Line2D([0], [0], color=RED,   linewidth=1.5, label="Sell signal"),
        Line2D([0], [0], color=BLUE,  linewidth=2,   label="Portfolio value"),
    ]
    axes[0].legend(handles=legend_elements, facecolor=PANEL,
                   edgecolor="#30363D", labelcolor=TEXT, fontsize=8,
                   loc="upper left")

    # Common x-axis formatting
    for ax in axes:
        ax.tick_params(axis="x", colors=GREY, rotation=30)
        ax.tick_params(axis="y", colors=GREY)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print(f"\n  Chart saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Alpha Score Backtester — MiF Deep Learning Competition (Group B)")
    print(f"  Strategy: Initialization ({INIT_MIN_POSITIONS}–{INIT_MAX_POSITIONS} stocks) + "
          f"Daily Rebalancing ({DAILY_TRADE_LIMIT} trades/day)")
    print(f"  Universe: {len(UNIVERSE)} stocks  |  Backtest window: {BACKTEST_DAYS} trading days")

    # ── Output folder: always save to user's Downloads folder ────────────────
    from pathlib import Path
    downloads = Path.home() / "Downloads"
    downloads.mkdir(exist_ok=True)
    print(f"  Output folder: {downloads}\n")

    # Download data
    data = download_data(UNIVERSE, lookback=LOOKBACK_DAYS + 5)

    if len(data) < 10:
        print("ERROR: Too few stocks downloaded. Check your internet connection.")
        sys.exit(1)

    # Run backtest
    bt     = Backtester(data)
    result = bt.run()

    daily_log = result["daily_log"]
    trade_log = result["trade_log"]

    if not daily_log:
        print("ERROR: No backtest days produced results.")
        sys.exit(1)

    # Stats & output
    stats = compute_stats(daily_log)
    print_summary(stats, daily_log, trade_log)

    chart_path     = downloads / "backtest_results.png"
    daily_log_path = downloads / "backtest_daily_log.csv"
    trades_path    = downloads / "backtest_trades.csv"

    plot_results(daily_log, trade_log, str(chart_path))

    # Save CSVs
    df_log = pd.DataFrame([{
        "date":      d["date"].strftime("%Y-%m-%d"),
        "portfolio": round(d["total_value"], 2),
        "cash":      round(d["cash"], 2),
        "holdings":  round(d["holdings"], 2),
        "daily_ret": round(d["daily_ret"], 4),
        "cum_ret":   round(d["ret_pct"], 4),
        "positions": "|".join(d["positions"]),
        "n_trades":  d["n_trades"],
    } for d in daily_log])
    df_log.to_csv(daily_log_path, index=False, encoding="utf-8-sig")

    df_trades = pd.DataFrame([{
        "date":    t["date"].strftime("%Y-%m-%d"),
        "action":  t["action"],
        "symbol":  t["symbol"],
        "price":   t["price"],
        "shares":  round(t["shares"], 2),
        "value":   round(t["value"], 2),
        "pnl_pct": round(t["pnl_pct"], 2) if t["pnl_pct"] is not None else "",
        "reason":  t["reason"],
    } for t in trade_log])
    df_trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print(f"\n  Logs saved to Downloads folder:")
    print(f"    {daily_log_path}")
    print(f"    {trades_path}")
    print(f"    {chart_path}")
    print()


if __name__ == "__main__":
    main()

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

Execution model (mirrors live system):
  - Signals computed from yesterday's close + today's open gap
  - Entry price  = today's OPEN  (code runs at market open)
  - Intraday stop  = if today's LOW < stop price, exit AT stop price
  - End-of-day mark = today's CLOSE (for P&L reporting)
  - Next day decision uses fresh alpha

Strategy:
  - Hold up to MAX_POSITIONS (2) at once, max 40% each
  - BUY  if alpha > BUY_THRESHOLD  and not already held
  - SELL if alpha < SELL_THRESHOLD or stop breached intraday
  - Rank by alpha descending, fill positions until cash runs out
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
STARTING_CAPITAL  = 1_000_000.0
MAX_POSITION_PCT  = 0.40
MAX_POSITIONS     = 2
TC_BPS            = 10.0        # transaction cost in basis points
STOP_LOSS_PCT     = -0.06       # -6% hard stop from entry price
BUY_THRESHOLD     = 0.45        # alpha score to open a position
SELL_THRESHOLD    = 0.10        # alpha below this → exit next open
LOOKBACK_DAYS     = 35          # history needed for indicators (SMA20 + buffer)
BACKTEST_DAYS     = 10          # number of trading days to simulate

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

class Backtester:

    def __init__(self, data: dict):
        self.data    = data
        self.tickers = list(data.keys())

        # Build aligned date index from all tickers
        all_dates = sorted(set(
            d for df in data.values() for d in df.index
        ))
        self.all_dates = all_dates

        # Last BACKTEST_DAYS dates are the backtest window
        # We need at least 21 prior dates for indicators
        if len(all_dates) < BACKTEST_DAYS + 21:
            raise ValueError("Not enough historical data — need at least 31 trading days.")
        self.bt_dates  = all_dates[-BACKTEST_DAYS:]
        self.hist_dates = all_dates[:-BACKTEST_DAYS]

    # ── Per-day signal computation ────────────────────────────────────────────

    def signals_for_day(self, day_idx: int) -> pd.DataFrame:
        """
        Compute alpha scores for all tickers on a given backtest day.
        day_idx: index into self.bt_dates (0 = first backtest day).

        Uses data available at market open:
          - All historical closes up to (not including) today
          - Today's open price
        """
        today    = self.bt_dates[day_idx]
        # Available history = all_dates up to (but not including) today
        hist_end = self.all_dates.index(today)   # exclusive upper bound

        rows = []
        for t in self.tickers:
            df = self.data[t]
            if today not in df.index:
                continue

            # Historical data available at open
            hist = df[df.index < today]
            if len(hist) < 21:
                continue

            today_row  = df.loc[today]
            today_open = float(today_row["Open"])
            today_high = float(today_row["High"])
            today_low  = float(today_row["Low"])
            today_close= float(today_row["Close"])
            today_vol  = float(today_row["Volume"])

            # Include today's volume in the volume signal
            hist_with_vol = hist.copy()

            sig = compute_alpha(hist["Close"], hist["Volume"], today_open)
            sig.update({
                "symbol":      t,
                "date":        today,
                "open":        today_open,
                "high":        today_high,
                "low":         today_low,
                "close":       today_close,
                "volume":      today_vol,
            })
            rows.append(sig)

        if not rows:
            return pd.DataFrame()
        df_sig = pd.DataFrame(rows).sort_values("alpha", ascending=False)
        return df_sig.reset_index(drop=True)

    # ── Main backtest loop ────────────────────────────────────────────────────

    def run(self) -> dict:
        cash       = STARTING_CAPITAL
        positions  = {}    # {symbol: {"shares": float, "entry": float, "stop": float, "entry_date": date}}
        daily_log  = []
        trade_log  = []

        print("=" * 72)
        print("  ALPHA SCORE BACKTEST — 10 Trading Days")
        print(f"  Universe: {len(self.tickers)} liquid stocks")
        print(f"  Start capital: ${STARTING_CAPITAL:,.0f}")
        print(f"  Max positions: {MAX_POSITIONS} × {MAX_POSITION_PCT*100:.0f}%")
        print(f"  Stop-loss: {STOP_LOSS_PCT*100:.0f}%   TC: {TC_BPS} bps")
        print(f"  Buy threshold α > {BUY_THRESHOLD}   Sell threshold α < {SELL_THRESHOLD}")
        print("=" * 72)

        for day_idx, today in enumerate(self.bt_dates):
            date_str = today.strftime("%Y-%m-%d (%a)")
            signals  = self.signals_for_day(day_idx)

            if signals.empty:
                print(f"\n  {date_str}  — no signal data, skipping")
                continue

            # ── Step 1: check intraday stops on held positions ─────────────
            stops_hit = []
            for sym, pos in list(positions.items()):
                row = signals[signals["symbol"] == sym]
                if row.empty:
                    continue
                row      = row.iloc[0]
                day_low  = row["low"]
                stop_px  = pos["stop"]

                if day_low <= stop_px:
                    # Stop hit intraday — exit at stop price
                    exit_px  = stop_px
                    proceeds = pos["shares"] * exit_px
                    tc       = proceeds * TC_BPS / 10_000
                    cash    += proceeds - tc
                    pnl_pct  = (exit_px / pos["entry"] - 1) * 100
                    days_held = (today - pos["entry_date"]).days
                    trade_log.append({
                        "date": today, "action": "SELL (STOP)", "symbol": sym,
                        "price": exit_px, "shares": pos["shares"],
                        "value": proceeds, "pnl_pct": pnl_pct,
                        "reason": (
                            f"Stop-loss triggered after {days_held} day(s). "
                            f"Intraday low of ${day_low:.2f} breached the hard stop "
                            f"at ${stop_px:.2f} ({STOP_LOSS_PCT*100:.0f}% below entry "
                            f"of ${pos['entry']:.2f}). "
                            f"Position closed with a {'gain' if pnl_pct >= 0 else 'loss'} "
                            f"of {abs(pnl_pct):.2f}%."
                        ),
                    })
                    stops_hit.append(sym)

            for sym in stops_hit:
                del positions[sym]

            # ── Step 2: sell positions where alpha dropped below threshold ──
            alpha_lookup = {r["symbol"]: r["alpha"] for _, r in signals.iterrows()}
            to_sell = []
            for sym in positions:
                alpha = alpha_lookup.get(sym, 0)
                if alpha < SELL_THRESHOLD:
                    to_sell.append((sym, alpha))

            for sym, alpha in to_sell:
                if sym not in positions:
                    continue
                row     = signals[signals["symbol"] == sym]
                if row.empty:
                    continue
                row     = row.iloc[0]
                exit_px = row["open"]    # exit at today's open
                proceeds= positions[sym]["shares"] * exit_px
                tc      = proceeds * TC_BPS / 10_000
                cash   += proceeds - tc
                pnl_pct = (exit_px / positions[sym]["entry"] - 1) * 100
                trade_log.append({
                        "date": today, "action": "SELL (fade)", "symbol": sym,
                        "price": exit_px, "shares": positions[sym]["shares"],
                        "value": proceeds, "pnl_pct": pnl_pct,
                        "reason": (
                            f"Momentum faded. Alpha score dropped to {alpha:.3f}, "
                            f"below the sell threshold of {SELL_THRESHOLD}. "
                            f"The 5-day price move has been absorbed and the signal "
                            f"is no longer strong enough to justify holding. "
                            f"Exited at open with a {'gain' if pnl_pct >= 0 else 'loss'} "
                            f"of {abs(pnl_pct):.2f}%."
                        ),
                    })
                del positions[sym]

            # ── Step 3: buy top alpha stocks ──────────────────────────────
            held = set(positions.keys())
            # Index ETFs excluded from direct purchase (used as benchmark context only)
            # Oil ETFs ARE tradeable in the backtest
            INDEX_ONLY = {"SPY", "QQQ", "IWM", "DIA", "VTI"}
            candidates = signals[
                (signals["alpha"] > BUY_THRESHOLD) &
                (~signals["symbol"].isin(INDEX_ONLY)) &
                (~signals["symbol"].isin(held))
            ].head(MAX_POSITIONS - len(positions))

            for _, row in candidates.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    break
                sym      = row["symbol"]
                entry_px = row["open"]
                if entry_px <= 0:
                    continue

                # Size: equal-weight remaining slots, capped at MAX_POSITION_PCT
                slots_left  = MAX_POSITIONS - len(positions)
                alloc_pct   = min(MAX_POSITION_PCT, 1.0 / max(slots_left, 1))
                total_value = cash + sum(
                    p["shares"] * (signals[signals["symbol"] == s].iloc[0]["close"]
                                   if not signals[signals["symbol"] == s].empty else p["entry"])
                    for s, p in positions.items()
                )
                alloc_value = alloc_pct * total_value
                if alloc_value > cash:
                    alloc_value = cash * 0.99   # use available cash

                shares  = alloc_value / entry_px
                tc      = alloc_value * TC_BPS / 10_000
                cash   -= alloc_value + tc

                stop_px = entry_px * (1 + STOP_LOSS_PCT)
                positions[sym] = {
                    "shares":     shares,
                    "entry":      entry_px,
                    "stop":       stop_px,
                    "entry_date": today,
                    "alpha_entry":row["alpha"],
                }
                trend_word = "above" if row['trend'] > 0 else "below"
                mom_word   = "strong upward" if row['momentum'] > 0.5 else ("moderate upward" if row['momentum'] > 0 else "weak")
                gap_word   = f"gap-up of {row['gap_raw']:+.1f}% at open" if row['gap_raw'] > 1 else (f"gap-down of {row['gap_raw']:+.1f}%" if row['gap_raw'] < -1 else "flat open")
                rsi_word   = "in the momentum sweet spot" if 40 <= row['rsi_raw'] <= 65 else ("approaching oversold" if row['rsi_raw'] < 40 else "overbought")
                trade_log.append({
                    "date": today, "action": "BUY", "symbol": sym,
                    "price": entry_px, "shares": shares,
                    "value": alloc_value, "pnl_pct": None,
                    "reason": (
                        f"Alpha score {row['alpha']:.3f} exceeded the buy threshold of {BUY_THRESHOLD}. "
                        f"5-day momentum is {mom_word} at {row['ret_5d']:+.1f}%, with a {gap_word}. "
                        f"RSI of {row['rsi_raw']:.0f} is {rsi_word}, and the stock is trading "
                        f"{trend_word} its 20-day moving average. "
                        f"Stop set at ${stop_px:.2f} ({STOP_LOSS_PCT*100:.0f}% below entry)."
                    ),
                })

            # ── Step 4: mark portfolio at today's close ────────────────────
            holdings_value = 0.0
            for sym, pos in positions.items():
                row = signals[signals["symbol"] == sym]
                close_px = row.iloc[0]["close"] if not row.empty else pos["entry"]
                holdings_value += pos["shares"] * close_px

            total_value = cash + holdings_value
            ret_vs_start = (total_value / STARTING_CAPITAL - 1) * 100
            ret_vs_prev  = ((total_value / daily_log[-1]["total_value"] - 1) * 100
                           if daily_log else 0.0)

            # Top 5 alpha for the day
            top5 = signals.head(5)[["symbol","alpha","momentum","gap_raw","rsi_raw","ret_5d"]]

            daily_log.append({
                "date":         today,
                "date_str":     date_str,
                "cash":         cash,
                "holdings":     holdings_value,
                "total_value":  total_value,
                "ret_pct":      ret_vs_start,
                "daily_ret":    ret_vs_prev,
                "positions":    list(positions.keys()),
                "n_trades":     len([t for t in trade_log if t["date"] == today]),
                "top5":         top5,
            })

            # ── Print day summary ─────────────────────────────────────────
            direction = "▲" if ret_vs_prev >= 0 else "▼"
            print(f"\n  ┌─ Day {day_idx+1:2d} │ {date_str} ─{'─'*36}┐")
            print(f"  │  Portfolio: ${total_value:>12,.0f}  ({ret_vs_start:+.2f}% vs start)  "
                  f"{direction} {abs(ret_vs_prev):.2f}% today  │")
            print(f"  │  Cash: ${cash:>12,.0f}   Holdings: ${holdings_value:>12,.0f}          │")
            held_str = ", ".join(f"{s}(α={alpha_lookup.get(s,0):.2f})" for s in positions) or "—"
            print(f"  │  Held: {held_str:<55} │")

            print(f"  │  Top 5 α today:                                                    │")
            for _, r in top5.iterrows():
                print(f"  │    {r['symbol']:6s}  α={r['alpha']:+.3f}  "
                      f"mom={r['momentum']:+.2f}  gap={r['gap_raw']:+.1f}%  "
                      f"RSI={r['rsi_raw']:5.1f}  5d={r['ret_5d']:+.1f}%        │")

            day_trades = [t for t in trade_log if t["date"] == today]
            if day_trades:
                print(f"  │  Trades today:                                                     │")
                for t in day_trades:
                    pnl_str = f"  P&L {t['pnl_pct']:+.1f}%" if t["pnl_pct"] is not None else ""
                    print(f"  │    {t['action']:12s} {t['symbol']:6s}  "
                          f"${t['price']:8.2f}  {t['shares']:8.1f}sh  "
                          f"${t['value']:>10,.0f}{pnl_str:<12}  │")
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
    print("  BACKTEST SUMMARY")
    print("=" * 72)
    print(f"  Total Return:          {stats['total_return']:>+8.2f}%")
    print(f"  Final Portfolio Value: ${stats['final_value']:>12,.0f}")
    print(f"  Max Drawdown:          {stats['max_drawdown']:>+8.2f}%")
    print(f"  Win Days / Loss Days:  {stats['win_days']} / {stats['loss_days']}")
    print(f"  Avg Daily Return:      {stats['avg_daily_ret']:>+8.3f}%")
    print(f"  Daily Std Dev:         {stats['daily_std']:>8.3f}%")
    print(f"  Annualised Sharpe:     {stats['annualised_sharpe']:>8.2f}")
    print(f"  Total Trades:          {len(trade_log)}")
    buys     = [t for t in trade_log if t["action"] == "BUY"]
    sells_pnl= [t for t in trade_log if t["pnl_pct"] is not None]
    if sells_pnl:
        wins  = [t for t in sells_pnl if t["pnl_pct"] > 0]
        losses= [t for t in sells_pnl if t["pnl_pct"] <= 0]
        avg_w = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
        avg_l = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
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
    fig.suptitle("Alpha Score Strategy — 10-Day Backtest",
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
    n_pos = [len(d["positions"]) for d in daily_log]
    ax3.fill_between(dates, n_pos, step="mid", color=BLUE, alpha=0.3)
    ax3.step(dates, n_pos, color=BLUE, linewidth=1.5, where="mid")
    ax3.set_ylim(0, MAX_POSITIONS + 0.5)
    ax3.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax3.set_title("Positions Held", color=TEXT, fontsize=10, pad=8)
    ax3.set_ylabel("# Positions", color=GREY, fontsize=8)
    ax3.set_xlim(dates[0], dates[-1])
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Alpha threshold lines ──────────────────────────────────────────────
    ax3.axhline(MAX_POSITIONS, color=GREEN, linewidth=0.8,
                linestyle=":", alpha=0.5, label=f"Max ({MAX_POSITIONS})")

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
    print("\n  Alpha Score Backtester — MiF Deep Learning Competition")
    print(f"  Universe: {len(UNIVERSE)} liquid stocks")
    print(f"  Lookback: {LOOKBACK_DAYS} days  |  Backtest window: {BACKTEST_DAYS} trading days")

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

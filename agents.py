"""
agents.py — Four-agent pipeline  ·  MiF AI Portfolio  ·  Group C

Alpha Score (same formula as backtest.py — live system = backtested system):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Component          Weight   Signal (each normalised to [-1, +1])   │
  │  5-day momentum      0.40    ret_5d / 0.20, clipped                 │
  │  Opening gap         0.25    (open/prev_close-1) / 0.05, clipped    │
  │  Trend vs SMA-20     0.15    +1 above, -1 below                     │
  │  RSI-14 sweet spot   0.10    +1 if 40-65, +0.5 if 35-40, -1 if >70 │
  │  Relative volume     0.10    (vol/avg_vol-1) / 2, clipped           │
  ├─────────────────────────────────────────────────────────────────────┤
  │  BUY threshold:  alpha > 0.45                                       │
  │  SELL threshold: alpha < 0.10                                       │
  └─────────────────────────────────────────────────────────────────────┘
"""

import json
import math
import os
import re

import anthropic

from database import get_positions, get_trades_today, record_trade
from portfolio_manager import get_market_snapshot, format_pct, format_dollar

# ── Competition parameters ────────────────────────────────────────────────────
MAX_POSITION_PCT = 0.40
MAX_TRADES_DAY   = 2
TC_BPS           = 10.0

# ── Alpha score weights — auto-loaded from optimal_weights.json if present ───
# Run optimize_weights.py first to generate that file.
# If not present, sensible defaults are used.
_DEFAULT_WEIGHTS = {
    "W_MOMENTUM": 0.40, "W_GAP": 0.25, "W_TREND": 0.15,
    "W_RSI": 0.10, "W_VOLUME": 0.10,
    "BUY_THRESHOLD": 0.45, "SELL_THRESHOLD": 0.10,
}

def _load_weights():
    wf = os.path.join(os.path.dirname(__file__), "optimal_weights.json")
    if os.path.exists(wf):
        try:
            with open(wf) as f:
                w = json.load(f)
            print(f"  ✅ Loaded optimal weights from optimal_weights.json "
                  f"(median {w.get('median_return_pct',0):+.2f}% over "
                  f"{w.get('n_windows','?')} windows)")
            return w
        except Exception as e:
            print(f"  ⚠️  Could not load optimal_weights.json: {e} — using defaults")
    return _DEFAULT_WEIGHTS

_w = _load_weights()
W_MOMENTUM = _w.get("W_MOMENTUM", _DEFAULT_WEIGHTS["W_MOMENTUM"])
W_GAP      = _w.get("W_GAP",      _DEFAULT_WEIGHTS["W_GAP"])
W_TREND    = _w.get("W_TREND",    _DEFAULT_WEIGHTS["W_TREND"])
W_RSI      = _w.get("W_RSI",      _DEFAULT_WEIGHTS["W_RSI"])
W_VOLUME   = _w.get("W_VOLUME",   _DEFAULT_WEIGHTS["W_VOLUME"])
BUY_THRESHOLD  = _w.get("BUY_THRESHOLD",  _DEFAULT_WEIGHTS["BUY_THRESHOLD"])
SELL_THRESHOLD = _w.get("SELL_THRESHOLD", _DEFAULT_WEIGHTS["SELL_THRESHOLD"])

WEIGHTS_SOURCE = ("optimal_weights.json"
                  if os.path.exists(os.path.join(os.path.dirname(__file__), "optimal_weights.json"))
                  else "defaults")

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL":"Tech",    "MSFT":"Tech",    "GOOGL":"Tech",   "META":"Tech",
    "NVDA":"AI/Semi", "AMD":"AI/Semi",  "ARM":"AI/Semi",  "AVGO":"AI/Semi",
    "MRVL":"AI/Semi", "QCOM":"AI/Semi", "MU":"AI/Semi",   "AMAT":"AI/Semi",
    "LRCX":"AI/Semi", "SMCI":"AI/Semi",
    "AMZN":"Consumer","TSLA":"Auto/EV",
    "COST":"Consumer","WMT":"Consumer", "HD":"Consumer",
    "NKE":"Consumer", "SBUX":"Consumer","MCD":"Consumer",
    "JPM":"Finance",  "GS":"Finance",   "MS":"Finance",   "V":"Finance",
    "BAC":"Finance",  "MA":"Finance",   "AXP":"Finance",  "C":"Finance",
    "UNH":"Health",   "LLY":"Health",   "JNJ":"Health",   "ABBV":"Health",
    "MRK":"Health",   "PFE":"Health",   "AMGN":"Health",  "ISRG":"Health",
    "XOM":"Energy",   "CVX":"Energy",   "OXY":"Energy",   "COP":"Energy",
    "LMT":"Defence",  "RTX":"Defence",  "NOC":"Defence",  "GD":"Defence",
    "PLTR":"AI/Data", "NET":"AI/Data",  "DDOG":"AI/Data", "SNOW":"AI/Data",
    "CRM":"AI/Data",  "NOW":"AI/Data",  "PANW":"AI/Data", "CRWD":"AI/Data",
    "COIN":"Crypto",  "MSTR":"Crypto",
    "UBER":"Growth",  "SHOP":"Growth",  "PYPL":"Growth",
    "SPY":"Index",    "QQQ":"Index",    "IWM":"Index",    "DIA":"Index",
    "TLT":"Bonds",    "GLD":"Commod",   "SLV":"Commod",
}


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Score helpers  (identical to backtest.py)
# ─────────────────────────────────────────────────────────────────────────────

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _rsi(prices: list, period: int = 14):
    if len(prices) < period + 1:
        return None
    changes  = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains    = [max(c, 0)     for c in changes[-period:]]
    losses   = [abs(min(c,0)) for c in changes[-period:]]
    avg_gain = sum(gains)  / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 1)


def _sma(prices: list, period: int):
    return sum(prices[-period:]) / period if len(prices) >= period else None


def _bollinger(prices: list, period: int = 20):
    if len(prices) < period:
        return None
    w   = prices[-period:]
    mid = sum(w) / period
    std = math.sqrt(sum((p - mid)**2 for p in w) / period)
    if std == 0:
        return None
    lo, hi = mid - 2*std, mid + 2*std
    return {"mid": round(mid,2), "upper": round(hi,2), "lower": round(lo,2),
            "position": round((prices[-1] - lo) / (hi - lo), 3)}


def _atr_pct(prices: list, period: int = 14):
    if len(prices) < period + 1:
        return None
    ranges = [abs(prices[i] - prices[i-1]) / prices[i-1]
              for i in range(1, len(prices))][-period:]
    return round(sum(ranges) / period * 100, 2)


def compute_alpha_score(prices: list, volumes: list, today_open: float = 0) -> dict:
    """
    Compute the Alpha Score for one stock.
    Identical formula to backtest.py — live system and backtest are the same model.

    prices:      list of daily closes, oldest → newest
    volumes:     list of daily volumes, same length
    today_open:  today's open price (0 = unavailable → gap component = 0)
    """
    empty = {"alpha": 0.0, "alpha_label": "➖ NEUTRAL",
             "momentum": 0, "gap": 0, "trend": 0, "rsi": 0, "volume": 0,
             "ret_5d": None, "gap_raw": 0, "rsi14": None,
             "sma20": None, "bb_pos": None, "bb_lower": None, "bb_upper": None,
             "atr_pct": None, "above_sma20": 0,
             "composite": 0.0, "rsi_signal": "neutral"}
    if len(prices) < 21:
        return empty

    cur     = prices[-1]
    prev_5d = prices[-6] if len(prices) >= 6 else prices[0]

    # 1. Momentum — 5-day return normalised at ±20%
    ret_5d    = (cur / prev_5d - 1) if prev_5d > 0 else 0
    mom_score = _clip(ret_5d / 0.20, -1.0, 1.0)

    # 2. Opening gap — today open vs yesterday close, normalised at ±5%
    gap       = (today_open / cur - 1) if (today_open > 0 and cur > 0) else 0
    gap_score = _clip(gap / 0.05, -1.0, 1.0)

    # 3. Trend vs SMA-20
    sma20       = _sma(prices, 20)
    trend_score = 1.0 if (sma20 and cur > sma20) else -1.0

    # 4. RSI sweet spot
    rsi14 = _rsi(prices, 14)
    if   rsi14 and 40 <= rsi14 <= 65: rsi_score =  1.0
    elif rsi14 and 35 <= rsi14 <  40: rsi_score =  0.5
    elif rsi14 and rsi14 > 70:        rsi_score = -1.0
    else:                             rsi_score =  0.0

    # 5. Relative volume
    avg_vol   = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
    today_vol = volumes[-1] if volumes else 0
    vol_score = _clip((today_vol / avg_vol - 1) / 2.0, -1.0, 1.0) if avg_vol > 0 else 0

    alpha = (W_MOMENTUM * mom_score
           + W_GAP      * gap_score
           + W_TREND    * trend_score
           + W_RSI      * rsi_score
           + W_VOLUME   * vol_score)

    # Supporting indicators for UI display
    bb  = _bollinger(prices, 20)
    atr = _atr_pct(prices, 14)

    if   alpha >= BUY_THRESHOLD: label = "🚀 BUY"
    elif alpha >= 0.30:           label = "📈 WATCH"
    elif alpha <= -0.20:          label = "📉 AVOID"
    else:                         label = "➖ NEUTRAL"

    return {
        # Primary signal
        "alpha":       round(alpha, 4),
        "alpha_label": label,
        # Component breakdown
        "momentum":    round(mom_score, 3),
        "gap":         round(gap_score, 3),
        "trend":       round(trend_score, 3),
        "rsi":         round(rsi_score, 3),
        "volume":      round(vol_score, 3),
        # Human-readable raw values
        "ret_5d":      round(ret_5d * 100, 2),
        "gap_raw":     round(gap * 100, 2),
        "rsi14":       rsi14,
        "sma20":       round(sma20, 2) if sma20 else None,
        "bb_pos":      bb["position"] if bb else None,
        "bb_lower":    bb["lower"]    if bb else None,
        "bb_upper":    bb["upper"]    if bb else None,
        "atr_pct":     atr,
        "above_sma20": 1 if (sma20 and cur > sma20) else -1,
        # Legacy aliases so existing UI code doesn't break
        "composite":   round(alpha, 4),
        "rsi_signal":  ("overbought" if rsi14 and rsi14 > 70
                        else "oversold" if rsi14 and rsi14 < 35
                        else "neutral"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Data Agent
# ─────────────────────────────────────────────────────────────────────────────

class DataAgent:
    """
    Enriches market snapshot with Alpha Scores.
    Returns (markdown_summary, enriched_snap_dict).
    """

    def run(self, market_snap: dict, sim_state: dict,
            price_history: dict = None) -> tuple:

        date        = sim_state["current_date"]
        total_value = sim_state["total_value"]
        cash        = sim_state["cash_balance"]
        positions   = get_positions()

        # Enrich every ticker with alpha score
        enriched = {}
        for ticker, m in market_snap.items():
            closes     = (price_history or {}).get(ticker, [])
            volumes    = (price_history or {}).get(f"{ticker}_vol", [])
            today_open = float(m.get("open", 0) or 0)
            sigs       = compute_alpha_score(closes, volumes, today_open)
            enriched[ticker] = {**m, **sigs}

        # Sort by alpha descending
        ranked = sorted(enriched.keys(),
                        key=lambda t: enriched[t].get("alpha", 0), reverse=True)

        buy_signals = [t for t in ranked if enriched[t]["alpha"] >= BUY_THRESHOLD]
        watch_list  = [t for t in ranked
                       if 0.30 <= enriched[t]["alpha"] < BUY_THRESHOLD]

        lines = [
            "## 📊 Data Agent — Alpha Score Table\n",
            f"**Date:** `{date}` | **Portfolio:** {format_dollar(total_value)} "
            f"| **Cash:** {format_dollar(cash)} ({cash/total_value*100:.1f}%)\n",
            "**Alpha = 0.40×momentum + 0.25×gap + 0.15×trend + 0.10×RSI + 0.10×volume**  "
            f"| BUY: α>{BUY_THRESHOLD}  SELL: α<{SELL_THRESHOLD}\n",
            "| Ticker | Sector | α Score | Label | mom | gap% | RSI | SMA20 | ATR% | 5d% |",
            "|--------|--------|---------|-------|-----|------|-----|-------|------|-----|",
        ]

        for ticker in ranked:
            m   = enriched[ticker]
            sec = SECTOR_MAP.get(ticker, "—")
            a   = m.get("alpha", 0)
            mom = f"{m['momentum']:+.2f}" if m.get("momentum") is not None else "—"
            gap = f"{m['gap_raw']:+.1f}%" if m.get("gap_raw")  is not None else "—"
            rsi = f"{m['rsi14']:.0f}"      if m.get("rsi14")   is not None else "—"
            s20 = f"${m['sma20']:,.0f}"    if m.get("sma20")                else "—"
            atr = f"{m['atr_pct']:.1f}%"  if m.get("atr_pct") is not None else "—"
            r5  = f"{m['ret_5d']:+.1f}%"  if m.get("ret_5d")  is not None else "—"
            lines.append(
                f"| **{ticker}** | {sec} | **{a:+.3f}** | {m['alpha_label']} | "
                f"{mom} | {gap} | {rsi} | {s20} | {atr} | {r5} |"
            )

        lines += [
            f"\n**🚀 BUY signals (α ≥ {BUY_THRESHOLD}):** {', '.join(buy_signals) or '—'}",
            f"**📈 Watch  (α 0.30–{BUY_THRESHOLD}):**  {', '.join(watch_list[:8]) or '—'}",
        ]

        # Current holdings block
        if positions:
            lines += [
                "\n**Holdings with live alpha:**",
                "| Ticker | Shares | Cost | Price | P/L% | α | Target | Stop |",
                "|--------|--------|------|-------|------|---|--------|------|",
            ]
            for pos in positions:
                sym  = pos["symbol"]
                cp   = enriched.get(sym, {}).get("price") or pos["avg_cost"]
                pnl  = (cp - pos["avg_cost"]) / pos["avg_cost"] * 100
                a    = enriched.get(sym, {}).get("alpha", 0)
                tgt  = f"${pos['target_price']:.2f}" if pos.get("target_price") else "—"
                stp  = f"${pos['stop_loss']:.2f}"    if pos.get("stop_loss")    else "—"
                warn = " ⚠️ EXIT" if a < SELL_THRESHOLD else ""
                lines.append(
                    f"| **{sym}** | {pos['shares']:.2f} | ${pos['avg_cost']:.2f} | "
                    f"${cp:,.2f} | {pnl:+.2f}% | **{a:+.3f}**{warn} | {tgt} | {stp} |"
                )
        else:
            lines.append("\n*No open positions — fully in cash.*")

        return "\n".join(lines), enriched


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Strategy Agent
# ─────────────────────────────────────────────────────────────────────────────

class StrategyAgent:
    """
    Claude + web search decides trades based on alpha scores.
    Decision rule mirrors backtest.py:
      BUY  if alpha > BUY_THRESHOLD  (0.45) and position not held
      SELL if alpha < SELL_THRESHOLD (0.10) or stop/target triggered
    Claude validates with live news before committing.
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def run(self, enriched_snap: dict, sim_state: dict,
            data_summary: str, allowed_assets: list) -> tuple:

        positions    = get_positions()
        total_value  = sim_state["total_value"]
        cash         = sim_state["cash_balance"]
        trades_today = get_trades_today(sim_state["current_date"])
        remaining    = MAX_TRADES_DAY - trades_today

        # ── Sector exposure ───────────────────────────────────────────────────
        sector_exp = {}
        for pos in positions:
            cp  = enriched_snap.get(pos["symbol"], {}).get("price") or pos["avg_cost"]
            val = pos["shares"] * cp
            sec = SECTOR_MAP.get(pos["symbol"], "Other")
            sector_exp[sec] = sector_exp.get(sec, 0) + val / total_value
        sector_text = ", ".join(
            f"{s} {w*100:.1f}%"
            for s, w in sorted(sector_exp.items(), key=lambda x: -x[1])
        ) or "none (all cash)"

        # ── Holdings summary ──────────────────────────────────────────────────
        held_symbols = [p["symbol"] for p in positions]
        positions_text = "\n".join(
            f"  - {p['symbol']} [{SECTOR_MAP.get(p['symbol'],'?')}]: "
            f"{p['shares']:.2f} sh @ ${p['avg_cost']:.2f}  "
            f"α={enriched_snap.get(p['symbol'],{}).get('alpha','?')}  "
            f"target={'$'+str(p['target_price']) if p.get('target_price') else 'none'}  "
            f"stop={'$'+str(p['stop_loss']) if p.get('stop_loss') else 'none'}"
            for p in positions
        ) or "  - (none — fully in cash)"

        # ── Top alpha candidates ──────────────────────────────────────────────
        ranked = sorted(
            [(t, m) for t, m in enriched_snap.items()
             if t not in ("SPY","QQQ","IWM","DIA") and t in allowed_assets],
            key=lambda x: x[1].get("alpha", 0), reverse=True
        )
        top_candidates = ranked[:12]

        alpha_table = "\n".join(
            f"  {t:6s}[{SECTOR_MAP.get(t,'?'):8s}] "
            f"α={m.get('alpha',0):+.3f} ({m.get('alpha_label','—')})  "
            f"mom={m.get('momentum',0):+.2f}  gap={m.get('gap_raw',0):+.1f}%  "
            f"RSI={m.get('rsi14','?')}  5d={m.get('ret_5d',0):+.1f}%  "
            f"ATR={m.get('atr_pct','?')}%  ${m.get('price',0):,.2f}"
            for t, m in top_candidates
        )

        # ── Overnight stop/target alerts ──────────────────────────────────────
        from database import get_target_alerts
        alerts       = get_target_alerts(enriched_snap)
        alert_lines  = []
        forced_sells = []
        for a in alerts:
            sym, kind = a["symbol"], a["type"]
            if kind == "STOP_HIT":
                alert_lines.append(
                    f"  🛑 STOP HIT   {sym}: price ${a['price']:.2f} ≤ stop ${a['level']:.2f} "
                    f"({a['pnl_pct']:+.1f}%) → MUST SELL"
                )
                forced_sells.append(sym)
            else:
                alert_lines.append(
                    f"  🎯 TARGET HIT {sym}: price ${a['price']:.2f} ≥ target ${a['level']:.2f} "
                    f"({a['pnl_pct']:+.1f}%) → consider selling"
                )
        # Positions whose alpha has dropped below sell threshold
        alpha_exits = [
            p["symbol"] for p in positions
            if enriched_snap.get(p["symbol"], {}).get("alpha", 1) < SELL_THRESHOLD
            and p["symbol"] not in forced_sells
        ]
        if alpha_exits:
            for sym in alpha_exits:
                a = enriched_snap.get(sym, {}).get("alpha", 0)
                alert_lines.append(
                    f"  📉 ALPHA EXIT {sym}: α={a:+.3f} < sell threshold {SELL_THRESHOLD} → exit"
                )

        alert_block = (
            "⚠️  EXITS REQUIRED — handle BEFORE new entries:\n" + "\n".join(alert_lines)
            if alert_lines else "✅  No exits required."
        )

        # Build strategy explanation block shown in Claude's reasoning
        weights_note = (f"weights optimised over {_w.get('n_windows','?')} "
                        f"× 14-day windows (median {_w.get('median_return_pct',0):+.2f}%)"
                        if WEIGHTS_SOURCE == "optimal_weights.json"
                        else "default weights — run optimize_weights.py to improve")

        prompt = f"""You are a quantitative AI portfolio manager in a 2-week paper trading competition ending 2026-03-27.

════════════════ CURRENT STRATEGY SUMMARY ════════════════
Strategy name:    Alpha Score Momentum
Execution model:  Once-daily at market open (daily close signals + today's open gap)
Universe:         ~60 liquid stocks across 8 sectors (no micro-caps)
Max positions:    2 simultaneous, up to {MAX_POSITION_PCT*100:.0f}% each
Stop-loss:        -6% hard stop (checked each morning vs yesterday's close)
Weights source:   {weights_note}

Alpha Score formula (each component normalised to [-1, +1]):
  α = {W_MOMENTUM:.2f} × momentum   +  {W_GAP:.2f} × opening_gap  +  {W_TREND:.2f} × trend_SMA20
    + {W_RSI:.2f} × RSI_sweet    +  {W_VOLUME:.2f} × rel_volume

  Component definitions:
    momentum    = clip(5d_return / 0.20, -1, +1)        ← biggest weight: sustained price move
    opening_gap = clip((open/prev_close - 1) / 0.05, -1, +1)  ← gap-up at open = momentum signal
    trend_SMA20 = +1 if price > SMA-20, else -1          ← confirms medium-term direction
    RSI_sweet   = +1.0 if RSI 40-65 | +0.5 if 35-40 | -1.0 if RSI>70  ← avoid overbought
    rel_volume  = clip((vol/avg_vol - 1) / 2, -1, +1)   ← high volume confirms the move

  Thresholds:  BUY if α > {BUY_THRESHOLD}  |  SELL if α < {SELL_THRESHOLD}
══════════════════════════════════════════════════════════

EXECUTION MODEL: Code runs ONCE per day at market open. You cannot monitor intraday.
All signals use yesterday's closing prices. The opening gap uses today's open vs yesterday's close.

═══════════════ SITUATION ═══════════════
Date:        {sim_state['current_date']}  (Day {sim_state['day_number']+1} of ~11)
Portfolio:   {format_dollar(total_value)}
Cash:        {format_dollar(cash)} ({cash/total_value*100:.1f}%)
Trades left: {remaining} of {MAX_TRADES_DAY}
Sectors:     {sector_text}

Holdings:
{positions_text}

{alert_block}

═══════════════ TOP ALPHA CANDIDATES ═══════════════
{alpha_table}

Full alpha table with all stocks:
{data_summary}

═══════════════ DECISION RULES ═══════════════
1. EXITS FIRST (mandatory before any new entries):
   • STOP HIT symbols → SELL immediately
   • Alpha exits (α < {SELL_THRESHOLD}) → SELL at open
   • Target hits → sell and rotate

2. NEW ENTRIES (for remaining trade slots):
   • Only consider stocks where alpha > {BUY_THRESHOLD}
   • NEVER buy RSI > 70 (overbought, even if alpha is borderline)
   • NEVER buy if alpha driven purely by gap with no momentum (gap > 0.8, momentum < 0)
   • Use web search to confirm the top 2-3 alpha candidates have real catalysts
   • ATR > 3% → cap position at 25% regardless of alpha

3. SIZING:
   • alpha > 0.70 → up to 40% (high conviction)
   • alpha 0.45–0.70 → 25–35%
   • ATR > 3% → max 25%

═══════════════ YOUR TASK ═══════════════
STEP 0 — EXITS (no web search needed):
  Process all forced exits from alerts above.

STEP 1 — WEB SEARCH (top alpha candidates only):
  For each stock where alpha > {BUY_THRESHOLD}, search "[TICKER] stock news {sim_state['current_date']}"
  Confirm: is there a real catalyst (earnings, upgrade, product news) or is this noise?
  Also search: "S&P500 market outlook {sim_state['current_date']}" for macro context.

STEP 2 — SELECT trades for remaining {remaining} slot(s).
  Pick the highest-alpha stock with confirmed news catalyst.
  Set target = entry × 1.10 to 1.15, stop = entry × 0.93 to 0.95.

STEP 3 — OUTPUT as JSON only (no other text after the block):

The "reason" field is the most important field — it will be stored permanently in the trade log
and shown in the dashboard. Write it as a thorough, human-readable explanation covering:
  • The alpha score breakdown (which components drove the signal)
  • The specific news catalyst found via web search (company name, event, source)
  • The macro context if relevant (sector tailwinds, Fed, earnings season)
  • Why you chose THIS stock over the other candidates
  • For SELLs: what changed since entry (alpha decay, stop hit, target reached, thesis broken)

Aim for 2-3 sentences. This is the record that explains every decision to the professor and to
the team. Make it informative enough that someone reading it months later understands exactly
why the trade was made.

Example BUY reason (use this style):
  "Alpha score 0.68 driven by 5d momentum +9.2% and gap-up +2.4% at open, RSI 54 in sweet spot,
  above SMA-20 confirming uptrend. Web search found Goldman Sachs upgraded the stock to Buy with
  $180 target citing accelerating data centre demand — strong fundamental catalyst backing the
  technical signal. Chosen over AVGO (α=0.51) because NVDA has higher momentum score and fresher
  analyst catalyst."

Example SELL reason (use this style):
  "Alpha score decayed to 0.07, below the 0.10 sell threshold. 5d momentum has faded to +1.2%
  after the initial move was absorbed. No new catalyst found in web search to justify holding.
  Exiting at open to free capital for stronger opportunities; held for 4 days, P&L approximately
  +3.2% from entry."

```json
[
  {{
    "action": "SELL",
    "symbol": "EXITED_TICKER",
    "target_pct": 0.0,
    "target_price": null,
    "stop_loss": null,
    "holding_days": null,
    "framework": "alpha_exit",
    "reason": "2-3 sentence explanation: alpha level, what changed, P&L if known"
  }},
  {{
    "action": "BUY",
    "symbol": "TOP_ALPHA_TICKER",
    "target_pct": 0.35,
    "target_price": 155.00,
    "stop_loss": 130.00,
    "holding_days": 6,
    "framework": "momentum",
    "reason": "2-3 sentence explanation: alpha breakdown, news catalyst, why chosen over alternatives"
  }}
]
```

`framework`: "momentum" | "alpha_exit" | "stop_exit" | "target_exit" | "sector_rotation"
`target_pct`: desired portfolio weight (0.0–{MAX_POSITION_PCT})
Allowed assets: {allowed_assets}
End with the JSON block."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        full_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        trades = self._parse_trades(full_text)
        return trades, full_text

    @staticmethod
    def _parse_trades(text: str) -> list:
        match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"(\[[\s\S]*?\"action\"[\s\S]*?\])", text)
        if not match:
            return []
        try:
            trades = json.loads(match.group(1))
            out = []
            for t in trades:
                if not all(k in t for k in ("action", "symbol")):
                    continue
                t["target_pct"]   = max(0.0, min(float(t.get("target_pct", 0)), MAX_POSITION_PCT))
                t["action"]       = t["action"].upper()
                t["target_price"] = float(t["target_price"]) if t.get("target_price") else None
                t["stop_loss"]    = float(t["stop_loss"])    if t.get("stop_loss")    else None
                t["holding_days"] = int(t["holding_days"])   if t.get("holding_days") else None
                out.append(t)
            return out
        except (json.JSONDecodeError, KeyError, ValueError):
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Risk Agent
# ─────────────────────────────────────────────────────────────────────────────

class RiskAgent:
    """Hard constraints + volatility-adjusted size cap."""

    def run(self, proposed: list, sim_state: dict,
            market_snap: dict) -> tuple:

        total_value = sim_state["total_value"]
        cash        = sim_state["cash_balance"]
        date_str    = sim_state["current_date"]
        trades_used = get_trades_today(date_str)
        budget      = MAX_TRADES_DAY - trades_used

        approved, rejected = [], []

        for t in proposed:
            symbol     = t["symbol"]
            action     = t["action"]
            target_pct = t.get("target_pct", 0.0)
            price      = market_snap.get(symbol, {}).get("price")

            if price is None:
                rejected.append((t, "No live price")); continue
            if len(approved) >= budget:
                rejected.append((t, f"Daily trade limit ({MAX_TRADES_DAY}) reached")); continue
            if target_pct > MAX_POSITION_PCT:
                rejected.append((t, f"Size {target_pct*100:.0f}% > cap {MAX_POSITION_PCT*100:.0f}%")); continue

            # ATR volatility cap
            atr = market_snap.get(symbol, {}).get("atr_pct")
            if atr and atr > 3.0 and target_pct > 0.25:
                t["target_pct"] = 0.25
                t["reason"]     = (t.get("reason", "") + f" [ATR {atr:.1f}% → capped 25%]")

            # Alpha sanity check for buys
            if action == "BUY":
                alpha = market_snap.get(symbol, {}).get("alpha", 1.0)
                if alpha < SELL_THRESHOLD:
                    rejected.append((t, f"Alpha {alpha:.3f} below sell threshold — risk veto")); continue
                required = t["target_pct"] * total_value
                if required > cash + 1e-2:
                    rejected.append((t, f"Insufficient cash: need {format_dollar(required)}")); continue

            approved.append(t)

        lines = [
            "## 🛡️ Risk Agent — Validation\n",
            f"**Proposals:** {len(proposed)} | **✅ Approved:** {len(approved)} | **❌ Rejected:** {len(rejected)}\n",
            "### ✅ Approved",
        ]
        if approved:
            lines += ["| # | Action | Symbol | Weight | α Score | Framework | Reason |",
                      "|---|--------|--------|--------|---------|-----------|--------|"]
            for i, t in enumerate(approved, 1):
                em    = "🟢" if t["action"] == "BUY" else "🔴"
                alpha = market_snap.get(t["symbol"], {}).get("alpha", "—")
                alpha_str = f"{alpha:+.3f}" if isinstance(alpha, float) else str(alpha)
                lines.append(
                    f"| {i} | {em} **{t['action']}** | `{t['symbol']}` | "
                    f"{t.get('target_pct',0)*100:.1f}% | {alpha_str} | "
                    f"{t.get('framework','—')} | {str(t.get('reason',''))[:60]} |"
                )
        else:
            lines.append("*No trades approved.*")

        if rejected:
            lines += ["\n### ❌ Rejected", "| Symbol | Reason |", "|--------|--------|"]
            for t, reason in rejected:
                lines.append(f"| `{t['symbol']}` | {reason} |")

        lines += [
            f"\n### 📋 Limits",
            f"- Max position: **{MAX_POSITION_PCT*100:.0f}%** (ATR >3% → 25%)",
            f"- Trades today: used {trades_used}, approved {len(approved)} more of {MAX_TRADES_DAY} max",
            f"- Alpha floor for buys: **{SELL_THRESHOLD}**",
        ]
        return approved, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Execution Agent
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionAgent:
    """Commits approved trades to SQLite."""

    def run(self, approved: list, sim_state: dict,
            market_snap: dict) -> tuple:

        date_str    = sim_state["current_date"]
        total_value = sim_state["total_value"]
        executed    = []

        for t in approved:
            symbol     = t["symbol"]
            target_pct = t.get("target_pct", 0.0)
            action     = t["action"]
            price      = market_snap.get(symbol, {}).get("price")

            if not price:
                continue
            shares = round(target_pct * total_value / price, 4)
            if shares <= 0:
                continue

            # Build threshold breakdown suffix — always appended so the
            # dashboard reason is self-contained and auditable
            snap = market_snap.get(symbol, {})
            alpha    = snap.get("alpha")
            mom      = snap.get("momentum")
            gap      = snap.get("gap_raw")
            rsi      = snap.get("rsi14")
            trend    = snap.get("above_sma20")
            atr      = snap.get("atr_pct")
            ret5d    = snap.get("ret_5d")

            if alpha is not None:
                threshold_line = (
                    f" | Thresholds crossed — "
                    f"alpha={alpha:+.3f} (buy>{BUY_THRESHOLD}) | "
                    f"5d-momentum={ret5d:+.1f}% | "
                    f"gap={gap:+.1f}% | "
                    f"RSI={rsi:.0f} (sweet-spot 40-65) | "
                    f"trend={'above' if trend else 'below'} SMA-20 | "
                    f"ATR={atr:.1f}%"
                ) if all(v is not None for v in [ret5d, gap, rsi, trend, atr]) else ""
            else:
                threshold_line = ""

            full_reason = t.get("reason", "Agent decision") + threshold_line

            record_trade(
                date_str=date_str, action=action, symbol=symbol,
                shares=shares, price=price, tc_bps=TC_BPS,
                reason=full_reason,
                agent="Execution",
                target_price=t.get("target_price") if action == "BUY" else None,
                stop_loss=t.get("stop_loss")        if action == "BUY" else None,
                holding_days=t.get("holding_days")  if action == "BUY" else None,
            )
            executed.append({
                "action":    action,
                "symbol":    symbol,
                "shares":    shares,
                "price":     price,
                "value":     shares * price,
                "tc":        shares * price * TC_BPS / 10_000,
                "framework": t.get("framework", "—"),
                "alpha":     market_snap.get(symbol, {}).get("alpha", "—"),
            })

        lines = ["## ⚙️ Execution Agent — Trade Log\n"]
        if executed:
            lines += [
                "| Action | Symbol | Shares | Price | Value | TC | α | Framework |",
                "|--------|--------|--------|-------|-------|----|---|-----------|",
            ]
            for e in executed:
                em        = "🟢" if e["action"] == "BUY" else "🔴"
                alpha_str = f"{e['alpha']:+.3f}" if isinstance(e["alpha"], float) else str(e["alpha"])
                lines.append(
                    f"| {em} **{e['action']}** | `{e['symbol']}` | {e['shares']:.2f} | "
                    f"${e['price']:.2f} | {format_dollar(e['value'])} | "
                    f"${e['tc']:.2f} | {alpha_str} | {e['framework']} |"
                )
            total_tc = sum(e["tc"] for e in executed)
            lines.append(f"\n**Total TC:** ${total_tc:.2f}  ({TC_BPS} bps)")
        else:
            lines.append("*No trades executed.*")

        return len(executed), "\n".join(lines)

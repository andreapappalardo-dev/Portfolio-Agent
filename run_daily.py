"""
run_daily.py — MiF AI Portfolio · Daily CLI Runner
===================================================
Run once per trading day:

    python run_daily.py

What it does (in order):
    1.  Loads / initialises the SQLite database (finance_game.db)
    2.  Advances the simulation date to the next business day
    3.  Fetches live prices via yfinance  (Data Agent)
    4.  Calls Claude claude-sonnet-4-20250514 + web search to propose trades  (Strategy Agent)
    5.  Validates proposals against hard risk constraints  (Risk Agent)
    6.  Commits approved trades to the database  (Execution Agent)
    7.  Prints a full daily summary to the terminal
    8.  Saves a daily snapshot so the Streamlit dashboard can plot performance

API key — set one of these before running:
    export ANTHROPIC_API_KEY="sk-ant-..."
    or add it to a .env file in this folder.

Competition: $1,000,000 virtual capital · March 13–27, 2026
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Check API key early ───────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not API_KEY:
    sys.exit(
        "\n❌  ANTHROPIC_API_KEY not set.\n"
        "    Add it to a .env file or run:\n"
        "        export ANTHROPIC_API_KEY='sk-ant-...'\n"
    )

import anthropic
import pandas as pd
import yfinance as yf

from database import (
    init_db, get_sim_state, get_positions, get_trades_today,
    record_trade, save_snapshot, reset_db,
)
from portfolio_manager import (
    fetch_prices, get_market_snapshot, get_portfolio_valuation,
    compute_kpis, get_snapshots, format_dollar, format_pct,
    STARTING_CAPITAL,
)

# ── Competition constants ─────────────────────────────────────────────────────
COMPETITION_START = "2026-03-13"
COMPETITION_END   = "2026-03-27"
ASSETS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "JPM", "GLD", "TLT", "SPY",
]
BENCHMARK        = "SPY"
MAX_POSITION_PCT = 0.15   # max 15% per position
MAX_TRADES_DAY   = 5      # raised to allow more daily trades
TC_BPS           = 10.0   # transaction cost in basis points


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
DIM   = "\033[2m"
RESET = "\033[0m"

def section(title: str) -> None:
    width = 64
    print(f"\n{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{'─' * width}{RESET}")

def ok(msg):  print(f"  {GREEN}✔{RESET}  {msg}")
def err(msg): print(f"  {RED}✘{RESET}  {msg}")
def info(msg):print(f"  {DIM}→{RESET}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — Data Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_data_agent(market_snap: dict, sim: dict) -> str:
    """
    Produces a plain-text market summary consumed by the Strategy Agent.
    Uses only prices up to sim['current_date'] — no future peeking.
    """
    positions = get_positions()
    cash      = sim["cash_balance"]
    total     = sim["total_value"]

    lines = [
        f"Date: {sim['current_date']}",
        f"Portfolio: {format_dollar(total)}  |  Cash: {format_dollar(cash)} ({cash/total*100:.1f}%)",
        "",
        f"{'Ticker':<8} {'Price':>9} {'1d':>8} {'5d':>8} {'20d':>8} {'AnnVol':>8}",
        "-" * 56,
    ]
    for ticker, m in market_snap.items():
        def _f(v): return "  n/a  " if v is None else f"{v*100:+.2f}%"
        vol = "  n/a " if not m["vol_20d"] else f"{m['vol_20d']*100:.1f}%"
        price_str = format_dollar(m["price"]) if m["price"] else "  n/a"
        lines.append(f"{ticker:<8} {price_str:>9} {_f(m['ret_1d']):>8} {_f(m['ret_5d']):>8} "
                     f"{_f(m['ret_20d']):>8} {vol:>8}")

    if positions:
        lines += ["", "Current Holdings:"]
        for p in positions:
            price = market_snap.get(p["symbol"], {}).get("price") or p["avg_cost"]
            mkt   = p["shares"] * price
            lines.append(f"  {p['symbol']:<6}  {p['shares']:.2f} shares  avg ${p['avg_cost']:.2f}"
                         f"  mkt {format_dollar(mkt)}")
    else:
        lines.append("\nHoldings: (none — fully in cash)")

    summary = "\n".join(lines)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — Strategy Agent  (Claude + web search)
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy_agent(
    market_snap: dict,
    sim: dict,
    data_summary: str,
) -> tuple[list[dict], str]:
    """
    Calls Claude claude-sonnet-4-20250514 with the web_search tool to:
      - research today's news for each asset
      - apply momentum reasoning
      - return structured BUY/SELL proposals + full reasoning text

    Returns: (proposed_trades, reasoning_text)
    """
    client       = anthropic.Anthropic(api_key=API_KEY)
    positions    = get_positions()
    cash         = sim["cash_balance"]
    total        = sim["total_value"]
    trades_used  = get_trades_today(sim["current_date"])
    remaining    = MAX_TRADES_DAY - trades_used
    max_pct      = int(MAX_POSITION_PCT * 100)

    held_txt = "\n".join(
        f"  - {p['symbol']}: {p['shares']:.2f} shares @ avg ${p['avg_cost']:.2f}"
        for p in positions
    ) or "  - (none — fully in cash)"

    # Compact market table (prices + 5d return only) to stay under rate limits
    mkt_lines = []
    for t, m in market_snap.items():
        r5 = f"{m['ret_5d']*100:+.1f}%" if m['ret_5d'] else "n/a"
        mkt_lines.append(f"{t}: ${m['price']:.0f} (5d {r5})")
    mkt_compact = " | ".join(mkt_lines)

    prompt = (
        f"You are an AI portfolio manager. Date: {sim['current_date']}. "
        f"Portfolio: {format_dollar(total)}, Cash: {format_dollar(cash)}. "
        f"Holdings: {held_txt}. "
        f"Market snapshot: {mkt_compact}. "
        f"Competition ends {COMPETITION_END} (~{remaining} trade(s) left today, max {max_pct}% per position). "
        "Use web search to find today's top market news. "
        "IMPORTANT REBALANCING RULES: "
        "1. If cash is insufficient to BUY a new position, you MUST first SELL a weaker existing holding to free up capital. "
        "2. Propose the SELL before the BUY in your JSON so cash is available when the BUY executes. "
        "3. To rotate: sell the holding with the weakest momentum or that has hit its stop-loss, then buy the better opportunity. "
        "4. Max 2 trades total per day (a SELL + BUY counts as 2 trades). "
        "Be bold — maximise returns. Reply ONLY with reasoning + this JSON at the end:\n"
        "```json\n"
        "[{\"action\":\"SELL\",\"symbol\":\"WEAK\",\"target_pct\":0.0,\"reason\":\"rotating out\"},{\"action\":\"BUY\",\"symbol\":\"X\",\"target_pct\":0.12,\"reason\":\"..\"}]\n"
        "```"
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    trades = _parse_trades(full_text)
    return trades, full_text


def _parse_trades(text: str) -> list[dict]:
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
            t["target_pct"] = max(0.0, min(float(t.get("target_pct", 0)), MAX_POSITION_PCT))
            t["action"]     = t["action"].upper()
            out.append(t)
        return out
    except (json.JSONDecodeError, ValueError):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — Risk Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_agent(
    proposed: list[dict],
    sim: dict,
    market_snap: dict,
) -> tuple[list[dict], list[tuple]]:
    """
    Enforces hard constraints. Returns (approved, rejected_with_reasons).
    """
    total       = sim["total_value"]
    cash        = sim["cash_balance"]
    trades_used = get_trades_today(sim["current_date"])
    budget      = MAX_TRADES_DAY - trades_used

    approved, rejected = [], []

    for t in proposed:
        symbol     = t["symbol"]
        action     = t["action"]
        target_pct = t.get("target_pct", 0.0)
        price      = market_snap.get(symbol, {}).get("price")

        if price is None:
            rejected.append((t, "No live price available")); continue

        if target_pct > MAX_POSITION_PCT:
            rejected.append((t, f"Target {target_pct*100:.0f}% > max {MAX_POSITION_PCT*100:.0f}%")); continue
        if action == "BUY":
            # Check combined weight (existing + new) doesn't exceed cap
            existing_pos = next((p for p in get_positions() if p["symbol"] == symbol), None)
            existing_pct = (existing_pos["shares"] * price / total) if existing_pos else 0.0
            combined_pct = existing_pct + target_pct
            if combined_pct > MAX_POSITION_PCT + 0.001:
                rejected.append((t, f"Combined weight {combined_pct*100:.1f}% > cap {MAX_POSITION_PCT*100:.0f}%")); continue
            required = target_pct * total
            if required > cash + 0.01:
                rejected.append((t, f"Insufficient cash: need {format_dollar(required)}, "
                                    f"have {format_dollar(cash)}")); continue
            cash -= required  # update running cash so subsequent BUYs see correct balance
        elif action == "SELL":
            # Credit cash from this sell so subsequent BUYs can use it
            position = next((p for p in get_positions() if p["symbol"] == symbol), None)
            if position:
                sell_value = position["shares"] * price
                tc_cost    = sell_value * TC_BPS / 10_000
                cash += sell_value - tc_cost

        approved.append(t)

    return approved, rejected


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — Execution Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_execution_agent(
    approved: list[dict],
    sim: dict,
    market_snap: dict,
) -> list[dict]:
    """
    Commits approved trades to SQLite. Returns list of executed trade dicts.
    """
    total    = sim["total_value"]
    executed = []

    positions = get_positions()
    pos_map   = {p["symbol"]: p for p in positions}

    for t in approved:
        symbol     = t["symbol"]
        target_pct = t.get("target_pct", 0.0)
        action     = t["action"]
        price = market_snap.get(symbol, {}).get("price")

        if action == "SELL":
            # Sell ALL shares of this position
            pos = pos_map.get(symbol)
            if not pos:
                continue
            # Fall back to avg_cost if live price unavailable
            if not price:
                price = pos["avg_cost"]
            shares = pos["shares"]
        else:
            # BUY: size based on target_pct of total portfolio
            if not price:
                continue
            shares = round(target_pct * total / price, 4)

        if shares <= 0:
            continue

        record_trade(
            date_str=sim["current_date"],
            action=action,
            symbol=symbol,
            shares=shares,
            price=price,
            tc_bps=TC_BPS,
            reason=t.get("reason", "Agent decision"),
            agent="Execution",
            target_price=t.get("target_price"),
            stop_loss=t.get("stop_loss"),
        )
        executed.append({
            "action": action,
            "symbol": symbol,
            "shares": shares,
            "price":  price,
            "value":  shares * price,
            "tc":     shares * price * TC_BPS / 10_000,
        })

    return executed


# ─────────────────────────────────────────────────────────────────────────────
# Main daily loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'='*64}{RESET}")
    print(f"{BOLD}  MiF AI Portfolio — Daily Runner{RESET}")
    print(f"  Competition: {COMPETITION_START} → {COMPETITION_END}")
    print(f"  Starting capital: {format_dollar(STARTING_CAPITAL)}")
    print(f"{BOLD}{'='*64}{RESET}")

    # ── 1. Database init ──────────────────────────────────────────────────────
    section("Step 1 — Database")
    init_db()
    sim = get_sim_state()

    if sim is None:
        info("No simulation found — initialising fresh database...")
        reset_db(STARTING_CAPITAL, COMPETITION_START)
        sim = get_sim_state()
        ok(f"Fresh simulation created · {format_dollar(STARTING_CAPITAL)} cash · date {sim['current_date']}")
    else:
        ok(f"Loaded existing simulation · Day {sim['day_number']} · {sim['current_date']}")

    # ── 2. Date check — NO auto-advance ──────────────────────────────────────
    # Date is advanced manually via the "Adv. Day" button in the dashboard.
    # run_daily.py always operates on the current sim date, whatever it is.
    section("Step 2 — Date Check")
    ok(f"Sim date: {sim['current_date']}  (Day {sim['day_number']})")
    days_left = max(0, (
        pd.Timestamp(COMPETITION_END) - pd.Timestamp(sim["current_date"])
    ).days)
    info(f"Days remaining in competition: {days_left}")

    # ── Idempotency guard — skip if daily trade limit already exhausted ────────
    trades_already = get_trades_today(sim["current_date"])

    # ── 3. Fetch market data ──────────────────────────────────────────────────
    section("Step 3 — Data Agent · Fetching Live Prices")
    info("Downloading prices via yfinance...")

    all_tickers = list(dict.fromkeys(ASSETS + [BENCHMARK]))
    prices_df   = fetch_prices(tuple(all_tickers), lookback_days=60)

    if prices_df.empty:
        sys.exit("❌  Could not fetch prices. Check internet connection.")

    asset_cols  = [a for a in ASSETS if a in prices_df.columns]
    market_snap = get_market_snapshot(prices_df[asset_cols])

    ok(f"Fetched prices for {len(market_snap)} assets")
    data_summary = run_data_agent(market_snap, sim)
    print()
    print(data_summary)

    # ── 3b. Run full 525-stock screener → screener_results.csv ───────────────
    section("Step 3b — Full Alpha Screener · 525 Stocks")
    try:
        from agents import compute_alpha_score
        from initialize_portfolio import UNIVERSE

        info(f"Scoring {len(UNIVERSE)} tickers in batches of 100…")
        scr_rows = []
        for i in range(0, len(UNIVERSE), 100):
            batch = UNIVERSE[i:i + 100]
            try:
                bdf = fetch_prices(tuple(batch), lookback_days=30)
            except Exception:
                continue
            for t in batch:
                if t not in bdf.columns:
                    continue
                col = bdf[t].dropna()
                if len(col) < 21:
                    continue
                price = float(col.iloc[-1])
                if price < 2:
                    continue
                prices_list = col.tolist()
                sigs = compute_alpha_score(prices_list, [], 0)
                ret_1d  = (prices_list[-1] / prices_list[-2]  - 1) if len(prices_list) >= 2  else 0
                ret_20d = (prices_list[-1] / prices_list[-21] - 1) if len(prices_list) >= 21 else 0
                scr_rows.append({
                    "ticker":  t,
                    "price":   round(price, 2),
                    "alpha":   round(sigs.get("alpha", 0), 4),
                    "ret_5d":  round(sigs.get("ret_5d") or 0, 2),
                    "ret_1d":  round(ret_1d * 100, 2),
                    "ret_20d": round(ret_20d * 100, 2),
                    "rsi14":   round(sigs.get("rsi14") or 0, 1),
                    "atr_pct": round(sigs.get("atr_pct") or 0, 2),
                })

        if scr_rows:
            scr_df = pd.DataFrame(scr_rows).sort_values("alpha", ascending=False)
            scr_df.to_csv("screener_results.csv", index=False)
            ok(f"Screener done — {len(scr_rows)} tickers scored, top: "
               f"{', '.join(scr_df.head(5)['ticker'].tolist())}")
        else:
            info("Screener returned no rows — watchlist only")
    except Exception as e:
        info(f"Screener failed ({e}) — watchlist only")

    # ── 3c. Extend market_snap with screener prices ───────────────────────────
    # The Strategy Agent picks from 525 tickers; the Risk Agent needs a price
    # for any ticker Claude proposes, not just the 25 in ASSETS.
    try:
        scr = pd.read_csv("screener_results.csv")
        scr.columns = [c.lstrip("\ufeff").strip() for c in scr.columns]
        added = 0
        for _, row in scr.iterrows():
            sym = str(row.get("ticker", "")).strip()
            px  = pd.to_numeric(row.get("price"), errors="coerce")
            if sym and pd.notna(px) and sym not in market_snap:
                market_snap[sym] = {
                    "price":   float(px),
                    "alpha":   float(pd.to_numeric(row.get("alpha",   0), errors="coerce") or 0),
                    "atr_pct": float(pd.to_numeric(row.get("atr_pct", 0), errors="coerce") or 0) or None,
                    "ret_5d":  float(pd.to_numeric(row.get("ret_5d",  0), errors="coerce") or 0),
                    "rsi14":   float(pd.to_numeric(row.get("rsi14",   0), errors="coerce") or 0) or None,
                    "ret_1d": None, "ret_20d": None, "vol_20d": None,
                }
                added += 1
        if added:
            info(f"market_snap extended: +{added} screener tickers → {len(market_snap)} total")
    except FileNotFoundError:
        info("screener_results.csv not found — market_snap limited to ASSETS watchlist")
    except Exception as e:
        info(f"Could not extend market_snap from screener: {e}")

    # ── 4. Strategy Agent (Claude + web search) ───────────────────────────────
    section("Step 4 — Strategy Agent · Claude + Web Search")
    info("Calling Claude claude-sonnet-4-20250514 with web search enabled… (20–45s)")

    proposed, reasoning_text = run_strategy_agent(market_snap, sim, data_summary)

    print(f"\n{DIM}── Agent Reasoning ──────────────────────────────────{RESET}")
    # Print first 2500 chars of reasoning (truncate for readability)
    preview = reasoning_text[:2500]
    if len(reasoning_text) > 2500:
        preview += f"\n{DIM}  … [truncated — {len(reasoning_text)} chars total]{RESET}"
    print(preview)
    print(f"{DIM}── End Reasoning ────────────────────────────────────{RESET}\n")

    if proposed:
        ok(f"{len(proposed)} trade(s) proposed:")
        for t in proposed:
            arrow = GREEN if t["action"] == "BUY" else RED
            print(f"    {arrow}{t['action']:4s}{RESET}  {t['symbol']:<6}  "
                  f"→ {t.get('target_pct', 0)*100:.0f}%   {t.get('reason', '')[:70]}")
    else:
        info("No trades proposed — agent recommends holding current positions")

    # ── 5. Risk Agent ──────────────────────────────────────────────────────────
    section("Step 5 — Risk Agent · Validating Proposals")
    approved, rejected = run_risk_agent(proposed, sim, market_snap)

    if approved:
        ok(f"{len(approved)} trade(s) approved:")
        for t in approved:
            arrow = GREEN if t["action"] == "BUY" else RED
            print(f"    {arrow}✔ {t['action']:4s}{RESET}  {t['symbol']:<6}  "
                  f"→ {t.get('target_pct', 0)*100:.0f}%")
    else:
        info("No trades approved")

    if rejected:
        print()
        for t, reason in rejected:
            err(f"REJECTED  {t['symbol']:<6}  {reason}")

    info(f"Constraints: max {MAX_POSITION_PCT*100:.0f}% per position · "
         f"max {MAX_TRADES_DAY} trades/day · cash ≥ $0")

    # ── 6. Execution Agent ────────────────────────────────────────────────────
    section("Step 6 — Execution Agent · Committing Trades")
    executed = run_execution_agent(approved, sim, market_snap)

    if executed:
        print(f"\n  {'Action':<6} {'Symbol':<8} {'Shares':>8} {'Price':>9} {'Value':>12} {'TC':>8}")
        print("  " + "-" * 56)
        for e in executed:
            arrow = GREEN if e["action"] == "BUY" else RED
            print(f"  {arrow}{e['action']:<6}{RESET} {e['symbol']:<8} "
                  f"{e['shares']:>8.2f} {e['price']:>9.2f} "
                  f"{format_dollar(e['value']):>12} ${e['tc']:>7.2f}")
        total_tc = sum(e["tc"] for e in executed)
        print(f"\n  Total transaction costs: ${total_tc:.2f}  ({TC_BPS} bps)\n")
        ok(f"{len(executed)} trade(s) executed")
    else:
        info("No trades executed — holding all current positions")

    # ── 7. Save daily snapshot ────────────────────────────────────────────────
    section("Step 7 — Saving Daily Snapshot")
    prices_dict = {t: m["price"] for t, m in market_snap.items() if m["price"]}
    total_val, holdings_val, cash_val = get_portfolio_valuation(prices_dict)
    sim_fresh = get_sim_state()

    save_snapshot(
        date_str        = sim["current_date"],
        cash            = cash_val,
        holdings_value  = holdings_val,
        total_equity    = total_val,
        day_number      = sim_fresh["day_number"],
        trades_count    = len(executed),
        agent_reasoning = reasoning_text[:2000],
        notes           = f"Day {sim_fresh['day_number']}: {len(executed)} trade(s) executed",
    )
    ok("Snapshot saved to finance_game.db")

    # ── 8. Daily summary ──────────────────────────────────────────────────────
    section("Daily Summary")
    delta_pct = (total_val - STARTING_CAPITAL) / STARTING_CAPITAL

    print(f"\n  {'Metric':<28} {'Value':>16}")
    print("  " + "-" * 46)
    print(f"  {'Total Portfolio Value':<28} {format_dollar(total_val):>16}")
    print(f"  {'Cash':<28} {format_dollar(cash_val):>16}")
    print(f"  {'Holdings (Market Value)':<28} {format_dollar(holdings_val):>16}")
    print(f"  {'Return vs $1M Start':<28} {format_pct(delta_pct):>16}")
    print(f"  {'Open Positions':<28} {len(get_positions()):>16}")
    print(f"  {'Trades Today':<28} {len(executed):>16}")
    print(f"  {'Days Remaining':<28} {days_left:>16}")

    # KPIs across all days so far
    snapshots = get_snapshots()
    kpis      = compute_kpis(snapshots)
    if kpis["total_ret"] is not None:
        print(f"\n  {'── Cumulative KPIs'}")
        print(f"  {'Max Drawdown':<28} {format_pct(kpis['max_dd']):>16}")
        win_str = f"{kpis['win_rate']*100:.0f}%" if kpis['win_rate'] else '—'
        print(f"  {'Win Rate (days)':<28} {win_str:>16}")
        print(f"  {'Ann. Volatility':<28} {format_pct(kpis['vol_ann'], sign=False) if kpis['vol_ann'] else '—':>16}")

    color = GREEN if delta_pct >= 0 else RED
    print(f"\n  {BOLD}Portfolio: {color}{format_dollar(total_val)}{RESET}  "
          f"({color}{format_pct(delta_pct)}{RESET} vs start)\n")
    print(f"  {DIM}Open Streamlit dashboard with: streamlit run app.py{RESET}")
    print(f"{BOLD}{'='*64}{RESET}\n")


if __name__ == "__main__":
    main()

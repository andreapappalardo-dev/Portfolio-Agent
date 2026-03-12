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
    record_trade, save_snapshot, advance_date, reset_db,
)
from portfolio_manager import (
    fetch_prices, get_market_snapshot, get_portfolio_valuation,
    compute_kpis, get_snapshots, format_dollar, format_pct,
    STARTING_CAPITAL,
)

# ── Competition constants ─────────────────────────────────────────────────────
COMPETITION_START = "2026-03-13"
COMPETITION_END   = "2026-03-27"
# ── Dashboard display assets (curated 26-stock watchlist shown in Streamlit) ──
ASSETS = [
    "NVDA", "AMD", "ARM", "AVGO", "MRVL",
    "AAPL", "MSFT", "GOOGL", "META", "AMZN",
    "JPM", "GS", "V",
    "LLY", "UNH",
    "XOM", "CVX",
    "LMT", "RTX",
    "PLTR", "NET",
    "SPY", "QQQ", "GLD", "TLT",
]

# ── Additional ETFs always included in the screener ──────────────────────────
SCREENER_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",          # broad market
    "XLK", "XLF", "XLE", "XLV", "XLI",  # sector SPDRs
    "XLC", "XLY", "XLP", "XLU", "XLRE", # remaining SPDRs
    "GLD", "SLV", "TLT", "IEF", "AGG",  # macro
    "ARKK", "SOXX", "SMH",              # thematic
]


def get_screener_universe() -> list[str]:
    """
    Returns the full screener universe: S&P 500 + SCREENER_ETFS.
    Tickers are hardcoded to avoid runtime HTTP dependencies.
    """
    SP500 = [
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB",
        "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP",
        "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL",
        "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY",
        "AXON","BKR","BALL","BAC","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","BA","BCR",
        "BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX",
        "CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CRL","SCHW",
        "CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS",
        "KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA",
        "CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DELL","DAL",
        "DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK",
        "DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM",
        "EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR",
        "XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F",
        "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD",
        "GIS","GM","GPC","GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC",
        "HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN",
        "HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
        "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K",
        "KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR","LHX","LH","LRCX",
        "LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB",
        "MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK",
        "META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR",
        "MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE",
        "NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
        "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PH","PAYX","PAYC","PYPL","PNR","PEP",
        "PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU","PLD","PEG",
        "PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG",
        "RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW",
        "SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
        "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY",
        "TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
        "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO",
        "VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","WRB","GWW","WAB","WBA","WMT","DIS",
        "WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL","XYL",
        "YUM","ZBRA","ZBH","ZTS","ARM","MRVL","NET","DDOG","SNOW","PANW","COIN","MSTR","SHOP",
    ]
    full = list(dict.fromkeys(SP500 + SCREENER_ETFS))
    info(f"S&P 500 + ETF universe loaded: {len(full)} tickers (hardcoded)")
    return full
BENCHMARK        = "SPY"
MAX_POSITION_PCT = 0.40   # max 40% per position
MAX_TRADES_DAY   = 2      # hard limit per project spec
TC_BPS           = 10.0   # transaction cost in basis points


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
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
    Calls Claude claude-sonnet-4-20250514 with the web_search tool.
    1. Fetches prices for the full S&P 500 + ETF universe.
    2. Scores every stock with the alpha formula.
    3. Passes the top 12 candidates to Claude for final selection.
    4. Writes screener_results.csv for the Streamlit dashboard.
    """
    from agents import compute_alpha_score, SECTOR_MAP, BUY_THRESHOLD, SELL_THRESHOLD

    client       = anthropic.Anthropic(api_key=API_KEY)
    positions    = get_positions()
    cash         = sim["cash_balance"]
    total        = sim["total_value"]
    trades_used  = get_trades_today(sim["current_date"])
    remaining    = MAX_TRADES_DAY - trades_used
    days_left    = max(1, 11 - sim["day_number"])
    cash_pct     = cash / total * 100

    # ── Step 1: Full universe price fetch ────────────────────────────────────
    universe = get_screener_universe()
    info(f"Screener universe: {len(universe)} tickers — fetching prices (this takes ~30s)…")

    # Fetch in batches of 100 to avoid yfinance timeouts
    BATCH = 100
    frames = []
    for i in range(0, len(universe), BATCH):
        batch = universe[i:i+BATCH]
        try:
            df_batch = fetch_prices(tuple(batch), lookback_days=30)
            if not df_batch.empty:
                frames.append(df_batch)
        except Exception as e:
            info(f"  Batch {i//BATCH+1} failed: {e}")

    if not frames:
        info("Price fetch failed for full universe — falling back to dashboard assets")
        universe = ASSETS
        prices_df_full = fetch_prices(tuple(ASSETS + [BENCHMARK]), lookback_days=30)
    else:
        prices_df_full = pd.concat(frames, axis=1)
        # De-duplicate columns (same ticker in multiple batches)
        prices_df_full = prices_df_full.loc[:, ~prices_df_full.columns.duplicated()]

    ok(f"Prices fetched for {len(prices_df_full.columns)} tickers")

    # ── Step 2: Score every ticker ────────────────────────────────────────────
    screener_rows = []
    enriched_full = {}

    for ticker in prices_df_full.columns:
        series = prices_df_full[ticker].dropna()
        if len(series) < 21:          # need 21 days for alpha score
            continue
        closes  = series.tolist()
        volumes = []                  # volume not in basic fetch; gap component will be 0
        sigs    = compute_alpha_score(closes, volumes, today_open=0)

        price   = closes[-1]
        ret_20d = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else None

        enriched_full[ticker] = {
            "price":   price,
            "ret_5d":  sigs.get("ret_5d"),
            "ret_20d": ret_20d,
            **sigs,
        }

        if ticker not in ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "IEF", "AGG"):
            screener_rows.append({
                "symbol":      ticker,
                "price":       price,
                "ret_5d":      (sigs.get("ret_5d") or 0) / 100,
                "ret_20d":     ret_20d,
                "vol_20d":     sigs.get("atr_pct"),
                "score":       sigs.get("alpha", 0),
                "alpha_label": sigs.get("alpha_label", "—"),
            })

    # ── Step 3: Write screener CSV for dashboard ──────────────────────────────
    if screener_rows:
        screener_df   = pd.DataFrame(screener_rows).sort_values("score", ascending=False)
        screener_path = Path(__file__).parent / "screener_results.csv"
        screener_df.to_csv(screener_path, index=False, encoding="utf-8-sig")
        ok(f"Screener results saved → screener_results.csv ({len(screener_rows)} stocks scored)")

    # ── Step 4: Select top 12 candidates for Claude ───────────────────────────
    ranked = sorted(
        [(t, m) for t, m in enriched_full.items()
         if t not in ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "IEF", "AGG")],
        key=lambda x: x[1].get("alpha", 0), reverse=True
    )
    top_candidates  = ranked[:12]
    buy_signals     = [(t, m) for t, m in top_candidates if m.get("alpha", 0) >= BUY_THRESHOLD]
    watch_list      = [(t, m) for t, m in top_candidates if 0.25 <= m.get("alpha", 0) < BUY_THRESHOLD]

    alpha_lines = []
    for t, m in top_candidates:
        sec  = SECTOR_MAP.get(t, "?")
        a    = m.get("alpha", 0)
        r5   = m.get("ret_5d") or 0
        rsi  = m.get("rsi14", "?")
        atr  = m.get("atr_pct", "?")
        lbl  = m.get("alpha_label", "—")
        alpha_lines.append(
            f"  {t:<6} [{sec:<8}]  α={a:+.3f} {lbl:<18}  "
            f"5d={r5:+.1f}%  RSI={rsi}  ATR={atr}%  ${m.get('price', 0):,.2f}"
        )
    alpha_table = "\n".join(alpha_lines)
    buy_txt  = ", ".join(f"{t}(α={m.get('alpha',0):+.3f})" for t, m in buy_signals) or "NONE"
    watch_txt= ", ".join(f"{t}(α={m.get('alpha',0):+.3f})" for t, m in watch_list[:5]) or "NONE"

    # ── Holdings summary ──────────────────────────────────────────────────────
    positions    = get_positions()
    held_symbols = [p["symbol"] for p in positions]
    held_txt     = "\n".join(
        f"  - {p['symbol']} [{SECTOR_MAP.get(p['symbol'],'?')}]: "
        f"{p['shares']:.2f} sh @ ${p['avg_cost']:.2f}  "
        f"α={enriched_full.get(p['symbol'],{}).get('alpha','?')}  "
        f"target={'$'+str(p['target_price']) if p.get('target_price') else 'none'}  "
        f"stop={'$'+str(p['stop_loss']) if p.get('stop_loss') else 'none'}"
        for p in positions
    ) or "  - (none — FULLY IN CASH, $0 deployed)"

    alpha_exits = [
        p["symbol"] for p in positions
        if enriched_full.get(p["symbol"], {}).get("alpha", 1.0) < SELL_THRESHOLD
    ]
    exit_block = ""
    if alpha_exits:
        exit_block = f"⚠️  ALPHA EXITS REQUIRED: {', '.join(alpha_exits)} (alpha < {SELL_THRESHOLD})\n"

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt = f"""You are a quantitative AI portfolio manager in a 2-week paper trading competition.

═══════════════════════════ SITUATION ═══════════════════════════
Date:         {sim["current_date"]}  (Day {sim["day_number"]} of 11)
Portfolio:    {format_dollar(total)}
Cash:         {format_dollar(cash)} ({cash_pct:.1f}%)
Positions:    {len(positions)} open
Trades left today: {remaining} of {MAX_TRADES_DAY}

Holdings:
{held_txt}

{exit_block}
════════════════ TOP 12 FROM S&P 500 + ETF SCREENER ════════════════
Screened {len(screener_rows)} stocks. Scoring formula:
α = 0.40×momentum + 0.25×gap + 0.15×trend + 0.10×RSI + 0.10×volume
BUY threshold: α > {BUY_THRESHOLD}  |  SELL threshold: α < {SELL_THRESHOLD}

{alpha_table}

🚀 BUY signals (α ≥ {BUY_THRESHOLD}): {buy_txt}
📈 Watch list (α 0.25–{BUY_THRESHOLD}):  {watch_txt}

═══════════════════════════ DECISION RULES ════════════════════════════════
1. SELL first: exit any position where alpha < {SELL_THRESHOLD} or stop hit
2. NEW ENTRIES: only consider stocks where alpha > {BUY_THRESHOLD}
   • NEVER buy RSI > 70 (overbought)
   • Use web search to confirm top candidates have real catalysts
   • ATR > 3% → cap position at 25%; otherwise up to 40% per position
   • Target = entry × 1.10 to 1.15  |  Stop = entry × 0.93 to 0.95
3. If no stock clears the alpha threshold, recommend holding — do not force trades.

STEP 1: Search "[TOP_TICKER] stock news {sim["current_date"]}" for the top 2 alpha candidates.
STEP 2: Output JSON with your decisions (empty list [] if no trades are warranted).

```json
[
  {{
    "action": "BUY",
    "symbol": "TICKER",
    "target_pct": 0.35,
    "target_price": 0.0,
    "stop_loss": 0.0,
    "holding_days": 5,
    "framework": "momentum",
    "reason": "2-3 sentence explanation with alpha breakdown + news catalyst"
  }}
]
```
End with the JSON block."""

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
        if len(approved) >= budget:
            rejected.append((t, f"Daily trade limit ({MAX_TRADES_DAY}) reached")); continue
        if target_pct > MAX_POSITION_PCT:
            rejected.append((t, f"Target {target_pct*100:.0f}% > max {MAX_POSITION_PCT*100:.0f}%")); continue
        if action == "BUY":
            required = target_pct * total
            if required > cash + 0.01:
                rejected.append((t, f"Insufficient cash: need {format_dollar(required)}, "
                                    f"have {format_dollar(cash)}")); continue

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

    for t in approved:
        symbol     = t["symbol"]
        target_pct = t.get("target_pct", 0.0)
        action     = t["action"]
        price      = market_snap.get(symbol, {}).get("price")

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
            target_price=t.get("target_price") if action == "BUY" else None,
            stop_loss=t.get("stop_loss")        if action == "BUY" else None,
            holding_days=t.get("holding_days")  if action == "BUY" else None,
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
# INITIALIZATION — Day 1 portfolio construction (10–20 stocks)
# ─────────────────────────────────────────────────────────────────────────────

def run_initialization(sim: dict) -> None:
    """
    Runs on Day 1 when the portfolio is empty.
    Screens the full S&P 500 + ETF universe, picks the top 10-20 stocks
    by alpha score, allocates capital equally across them, executes all
    entry trades (bypasses the 2-trade/day limit for this one-time setup),
    and writes prices.csv with the chosen holdings.
    """
    from agents import compute_alpha_score, SECTOR_MAP

    N_MIN, N_MAX   = 10, 20      # target portfolio size
    DEPLOY_FRAC    = 0.90        # deploy 90% of cash; keep 10% buffer
    MAX_POS        = 0.40        # hard cap per position

    section("INITIALIZATION — Building Day 1 Portfolio (10-20 stocks)")
    info("Running full screener to select initial holdings...")

    # Fetch prices for full universe
    universe = get_screener_universe()
    BATCH    = 100
    frames   = []
    for i in range(0, len(universe), BATCH):
        batch = universe[i:i+BATCH]
        try:
            df_b = fetch_prices(tuple(batch), lookback_days=30)
            if not df_b.empty:
                frames.append(df_b)
        except Exception as e:
            info(f"  Batch {i//BATCH+1} failed: {e}")

    if not frames:
        err("Could not fetch prices for initialization -- aborting init")
        return

    prices_df_full = pd.concat(frames, axis=1)
    prices_df_full = prices_df_full.loc[:, ~prices_df_full.columns.duplicated()]
    ok(f"Prices fetched: {len(prices_df_full.columns)} tickers")

    # Score every ticker
    scored = []
    for ticker in prices_df_full.columns:
        series = prices_df_full[ticker].dropna()
        if len(series) < 21:
            continue
        closes = series.tolist()
        sigs   = compute_alpha_score(closes, [], today_open=0)
        price  = closes[-1]
        alpha  = sigs.get("alpha", 0)

        # Exclude broad index ETFs from direct equity holdings
        if ticker in ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "IEF", "AGG"):
            continue

        scored.append({
            "symbol":  ticker,
            "price":   price,
            "alpha":   alpha,
            "ret_5d":  sigs.get("ret_5d", 0),
            "rsi":     sigs.get("rsi14", 50),
            "sector":  SECTOR_MAP.get(ticker, "Other"),
        })

    if not scored:
        err("No stocks scored -- initialization aborted")
        return

    # Sort by alpha, filter negatives, clamp to N_MAX
    scored.sort(key=lambda x: x["alpha"], reverse=True)
    scored = [s for s in scored if s["alpha"] > 0]
    n_pick = min(N_MAX, max(N_MIN, len(scored)))
    picks  = scored[:n_pick]

    ok(f"Selected {len(picks)} stocks for initial portfolio")
    for p in picks[:5]:
        print(f"    {p['symbol']:<6}  alpha={p['alpha']:+.3f}  RSI={p['rsi']}  ${p['price']:.2f}")
    if len(picks) > 5:
        info(f"  ... and {len(picks)-5} more")

    # Allocate capital — rank-based weighting
    # Stock ranked #1 gets weight N, #2 gets N-1, ..., #N gets 1.
    # Normalized so all weights sum to DEPLOY_FRAC (90%), then capped at MAX_POS (40%).
    # After capping, surplus is redistributed proportionally to uncapped stocks.
    sim_now   = get_sim_state()
    cash      = sim_now["cash_balance"]
    total     = sim_now["total_value"]
    n         = len(picks)

    # Raw rank scores: best stock = n points, worst = 1 point
    rank_scores = [n - i for i in range(n)]           # [n, n-1, ..., 1]
    rank_sum    = sum(rank_scores)                     # n*(n+1)/2
    raw_weights = [s / rank_sum * DEPLOY_FRAC for s in rank_scores]

    # Iteratively cap at MAX_POS and redistribute surplus
    weights = raw_weights[:]
    for _ in range(10):
        surplus    = sum(max(0, w - MAX_POS) for w in weights)
        weights    = [min(w, MAX_POS) for w in weights]
        uncapped   = [i for i, w in enumerate(weights) if w < MAX_POS]
        if surplus < 1e-6 or not uncapped:
            break
        uncapped_sum = sum(weights[i] for i in uncapped)
        for i in uncapped:
            weights[i] += surplus * (weights[i] / uncapped_sum) if uncapped_sum else surplus / len(uncapped)

    # Log the weight plan
    info("Rank-based weight allocation:")
    for i, (pick, w) in enumerate(zip(picks, weights)):
        print(f"    #{i+1:<2} {pick['symbol']:<6}  alpha={pick['alpha']:+.3f}  weight={w*100:.1f}%")

    executed_init = []
    prices_rows   = []

    for pick, target_pct in zip(picks, weights):
        sym         = pick["symbol"]
        price       = pick["price"]

        # Fetch a fresh live price
        try:
            raw = yf.download(sym, period="2d", progress=False, auto_adjust=True)
            if not raw.empty:
                price = float(raw["Close"].iloc[-1])
        except Exception:
            pass

        required = target_pct * total
        if required > cash:
            info(f"  Skipping {sym} -- insufficient cash")
            continue

        shares       = round(required / price, 4)
        target_price = round(price * 1.12, 2)
        stop_loss    = round(price * 0.94, 2)

        record_trade(
            date_str     = sim_now["current_date"],
            action       = "BUY",
            symbol       = sym,
            shares       = shares,
            price        = price,
            tc_bps       = TC_BPS,
            reason       = (
                f"Initialization: Day 1 portfolio construction. "
                f"Alpha={pick['alpha']:+.3f}, RSI={pick['rsi']}, "
                f"sector={pick['sector']}. Equal-weight entry across "
                f"{n} top-alpha stocks. Target +12%, stop -6%."
            ),
            agent        = "Init",
            target_price = target_price,
            stop_loss    = stop_loss,
            holding_days = 11,
        )

        sim_now = get_sim_state()
        cash    = sim_now["cash_balance"]

        executed_init.append({
            "symbol": sym, "shares": shares, "price": price,
            "value": shares * price, "alpha": pick["alpha"],
            "weight_pct": target_pct * 100,
        })
        prices_rows.append({
            "symbol":       sym,
            "price":        round(price, 4),
            "alpha":        round(pick["alpha"], 4),
            "weight_pct":   round(target_pct * 100, 2),
            "shares":       shares,
            "value":        round(shares * price, 2),
            "target_price": target_price,
            "stop_loss":    stop_loss,
            "sector":       pick["sector"],
            "date":         sim_now["current_date"],
        })

    if executed_init:
        print(f"\n  {'Symbol':<8} {'Shares':>8} {'Price':>9} {'Value':>12} {'Alpha':>8} {'Wt%':>6}")
        print("  " + "-" * 58)
        for e in executed_init:
            print(f"  {e['symbol']:<8} {e['shares']:>8.2f} {e['price']:>9.2f} "
                  f"{format_dollar(e['value']):>12} {e['alpha']:>+8.3f} {e['weight_pct']:>5.1f}%")
        ok(f"Initialized {len(executed_init)} positions")

        prices_csv_path = Path(__file__).parent / "prices.csv"
        pd.DataFrame(prices_rows).to_csv(prices_csv_path, index=False, encoding="utf-8-sig")
        ok(f"prices.csv written -> {prices_csv_path} ({len(prices_rows)} stocks)")
    else:
        info("No positions executed during initialization")

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

    is_fresh = sim is None
    if sim is None:
        info("No simulation found — initialising fresh database...")
        reset_db(STARTING_CAPITAL, COMPETITION_START)
        sim = get_sim_state()
        ok(f"Fresh simulation created · {format_dollar(STARTING_CAPITAL)} cash · date {sim['current_date']}")
    else:
        ok(f"Loaded existing simulation · Day {sim['day_number']} · {sim['current_date']}")

    # ── 1b. Initialization phase (Day 1 only) ─────────────────────────────────
    existing_positions = get_positions()
    if is_fresh or (sim.get("day_number", 1) == 1 and not existing_positions):
        info("Day 1 detected with no positions — running initialization phase...")
        run_initialization(sim)
        sim = get_sim_state()
        ok("Initialization complete — portfolio built")

    # ── 2. Advance date ───────────────────────────────────────────────────────
    section("Step 2 — Advancing Date")
    today_real = datetime.today().strftime("%Y-%m-%d")
    if sim["current_date"] < today_real:
        new_date = advance_date(sim["current_date"])
        sim      = get_sim_state()
        ok(f"Advanced to {new_date}  (Day {sim['day_number']})")
    else:
        ok(f"Already on {sim['current_date']} — no advance needed")

    days_left = max(0, 11 - sim["day_number"])
    info(f"Days remaining in competition: {days_left}")

    # ── 3. Fetch market data ──────────────────────────────────────────────────
    section("Step 3 — Data Agent · Fetching Live Prices")
    info("Loading S&P 500 + ETF screener universe…")
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

    # ── Forced-trade fallback ─────────────────────────────────────────────────
    # If Claude returned no BUY trades but cash is above 20%, we force-buy the
    # top alpha stock. In a short competition, idle cash is a guaranteed loss.
    # This only triggers when:
    #   1. No BUY was proposed by Claude
    #   2. Cash > 20% of portfolio
    #   3. At least one trade slot is still available today
    #   4. There is at least one scoreable stock in the screener results
    trades_used_now = get_trades_today(sim["current_date"])
    cash_pct_now    = sim["cash_balance"] / sim["total_value"] if sim["total_value"] else 0
    has_buy         = any(t["action"] == "BUY" for t in proposed)

    if not has_buy and cash_pct_now > 0.20 and trades_used_now < MAX_TRADES_DAY:
        screener_path = Path(__file__).parent / "screener_results.csv"
        fallback_trade = None

        if screener_path.exists():
            try:
                sc_df = pd.read_csv(screener_path)
                # Filter out already-held symbols and ETFs
                held_syms = [p["symbol"] for p in get_positions()]
                exclude   = set(held_syms + ["SPY","QQQ","IWM","DIA","GLD","TLT","IEF","AGG"])
                sc_df     = sc_df[~sc_df["symbol"].isin(exclude)]
                sc_df     = sc_df.sort_values("score", ascending=False)

                if not sc_df.empty:
                    top = sc_df.iloc[0]
                    sym   = top["symbol"]
                    score = float(top["score"])
                    price = market_snap.get(sym, {}).get("price")

                    # Only force-buy if price is available and score isn't deeply negative
                    if price and score > -0.10:
                        # Size: deploy up to 35% per slot, respect cash available
                        slots_left  = MAX_TRADES_DAY - trades_used_now
                        # If 2 slots free, split remaining cash ~50/50 across both
                        # If 1 slot free, deploy all remaining above 10% buffer
                        if slots_left >= 2:
                            target_pct = min(0.35, (cash_pct_now - 0.10) / slots_left)
                        else:
                            target_pct = min(0.35, cash_pct_now - 0.10)
                        target_pct = round(max(0.10, target_pct), 2)

                        fallback_trade = {
                            "action":       "BUY",
                            "symbol":       sym,
                            "target_pct":   target_pct,
                            "target_price": round(price * 1.10, 2),
                            "stop_loss":    round(price * 0.94, 2),
                            "holding_days": 5,
                            "framework":    "forced_momentum",
                            "reason":       (
                                f"Forced-trade fallback: Claude returned no BUY despite "
                                f"{cash_pct_now*100:.0f}% cash and {max(0,11-sim['day_number'])} "
                                f"days remaining. Top alpha stock from screener: "
                                f"{sym} (α={score:+.3f}). "
                                f"Target +10%, stop -6%."
                            ),
                        }
            except Exception as e:
                info(f"Forced-trade fallback failed to read screener: {e}")

        if fallback_trade:
            proposed.append(fallback_trade)
            print(f"\n  {YELLOW}⚡ FORCED TRADE FALLBACK triggered — Claude held cash above 20%{RESET}")
            print(f"    {GREEN}BUY {RESET} {fallback_trade['symbol']:<6}  "
                  f"→ {fallback_trade['target_pct']*100:.0f}%   {fallback_trade['reason'][:80]}")
        else:
            info("Forced-trade fallback: no eligible stock found — holding cash")

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

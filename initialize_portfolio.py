"""
initialize_portfolio.py — MiF AI Portfolio · Day-1 Initialization
==================================================================
Run ONCE on Day 1, BEFORE run_daily.py.

What it does:
    1.  Runs the full 525-ticker screener using the live alpha formula
    2.  Picks the top N stocks (default 15) with alpha > BUY_THRESHOLD
    3.  Allocates capital proportionally to alpha score (leaving 5% cash reserve)
    4.  Caps any single position at MAX_SINGLE_PCT (default 15%)
    5.  Executes all buys directly (bypasses the 2-trade/day competition limit)
    6.  Writes  prices.csv  so the dashboard has historical price data
    7.  Saves the daily snapshot

Usage:
    python initialize_portfolio.py

After this completes, run the normal daily runner each subsequent day:
    python run_daily.py
"""

import math
import os
import sys
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

from database import (
    init_db, get_sim_state, record_trade, save_snapshot,
    reset_db, get_positions,
)
from portfolio_manager import (
    fetch_prices, get_market_snapshot, get_portfolio_valuation,
    format_dollar, format_pct, STARTING_CAPITAL,
)

# ── Config ────────────────────────────────────────────────────────────────────
COMPETITION_START = "2026-03-13"
COMPETITION_END   = "2026-03-27"
TODAY             = datetime.today().strftime("%Y-%m-%d")
DATE_STR          = max(TODAY, COMPETITION_START)   # never pre-date the competition

TOP_N            = 15     # number of stocks to buy (10–20 recommended)
MAX_SINGLE_PCT   = 0.15   # hard cap per position  (15%)
CASH_RESERVE_PCT = 0.05   # keep 5% in cash for flexibility
TC_BPS           = 10.0   # transaction costs in basis points
BUY_THRESHOLD    = 0.45   # only buy stocks above this alpha

# ── Colours ───────────────────────────────────────────────────────────────────
BOLD  = "\033[1m"; GREEN = "\033[92m"; RED   = "\033[91m"
CYAN  = "\033[96m"; DIM  = "\033[2m";  RESET = "\033[0m"

def section(t): print(f"\n{CYAN}{'─'*64}{RESET}\n{BOLD}{CYAN}  {t}{RESET}\n{CYAN}{'─'*64}{RESET}")
def ok(m):   print(f"  {GREEN}✔{RESET}  {m}")
def err(m):  print(f"  {RED}✘{RESET}  {m}")
def info(m): print(f"  {DIM}→{RESET}  {m}")

# ── Full S&P 500 + ETF universe (525 tickers, hardcoded) ─────────────────────
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI",
    "ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
    "T","ATO","ADSK","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BAX","BDX",
    "WRB","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR","BSX","BMY","AVGO",
    "BR","BRO","BF.B","BLDR","BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH",
    "KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP",
    "CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO",
    "C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP",
    "ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA",
    "CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DECK","DE","DAL",
    "DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DDOG","DOV","DOW",
    "DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ENPH",
    "ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
    "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX",
    "BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM",
    "GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE",
    "HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII",
    "IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG",
    "INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ",
    "JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC",
    "KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV",
    "LKQ","LMT","L","LOW","LULU","LYB","MTB","MPC","MKTX","MAR","MMC","MLM",
    "MAS","MA","MTCH","MKC","MCD","MCK","MDT","MET","META","MTD","MGM","MCHP",
    "MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO",
    "MS","MOS","MSI","MSCI","NDAQ","NTAP","NWS","NWSA","NEM","NFLX","NI","NK",
    "NEE","NKE","NI","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY",
    "OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","OGN","PCAR","PKG","PANW",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC",
    "POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM",
    "QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG",
    "RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB",
    "STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV",
    "SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS",
    "TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN",
    "TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB",
    "UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN",
    "VRSK","VZ","VRTX","VTRS","VICI","V","VMC","WRK","WAB","WBA","WMT","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WRB","WY","WHR","WMB","WTW",
    "GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS","MRO","JNPR","HES","K",
]

ETF_TICKERS = [
    "SPY","QQQ","IWM","DIA","GLD","SLV","GDX","USO","UCO","BNO","DBO",
    "TLT","IEF","HYG","LQD","XLK","XLF","XLE","XLV","XLC","XLI","XLY",
    "ARKK","SOXL","TQQQ",
]

UNIVERSE = list(dict.fromkeys(SP500_TICKERS + ETF_TICKERS))

# ── Alpha helpers (mirror of agents.py) ──────────────────────────────────────
def _clip(x, lo, hi): return max(lo, min(hi, x))

def _sma(prices, n):
    return sum(prices[-n:]) / n if len(prices) >= n else None

def _rsi(prices, n=14):
    if len(prices) < n + 1: return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = [max(d, 0) for d in deltas[-n:]]
    losses = [-min(d, 0) for d in deltas[-n:]]
    ag, al = sum(gains)/n, sum(losses)/n
    return 100 - 100/(1 + ag/al) if al > 0 else 100.0

def _atr_pct(prices, n=14):
    if len(prices) < n + 1: return None
    trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    atr = sum(trs[-n:]) / n
    return round(atr / prices[-1] * 100, 2) if prices[-1] > 0 else None

def compute_alpha(prices: list, volumes: list, today_open: float = 0) -> dict:
    empty = {"alpha": 0.0, "price": None, "atr_pct": None, "rsi14": None,
             "ret_5d": 0.0, "alpha_label": "—"}
    if len(prices) < 21:
        return empty
    cur    = prices[-1]
    p5     = prices[-6] if len(prices) >= 6 else prices[0]
    ret_5d = (cur / p5 - 1) if p5 > 0 else 0
    mom    = _clip(ret_5d / 0.20, -1.0, 1.0)
    gap    = _clip(((today_open/cur - 1) if today_open > 0 else 0) / 0.05, -1.0, 1.0)
    sma20  = _sma(prices, 20)
    trend  = 1.0 if (sma20 and cur > sma20) else -1.0
    rsi14  = _rsi(prices, 14)
    if   rsi14 and 40 <= rsi14 <= 65: rsi_s =  1.0
    elif rsi14 and 35 <= rsi14 <  40: rsi_s =  0.5
    elif rsi14 and rsi14 > 70:        rsi_s = -1.0
    else:                             rsi_s =  0.0
    avg_vol   = sum(volumes[-20:])/min(len(volumes),20) if volumes else 0
    vol_s     = _clip((volumes[-1]/avg_vol - 1)/2.0, -1.0, 1.0) if avg_vol > 0 else 0
    alpha     = 0.40*mom + 0.25*gap + 0.15*trend + 0.10*rsi_s + 0.10*vol_s
    alpha     = round(alpha, 4)
    label     = ("🚀 BUY" if alpha >= BUY_THRESHOLD else
                 "📈 WATCH" if alpha >= 0.30 else
                 "📉 AVOID" if alpha <= -0.20 else "➖ NEUTRAL")
    return {
        "alpha": alpha, "alpha_label": label,
        "price": round(cur, 2), "atr_pct": _atr_pct(prices, 14),
        "rsi14": rsi14, "ret_5d": round(ret_5d*100, 2),
        "sma20": round(sma20, 2) if sma20 else None,
        # Component scores (each in [-1, +1])
        "momentum_score": round(mom,   4),
        "gap_score":      round(gap,   4),
        "trend_score":    round(trend, 4),
        "rsi_score":      round(rsi_s, 4),
        "vol_score":      round(vol_s,  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*64}{RESET}")
    print(f"{BOLD}  MiF AI Portfolio — Day-1 Initialization{RESET}")
    print(f"  Date: {DATE_STR}  |  Universe: {len(UNIVERSE)} tickers")
    print(f"  Top N: {TOP_N}  |  Cash reserve: {CASH_RESERVE_PCT*100:.0f}%")
    print(f"{BOLD}{'='*64}{RESET}")

    # ── 1. Database ───────────────────────────────────────────────────────────
    section("Step 1 — Database")
    init_db()
    sim = get_sim_state()
    if sim is None:
        reset_db(STARTING_CAPITAL, DATE_STR)
        sim = get_sim_state()
        ok(f"Fresh simulation created · {format_dollar(STARTING_CAPITAL)}")
    else:
        existing = get_positions()
        if existing:
            print(f"\n  {RED}{BOLD}⚠  Portfolio already has {len(existing)} open position(s).{RESET}")
            print("  Running initialization again would double-buy existing positions.")
            ans = input("  Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("  Aborted."); return
        ok(f"Loaded simulation · Day {sim['day_number']} · {sim['current_date']}")

    # ── 2. Screener: fetch all prices ─────────────────────────────────────────
    section("Step 2 — Screener · Fetching 525-Ticker Universe")
    info(f"Downloading {len(UNIVERSE)} tickers via yfinance (takes ~60s)…")

    BATCH = 50
    all_prices: dict[str, pd.Series] = {}
    failed = []
    for i in range(0, len(UNIVERSE), BATCH):
        batch = UNIVERSE[i:i+BATCH]
        try:
            start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
            raw   = yf.download(batch, start=start, auto_adjust=True,
                                progress=False, threads=True)
            if raw.empty:
                failed += batch; continue
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": batch[0]})
            vol   = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]].rename(columns={"Volume": batch[0]})
            for t in batch:
                if t in close.columns:
                    s = close[t].dropna()
                    v = vol[t].dropna() if t in vol.columns else pd.Series([], dtype=float)
                    if len(s) >= 21:
                        all_prices[t] = {"close": list(s.values), "volume": list(v.values)}
        except Exception as e:
            info(f"Batch {i//BATCH+1} error: {e}")
            failed += batch
        n_done = min(i+BATCH, len(UNIVERSE))
        print(f"  … {n_done}/{len(UNIVERSE)} tickers processed", end="\r")
    print()
    ok(f"Price history loaded for {len(all_prices)} tickers · {len(failed)} failed/delisted")

    # ── 3. Score every ticker ─────────────────────────────────────────────────
    section("Step 3 — Alpha Screener · Scoring All Tickers")
    scored = []
    for t, hist in all_prices.items():
        scores = compute_alpha(hist["close"], hist["volume"])
        if scores["price"] and scores["price"] > 0:
            scored.append({"ticker": t, **scores})

    scored.sort(key=lambda x: x["alpha"], reverse=True)
    ok(f"Scored {len(scored)} tickers")

    # Always target TOP_N positions. Use BUY_THRESHOLD as soft preference:
    # take all stocks above threshold first, then fill remainder from the next
    # best by alpha until we reach TOP_N (or run out of positively-scored stocks).
    MIN_POSITIONS = 10
    above      = [s for s in scored if s["alpha"] >= BUY_THRESHOLD]
    below      = [s for s in scored if 0 < s["alpha"] < BUY_THRESHOLD]
    candidates = (above + below)[:TOP_N]

    if len(candidates) < MIN_POSITIONS:
        # Extremely unusual — take any positive-alpha stock
        candidates = [s for s in scored if s["alpha"] > 0][:TOP_N]

    info(f"{len(above)} stocks above threshold (α ≥ {BUY_THRESHOLD}), "
         f"{max(0, len(candidates) - len(above))} filled from next-best — "
         f"{len(candidates)} total selected")

    # Take top N
    selected = candidates[:TOP_N]

    # ── 4. Compute proportional weights ───────────────────────────────────────
    section("Step 4 — Weight Allocation · Alpha-Proportional")
    deployable = 1.0 - CASH_RESERVE_PCT
    alphas     = [max(s["alpha"], 0.01) for s in selected]
    raw_wts    = [a / sum(alphas) for a in alphas]
    # Cap each weight at MAX_SINGLE_PCT, redistribute excess to others
    for _ in range(20):                    # iterate to convergence
        capped    = [min(w, MAX_SINGLE_PCT) for w in raw_wts]
        excess    = sum(w - c for w, c in zip(raw_wts, capped))
        uncapped  = [i for i, (w, c) in enumerate(zip(raw_wts, capped)) if w < MAX_SINGLE_PCT]
        if excess < 1e-6 or not uncapped:
            raw_wts = capped; break
        extra = excess / len(uncapped)
        raw_wts = [min(c + (extra if i in uncapped else 0), MAX_SINGLE_PCT) for i, c in enumerate(capped)]
    final_wts  = [w * deployable for w in raw_wts]

    print(f"\n  {'#':<3} {'Ticker':<7} {'α Score':>8} {'Label':<14} {'Weight':>7} {'Price':>9} {'ATR%':>6} {'5d%':>7}")
    print("  " + "─"*62)
    for i, (s, w) in enumerate(zip(selected, final_wts), 1):
        atr = f"{s['atr_pct']:.1f}%" if s.get("atr_pct") else "  —"
        r5  = f"{s['ret_5d']:+.1f}%"
        print(f"  {i:<3} {s['ticker']:<7} {s['alpha']:>+8.4f} {s['alpha_label']:<14} "
              f"{w*100:>6.1f}% ${s['price']:>8.2f} {atr:>6} {r5:>7}")

    total_alloc = sum(final_wts) * 100
    print(f"\n  Total allocated: {total_alloc:.1f}%  |  Cash reserve: {(1-sum(final_wts))*100:.1f}%")

    # ── 5. Execute all buys ───────────────────────────────────────────────────
    section("Step 5 — Execution · Buying All Positions")
    sim       = get_sim_state()
    total_val = sim["total_value"]
    cash_avail= sim["cash_balance"]

    print(f"\n  {'Action':<6} {'Ticker':<8} {'Shares':>9} {'Price':>9} {'Value':>13} {'TC':>9}")
    print("  " + "─"*58)

    executed = []
    for s, w in zip(selected, final_wts):
        ticker = s["ticker"]
        price  = s["price"]
        target_val = w * total_val
        shares     = math.floor(target_val / price * 100) / 100   # round down to 2dp
        value      = shares * price
        tc         = value * TC_BPS / 10_000

        if shares <= 0 or value + tc > cash_avail:
            err(f"Skipping {ticker}: insufficient cash (need {format_dollar(value+tc)}, have {format_dollar(cash_avail)})")
            continue

        record_trade(
            date_str    = DATE_STR,
            action      = "BUY",
            symbol      = ticker,
            shares      = shares,
            price       = price,
            tc_bps      = TC_BPS,
            reason      = f"Initialization: α={s['alpha']:+.4f}, rank #{selected.index(s)+1}/{len(selected)}",
            agent       = "Init",
        )
        cash_avail -= (value + tc)
        executed.append({**s, "shares": shares, "value": value, "tc": tc, "weight": w})
        print(f"  {GREEN}BUY{RESET}    {ticker:<8} {shares:>9.2f} ${price:>8.2f} "
              f"{format_dollar(value):>13} ${tc:>8.2f}")

    total_tc = sum(e["tc"] for e in executed)
    print(f"\n  {BOLD}Executed: {len(executed)} buys  |  Total TC: ${total_tc:.2f}{RESET}")
    ok(f"Cash remaining: {format_dollar(cash_avail)}")

    # ── 6. Write prices.csv — one row per position, metrics at purchase time ──
    section("Step 6 — Writing prices.csv")

    # Full company names via yfinance
    info("Fetching company names…")
    name_map = {}
    for t in [e["ticker"] for e in executed]:
        try:
            info_data = yf.Ticker(t).info
            name_map[t] = info_data.get("longName") or info_data.get("shortName") or t
        except Exception:
            name_map[t] = t

    rows = []
    for e in executed:
        t = e["ticker"]
        rows.append({
            "ticker":           t,
            "name":             name_map.get(t, t),
            "purchase_date":    DATE_STR,
            "purchase_price":   round(e["price"], 4),
            "shares":           round(e["shares"], 4),
            "position_value":   round(e["value"], 2),
            "weight_pct":       round(e["weight"] * 100, 2),
            # Risk levels
            "stop_loss":        round(e["price"] * 0.92, 4),   # −8%
            "target_price":     round(e["price"] * 1.15, 4),   # +15%
            "tc_paid":          round(e["tc"], 4),
            # Alpha composite score
            "alpha":            round(e["alpha"], 4),
            "alpha_label":      e["alpha_label"],
            # Alpha component scores (each normalised to [−1, +1])
            "momentum_score":   round(e.get("momentum_score", 0) or 0, 4),
            "gap_score":        round(e.get("gap_score",      0) or 0, 4),
            "trend_score":      round(e.get("trend_score",    0) or 0, 4),
            "rsi_score":        round(e.get("rsi_score",      0) or 0, 4),
            "vol_score":        round(e.get("vol_score",      0) or 0, 4),
            # Raw market indicators at purchase
            "rsi14":            round(e["rsi14"],   2) if e.get("rsi14")   else None,
            "atr_pct":          round(e["atr_pct"], 2) if e.get("atr_pct") else None,
            "ret_5d_pct":       round(e["ret_5d"],  2) if e.get("ret_5d")  else None,
            "sma20":            round(e["sma20"],   4) if e.get("sma20")   else None,
        })

    df_positions = pd.DataFrame(rows)
    df_positions.to_csv("prices.csv", index=False)

    ok(f"prices.csv written — {len(rows)} positions")
    info("Columns: ticker, name, purchase_price, weight_pct, alpha, rsi14, atr_pct, ret_5d_pct, sma20, stop_loss, target_price")

    # ── 7. Save snapshot ──────────────────────────────────────────────────────
    section("Step 7 — Saving Daily Snapshot")
    prices_dict = {e["ticker"]: e["price"] for e in executed}
    prices_dict["SPY"] = all_prices.get("SPY", {}).get("close", [None])[-1] or 0
    total_val2, holdings_val, cash_val = get_portfolio_valuation(prices_dict)
    sim2 = get_sim_state()
    save_snapshot(
        date_str       = DATE_STR,
        cash           = cash_val,
        holdings_value = holdings_val,
        total_equity   = total_val2,
        day_number     = sim2["day_number"],
        trades_count   = len(executed),
        agent_reasoning= f"Initialization: bought {len(executed)} stocks, α-proportional weights.",
        notes          = f"Day-1 init: {len(executed)} positions opened.",
    )
    ok("Snapshot saved to finance_game.db")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    section("Initialization Complete")
    delta = (total_val2 - STARTING_CAPITAL) / STARTING_CAPITAL

    print(f"\n  {'Total Portfolio Value':<30} {format_dollar(total_val2):>14}")
    print(f"  {'Cash':<30} {format_dollar(cash_val):>14}")
    print(f"  {'Holdings (Market Value)':<30} {format_dollar(holdings_val):>14}")
    print(f"  {'Positions Opened':<30} {len(executed):>14}")
    print(f"  {'Return vs $1M Start':<30} {format_pct(delta):>14}")

    print(f"\n  {BOLD}Next step:{RESET}")
    print(f"  Run {CYAN}python run_daily.py{RESET} each morning from tomorrow onwards.")
    print(f"  Dashboard: {CYAN}python -m streamlit run app.py{RESET}")
    print(f"\n{BOLD}{'='*64}{RESET}\n")


if __name__ == "__main__":
    main()

# 🏆 MiF AI Portfolio — Competition Agent

> **MiF Deep Learning · Dr. Jaume Manero · IE Business School**  
> $1,000,000 virtual capital · Competition: March 13–27, 2026

---

## Overview

A multi-agent AI system that autonomously manages a stock portfolio over the competition period.
A four-agent pipeline (Data → Strategy → Risk → Execution) runs once per day, using **Claude + live web search** to research markets, propose trades, validate risk constraints, and commit allocations to a SQLite database.

The app has two entry points — use whichever suits your workflow:

| Entry point | When to use |
|---|---|
| `python run_daily.py` | **Daily CLI runner** — run once each morning, no browser needed |
| `streamlit run app.py` | **Streamlit dashboard** — interactive UI to step through agents and view performance |

---

## Architecture

```
run_daily.py  ──or──  app.py (Streamlit)
         │
         ▼
  Four-Agent Pipeline
  ┌─────────────────────────────────────┐
  │  1. Data Agent     (The Analyst)    │  ← yfinance live prices
  │  2. Strategy Agent (The Trader)     │  ← Claude + web search
  │  3. Risk Agent     (The Validator)  │  ← constraint enforcement
  │  4. Execution Agent                 │  ← commits trades to DB
  └──────────────┬──────────────────────┘
                 │
         SQLite  (finance_game.db)
  ┌──────────────────────────────┐
  │ simulation · positions       │
  │ transactions · daily_snapshots│
  └──────────────────────────────┘
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install streamlit plotly anthropic yfinance pandas numpy python-dotenv
```

### 2. Set your Anthropic API key

```bash
# Option A — environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B — .env file (copy the example)
cp .env.example .env
# then edit .env and paste your key
```

> Get a key at [console.anthropic.com](https://console.anthropic.com)

### 3. Run

**CLI (recommended for daily use):**
```bash
python run_daily.py
```

**Streamlit dashboard (for interactive exploration):**
```bash
streamlit run app.py
```

---

## Daily Workflow

Each morning during the competition (March 13–27):

```bash
python run_daily.py
```

That's it. The script will:
1. Advance the simulation date to today
2. Fetch live prices for all assets
3. Call Claude to research today's news and propose trades
4. Validate proposals against risk constraints
5. Execute approved trades in the database
6. Print a full summary to the terminal

Then optionally open the dashboard to visualise performance:
```bash
streamlit run app.py
```

---

## Competition Rules

| Rule | Value |
|---|---|
| Starting capital | **$1,000,000 USD** (virtual) |
| Competition start | Session 9 — **March 13, 2026** |
| Competition end | **March 27, 2026** (~11 trading days) |
| Run cadence | **Once per day** |
| Max position size | **40%** of portfolio value |
| Max trades per day | **2** |
| Transaction cost | **10 basis points** |

---

## Files

| File | Purpose |
|---|---|
| `run_daily.py` | CLI daily runner — the main entry point |
| `app.py` | Streamlit interactive dashboard |
| `agents.py` | Four agent classes (Data, Strategy, Risk, Execution) |
| `database.py` | SQLite schema, read/write helpers |
| `portfolio_manager.py` | Price fetching, valuation, KPI computation |
| `requirements.txt` | Python dependencies |
| `.env.example` | API key template |

---

## Agents

### 1. Data Agent — The Analyst
Fetches real-time prices via `yfinance` and computes 1d / 5d / 20d returns and annualised volatility. No future peeking — only uses data available as of the simulation date.

### 2. Strategy Agent — The Trader
Calls **Claude claude-sonnet-4-20250514** with the **web search tool** enabled. The agent:
- Searches for current news on each asset in the universe
- Looks for macro / Fed / sector rotation signals
- Applies momentum reasoning to propose BUY/SELL trades with target weights
- Returns full reasoning text + structured JSON trade list

### 3. Risk Agent — The Validator
Enforces hard constraints before any trade is executed:
- Max position size ≤ 40% of portfolio value
- Max 2 trades per day
- Cash balance must remain ≥ 0

### 4. Execution Agent
Commits approved trades to SQLite, updates cash balance (deducting transaction costs), and recalculates total portfolio value.

---

## Database Schema

```sql
simulation       -- current_date, cash_balance, total_value, day_number
positions        -- symbol, shares, avg_cost
transactions     -- date, action, symbol, shares, price, value, tc_cost, reason, agent
daily_snapshots  -- date, cash, holdings_value, total_equity, trades_count, notes
```

---

## Team

MiF Class 2025 · IE Business School · Deep Learning and Generative AI in Finance

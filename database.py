"""
database.py — SQLite layer for the MiF AI Investment Application

Tables:
    simulation      — game state (date, cash, total value)
    positions       — current holdings
    transactions    — full accounting log of every trade
    daily_snapshots — one row per day for the performance chart
"""

import sqlite3
from pathlib import Path

DB_PATH = "finance_game.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulation (
            id              INTEGER PRIMARY KEY,
            current_date    TEXT    NOT NULL,
            cash_balance    REAL    NOT NULL DEFAULT 1000000.0,
            total_value     REAL    NOT NULL DEFAULT 1000000.0,
            day_number      INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS positions (
            symbol          TEXT PRIMARY KEY,
            shares          REAL    NOT NULL DEFAULT 0.0,
            avg_cost        REAL    NOT NULL DEFAULT 0.0,
            target_price    REAL    DEFAULT NULL,
            stop_loss       REAL    DEFAULT NULL,
            entry_date      TEXT    DEFAULT NULL,
            holding_days    INTEGER DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            action      TEXT    NOT NULL,   -- BUY | SELL
            symbol      TEXT    NOT NULL,
            shares      REAL    NOT NULL,
            price       REAL    NOT NULL,
            value       REAL    NOT NULL,
            tc_cost     REAL    NOT NULL DEFAULT 0.0,
            reason      TEXT,
            agent       TEXT    DEFAULT 'Execution'
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date            TEXT PRIMARY KEY,
            cash            REAL    NOT NULL,
            holdings_value  REAL    NOT NULL DEFAULT 0.0,
            total_equity    REAL    NOT NULL,
            day_number      INTEGER NOT NULL DEFAULT 0,
            trades_count    INTEGER NOT NULL DEFAULT 0,
            agent_reasoning TEXT,
            notes           TEXT
        );
    """)
    conn.commit()
    conn.close()


def migrate_db() -> None:
    """Add new columns to existing DBs without wiping data."""
    conn = get_connection()
    existing = [row[1] for row in conn.execute("PRAGMA table_info(positions)").fetchall()]
    migrations = {
        "target_price": "ALTER TABLE positions ADD COLUMN target_price REAL DEFAULT NULL",
        "stop_loss":    "ALTER TABLE positions ADD COLUMN stop_loss REAL DEFAULT NULL",
        "entry_date":   "ALTER TABLE positions ADD COLUMN entry_date TEXT DEFAULT NULL",
        "holding_days": "ALTER TABLE positions ADD COLUMN holding_days INTEGER DEFAULT NULL",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
    conn.commit()
    conn.close()


def get_target_alerts(market_snap: dict) -> list[dict]:
    """
    Return positions where current price has crossed target or stop-loss.
    market_snap: {symbol: {"price": float, ...}}
    """
    positions = get_positions()
    alerts = []
    for p in positions:
        sym   = p["symbol"]
        price = (market_snap.get(sym) or {}).get("price")
        if not price:
            continue
        target = p.get("target_price")
        stop   = p.get("stop_loss")
        if target and price >= target:
            pnl_pct = (price - p["avg_cost"]) / p["avg_cost"] * 100
            alerts.append({"symbol": sym, "type": "TARGET_HIT",
                           "price": price, "level": target, "pnl_pct": pnl_pct})
        elif stop and price <= stop:
            pnl_pct = (price - p["avg_cost"]) / p["avg_cost"] * 100
            alerts.append({"symbol": sym, "type": "STOP_HIT",
                           "price": price, "level": stop, "pnl_pct": pnl_pct})
    return alerts



def reset_db(starting_cash: float = 1_000_000.0,
             start_date: str = "2026-03-13") -> None:
    """Wipe all tables and create a fresh simulation row."""
    conn = get_connection()
    conn.executescript("""
        DELETE FROM simulation;
        DELETE FROM positions;
        DELETE FROM transactions;
        DELETE FROM daily_snapshots;
    """)
    conn.execute(
        "INSERT INTO simulation (current_date, cash_balance, total_value, day_number) "
        "VALUES (?, ?, ?, 0)",
        (start_date, starting_cash, starting_cash)
    )
    conn.commit()
    conn.close()


# ─── Read helpers ─────────────────────────────────────────────────────────────

def get_sim_state() -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM simulation ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_positions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM positions WHERE shares > 0.001"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_transactions(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trades_today(date_str: str) -> int:
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE date=?", (date_str,)
    ).fetchone()[0]
    conn.close()
    return count


def get_snapshots() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Write helpers ────────────────────────────────────────────────────────────

def record_trade(date_str: str, action: str, symbol: str,
                 shares: float, price: float, tc_bps: float,
                 reason: str = "", agent: str = "Execution",
                 target_price: float = None, stop_loss: float = None,
                 holding_days: int = None) -> None:
    """Write a transaction and update cash + position atomically."""
    value   = shares * price
    tc_cost = value * tc_bps / 10_000

    conn = get_connection()
    # Log the transaction
    conn.execute(
        "INSERT INTO transactions "
        "(date, action, symbol, shares, price, value, tc_cost, reason, agent) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (date_str, action, symbol, shares, price, value, tc_cost, reason, agent)
    )

    if action == "BUY":
        existing = conn.execute(
            "SELECT shares, avg_cost FROM positions WHERE symbol=?", (symbol,)
        ).fetchone()
        if existing and existing["shares"] > 0:
            new_shares = existing["shares"] + shares
            new_cost   = (existing["shares"] * existing["avg_cost"] + shares * price) / new_shares
            conn.execute(
                "UPDATE positions SET shares=?, avg_cost=? WHERE symbol=?",
                (new_shares, new_cost, symbol)
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO positions (symbol, shares, avg_cost, target_price, stop_loss, entry_date, holding_days) VALUES (?,?,?,?,?,?,?)",
                (symbol, shares, price, target_price, stop_loss, date_str, holding_days)
            )
        # Debit cash
        conn.execute(
            "UPDATE simulation SET cash_balance = cash_balance - ?",
            (value + tc_cost,)
        )

    elif action == "SELL":
        conn.execute(
            "UPDATE positions SET shares = MAX(0, shares - ?) WHERE symbol=?",
            (shares, symbol)
        )
        # Credit cash
        conn.execute(
            "UPDATE simulation SET cash_balance = cash_balance + ?",
            (value - tc_cost,)
        )

    conn.commit()
    conn.close()


def update_total_value(new_total: float) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE simulation SET total_value=?, day_number=day_number+1",
        (new_total,)
    )
    conn.commit()
    conn.close()


def save_snapshot(date_str: str, cash: float, holdings_value: float,
                  total_equity: float, day_number: int,
                  trades_count: int, agent_reasoning: str = "",
                  notes: str = "") -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO daily_snapshots "
        "(date, cash, holdings_value, total_equity, day_number, "
        " trades_count, agent_reasoning, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (date_str, cash, holdings_value, total_equity,
         day_number, trades_count, agent_reasoning, notes)
    )
    conn.commit()
    conn.close()


def advance_date(current_date_str: str) -> str:
    """Move simulation date forward by one business day and return new date string."""
    import pandas as pd
    next_day = (pd.Timestamp(current_date_str) + pd.tseries.offsets.BDay(1)).date().isoformat()
    conn = get_connection()
    conn.execute("UPDATE simulation SET current_date=?", (next_day,))
    conn.commit()
    conn.close()
    return next_day

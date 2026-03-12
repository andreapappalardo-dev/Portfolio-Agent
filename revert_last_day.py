"""
revert_last_day.py  —  Undo the most recent trading day
Usage:  python revert_last_day.py
"""
import sqlite3, sys

DB_PATH = "finance_game.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# ── 1. Find the last trading date ─────────────────────────────────────────────
last = cur.execute("SELECT MAX(date) AS d FROM transactions").fetchone()["d"]
if not last:
    print("No transactions found — nothing to revert.")
    conn.close()
    sys.exit(0)

trades = cur.execute(
    "SELECT * FROM transactions WHERE date = ? ORDER BY id ASC", (last,)
).fetchall()

print(f"\n  Reverting {len(trades)} trade(s) from {last}:\n")
for t in trades:
    print(f"    {t['action']:<6} {t['symbol']:<8} {t['shares']:.2f} shares @ ${t['price']:.2f}  (value ${t['value']:.2f})")

confirm = input("\n  Type YES to confirm revert: ").strip()
if confirm != "YES":
    print("  Aborted.")
    conn.close()
    sys.exit(0)

# ── 2. Reverse each trade ─────────────────────────────────────────────────────
for t in trades:
    action = t["action"]
    symbol = t["symbol"]
    shares = t["shares"]
    price  = t["price"]
    value  = t["value"]
    tc     = t["tc_cost"]

    if action == "BUY":
        # Reverse BUY: remove shares from positions, restore cash
        cur.execute(
            "UPDATE positions SET shares = shares - ? WHERE symbol = ?",
            (shares, symbol)
        )
        cur.execute("DELETE FROM positions WHERE symbol = ? AND shares <= 0.0001", (symbol,))
        cur.execute("UPDATE simulation SET cash_balance = cash_balance + ?", (value + tc,))

    elif action == "SELL":
        # Reverse SELL: restore shares to positions, deduct cash
        existing = cur.execute(
            "SELECT shares, average_cost FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if existing:
            cur.execute(
                "UPDATE positions SET shares = shares + ? WHERE symbol = ?",
                (shares, symbol)
            )
        else:
            cur.execute(
                "INSERT INTO positions (symbol, shares, average_cost) VALUES (?, ?, ?)",
                (symbol, shares, price)
            )
        cur.execute("UPDATE simulation SET cash_balance = cash_balance - ?", (value - tc,))

# ── 3. Remove transactions and snapshot for that day ─────────────────────────
cur.execute("DELETE FROM transactions WHERE date = ?", (last,))
cur.execute("DELETE FROM daily_snapshots WHERE date = ?", (last,))

# ── 4. Roll back simulation date by 1 day (business day) ─────────────────────
prev = cur.execute(
    "SELECT date FROM daily_snapshots ORDER BY date DESC LIMIT 1"
).fetchone()

if prev:
    cur.execute("UPDATE simulation SET current_date = ?", (prev["date"],))
    print(f"\n  Simulation date rolled back to {prev['date']}")
else:
    print(f"\n  No earlier snapshot found — date not changed.")

conn.commit()
conn.close()
print(f"  Done. Day {last} has been reverted.\n")

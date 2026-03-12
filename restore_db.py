"""
restore_db.py  — Run once to undo the accidental 'New Sim' reset.

Since the New Sim wiped all transactions and set portfolio back to $1,000,000 at Day 0,
the DB is already in a clean initial state. This script just confirms and displays the
current state. If you had a backup (finance_game_backup.db), it would restore from that.

Usage:
    python restore_db.py
"""
import sqlite3, shutil, os

DB = "finance_game.db"
BACKUP = "finance_game_backup.db"

if os.path.exists(BACKUP):
    shutil.copy(BACKUP, DB)
    print(f"✅ Restored from {BACKUP}")
else:
    print("ℹ️  No backup found — DB is currently in clean Day 0 state ($1,000,000).")
    print("   The accidental New Sim reset the portfolio to the start.")
    print("   Your previous trades cannot be recovered without a backup file.")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
sim = conn.execute("SELECT * FROM simulation ORDER BY id DESC LIMIT 1").fetchone()
if sim:
    print(f"\nCurrent state:")
    print(f"  Date:       {sim['current_date']}")
    print(f"  Cash:       ${sim['cash_balance']:,.2f}")
    print(f"  Total:      ${sim['total_value']:,.2f}")
    print(f"  Day:        {sim['day_number']}")

txns = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
pos  = conn.execute("SELECT COUNT(*) FROM positions WHERE shares > 0").fetchone()[0]
print(f"  Positions:  {pos}")
print(f"  Total txns: {txns}")
conn.close()

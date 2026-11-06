"""SQLite storage with conflict detection. (PostgreSQL is overkill at this scale.)"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/processed/loto.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    draw_date TEXT PRIMARY KEY,
    n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER, n6 INTEGER,
    source TEXT
);
CREATE TABLE IF NOT EXISTS conflicts (
    draw_date TEXT, existing TEXT, incoming TEXT, existing_source TEXT, incoming_source TEXT
);
CREATE TABLE IF NOT EXISTS rejected (
    raw TEXT, reason TEXT, source TEXT
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert_draws(con: sqlite3.Connection, rows: list[tuple], source: str) -> dict:
    """rows: (date, (n1..n6)). Returns counts of inserted/duplicate/conflict."""
    stats = {"inserted": 0, "duplicate": 0, "conflict": 0}
    for d, nums in rows:
        key = d.isoformat()
        cur = con.execute("SELECT n1,n2,n3,n4,n5,n6,source FROM draws WHERE draw_date=?", (key,)).fetchone()
        if cur is None:
            con.execute("INSERT INTO draws VALUES (?,?,?,?,?,?,?,?)", (key, *nums, source))
            stats["inserted"] += 1
        elif tuple(cur[:6]) == tuple(nums):
            stats["duplicate"] += 1
        else:
            con.execute("INSERT INTO conflicts VALUES (?,?,?,?,?)",
                        (key, str(cur[:6]), str(nums), cur[6], source))
            stats["conflict"] += 1
    con.commit()
    return stats


def log_rejected(con, raw: str, reason: str, source: str):
    con.execute("INSERT INTO rejected VALUES (?,?,?)", (raw[:500], reason, source))
    con.commit()


def load_draws(path: Path = DB_PATH) -> pd.DataFrame:
    con = connect(path)
    df = pd.read_sql_query("SELECT * FROM draws ORDER BY draw_date", con)
    con.close()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    return df

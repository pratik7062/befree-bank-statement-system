import re
import sqlite3
from pathlib import Path
import pandas as pd

DB = Path("data/history.db")
DB.parent.mkdir(parents=True, exist_ok=True)

TRANSACTION_SCHEMA = {
    "date": "TEXT", "value_date": "TEXT", "post_date": "TEXT", "description": "TEXT",
    "debit": "REAL", "credit": "REAL", "balance": "REAL", "category": "TEXT",
    "classification_method": "TEXT", "confidence": "REAL", "validation_issues": "TEXT",
}


def _quote_identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
        raise ValueError(f"Unsupported database column name: {name!r}")
    return '"' + name + '"'


def _sqlite_type(series=None, fallback="TEXT"):
    if series is None:
        return fallback
    dtype = series.dtype if isinstance(series, pd.Series) else type(series)
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_numeric_dtype(dtype):
        return "REAL"
    return "TEXT"


def _ensure_columns(con, table, columns):
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({_quote_identifier(table)})")}
    for name, sql_type in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {_quote_identifier(table)} ADD COLUMN {_quote_identifier(name)} {sql_type}")
            existing.add(name)
    return existing


def migrate_database(db_path=DB, transaction_columns=(), account_columns=()):
    """Idempotently add current columns while preserving all existing SQLite rows."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE IF NOT EXISTS "transactions" ("id" INTEGER PRIMARY KEY AUTOINCREMENT)')
        con.execute('CREATE TABLE IF NOT EXISTS "runs" ("id" INTEGER PRIMARY KEY AUTOINCREMENT)')
        tx_columns = dict(TRANSACTION_SCHEMA)
        for name, series in transaction_columns:
            tx_columns.setdefault(name, _sqlite_type(series))
        run_columns = {"filename": "TEXT"}
        for name, series in account_columns:
            run_columns.setdefault(name, _sqlite_type(series))
        _ensure_columns(con, "transactions", tx_columns)
        _ensure_columns(con, "runs", run_columns)


def save_run(filename, transactions, account, db_path=DB):
    migrate_database(db_path, transactions.items(), account.items())
    with sqlite3.connect(db_path) as con:
        transactions.to_sql("transactions", con, if_exists="append", index=False)
        pd.DataFrame([{"filename": filename, **account}]).to_sql("runs", con, if_exists="append", index=False)


def load_runs(db_path=DB):
    if not Path(db_path).exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        try:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY rowid DESC LIMIT 20", con)
        except (sqlite3.Error, pd.errors.DatabaseError):
            return pd.DataFrame()

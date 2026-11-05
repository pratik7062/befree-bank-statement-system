import sqlite3
import pandas as pd
from src.storage import migrate_database, save_run


def test_legacy_schema_migration_adds_new_columns_and_preserves_history(tmp_path):
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE transactions (id INTEGER PRIMARY KEY, date TEXT, description TEXT, debit REAL, credit REAL, balance REAL)")
        con.execute("INSERT INTO transactions (date, description, debit, credit, balance) VALUES ('2026-01-01', 'Old row', 1, NULL, 9)")
        con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, filename TEXT)")
        con.execute("INSERT INTO runs (filename) VALUES ('old.pdf')")
    migrate_database(path)
    migrate_database(path)
    with sqlite3.connect(path) as con:
        tx_columns = {row[1] for row in con.execute("PRAGMA table_info(transactions)")}
        old_row = con.execute("SELECT description FROM transactions WHERE id=1").fetchone()
        run_row = con.execute("SELECT filename FROM runs WHERE id=1").fetchone()
    assert {"value_date", "post_date", "classification_method", "validation_issues"}.issubset(tx_columns)
    assert old_row == ("Old row",)
    assert run_row == ("old.pdf",)


def test_save_run_persists_new_transaction_columns_after_migration(tmp_path):
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE transactions (id INTEGER PRIMARY KEY, date TEXT, description TEXT)")
    tx = pd.DataFrame([{
        "date": "2026-01-02", "value_date": "2026-01-03", "post_date": "2026-01-02",
        "description": "UPI/DR/123 SWIGGY", "debit": 12.5, "credit": None,
        "balance": 87.5, "category": "Food & Dining", "classification_method": "Rule-based",
        "confidence": 1.0, "validation_issues": "",
    }])
    save_run("new.pdf", tx, {"account_holder": "Synthetic"}, db_path=path)
    with sqlite3.connect(path) as con:
        row = con.execute("SELECT value_date, post_date, category, classification_method, confidence FROM transactions WHERE description LIKE '%SWIGGY%'").fetchone()
        runs = con.execute("SELECT filename, account_holder FROM runs").fetchone()
    assert row == ("2026-01-03", "2026-01-02", "Food & Dining", "Rule-based", 1.0)
    assert runs == ("new.pdf", "Synthetic")

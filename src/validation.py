import pandas as pd
from dateutil import parser as date_parser


def _is_valid_date(value):
    """Validate individual statement dates without pandas inference warnings."""
    if pd.isna(value) or not str(value).strip():
        return False
    try:
        date_parser.parse(str(value).strip(), dayfirst=True, fuzzy=False)
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def reconcile(transactions: pd.DataFrame, opening_balance=None, statement_summary=None) -> dict:
    df = transactions.copy()
    debit = pd.to_numeric(df.get("debit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    credit = pd.to_numeric(df.get("credit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    balances = pd.to_numeric(df.get("balance", pd.Series(dtype=float)), errors="coerce")
    calc = float(opening_balance) + credit.sum() - debit.sum() if opening_balance is not None else None
    summary = statement_summary or {}
    close = summary.get("closing_balance")
    if close is None:
        close = float(balances.iloc[-1]) if len(balances) and pd.notna(balances.iloc[-1]) else None
    balance_check = bool(abs(calc - close) <= .01) if calc is not None and close is not None else None
    debit_check = _matches(debit.sum(), summary.get("total_debits"))
    credit_check = _matches(credit.sum(), summary.get("total_credits"))
    running_check = _running_check(df, opening_balance)
    checks = [balance_check, debit_check, credit_check, running_check]
    status = "Not Available" if opening_balance is None else ("PASS" if all(x is True for x in checks if x is not None) and any(x is True for x in checks) else "FAIL")
    if opening_balance is not None and any(x is False for x in checks):
        status = "FAIL"
    return {"opening_balance": opening_balance, "transaction_count": len(df),
            "debit_count": int((debit > 0).sum()), "credit_count": int((credit > 0).sum()),
            "total_debits": round(float(debit.sum()), 2), "total_credits": round(float(credit.sum()), 2),
            "closing_balance": close, "calculated_closing_balance": round(float(calc), 2) if calc is not None else None,
            "balance_check": balance_check, "debit_total_check": debit_check,
            "credit_total_check": credit_check, "running_balance_check": running_check,
            "status": status}


def _matches(actual, expected):
    return bool(abs(float(actual) - float(expected)) <= .01) if expected is not None else None


def _running_check(df, opening_balance):
    if opening_balance is None or df.empty or not {"debit", "credit", "balance"}.issubset(df.columns): return None
    expected = float(opening_balance)
    for row in df.itertuples(index=False):
        d = pd.to_numeric(getattr(row, "debit", None), errors="coerce")
        c = pd.to_numeric(getattr(row, "credit", None), errors="coerce")
        b = pd.to_numeric(getattr(row, "balance", None), errors="coerce")
        if pd.isna(b) or (pd.isna(d) and pd.isna(c)): return False
        expected += (0 if pd.isna(c) else c) - (0 if pd.isna(d) else d)
        if abs(expected - b) > .01: return False
    return True


def validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy().reset_index(drop=True)
    dates = df.get("date", pd.Series(index=df.index, dtype=object))
    debit = pd.to_numeric(df.get("debit", pd.Series(index=df.index, dtype=float)), errors="coerce")
    credit = pd.to_numeric(df.get("credit", pd.Series(index=df.index, dtype=float)), errors="coerce")
    balance = pd.to_numeric(df.get("balance", pd.Series(index=df.index, dtype=float)), errors="coerce")
    issues = []
    for i, row in df.iterrows():
        item = []
        if not _is_valid_date(dates.iloc[i]): item.append("Invalid or missing date")
        if not str(row.get("description", "")).strip() or str(row.get("description", "")).lower() == "nan": item.append("Missing description")
        d, c = debit.iloc[i], credit.iloc[i]
        if pd.isna(d) and pd.isna(c): item.append("Missing amount")
        if (pd.notna(d) and d < 0) or (pd.notna(c) and c < 0): item.append("Negative amount")
        if pd.notna(d) and d != 0 and pd.notna(c) and c != 0: item.append("Ambiguous debit and credit")
        if pd.isna(balance.iloc[i]): item.append("Invalid or missing balance")
        issues.append("; ".join(item))
    df["validation_issues"] = issues
    keys = [c for c in ["date", "description", "debit", "credit", "balance"] if c in df]
    duplicates = df.duplicated(subset=keys, keep=False)
    df.loc[duplicates, "validation_issues"] = df.loc[duplicates, "validation_issues"].apply(lambda x: (x + "; " if x else "") + "Duplicate transaction")
    return df

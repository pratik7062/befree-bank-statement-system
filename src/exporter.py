from io import BytesIO
import pandas as pd
from .validation import reconcile, validate_transactions


def to_excel(transactions: pd.DataFrame, account: dict, reconciliation=None) -> bytes:
    output = BytesIO()
    tx = validate_transactions(transactions)
    if "transaction_direction" not in tx:
        tx["transaction_direction"] = ""
    rec = reconciliation or reconcile(tx, account.get("opening_balance"))
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([account]).to_excel(writer, sheet_name="Account Details", index=False)
        tx.to_excel(writer, sheet_name="Transactions", index=False)
        if "category" in tx and not tx.empty:
            summary = tx.groupby("category", dropna=False).agg(
                transaction_count=("description", "count"),
                debit_total=("debit", "sum"), credit_total=("credit", "sum")).reset_index()
        else:
            summary = pd.DataFrame(columns=["category", "transaction_count", "debit_total", "credit_total"])
        summary.to_excel(writer, sheet_name="Category Summary", index=False)
        tx.loc[tx["validation_issues"].ne(""), ["date", "description", "validation_issues"]].to_excel(writer, sheet_name="Validation Report", index=False)
        pd.DataFrame([rec]).to_excel(writer, sheet_name="Reconciliation Report", index=False)
    return output.getvalue()

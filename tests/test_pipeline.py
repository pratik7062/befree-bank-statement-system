from io import BytesIO
from pathlib import Path
import pytest
import pandas as pd
from src.classifier import classify_transactions
from src.extractor import extract_account_details, extract_transactions, _parse_date
from src.validation import reconcile, validate_transactions
from src.exporter import to_excel
from src.pipeline import extract_statement_summary
from src.pipeline import process_pdf


SBI_TEXT = """STATE BANK OF INDIA
Account Holder Name: Asha Rao
Account No: 12345678901
IFSC: SBIN0001234
Statement Period: 01-10-2026 to 31-10-2026
Txn Date Value Date Description Debit Credit Balance
01-10-2026 01-10-2026 UPI-SWIGGY-ORDER 450.00 0.00 20289.82
02-10-2026 02-10-2026 SALARY ACME 0.00 17737.00 38026.82
03-10-2026 03-10-2026 PETROL PUMP 300.00 0.00 37726.82
04-10-2026 04-10-2026 AMAZON MARKETPLACE 156.00 0.00 37570.82
Opening Balance 20739.82
Total Credits 17737.00
Closing Balance 37570.82
"""


def test_sbi_extraction_and_account_details():
    tx = extract_transactions(SBI_TEXT, bank="sbi")
    account = extract_account_details(SBI_TEXT)
    assert len(tx) == 4
    assert tx.loc[0, "debit"] == 450
    assert tx.loc[0, "credit"] is None or pd.isna(tx.loc[0, "credit"])
    assert tx.loc[1, "credit"] == 17737
    assert tx.loc[1, "debit"] is None or pd.isna(tx.loc[1, "debit"])
    assert account["account_holder"] == "Asha Rao"
    assert account["account_number"] == "12345678901"
    assert account["ifsc"] == "SBIN0001234"
    assert "01-10-2026" in account["statement_period"]


def test_multiline_description_and_date_parser():
    text = """STATE BANK OF INDIA
Txn Date Value Date Description Debit Credit Balance
01-10-2026 01-10-2026 UPI-REF123 SWIGGY 100.00 0.00 900.00
FOOD ORDER SECOND LINE
"""
    tx = extract_transactions(text, bank="sbi")
    assert len(tx) == 1
    assert "FOOD ORDER SECOND LINE" in tx.loc[0, "description"]
    assert tx.loc[0, "date"] == "2026-10-01"
    assert _parse_date("31/10/2026") == "2026-10-31"


def test_sbi_multiline_narration_keeps_numeric_reference_and_separates_rows():
    text = """STATE BANK OF INDIA
Txn Date Value Date Description Debit Credit Balance
01-10-2026 02-10-2026 UPI/DR/667132765927/VIJAY 100.00 0.00 900.00
BANK ID: SBIN0001234 PAYMENT REF 887766
02-10-2026 03-10-2026 UPI/CR/123456789/SALARY 0.00 200.00 1100.00
EMPLOYER PAYROLL REF 998877
Brought Forward: 1,000.00 CR
Total Debits 100.00 Total Credits 200.00 Closing Balance: 1,100.00 CR
1,000.00 CR 1 1 100.00 200.00 1,100.00 CR
Page no. 1
"""
    tx = extract_transactions(text, bank="sbi")
    assert len(tx) == 2
    assert "SBIN0001234 PAYMENT REF 887766" in tx.loc[0, "description"]
    assert "EMPLOYER PAYROLL REF 998877" in tx.loc[1, "description"]
    assert tx.loc[0, "value_date"] == "2026-10-02"
    assert tx.loc[0, "post_date"] == "2026-10-01"
    assert tx.loc[0, "debit"] == 100 and tx.loc[1, "credit"] == 200
    assert tx.loc[0, "balance"] == 900 and tx.loc[1, "balance"] == 1100
    assert "1,000.00" not in tx.loc[1, "description"]
    assert "Page no." not in tx.loc[1, "description"]


def test_classification_merchant_precedes_upi_and_ml_fallback():
    df = pd.DataFrame({"description": ["UPI SWIGGY ORDER", "PETROL PUMP", "AMAZON", "SALARY CREDIT", "UPI TRANSFER TO OWN ACCOUNT", "MYSTERY XYZ"]})
    out = classify_transactions(df)
    assert out.loc[0, "category"] == "Food & Dining"
    assert out.loc[1, "category"] == "Transportation"
    assert out.loc[2, "category"] == "Shopping"
    assert out.loc[3, "category"] == "Salary"
    assert out.loc[4, "category"] == "Transfers"
    assert "classification_method" in out and "confidence" in out
    assert out.loc[5, "classification_method"] == "TF-IDF + Naive Bayes"
    assert pd.isna(out.loc[0, "confidence"])


def test_transfer_intent_phrases_are_classified_as_transfers():
    df = pd.DataFrame({"description": ["TRANSFER TO OWN ACCOUNT", "SELF TRANSFER", "OWN ACCOUNT TRANSFER"]})
    out = classify_transactions(df)
    assert out["category"].tolist() == ["Transfers", "Transfers", "Transfers"]
    assert out["classification_method"].tolist() == ["Rule-based"] * 3


def test_complete_narration_drives_merchant_classification_and_unknown_upi_review():
    df = pd.DataFrame({"description": [
        "UPI/DR/6671/VIJAY SWIGGY FOOD ORDER", "UPI/DR/22 PETROL PUMP", "UPI/DR/33 AMAZON",
        "NEFT SALARY ACME", "UPI/DR/44 ELECTRICITY BILL", "UPI/DR/667132765927/VIJAY",
    ]})
    out = classify_transactions(df)
    assert out["category"].tolist() == [
        "Food & Dining", "Transportation", "Shopping", "Salary", "Utilities", "Needs Review"
    ]
    assert out["transaction_direction"].tolist() == [
        "Debit/Outgoing", "Debit/Outgoing", "Debit/Outgoing", "", "Debit/Outgoing", "Debit/Outgoing"
    ]


def test_upi_credit_direction_is_not_a_spending_category():
    out = classify_transactions(pd.DataFrame({"description": ["UPI/CR/778899/UNKNOWN PERSON"]}))
    assert out.loc[0, "transaction_direction"] == "Credit/Incoming"
    assert out.loc[0, "category"] == "Needs Review"


def test_brought_forward_and_closing_balance_summary_extraction():
    summary = extract_statement_summary(
        "Brought Forward( ) Dr Count Cr Count Total Debits( ) Total Credits( ) Closing Balance( )\n"
        "20,739.82CR 9 1 906.00 17,737.00 37,570.82CR"
    )
    assert summary["opening_balance"] == 20739.82
    assert summary["closing_balance"] == 37570.82
    assert summary["total_debits"] == 906
    assert summary["total_credits"] == 17737


def test_statement_total_and_running_balance_reconciliation_passes():
    tx = pd.DataFrame({"debit": [10, None], "credit": [None, 20], "balance": [90, 110]})
    summary = {"total_debits": 10, "total_credits": 20, "closing_balance": 110}
    result = reconcile(tx, 100, summary)
    assert result["status"] == "PASS"
    assert result["debit_total_check"] is True
    assert result["credit_total_check"] is True
    assert result["running_balance_check"] is True


def test_reconciliation_is_not_available_without_opening_balance():
    result = reconcile(pd.DataFrame({"debit": [1], "credit": [None], "balance": [9]}))
    assert result["status"] == "Not Available"


def test_duplicate_and_invalid_data_flags():
    df = pd.DataFrame([{"date": "bad", "description": "", "debit": -2, "credit": 5, "balance": None}] * 2)
    out = validate_transactions(df)
    assert out.validation_issues.str.contains("Invalid or missing date").all()
    assert out.validation_issues.str.contains("Duplicate transaction").all()
    assert out.validation_issues.str.contains("Negative amount").all()
    assert out.validation_issues.str.contains("Ambiguous debit and credit").all()


def test_negative_debit_and_positive_credit_are_ambiguous_without_modification():
    df = pd.DataFrame([{"date": "01-10-2026", "description": "Review", "debit": -2, "credit": 5, "balance": 3}])
    out = validate_transactions(df)
    assert "Ambiguous debit and credit" in out.loc[0, "validation_issues"]
    assert "Negative amount" in out.loc[0, "validation_issues"]
    assert out.loc[0, "debit"] == -2
    assert out.loc[0, "credit"] == 5


def test_reconciliation():
    df = pd.DataFrame({"debit": [906], "credit": [17737], "balance": [37570.82]})
    result = reconcile(df, 20739.82)
    assert result["calculated_closing_balance"] == 37570.82
    assert result["balance_check"] is True
    assert result["total_debits"] == 906
    assert result["total_credits"] == 17737
    assert type(result["balance_check"]) is bool


def test_date_validation_accepts_statement_formats_without_inference():
    df = pd.DataFrame([
        {"date": "31/10/2026", "description": "Valid date", "debit": 1, "credit": None, "balance": 9},
        {"date": "2026-10-31", "description": "ISO date", "debit": 1, "credit": None, "balance": 8},
    ])
    out = validate_transactions(df)
    assert not out.validation_issues.str.contains("Invalid or missing date").any()


def test_excel_has_required_sheets_and_corrected_values():
    tx = pd.DataFrame([{"date": "2026-10-01", "description": "SWIGGY full narration", "debit": 10, "credit": None, "balance": 90, "category": "Food & Dining", "transaction_direction": "Debit/Outgoing"}])
    content = to_excel(tx, {"account_holder": "Test"}, {"balance_check": True})
    workbook = pd.ExcelFile(BytesIO(content))
    sheets = workbook.sheet_names
    assert sheets == ["Account Details", "Transactions", "Category Summary", "Validation Report", "Reconciliation Report"]
    assert "transaction_direction" in pd.read_excel(BytesIO(content), sheet_name="Transactions").columns
    assert pd.read_excel(BytesIO(content), sheet_name="Transactions").loc[0, "description"] == "SWIGGY full narration"


LOCAL_SBI_PDF = Path(__file__).parents[1] / "data" / "uploads" / "AccountStatement_04112026_211024_unlocked.pdf"


@pytest.mark.skipif(not LOCAL_SBI_PDF.exists(), reason="Local SBI statement is not available")
def test_local_sbi_statement_financial_and_narration_regression():
    result = process_pdf(str(LOCAL_SBI_PDF))
    tx = result["transactions"]
    rec = result["reconciliation"]
    assert len(tx) == 10
    assert rec["debit_count"] == 9
    assert rec["credit_count"] == 1
    assert rec["total_debits"] == 906
    assert rec["total_credits"] == 17737
    assert result["opening_balance"] == 20739.82
    assert result["closing_balance"] == 37570.82
    assert rec["status"] == "PASS"
    assert rec["running_balance_check"] is True
    assert tx["description"].str.len().gt(30).all()
    assert tx["transaction_direction"].value_counts().to_dict() == {"Debit/Outgoing": 9, "Credit/Incoming": 1}

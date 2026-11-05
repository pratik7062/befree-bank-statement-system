from io import StringIO
import shutil
import sqlite3

import pandas as pd
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.exporter import to_excel
from src.pipeline import process_pdf
from src.storage import load_runs, save_run
from src.validation import reconcile
from src.classifier import classify_transactions


def _write_text_pdf(path, lines):
    """Write a tiny single-page text PDF without adding a test dependency."""
    commands = ["BT", "/F1 10 Tf", "50 750 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            commands.append("0 -16 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    path.write_bytes(content)


def test_text_based_pdf_pipeline_with_synthetic_statement(tmp_path):
    pdf = tmp_path / "synthetic_text_statement.pdf"
    _write_text_pdf(pdf, [
        "BANK STATEMENT TRANSACTION DATE VALUE DATE DESCRIPTION DEBIT CREDIT BALANCE",
        "01-01-2026 01-01-2026 SWIGGY FOOD 10.00 0.00 90.00",
    ])
    result = process_pdf(str(pdf), enable_ocr=False)
    assert result["pdf_type"] == "text"
    assert len(result["transactions"]) == 1
    assert result["transactions"].loc[0, "debit"] == 10
    assert result["transactions"].loc[0, "category"] == "Food & Dining"
    assert result["transactions"].loc[0, "transaction_direction"] == "Debit/Outgoing"


def test_scanned_pdf_ocr_with_synthetic_image(tmp_path):
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract executable is not installed or available on PATH")
    image = Image.new("RGB", (1500, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), "SYNTHETIC BANK STATEMENT OCR CHECK", fill="black", font=ImageFont.load_default(size=48))
    pdf = tmp_path / "synthetic_scanned_statement.pdf"
    image.save(pdf, "PDF", resolution=150)
    result = process_pdf(str(pdf), enable_ocr=True)
    assert result["pdf_type"] == "image"
    assert result["method"] == "Tesseract OCR"
    assert "SYNTHETIC" in result["raw_text"].upper()


def test_ml_fallback_and_unknown_upi_review():
    out = classify_transactions(pd.DataFrame({"description": ["meal purchase", "UPI/DR/123456/UNKNOWN PERSON"]}))
    assert out.loc[0, "classification_method"] == "TF-IDF + Naive Bayes"
    assert 0 <= out.loc[0, "confidence"] <= 1
    assert out.loc[1, "category"] == "Needs Review"


def test_manual_edit_updates_reconciliation_and_csv_fields():
    original = pd.DataFrame([{
        "date": "2026-01-01", "value_date": "2026-01-01", "post_date": "2026-01-01",
        "description": "SWIGGY FOOD ORDER FULL NARRATION", "debit": 10, "credit": None,
        "balance": 90, "category": "Food & Dining", "transaction_direction": "Debit/Outgoing",
        "classification_method": "Rule-based", "confidence": None, "validation_issues": "",
    }])
    summary = {"total_debits": 10, "total_credits": 0, "closing_balance": 90}
    assert reconcile(original, 100, summary)["status"] == "PASS"
    edited = original.copy()
    edited.loc[0, "debit"] = 20
    edited.loc[0, "balance"] = 80
    edited_rec = reconcile(edited, 100, summary)
    assert edited_rec["status"] == "FAIL"
    assert edited_rec["calculated_closing_balance"] == 80
    csv_roundtrip = pd.read_csv(StringIO(edited.to_csv(index=False)))
    assert set(original.columns).issubset(csv_roundtrip.columns)
    assert csv_roundtrip.loc[0, "description"] == "SWIGGY FOOD ORDER FULL NARRATION"
    assert csv_roundtrip.loc[0, "transaction_direction"] == "Debit/Outgoing"


def test_sqlite_history_survives_a_new_connection(tmp_path):
    db = tmp_path / "history.db"
    tx = pd.DataFrame([{
        "date": "2026-01-01", "value_date": "2026-01-01", "post_date": "2026-01-01",
        "description": "Synthetic row", "debit": 2, "credit": None, "balance": 8,
        "category": "Needs Review", "transaction_direction": "Debit/Outgoing",
    }])
    save_run("synthetic.pdf", tx, {"account_holder": "Synthetic"}, db_path=db)
    # load_runs opens a fresh SQLite connection, modelling a later app launch.
    runs = load_runs(db)
    with sqlite3.connect(db) as con:
        stored = con.execute("SELECT description, transaction_direction FROM transactions").fetchone()
    assert runs.loc[0, "filename"] == "synthetic.pdf"
    assert runs.loc[0, "account_holder"] == "Synthetic"
    assert stored == ("Synthetic row", "Debit/Outgoing")


def test_excel_export_has_all_sheets_and_direction(tmp_path):
    tx = pd.DataFrame([{
        "date": "2026-01-01", "description": "Full synthetic narration", "debit": 1,
        "credit": None, "balance": 9, "category": "Needs Review",
        "transaction_direction": "Debit/Outgoing",
    }])
    from io import BytesIO
    payload = to_excel(tx, {"account_holder": "Synthetic"}, {"status": "Not Available"})
    book = pd.ExcelFile(BytesIO(payload))
    assert book.sheet_names == ["Account Details", "Transactions", "Category Summary", "Validation Report", "Reconciliation Report"]
    exported = pd.read_excel(BytesIO(payload), sheet_name="Transactions")
    assert exported.loc[0, "transaction_direction"] == "Debit/Outgoing"
    assert exported.loc[0, "description"] == "Full synthetic narration"

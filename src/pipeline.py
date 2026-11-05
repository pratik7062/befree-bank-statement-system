from pathlib import Path
import pdfplumber
from .ocr import ocr_pdf
import re
from .extractor import extract_account_details, extract_transactions
from .classifier import classify_transactions
from .validation import validate_transactions, reconcile


def extract_statement_summary(text):
    """Read balance and total lines from statement summaries without fixed values."""
    amount = r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?)"
    fields = {
        "opening_balance": r"(?:Brought\s+Forward|Opening\s+Balance)",
        "total_debits": r"Total\s+Debits?",
        "total_credits": r"Total\s+Credits?",
        "closing_balance": r"Closing\s+Balance",
    }
    result = {key: None for key in fields}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        for key, label in fields.items():
            match = re.search(rf"\b{label}\b\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*{amount}\s*(CR|DR)?\b", line, re.I)
            if match:
                value = float(match.group(1).replace(",", ""))
                if key in {"opening_balance", "closing_balance"} and (match.group(2) or "").upper() == "DR":
                    value = -value
                result[key] = value
        # SBI sometimes renders summary column labels and their values on adjacent rows.
        if re.search(r"Brought\s+Forward.*Dr\s+Count.*Cr\s+Count.*Total\s+Debits.*Total\s+Credits.*Closing\s+Balance", line, re.I):
            values_line = next((candidate for candidate in lines[index + 1:] if candidate.strip()), "")
            tokens = re.findall(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?(?:CR|DR)?", values_line, re.I)
            if len(tokens) >= 6:
                def parse_token(token, signed=False):
                    value = float(re.sub(r"(?:CR|DR)$", "", token, flags=re.I).replace(",", ""))
                    return -value if signed and token.upper().endswith("DR") else value
                result["opening_balance"] = parse_token(tokens[0], signed=True)
                result["total_debits"] = parse_token(tokens[3])
                result["total_credits"] = parse_token(tokens[4])
                result["closing_balance"] = parse_token(tokens[5], signed=True)
    return result

def process_pdf(path: str, enable_ocr: bool = True):
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError("Please provide an existing PDF file.")
    page_texts = []
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            page_texts.append(page.extract_text() or "")
    raw_text = "\n\n".join(page_texts)
    text_chars = sum(len(x.strip()) for x in page_texts)
    if text_chars >= 40:
        pdf_type, method = "text", "pdfplumber"
    elif enable_ocr:
        raw_text = ocr_pdf(str(path))
        pdf_type, method = "image", "Tesseract OCR"
    else:
        raise ValueError("This appears to be a scanned PDF. Enable OCR and install Tesseract.")
    account = extract_account_details(raw_text)
    bank = "SBI" if re.search(r"\bSTATE BANK OF INDIA\b|\bSBIN\b", raw_text, re.I) else "generic"
    tx = extract_transactions(raw_text, bank=bank)
    if not tx.empty:
        tx = classify_transactions(tx)
    summary = extract_statement_summary(raw_text)
    opening = summary["opening_balance"]
    tx = validate_transactions(tx)
    reconciliation = reconcile(tx, opening, summary)
    return {"account": account, "transactions": tx, "pdf_type": pdf_type,
            "method": method, "pages": pages, "raw_text": raw_text,
            "bank": bank, "opening_balance": opening,
            "closing_balance": summary["closing_balance"],
            "statement_summary": summary, "reconciliation": reconciliation}

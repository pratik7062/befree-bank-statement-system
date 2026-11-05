import re
import pandas as pd
from dateutil import parser as date_parser

DATE_TOKEN = r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4})"
DATE_RE = re.compile(rf"(?<!\w)({DATE_TOKEN})(?!\w)")
AMOUNT_RE = re.compile(r"(?<![\w/])(?:₹|Rs\.?|INR)?\s*-?\d[\d,]*(?:\.\d{1,2})?(?!\d)", re.I)
TX_COLUMNS = ["date", "value_date", "post_date", "description", "debit", "credit", "balance"]


def _parse_date(value):
    try:
        return date_parser.parse(value, dayfirst=True, fuzzy=False).date().isoformat()
    except (ValueError, TypeError, OverflowError):
        return ""


def _money(raw):
    raw = re.sub(r"^(?:₹|Rs\.?|INR)\s*", "", raw.strip(), flags=re.I).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _amounts(text):
    return [(m.start(), _money(m.group())) for m in AMOUNT_RE.finditer(text)]


def extract_account_details(text: str) -> dict:
    """Extract common SBI and generic account metadata without treating labels as names."""
    patterns = {
        "account_holder": [r"(?:Account Holder(?: Name)?|Customer Name|Name of Customer)\s*[:\-]?\s*([^\n]{2,70})"],
        "account_number": [r"(?:Account No\.?|A/C No\.?|Account Number)\s*[:\-]?\s*([Xx*0-9\-]{5,25})"],
        "ifsc": [r"\b([A-Z]{4}0[A-Z0-9]{6})\b"],
        "statement_period": [r"(?:Statement Period|Period)\s*[:\-]?\s*([^\n]{5,70})", r"From\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+To\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"],
    }
    out = {key: "" for key in patterns}
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, text, re.I)
            if match:
                value = " ".join(part.strip() for part in match.groups())
                if key == "account_holder":
                    value = re.split(r"\s{2,}|\b(?:ACCOUNT|BRANCH|IFSC|CIF)\b", value, maxsplit=1, flags=re.I)[0].strip(" :-")
                out[key] = value.upper() if key == "ifsc" else value
                break
    return out


def _sbi_transactions(text):
    """Parse SBI statement rows: dates, narration/reference, debit, credit, balance."""
    rows, current = [], None
    summary_started = False
    header_seen = bool(re.search(r"(?:Txn|Transaction|Post)\s*Date.*(?:Debit|Withdrawal).*(?:Credit|Deposit).*Balance|(?:Debit|Credit).*Balance", text, re.I | re.S))
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.search(r"\bBrought\s+Forward\b|\bStatement\s+Summary\b|\bTotal\s+Debits?\b.*\bClosing\s+Balance\b", line, re.I):
            summary_started = True
        if summary_started:
            continue
        if _is_statement_boilerplate(line):
            continue
        dates = list(DATE_RE.finditer(line))
        is_transaction_row = False
        if len(dates) >= 2:
            start = dates[0]
            tail = line[start.end():]
            second = DATE_RE.match(tail.lstrip())
            if second:
                leading = len(tail) - len(tail.lstrip())
                after_dates = tail[leading + second.end():]
                vals = _amounts(after_dates)
                # Only a dated row with debit, credit and balance starts a new transaction.
                is_transaction_row = len([v for _, v in vals if v is not None]) >= 3
            if is_transaction_row:
                if current:
                    rows.append(current)
                date1 = _parse_date(start.group(1))
                date2 = _parse_date(second.group(1))
                description = after_dates[:vals[0][0]].strip(" -|:")
                # Ignore reference identifiers that occur as narration suffixes.
                nums = [v for _, v in vals if v is not None]
                if len(nums) >= 3:
                    debit, credit, balance = nums[-3:]
                    debit = debit or None
                    credit = credit or None
                    # SBI often prints a blank side as zero. Keep transaction amount semantics.
                    if debit is not None and credit is not None:
                        debit = credit = None
                    current = {"date": date1, "value_date": date2, "post_date": date1,
                               "description": description, "debit": debit, "credit": credit,
                               "balance": balance}
                continue
        if current and not is_transaction_row and not _is_table_header_or_metadata(line):
            # Narration continuations may contain numeric references, so do not reject
            # them merely because an amount-like token appears in the line.
            current["description"] = (current["description"] + " " + line).strip()
    if current:
        rows.append(current)
    return rows if header_seen else []


def _is_statement_boilerplate(line):
    return bool(re.search(
        r"\b(?:brought forward|carried forward|opening balance|closing balance|"
        r"total debits?|total credits?|statement summary|page\s*(?:no\.?\s*)?\d+(?:\s+of\s+\d+)?|"
        r"end of statement|state bank of india|account statement|please do not share|"
        r"customer care|printed on)\b", line, re.I))


def _is_table_header_or_metadata(line):
    return bool(re.search(
        r"\b(?:txn|transaction|post)\s*date\b|\b(?:value\s*date|narration|description)\s+"
        r"(?:debit|credit|balance)\b|\b(?:account holder|account no\.?|ifsc|statement period)\b",
        line, re.I))


def _generic_transactions(text):
    rows = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 10 or re.search(r"\b(?:opening balance|closing balance|total debits?|total credits?)\b", line, re.I):
            continue
        dates = list(DATE_RE.finditer(line))
        if not dates:
            continue
        date = _parse_date(dates[0].group(1))
        if not date:
            continue
        tail = line[dates[-1].end():]
        amounts = [(pos, val) for pos, val in _amounts(tail) if val is not None]
        if len(amounts) < 2:
            continue
        description = tail[:amounts[0][0]].strip(" -|:")
        # Generic format inference only if explicit DR/CR or a three-column amount sequence.
        vals = [v for _, v in amounts]
        debit = credit = None
        balance = vals[-1]
        if len(vals) >= 3:
            debit, credit = vals[-3], vals[-2]
            debit, credit = debit or None, credit or None
            if debit is not None and credit is not None:
                debit = credit = None
        elif re.search(r"\b(?:DR|DEBIT)\b", line, re.I):
            debit = vals[-2]
        elif re.search(r"\b(?:CR|CREDIT)\b", line, re.I):
            credit = vals[-2]
        else:
            # Unknown direction stays blank for validation/manual review.
            pass
        rows.append({"date": date, "value_date": date, "post_date": date,
                     "description": description or line, "debit": debit, "credit": credit,
                     "balance": balance})
    return rows


def _sbi_narrations(text):
    """Collect narration text between SBI table row boundaries, including wrapped lines."""
    narrations = []
    active = False
    current = None
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.search(r"\bBrought\s+Forward\b|\bStatement\s+Summary\b", line, re.I):
            break
        if _is_statement_boilerplate(line) or _is_table_header_or_metadata(line):
            continue

        dates = list(DATE_RE.finditer(line))
        row_description = None
        if len(dates) >= 2:
            tail = line[dates[0].end():]
            second = DATE_RE.match(tail.lstrip())
            if second:
                leading = len(tail) - len(tail.lstrip())
                after_dates = tail[leading + second.end():]
                amounts = _amounts(after_dates)
                # SBI text extraction often omits the zero side of debit/credit,
                # leaving the nonzero amount and running balance only.
                row_description = after_dates[:amounts[0][0]].strip(" -|:") if amounts else ""
                if len([value for _, value in amounts if value is not None]) >= 2 and row_description:
                    if current is not None:
                        narrations.append(current)
                    active = True
                    current = row_description
                    continue
        if active and current is not None:
            # Continuation lines can contain numeric references and must be retained.
            current = (current + " " + line).strip()
    if current is not None:
        narrations.append(current)
    return narrations


def extract_transactions(text: str, bank: str = "auto") -> pd.DataFrame:
    sbi = bank.lower() in {"auto", "sbi"}
    rows = _sbi_transactions(text) if sbi else []
    if not rows:
        rows = _generic_transactions(text)
        # Keep the known-good generic financial columns intact; replace only the
        # narration when the SBI table pass finds a one-to-one set of row descriptions.
        narrations = _sbi_narrations(text) if sbi else []
        if len(narrations) == len(rows):
            for row, narration in zip(rows, narrations):
                row["description"] = narration
    df = pd.DataFrame(rows, columns=TX_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates().reset_index(drop=True)
    return df

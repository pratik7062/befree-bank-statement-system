# Befree Bank Statement Processing & Classification System

A Python/Streamlit prototype for extracting transactions from text-based and scanned bank-statement PDFs, classifying transaction descriptions without LLMs, and exporting Excel/CSV.

## Features
- Text PDF extraction using `pdfplumber`
- Scanned PDF OCR using Tesseract
- SBI table-aware transaction and account-detail extraction with generic fallback
- Rule-based classification plus TF-IDF / Multinomial Naive Bayes fallback
- Editable transaction grid
- Data-quality review and balance reconciliation
- CSV and multi-sheet Excel export
- Category and monthly cash-flow charts
- Optional local SQLite processing history

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Tesseract OCR for scanned PDFs (text PDFs work without it)

## Setup in VS Code (Windows PowerShell)

```powershell
# 1. Open this project folder in VS Code
# 2. Create virtual environment
py -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, run this once for the current terminal:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Start app
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## OCR setup
Install Tesseract OCR for Windows. After installation, if Python cannot find it, add its installation directory to PATH or set the executable path in `src/ocr.py`, for example:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Extraction and validation
The SBI parser uses the statement transaction table structure and retains ambiguous or invalid records for review. Other formats use a conservative generic fallback; validate its results against the source statement before relying on them. Reconciliation reports totals and running-balance checks but never changes extracted amounts.

Run the automated suite with `python -m pytest -q`.

Never upload real account statements to a public repository. Use synthetic or redacted statements for demos.

## Project structure
```text
befree_bank_statement_system/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── pipeline.py
│   ├── ocr.py
│   ├── extractor.py
│   ├── classifier.py
│   ├── exporter.py
│   ├── storage.py
│   └── analytics.py
└── data/
```

## Classification design
- Exact keyword matches use deterministic rules.
- Remaining descriptions are classified using a TF-IDF vectorizer and Multinomial Naive Bayes.
- The training examples included in `src/classifier.py` are illustrative seed data. Replace/expand them with labeled transaction descriptions before claiming measured accuracy.
- No LLM or generative AI API is used.

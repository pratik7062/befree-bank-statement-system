import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.pipeline import process_pdf
from src.storage import save_run, load_runs
from src.analytics import category_summary, monthly_summary
from src.validation import validate_transactions, reconcile

st.set_page_config(page_title="Befree | Bank Statement Intelligence", page_icon="🏦", layout="wide")
st.title("🏦 Bank Statement Processing & Classification System")
st.caption("PDF extraction • OCR • Non-LLM classification • Excel / CSV export")

with st.sidebar:
    st.header("Processing settings")
    use_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True)
    save_history = st.checkbox("Save processing history locally", value=True)
    st.info("Classification uses deterministic rules plus TF-IDF + Multinomial Naive Bayes. No LLM is used.")

uploaded = st.file_uploader("Upload a bank statement PDF", type=["pdf"])
if uploaded:
    st.write(f"**File:** {uploaded.name} · **Size:** {uploaded.size/1024:.1f} KB")
    if st.button("Process statement", type="primary"):
        with st.spinner("Reading PDF, extracting transactions and classifying..."):
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = temp_dir / uploaded.name
            pdf_path.write_bytes(uploaded.getvalue())
            try:
                result = process_pdf(str(pdf_path), enable_ocr=use_ocr)
                st.session_state["result"] = result
                st.session_state["filename"] = uploaded.name
                if save_history and not result["transactions"].empty:
                    try:
                        save_run(uploaded.name, result["transactions"], result["account"])
                    except Exception as exc:
                        st.warning(f"Statement processed, but local history could not be saved: {exc}")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)

result = st.session_state.get("result")
if result:
    account = result["account"]
    tx = result["transactions"].copy()
    account["opening_balance"] = result.get("opening_balance")
    account["closing_balance"] = result.get("closing_balance")
    st.caption(f"Detected bank: {result.get('bank', 'unknown')} · PDF type: {result['pdf_type']} · Extraction: {result['method']}")
    rec = reconcile(tx, result.get("opening_balance"), result.get("statement_summary"))
    st.subheader("Reconciliation")
    st.metric("Reconciliation status", rec.get("status", "Not Available"))
    st.json(rec)
    st.write("Opening balance:", result.get("opening_balance") if result.get("opening_balance") is not None else "Not Available")
    st.write("Closing balance:", result.get("closing_balance") if result.get("closing_balance") is not None else "Not Available")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", len(tx))
    m2.metric("Total debits", f"₹{tx['debit'].fillna(0).sum():,.2f}" if not tx.empty else "₹0")
    m3.metric("Total credits", f"₹{tx['credit'].fillna(0).sum():,.2f}" if not tx.empty else "₹0")
    m4.metric("PDF type", result["pdf_type"].upper())

    st.subheader("Account information")
    st.json({k: v for k, v in account.items() if v})
    st.subheader("Extracted and classified transactions")
    if tx.empty:
        st.warning("No transactions were detected. This statement may use an unsupported layout; inspect extraction text and add a bank-specific parser.")
    else:
        st.subheader("Classification distribution")
        if "category" in tx:
            st.bar_chart(tx["category"].fillna("Needs Review").value_counts())
        if "transaction_direction" in tx:
            st.caption("Direction reflects incoming or outgoing movement; category reflects the transaction purpose.")
        pending = tx[tx.get("category", pd.Series(index=tx.index, dtype=object)).eq("Needs Review")]
        if not pending.empty:
            st.warning(f"{len(pending)} transaction(s) need category review.")
            st.dataframe(pending, use_container_width=True)
        edited = st.data_editor(
            tx, use_container_width=True, num_rows="dynamic", key="tx_editor",
            column_config={"description": st.column_config.TextColumn("Full narration", width="large")},
        )
        st.session_state["edited_tx"] = edited
        edited = validate_transactions(edited)
        st.subheader("Transactions requiring manual review")
        review = edited[edited["validation_issues"].ne("") | edited.get("category", pd.Series(index=edited.index, dtype=object)).eq("Needs Review")]
        st.dataframe(review, use_container_width=True)
        edited_rec = reconcile(edited, result.get("opening_balance"), result.get("statement_summary"))
        st.subheader("Reconciliation for edited transactions")
        st.json(edited_rec)
        csv = edited.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ Download CSV", csv, file_name="classified_transactions.csv", mime="text/csv")
        from src.exporter import to_excel
        xlsx = to_excel(edited, account, edited_rec)
        st.download_button("⬇ Download Excel", xlsx, file_name="classified_bank_statement.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        left, right = st.columns(2)
        with left:
            st.subheader("Category breakdown")
            st.bar_chart(category_summary(edited))
        with right:
            st.subheader("Monthly cash flow")
            st.line_chart(monthly_summary(edited))

    with st.expander("Extraction diagnostics"):
        st.write("Detected PDF type:", result["pdf_type"])
        st.write("Extraction method:", result["method"])
        st.write("Pages:", result["pages"])
        st.text_area("Extracted text (first 15,000 characters)", result["raw_text"][:15000], height=250)

st.divider()
with st.expander("Previous local processing runs"):
    try:
        history = load_runs()
    except Exception as exc:
        history = pd.DataFrame()
        st.warning(f"Local processing history is unavailable: {exc}")
    if history.empty:
        st.caption("No saved runs yet.")
    else:
        st.dataframe(history, use_container_width=True)

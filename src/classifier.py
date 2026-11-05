import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

TRAINING = [
    ("UPI/DR SWIGGY FOOD ORDER restaurant meal", "Food & Dining"),
    ("ZOMATO order cafe coffee pizza restaurant", "Food & Dining"),
    ("grocery store supermarket blinkit zepto", "Food & Dining"),
    ("UBER ride fare taxi cab", "Transportation"),
    ("PETROL pump fuel station diesel", "Transportation"),
    ("IRCTC railway metro bus transport fare", "Transportation"),
    ("AMAZON marketplace order shopping", "Shopping"),
    ("FLIPKART online purchase retail", "Shopping"),
    ("MYNTRA clothing fashion purchase", "Shopping"),
    ("SALARY payroll employer monthly wages", "Salary"),
    ("salary credit company payroll", "Salary"),
    ("RENT landlord lease house monthly rent", "Rent"),
    ("ELECTRICITY power bill utility payment", "Utilities"),
    ("water gas broadband mobile recharge bill", "Utilities"),
    ("JIO AIRTEL telephone internet bill", "Utilities"),
    ("hospital pharmacy doctor medical clinic", "Healthcare"),
    ("medicine prescription diagnostic health", "Healthcare"),
    ("school college university tuition course fees", "Education"),
    ("ATM cash withdrawal cash deposit", "ATM / Cash"),
    ("NEFT IMPS RTGS fund transfer own account", "Transfers"),
    ("self transfer account transfer beneficiary", "Transfers"),
    ("interest dividend refund cashback", "Income / Refund"),
    ("insurance premium policy renewal", "Insurance"),
    ("netflix spotify subscription membership", "Subscriptions"),
    ("unknown transaction narration", "Others"),
]
RULES = {
"Food & Dining": ["swiggy","zomato","restaurant","cafe","dominos","pizza","food","blinkit","zepto","grocery","groceries"],
"Transportation": ["uber","ola","rapido","petrol","fuel","metro","irctc","railway","bus ticket","bus fare","diesel"],
"Shopping": ["amazon","flipkart","myntra","ajio","mall","retail"],
"Salary": ["salary","payroll","wages"],
"Rent": ["rent","landlord","lease"],
"Utilities": ["electricity","electric bill","water bill","gas bill","broadband","recharge","jio","airtel"],
"Healthcare": ["hospital","pharmacy","medical","apollo","doctor","clinic"],
"Education": ["school","college","university","tuition","udemy","coursera"],
"ATM / Cash": ["atm","cash withdrawal","cash wdl"],
"Transfers": ["neft","imps","rtgs","fund transfer","self transfer","own account transfer","account transfer"],
"Income / Refund": ["interest","dividend","refund","cashback"],
"Insurance": ["insurance","premium","lic"],
"Subscriptions": ["netflix","spotify","subscription","prime video"],
}

def _rule(text):
    t = str(text).lower()
    # Specific merchant and purpose terms outrank transfer intent and payment rails.
    for category, terms in RULES.items():
        if category == "Transfers":
            continue
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", t) for term in terms):
            return category
    transfer_intents = (
        "transfer to own account", "self transfer", "own account transfer",
        "fund transfer", "account transfer", "transfer to self",
    )
    if any(phrase in t for phrase in transfer_intents) or any(
        re.search(rf"\b{re.escape(term)}\b", t) for term in ("neft", "imps", "rtgs")
    ):
        return "Transfers"
    return None

def _model():
    x = [x for x, _ in TRAINING]
    y = [y for _, y in TRAINING]
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
                      ("nb", MultinomialNB(alpha=0.35))])
    model.fit(x, y)
    return model

MODEL = _model()

MODEL_EVIDENCE = {
    token
    for example, _ in TRAINING
    for token in re.findall(r"[a-z]{3,}", example.lower())
} - {"upi", "payment", "transaction", "unknown", "narration"}


def _direction(description, row):
    if re.search(r"\bUPI\s*[/ -]\s*DR\b", description, re.I):
        return "Debit/Outgoing"
    if re.search(r"\bUPI\s*[/ -]\s*CR\b", description, re.I):
        return "Credit/Incoming"
    debit = pd.to_numeric(row.get("debit"), errors="coerce")
    credit = pd.to_numeric(row.get("credit"), errors="coerce")
    if pd.notna(debit) and debit > 0 and not (pd.notna(credit) and credit > 0):
        return "Debit/Outgoing"
    if pd.notna(credit) and credit > 0 and not (pd.notna(debit) and debit > 0):
        return "Credit/Incoming"
    return ""

def classify_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    descriptions = df["description"].fillna("").astype(str)
    df["transaction_direction"] = [
        _direction(desc, df.iloc[i]) for i, desc in enumerate(descriptions)
    ]
    categories, methods, confidences = [], [], []
    probs = MODEL.predict_proba(descriptions)
    classes = MODEL.classes_
    for i, desc in enumerate(descriptions):
        rule = _rule(desc)
        if rule:
            # A deterministic keyword match is not a calibrated probability.
            categories.append(rule); methods.append("Rule-based"); confidences.append(None)
        else:
            idx = probs[i].argmax()
            confidence = float(probs[i][idx])
            words = set(re.findall(r"[a-z]{3,}", desc.lower()))
            rail_only = bool(re.search(r"\bUPI\b", desc, re.I)) and not (words & MODEL_EVIDENCE)
            categories.append(classes[idx] if confidence >= 0.38 and not rail_only else "Needs Review")
            methods.append("TF-IDF + Naive Bayes")
            confidences.append(round(confidence, 3))
    df["category"] = categories
    df["classification_method"] = methods
    df["confidence"] = confidences
    return df

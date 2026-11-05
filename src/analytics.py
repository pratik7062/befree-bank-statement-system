import pandas as pd

def category_summary(df):
    if df.empty: return pd.DataFrame()
    return df.assign(debit=df["debit"].fillna(0)).groupby("category")["debit"].sum().sort_values(ascending=False)

def monthly_summary(df):
    if df.empty: return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["month"] = work["date"].dt.to_period("M").astype(str)
    work["debit"] = work["debit"].fillna(0)
    work["credit"] = work["credit"].fillna(0)
    return work.groupby("month")[["debit","credit"]].sum()

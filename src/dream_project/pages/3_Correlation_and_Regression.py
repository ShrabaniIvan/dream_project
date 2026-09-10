"""Correlation & Regression page — wheat dataset, numerical columns only."""
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from dream_project.modules.theme import apply_sidebar_style

st.set_page_config(page_title="Correlation & Regression", layout="wide")
apply_sidebar_style()
st.title("Correlation & Regression")

# ── 1. Load data ────────────────────────────────────────────────────────────

DATA_PATH = pathlib.Path(__file__).parents[3] / "data" / "wheat_data_Correlation_Regression.csv"

st.caption("Using default dataset: `wheat_data_Correlation_Regression.csv`. Upload a different file to override.")
uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded:
    name = uploaded.name.lower()
    df = pd.read_csv(uploaded) if name.endswith(".csv") else pd.read_excel(uploaded)
    source_name = uploaded.name
else:
    df = pd.read_csv(DATA_PATH)
    source_name = DATA_PATH.name

st.success(f"Loaded **{source_name}** — {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Column type check ─────────────────────────────────────────────────────

st.divider()
st.subheader("Column Type Check")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

if non_numeric_cols:
    st.warning(
        f"**{len(non_numeric_cols)} non-numeric column(s) found — excluded from analysis:** "
        + ", ".join(f"`{c}`" for c in non_numeric_cols)
    )
else:
    st.success("All columns are numeric.")

st.info(
    f"**{len(numeric_cols)} numeric column(s) selected for correlation:** "
    + ", ".join(f"`{c}`" for c in numeric_cols)
)

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to compute a correlation matrix.")
    st.stop()

# ── 3. Correlation matrix heatmap ────────────────────────────────────────────

st.divider()
st.subheader("Correlation Matrix")

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) - 1)))
fig.patch.set_facecolor("#f8f8f8")
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Pearson Correlation Matrix", fontsize=13)
ax.set_facecolor("#f0f0f0")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Values show Pearson r (−1 to +1). Red = positive correlation, Blue = negative. "
    "Strong correlations: |r| ≥ 0.70. Moderate: 0.30–0.69. Weak: < 0.30."
)

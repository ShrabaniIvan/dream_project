"""Correlation Analysis page — Pearson / Spearman / Chi-square."""
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from dream_project.modules.descriptive_stats import load_data
from dream_project.modules.correlation import (
    VARIABLE_TYPES, determine_method, detect_corrupt_pair,
    validate_bins, prepare_series,
    run_pearson, run_spearman, run_chi_square,
    strength_label, interpret,
)

st.set_page_config(page_title="Descriptive Statistics - II: Correlation Analysis", layout="wide")
st.title("Descriptive Statistics - II: Correlation Analysis")

# ── 1. File Upload ─────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

df = load_data(uploaded)
st.success(f"Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

cols = list(df.columns)

# ── 2. Column Selection ────────────────────────────────────────────────────

st.divider()
col_a = st.selectbox("Variable A", ["— select —"] + cols)
if col_a == "— select —":
    st.stop()

col_b_list = st.multiselect("Variable B (one or more)", [c for c in cols if c != col_a])
if not col_b_list:
    st.stop()

all_cols = list(dict.fromkeys([col_a] + col_b_list))  # preserve order, no duplicates

# ── 3. Variable Types ──────────────────────────────────────────────────────

st.divider()
st.subheader("Variable Types")

types_map = {}
order_map = {}
bins_map = {}

for col in all_cols:
    st.markdown(f"**{col}**")
    c1, c2 = st.columns([1, 3])
    vtype = c1.selectbox("Type", VARIABLE_TYPES, key=f"type_{col}")
    types_map[col] = vtype

    if vtype == "Ordinal":
        unique_vals = sorted(df[col].dropna().astype(str).unique())
        c2.caption(f"Unique values found: {', '.join(unique_vals)}")
        order_map[col] = c2.text_input(
            "Rank order — lowest to highest (comma-separated)",
            key=f"order_{col}",
            placeholder="e.g. Low, Medium, High",
        )

    elif vtype == "Nominal":
        numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
        if numeric_ratio > 0.5:
            n_unique = df[col].nunique()
            c2.caption(f"Column appears numeric ({n_unique} unique values). Provide bin boundaries.")
            bins_input = c2.text_input(
                "Bin boundaries (Python list)",
                key=f"bins_{col}",
                placeholder="e.g. [0, 50, 100, 200]",
            )
            if bins_input:
                ok, msg, bins = validate_bins(bins_input)
                if ok:
                    bins_map[col] = bins
                else:
                    c2.error(msg)
        else:
            c2.caption(f"Unique categories: {', '.join(sorted(df[col].dropna().astype(str).unique()))}")

# ── 4. Data Cleanliness ────────────────────────────────────────────────────

st.divider()
st.subheader("Data Cleanliness")
for col_b in col_b_list:
    clean_df, corrupt_df = detect_corrupt_pair(df, col_a, col_b)
    if corrupt_df.empty:
        st.success(f"**{col_a} × {col_b}** — clean ({len(clean_df)} valid rows)")
    else:
        st.warning(
            f"**{col_a} × {col_b}** — {len(corrupt_df)} corrupt/missing rows excluded "
            f"({len(clean_df)} valid rows remain)"
        )

# ── 5. Run Analysis ────────────────────────────────────────────────────────

st.divider()
if st.button("Run Analysis", type="primary"):
    st.session_state["corr_ready"] = True
    # Reset pair decisions when re-running
    for col_b in col_b_list:
        key = f"decision_{col_a}__{col_b}"
        if key in st.session_state:
            del st.session_state[key]

if not st.session_state.get("corr_ready"):
    st.stop()

# ── 6. Results — one section per pair ─────────────────────────────────────

for col_b in col_b_list:
    st.divider()
    st.subheader(f"{col_a}  ×  {col_b}")

    type_a = types_map[col_a]
    type_b = types_map.get(col_b, "Continuous")
    method = determine_method(type_a, type_b)
    method_label = "Chi-square / Cramér's V" if method == "chi-square" else method.title()
    st.caption(f"Method: **{method_label}** | {col_a}: {type_a}  ·  {col_b}: {type_b}")

    # Prepare clean data for this pair
    clean_df, _ = detect_corrupt_pair(df, col_a, col_b)

    # Prepare series A
    ok_a, err_a, series_a = prepare_series(
        clean_df[col_a],
        type_a,
        order_str=order_map.get(col_a, ""),
        bins=bins_map.get(col_a),
    )
    if not ok_a:
        st.error(f"**{col_a}**: {err_a}")
        continue

    # Prepare series B
    ok_b, err_b, series_b = prepare_series(
        clean_df[col_b],
        type_b,
        order_str=order_map.get(col_b, ""),
        bins=bins_map.get(col_b),
    )
    if not ok_b:
        st.error(f"**{col_b}**: {err_b}")
        continue

    # Align and drop remaining NaN
    combined = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if combined.empty:
        st.error("No valid data rows after preparation. Check types and inputs.")
        continue
    sa, sb = combined["a"], combined["b"]

    pair_key = f"{col_a}__{col_b}"

    # Chi-square: show contingency table → Continue / Skip gate
    if method == "chi-square":
        st.markdown("**Contingency Table**")
        st.dataframe(pd.crosstab(sa, sb), use_container_width=True)

        decision = st.session_state.get(f"decision_{pair_key}")
        if decision is None:
            c1, c2, _ = st.columns([1, 1, 5])
            if c1.button("Continue", key=f"cont_{pair_key}"):
                st.session_state[f"decision_{pair_key}"] = "continue"
                st.rerun()
            if c2.button("Skip", key=f"skip_{pair_key}"):
                st.session_state[f"decision_{pair_key}"] = "skip"
                st.rerun()
            continue

        if decision == "skip":
            st.info("Skipped.")
            continue

    # ── Run the test ──────────────────────────────────────────────────────
    if method == "pearson":
        result = run_pearson(sa, sb)
    elif method == "spearman":
        result = run_spearman(sa, sb)
    else:
        result = run_chi_square(sa, sb)

    is_chi = method == "chi-square"
    strength = strength_label(result["coefficient"], is_cramers=is_chi)

    # ── Stat card ─────────────────────────────────────────────────────────
    if is_chi:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("χ²", result["chi2"])
        c2.metric("Cramér's V", result["coefficient"])
        c3.metric("p-value", result["p_value"])
        c4.metric("DoF", result["dof"])
        c5.metric("N", result["n"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Coefficient ({result['label']})", result["coefficient"])
        c2.metric("p-value", result["p_value"])
        c3.metric("N", result["n"])

    # Strength indicator
    if strength == "Strong":
        st.success(f"Association Strength: {strength}")
    elif strength == "Moderate":
        st.warning(f"Association Strength: {strength}")
    else:
        st.error(f"Association Strength: {strength}")

    # Interpretation
    st.info(interpret(result, col_a, col_b))

    # ── Scatterplot (non-nominal pairs only) ──────────────────────────────
    if method != "chi-square":
        want_scatter = st.radio(
            "Want scatterplot?", ["No", "Yes"],
            horizontal=True, key=f"scatter_{pair_key}"
        )
        if want_scatter == "Yes":
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#f8f8f8")
            sns.scatterplot(x=sa, y=sb, ax=ax, color="#FF69B4", s=60, edgecolor="white")
            ax.set_xlabel(col_a)
            ax.set_ylabel(col_b)
            ax.set_title(f"{col_a} vs {col_b}")
            ax.set_facecolor("#f0f0f0")
            ax.grid(color="white", linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

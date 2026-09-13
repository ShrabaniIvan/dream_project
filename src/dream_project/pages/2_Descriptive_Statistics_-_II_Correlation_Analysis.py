"""Correlation Analysis page — Pearson / Spearman / Chi-square."""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from dream_project.modules.theme import apply_sidebar_style
from dream_project.modules.descriptive_stats import load_data
from dream_project.modules.correlation import (
    VARIABLE_TYPES, determine_method, detect_corrupt_pair,
    validate_bins, prepare_series,
    run_pearson, run_spearman, run_chi_square, run_point_biserial, run_anova,
    strength_label, interpret, build_correlation_prompt,
)
from dream_project.modules.ai_narrative import generate_narrative

st.set_page_config(page_title="Descriptive Statistics - II: Correlation Analysis", layout="wide")
apply_sidebar_style()
st.title("Descriptive Statistics - II: Correlation Analysis")

with st.expander("How to use this page — variable types and test selection", expanded=False):
    st.markdown("""
**Variable types**

| Type | Definition | Examples |
|---|---|---|
| **Numerical (Continuous)** | Measured on a continuous scale; any value within a range is possible | Rainfall (mm), Yield (kg/ha), Temperature |
| **Ordinal** | Categorical with a meaningful order, but gaps between levels are not equal | Soil Quality (Low / Medium / High), Likert scales |
| **Nominal** | Categorical with no intrinsic order; labels only | Irrigation Type, Disease Present (Yes/No), Crop Variety |

---

**Which test does the dashboard apply?**

| Variable A | Variable B | Test used |
|---|---|---|
| Continuous | Continuous | **Pearson r** — linear association; use Spearman when the relationship is monotonic but not linear |
| Ordinal | Continuous or Ordinal | **Spearman ρ** — rank-based, monotonic association |
| Nominal | Nominal or Ordinal | **Chi-square / Cramér's V** — tests independence of category counts |
| Binary Nominal (2 groups) | Continuous | **Point-biserial r** — correlation between a dichotomy and a continuous variable |
| Nominal (3 + groups) | Continuous | **ANOVA + η²** — tests whether group means differ; η² measures effect size |

> For **Nominal × Continuous** pairs the key question is how many unique categories the nominal variable has.
> Two categories → point-biserial correlation. Three or more → one-way ANOVA.

---

**Effect-size benchmarks**

| Measure | Weak | Moderate | Strong |
|---|---|---|---|
| Pearson / Spearman / Point-biserial r | abs(r) < 0.30 | 0.30 – 0.69 | ≥ 0.70 |
| Cramér's V | V < 0.10 | 0.10 – 0.29 | ≥ 0.30 |
| η² (ANOVA) | η² < 0.06 | 0.06 – 0.13 | ≥ 0.14 |
""")

# ── 1. File Upload ─────────────────────────────────────────────────────────

st.caption("Upload your dataset. Both CSV and Excel formats are supported.")
uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

df = load_data(uploaded)
st.success(f"Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

cols = list(df.columns)

# ── 1b. All-Numeric Correlation Matrix (Heatmap) ───────────────────────────

st.divider()
st.subheader("Correlation Matrix (All Numeric Variables)")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) < 2:
    st.info("Need at least 2 numeric columns to compute a correlation matrix.")
else:
    st.caption(
        "Pearson r across every numeric column. Values run −1 to +1 — "
        "red = positive, blue = negative. Strong: |r| ≥ 0.70 · Moderate: 0.30–0.69 · Weak: < 0.30."
    )
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) - 1)))
    fig.patch.set_facecolor("#f8f8f8")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix", fontsize=13)
    ax.set_facecolor("#f0f0f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── 2. Column Selection ────────────────────────────────────────────────────

st.divider()
st.caption("Select the primary variable (A) and one or more comparison variables (B). The dashboard tests each A × B pair independently — the method is chosen automatically based on the variable types you declare below.")
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
st.caption("Declare the type of each variable. This determines which statistical test is applied — see the guide at the top of the page. For Ordinal variables, provide the rank order from lowest to highest.")

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
st.caption("Rows with missing or blank values in either column of a pair are automatically excluded. Check the counts here before running the analysis.")
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
st.caption("Click **Run Analysis** to compute all selected pairs. Chi-square pairs show a contingency table first — you can review it and then choose to continue or skip. All other test types run immediately.")
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
    n_cats_a = df[col_a].nunique() if type_a == "Nominal" else None
    n_cats_b = df[col_b].nunique() if type_b == "Nominal" else None
    method = determine_method(type_a, type_b, n_cats_a, n_cats_b)
    METHOD_LABELS = {
        "chi-square": "Chi-square / Cramér's V",
        "pearson": "Pearson r",
        "spearman": "Spearman ρ",
        "point-biserial": "Point-biserial r",
        "anova": "ANOVA + η²",
    }
    method_label = METHOD_LABELS.get(method, method.title())
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
    elif method == "point-biserial":
        nom_s, cont_s = (sa, sb) if type_a == "Nominal" else (sb, sa)
        result = run_point_biserial(nom_s, cont_s)
    elif method == "anova":
        nom_s, cont_s = (sa, sb) if type_a == "Nominal" else (sb, sa)
        result = run_anova(nom_s, cont_s)
    else:
        result = run_chi_square(sa, sb)

    is_chi = method == "chi-square"
    is_anova = method == "anova"
    strength = strength_label(result["coefficient"], is_cramers=is_chi, is_eta2=is_anova)

    # ── Stat card ─────────────────────────────────────────────────────────
    if is_chi:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("χ²", result["chi2"])
        c2.metric("Cramér's V", result["coefficient"])
        c3.metric("p-value", result["p_value"])
        c4.metric("DoF", result["dof"])
        c5.metric("N", result["n"])
    elif is_anova:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F-statistic", result["f_stat"])
        c2.metric("η² (eta-squared)", result["coefficient"])
        c3.metric("p-value", result["p_value"])
        c4.metric("N", result["n"])
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

    # ── AI Narrative ──────────────────────────────────────────────────────
    narrative_key = f"narrative_{pair_key}"
    if st.button("Generate AI Explanation", key=f"ai_{pair_key}"):
        max_tokens = int(os.environ.get("dp_OPENAI_MAX_TOKENS", 500))
        prompt = build_correlation_prompt(result, col_a, type_a, col_b, type_b, max_tokens)
        with st.spinner("Generating explanation..."):
            st.session_state[narrative_key] = generate_narrative(prompt, model="gpt-4.1-mini")
    if st.session_state.get(narrative_key):
        st.markdown(st.session_state[narrative_key])

    # ── Chart ────────────────────────────────────────────────────────────
    if method == "chi-square":
        want_chart = st.radio(
            "Want heatmap?", ["No", "Yes"],
            horizontal=True, key=f"heatmap_{pair_key}"
        )
        if want_chart == "Yes":
            ct = result["contingency_table"]
            ct_norm = ct.div(ct.sum(axis=1), axis=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#f8f8f8")
            sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                        linewidths=0.5, cbar_kws={"label": "Row proportion"})
            ax.set_title(f"{col_a} vs {col_b} (row proportions)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    elif method in ("pearson", "spearman"):
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
    else:  # point-biserial or anova — boxplot
        want_box = st.radio(
            "Want boxplot?", ["No", "Yes"],
            horizontal=True, key=f"box_{pair_key}"
        )
        if want_box == "Yes":
            nom_col = col_a if type_a == "Nominal" else col_b
            cont_col = col_b if type_a == "Nominal" else col_a
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#f8f8f8")
            sns.boxplot(data=clean_df, x=nom_col, y=cont_col, ax=ax, palette="YlOrRd")
            ax.set_title(f"{cont_col} by {nom_col}")
            ax.set_facecolor("#f0f0f0")
            ax.grid(color="white", linewidth=0.8, axis="y")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

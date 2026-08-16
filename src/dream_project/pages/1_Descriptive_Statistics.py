"""Descriptive Statistics page — user-uploaded dataset."""
import os
import io
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from dotenv import load_dotenv

from dream_project.modules.theme import apply_sidebar_style
from dream_project.modules.descriptive_stats import (
    load_data, detect_corrupt, compute_stats, to_download_bytes, build_narrative_prompt,
)
from dream_project.modules.ai_narrative import generate_narrative

load_dotenv()

st.set_page_config(page_title="Descriptive Statistics", layout="wide")
apply_sidebar_style()
st.title("Descriptive Statistics")

with st.expander("How to use this page — statistics and charts explained", expanded=False):
    st.markdown("""
**What this module does**

Computes summary statistics for a single numeric column in your uploaded dataset and
visualises its distribution. Use it to check data quality and understand the shape of
a variable before running association tests.

---

**Statistics computed**

| Statistic | What it tells you |
|---|---|
| **Count** | Number of valid (non-missing) observations used in the analysis |
| **Mean** | Arithmetic average — sensitive to outliers |
| **Std Dev** | Standard deviation — average spread around the mean |
| **Variance** | Std Dev squared — useful for comparing variability across variables |
| **Min / Max** | Smallest and largest values in the dataset |
| **25th / 50th / 75th Percentile** | Values below which 25%, 50%, and 75% of observations fall; the 50th percentile is the median |
| **Skewness** | Symmetry of the distribution. Near 0 = symmetric. Positive = right tail (high outliers). Negative = left tail (low outliers). Values beyond ±1 indicate notable skew |
| **Kurtosis** | Tail heaviness relative to a normal distribution. 0 = normal-like. Positive (leptokurtic) = heavier tails / more outliers. Negative (platykurtic) = lighter tails |
| **CV%** | Coefficient of Variation — Std Dev as a percentage of the Mean. Useful for comparing variability between variables measured in different units |

---

**Charts**

| Chart | Best used for |
|---|---|
| **Histogram** | Seeing the overall frequency distribution and spotting gaps or clusters |
| **KDE Plot** | A smoothed version of the histogram — better for comparing shapes across groups |
| **Boxplot** | Quickly spotting the median, IQR, and outliers (points beyond the whiskers) |
| **Violin Plot** | Like a boxplot but also shows the full distribution shape — useful when data is multimodal |
| **QQ-Plot** | Assessing normality — points close to the diagonal line suggest a normal distribution; S-curves or heavy departures suggest non-normality |

---

**Interpreting skewness and kurtosis for agricultural data**

Many agricultural variables (yield, rainfall, pest counts) are right-skewed.
A strong positive skew (> 1) often indicates a few unusually high observations — worth checking whether these are real measurements or data entry errors.
High kurtosis alongside high skew can signal that parametric tests (which assume normality) may give unreliable results on this variable.
""")


CHART_OPTIONS = ["Histogram", "Boxplot", "Violin Plot", "KDE Plot", "QQ-Plot"]

# ── chart styling helpers ────────────────────────────────────────────────────

def _apply_style(ax, fontsize: int):
    ax.set_facecolor("#f0f0f0")
    ax.grid(color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)
    ax.tick_params(labelsize=fontsize, colors="black")
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    if ax.get_title():
        ax.title.set_size(fontsize + 1)


def _render_chart(chart_type: str, series, col_name: str, fontsize: int):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#f8f8f8")

    if chart_type == "Histogram":
        sns.histplot(series, kde=False, ax=ax, color="#4C8BE2", edgecolor="white")
        ax.set_xlabel(col_name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram — {col_name}")

    elif chart_type == "Boxplot":
        sns.boxplot(y=series, ax=ax, color="#4C8BE2", width=0.4)
        ax.set_ylabel(col_name)
        ax.set_title(f"Boxplot — {col_name}")

    elif chart_type == "Violin Plot":
        sns.violinplot(y=series, ax=ax, color="#4C8BE2")
        ax.set_ylabel(col_name)
        ax.set_title(f"Violin Plot — {col_name}")

    elif chart_type == "KDE Plot":
        sns.kdeplot(series, ax=ax, color="#4C8BE2", fill=True, alpha=0.4)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Density")
        ax.set_title(f"KDE Plot — {col_name}")

    elif chart_type == "QQ-Plot":
        fig_qq = sm.qqplot(series, line="s", alpha=0.6, fit=True)
        plt.close(fig)
        fig = fig_qq
        ax = fig.axes[0]
        ax.set_title(f"QQ-Plot — {col_name}")
        ax.set_facecolor("#f0f0f0")
        ax.grid(color="white", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        ax.tick_params(labelsize=fontsize, colors="black")
        st.pyplot(fig)
        plt.close(fig)
        return

    _apply_style(ax, fontsize)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 1. File upload ────────────────────────────────────────────────────────────

st.caption("Upload your dataset as a CSV or Excel file. Each column will be analyzed one at a time.")
uploaded = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

df = load_data(uploaded)
st.success(f"Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Column selection ───────────────────────────────────────────────────────

st.caption("Select the numeric column to analyze. Non-numeric columns will not produce meaningful statistics.")
col_name = st.selectbox("Choose Column", ["— select —"] + list(df.columns))
if col_name == "— select —":
    st.stop()

# Corrupt data check (automatic on column select)
clean_series, corrupt_df = detect_corrupt(df, col_name)

st.divider()
if corrupt_df.empty:
    st.success("Data is clean — no missing or corrupt values found.")
else:
    st.warning(f"{len(corrupt_df)} corrupt / missing row(s) found and excluded from analysis.")
    with st.expander("Preview corrupt rows"):
        st.dataframe(corrupt_df, use_container_width=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    filename = c1.text_input("Download filename", value="corrupt_rows")
    fmt = c2.selectbox("Format", ["csv", "xlsx"])
    data_bytes, mime = to_download_bytes(corrupt_df, fmt)
    c3.markdown("&nbsp;", unsafe_allow_html=True)  # vertical align
    c3.download_button(
        label="Download corrupt rows",
        data=data_bytes,
        file_name=f"{filename}.{fmt}",
        mime=mime,
    )

st.divider()

# ── 3. Run Analysis (gated) ───────────────────────────────────────────────────

st.caption("Click **Run Analysis** to compute statistics. Re-run any time you change the column selection.")
if st.button("Run Analysis", type="primary"):
    st.session_state["analysis_ready"] = True
    st.session_state["analysis_col"] = col_name
    st.session_state["clean_series"] = clean_series

if not st.session_state.get("analysis_ready") or st.session_state.get("analysis_col") != col_name:
    st.stop()

series = st.session_state["clean_series"]

# ── 4. Descriptive statistics table ──────────────────────────────────────────

st.subheader(f"Descriptive Statistics — {col_name}")
stats_df = compute_stats(series, col_name)
st.dataframe(stats_df, use_container_width=True)

st.divider()

# ── 5. Visualization ──────────────────────────────────────────────────────────

st.subheader("Visualization")
st.caption("Choose one or more chart types. Use a **Histogram** or **KDE Plot** to see the distribution shape; a **Boxplot** or **Violin Plot** to spot outliers and spread; a **QQ-Plot** to assess normality.")
selected_charts = st.multiselect("Select chart type(s)", CHART_OPTIONS, default=["Histogram"])
fontsize = st.slider("Label / tick font size", min_value=8, max_value=20, value=11)

if selected_charts:
    cols = st.columns(min(len(selected_charts), 2))
    for i, chart_type in enumerate(selected_charts):
        with cols[i % 2]:
            _render_chart(chart_type, series, col_name, fontsize)

st.divider()

# ── 6. AI Narrative (premium) ─────────────────────────────────────────────────

st.subheader("AI Narrative Interpretation")
st.caption("Generate a plain-language summary of the statistics above. Requires an OpenAI API key configured in the environment.")
if st.button("Generate AI Narrative"):
    max_tokens = int(os.environ.get("dp_OPENAI_MAX_TOKENS", 1000))
    with st.spinner("Generating interpretation..."):
        prompt = build_narrative_prompt(stats_df, col_name, max_tokens)
        narrative = generate_narrative(prompt)
    st.markdown(narrative)

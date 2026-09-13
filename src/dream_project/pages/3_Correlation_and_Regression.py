"""Regression page — wheat dataset, numerical columns only."""
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dream_project.modules.theme import apply_sidebar_style
from dream_project.modules.regression import prepare_xy, run_polynomial_regression, run_xgboost_regression

st.set_page_config(page_title="Regression", layout="wide")
apply_sidebar_style()
st.title("Regression")

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

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to run a regression.")
    st.stop()

# ── 3. Variable selection ────────────────────────────────────────────────────

st.divider()
st.subheader("Select Variables")
st.caption(
    "Choose the target (dependent) variable and one or more predictors (independent variables). "
    "Model type and hyperparameters are selected automatically via cross-validation — no tuning needed."
)

y_col = st.selectbox("Dependent variable (target)", ["— select —"] + numeric_cols)
if y_col == "— select —":
    st.stop()

x_cols = st.multiselect("Independent variable(s)", [c for c in numeric_cols if c != y_col])
if not x_cols:
    st.stop()

X, y = prepare_xy(df, x_cols, y_col)
if len(X) < 20:
    st.error(f"Only {len(X)} valid rows after removing missing values — need at least 20 to fit a regression.")
    st.stop()

st.info(f"**{len(X)} valid rows** used for training/testing (80/20 split).")

# ── 4. Run regression ─────────────────────────────────────────────────────────

st.divider()
if st.button("Run Regression", type="primary"):
    st.session_state["reg_ready"] = True

if not st.session_state.get("reg_ready"):
    st.stop()

tab_poly, tab_xgb = st.tabs(["Polynomial + Regularization", "XGBoost"])

with tab_poly:
    with st.spinner("Fitting polynomial + ElasticNet (auto-selecting degree and alpha)..."):
        poly_result = run_polynomial_regression(X, y)

    st.caption(
        f"Auto-selected: degree **{poly_result['degree']}**, alpha **{poly_result['alpha']}**, "
        f"l1_ratio **{poly_result['l1_ratio']}** (0 = Ridge, 1 = Lasso)"
    )
    if poly_result["skipped_degrees"]:
        st.caption(
            f"Degree(s) {', '.join(map(str, poly_result['skipped_degrees']))} were skipped — "
            "too few training rows for that many polynomial features."
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", poly_result["r2"])
    c2.metric("RMSE", poly_result["rmse"])
    c3.metric("Test rows", poly_result["n_test"])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.scatter(poly_result["y_test"], poly_result["y_pred"], color="#FF69B4", edgecolor="white", s=60)
    lims = [min(poly_result["y_test"].min(), poly_result["y_pred"].min()),
            max(poly_result["y_test"].max(), poly_result["y_pred"].max())]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Actual {y_col}")
    ax.set_ylabel(f"Predicted {y_col}")
    ax.set_title("Predicted vs Actual")
    ax.set_facecolor("#f0f0f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_xgb:
    if len(X) < 100:
        st.warning(
            f"Only {len(X)} rows available. Tree-based models like XGBoost need many more rows "
            "to learn a smooth, reliable trend — with this little data the fitted curve can look "
            "jagged or overfit. Prefer the **Polynomial + Regularization** tab for small datasets."
        )

    with st.spinner("Fitting XGBoost (auto-selecting hyperparameters via grid search)..."):
        xgb_result = run_xgboost_regression(X, y)

    st.caption(f"Auto-selected parameters: {xgb_result['best_params']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", xgb_result["r2"])
    c2.metric("RMSE", xgb_result["rmse"])
    c3.metric("Test rows", xgb_result["n_test"])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.scatter(xgb_result["y_test"], xgb_result["y_pred"], color="#FF69B4", edgecolor="white", s=60)
    lims = [min(xgb_result["y_test"].min(), xgb_result["y_pred"].min()),
            max(xgb_result["y_test"].max(), xgb_result["y_pred"].max())]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Actual {y_col}")
    ax.set_ylabel(f"Predicted {y_col}")
    ax.set_title("Predicted vs Actual")
    ax.set_facecolor("#f0f0f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Feature importance**")
    importance_df = pd.DataFrame(
        sorted(xgb_result["feature_importance"].items(), key=lambda kv: -kv[1]),
        columns=["Feature", "Importance"],
    )
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

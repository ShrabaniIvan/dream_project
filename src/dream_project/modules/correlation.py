"""Correlation analysis — Pearson, Spearman, Chi-square / Cramér's V."""
import ast
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

VARIABLE_TYPES = ["Continuous", "Ordinal", "Nominal"]


def determine_method(type_a: str, type_b: str) -> str:
    """Return analysis method based on variable types."""
    if "Nominal" in (type_a, type_b):
        return "chi-square"
    if "Ordinal" in (type_a, type_b):
        return "spearman"
    return "pearson"


def detect_corrupt_pair(df: pd.DataFrame, col_a: str, col_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, corrupt_df) for a column pair. Corrupt = null or blank."""
    work = df[[col_a, col_b]].copy()
    corrupt_mask = work.isnull().any(axis=1)
    corrupt_mask |= work.apply(lambda col: col.astype(str).str.strip() == "").any(axis=1)
    clean_df = work[~corrupt_mask].reset_index(drop=True)
    corrupt_df = df[corrupt_mask].copy()
    return clean_df, corrupt_df


def validate_ordinal_order(series: pd.Series, order_str: str) -> tuple[bool, str]:
    """Check all unique values in series are covered by user-provided order string."""
    order = [v.strip() for v in order_str.split(",")]
    unique_vals = set(series.dropna().astype(str).unique())
    order_set = set(order)
    missing = unique_vals - order_set
    extra = order_set - unique_vals
    if missing:
        return False, f"Values in data not in order list: {missing}"
    if extra:
        return False, f"Values in order list not found in data: {extra}"
    return True, ""


def encode_ordinal(series: pd.Series, order_str: str) -> pd.Series:
    """Map ordinal text values to integer ranks (1 = lowest)."""
    order = [v.strip() for v in order_str.split(",")]
    rank_map = {v: i + 1 for i, v in enumerate(order)}
    return series.astype(str).map(rank_map)


def validate_bins(bins_str: str) -> tuple[bool, str, list]:
    """Parse and validate user-provided bin boundary string."""
    try:
        bins = ast.literal_eval(bins_str)
    except Exception:
        return False, "Could not parse — enter a valid Python list e.g. [0, 10, 20, 30]", []
    if not isinstance(bins, list) or len(bins) < 2:
        return False, "Must be a list with at least 2 values.", []
    if not all(isinstance(b, (int, float)) for b in bins):
        return False, "All bin boundaries must be numbers.", []
    if bins != sorted(bins):
        return False, "Bin boundaries must be in ascending order.", []
    return True, "", bins


def bin_continuous(series: pd.Series, bins: list) -> pd.Series:
    """Bin a continuous series into nominal categories."""
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins).astype(str)


def prepare_series(series: pd.Series, vtype: str, order_str: str = "", bins: list = None) -> tuple[bool, str, pd.Series]:
    """
    Transform a series according to its declared type.
    Returns (ok, error_message, transformed_series).
    """
    if vtype == "Continuous":
        out = pd.to_numeric(series, errors="coerce")
        return True, "", out

    if vtype == "Ordinal":
        if not order_str:
            return False, "Rank order not provided.", series
        ok, msg = validate_ordinal_order(series, order_str)
        if not ok:
            return False, msg, series
        return True, "", encode_ordinal(series, order_str)

    # Nominal
    if bins:
        return True, "", bin_continuous(series, bins)
    return True, "", series.astype(str)


def run_pearson(a: pd.Series, b: pd.Series) -> dict:
    r, p = stats.pearsonr(a.astype(float), b.astype(float))
    return {"method": "Pearson", "label": "r", "coefficient": round(r, 4), "p_value": round(p, 4), "n": len(a)}


def run_spearman(a: pd.Series, b: pd.Series) -> dict:
    rho, p = stats.spearmanr(a.astype(float), b.astype(float))
    return {"method": "Spearman", "label": "ρ", "coefficient": round(rho, 4), "p_value": round(p, 4), "n": len(a)}


def run_chi_square(a: pd.Series, b: pd.Series) -> dict:
    ct = pd.crosstab(a, b)
    chi2, p, dof, _ = chi2_contingency(ct)
    n = int(ct.values.sum())
    min_dim = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
    return {
        "method": "Chi-square / Cramér's V", "label": "V",
        "chi2": round(chi2, 4), "coefficient": round(v, 4),
        "p_value": round(p, 4), "dof": int(dof), "n": n,
        "contingency_table": ct,
    }


def strength_label(coef: float, is_cramers: bool = False) -> str:
    v = abs(coef)
    if is_cramers:
        if v >= 0.3: return "Strong"
        if v >= 0.1: return "Moderate"
        return "Weak"
    if v >= 0.7: return "Strong"
    if v >= 0.3: return "Moderate"
    return "Weak"


def build_correlation_prompt(result: dict, col_a: str, type_a: str, col_b: str, type_b: str, max_tokens: int) -> str:
    """Build the GPT prompt for a single correlation pair."""
    method = result["method"]
    is_chi = "Chi" in method

    stats_lines = [
        f"  - Method: {method}",
        f"  - {result['label']}: {result['coefficient']}",
        f"  - p-value: {result['p_value']}",
        f"  - N: {result['n']}",
    ]
    if is_chi:
        stats_lines.insert(2, f"  - χ²: {result['chi2']}")
        stats_lines.insert(3, f"  - Degrees of freedom: {result['dof']}")
        ct_str = f"\nContingency table:\n{result['contingency_table'].to_string()}"
    else:
        ct_str = ""

    return (
        f"The following correlation analysis was performed on an agricultural dataset.\n\n"
        f"Variable A: {col_a} (type: {type_a})\n"
        f"Variable B: {col_b} (type: {type_b})\n\n"
        f"Results:\n" + "\n".join(stats_lines) + ct_str + "\n\n"
        f"Provide an interpretation as a numbered list. Cover:\n"
        f"1. Whether the association is statistically significant and what the p-value implies\n"
        f"2. The strength and direction (if applicable) of the association\n"
        f"3. What this means in practical/agricultural terms\n"
        f"4. Any caveats or limitations the analyst should be aware of\n\n"
        f"Be concise — your response must fit within {max_tokens} tokens total."
    )


def interpret(result: dict, col_a: str, col_b: str) -> str:
    """Return a pre-determined rule-based interpretation string."""
    p, coef, method = result["p_value"], result["coefficient"], result["method"]
    is_chi = "Chi" in method

    if p >= 0.05:
        return (
            f"No statistically significant association between **{col_a}** and **{col_b}** "
            f"(p = {p}). The result may be due to chance."
        )

    strength = strength_label(coef, is_cramers=is_chi)

    if is_chi:
        return (
            f"**{strength} association** between **{col_a}** and **{col_b}** "
            f"(Cramér's V = {coef}, p = {p}). The categories are not independent."
        )

    direction = "positive" if coef > 0 else "negative"
    if strength == "Strong":
        verb = "increases substantially" if direction == "positive" else "decreases substantially"
        return f"**Strong {direction} correlation** — as **{col_a}** increases, **{col_b}** tends to {verb}."
    if strength == "Moderate":
        together = "increase together" if direction == "positive" else "move in opposite directions"
        return f"**Moderate {direction} correlation** — a moderate tendency for **{col_a}** and **{col_b}** to {together}."
    return f"**Weak {direction} correlation** — little reliable association between **{col_a}** and **{col_b}**."

"""Correlation analysis — Pearson, Spearman, Chi-square / Cramér's V."""
import ast
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

VARIABLE_TYPES = ["Continuous", "Ordinal", "Nominal"]


def determine_method(type_a: str, type_b: str, n_cats_a: int = None, n_cats_b: int = None) -> str:
    """Return analysis method based on variable types.

    Nominal × Nominal/Ordinal  → chi-square
    Binary Nominal × Continuous → point-biserial
    Nominal (3+) × Continuous  → anova
    Ordinal × anything else    → spearman
    Continuous × Continuous    → pearson
    """
    nom_a, nom_b = type_a == "Nominal", type_b == "Nominal"
    cont_a, cont_b = type_a == "Continuous", type_b == "Continuous"

    if nom_a and nom_b:
        return "chi-square"
    if nom_a and type_b == "Ordinal":
        return "chi-square"
    if type_a == "Ordinal" and nom_b:
        return "chi-square"
    if nom_a and cont_b:
        return "point-biserial" if (n_cats_a or 0) == 2 else "anova"
    if cont_a and nom_b:
        return "point-biserial" if (n_cats_b or 0) == 2 else "anova"
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


def run_point_biserial(nominal: pd.Series, continuous: pd.Series) -> dict:
    """Point-biserial correlation for binary nominal × continuous."""
    cats = sorted(nominal.astype(str).unique())
    binary = (nominal.astype(str) == cats[1]).astype(int)
    r, p = stats.pointbiserialr(binary, continuous.astype(float))
    return {
        "method": "Point-biserial",
        "label": "r_pb",
        "coefficient": round(float(r), 4),
        "p_value": round(float(p), 4),
        "n": len(nominal),
        "categories": cats,  # index 0 coded as 0, index 1 coded as 1
    }


def run_anova(nominal: pd.Series, continuous: pd.Series) -> dict:
    """One-way ANOVA + η² for nominal (3+ groups) × continuous."""
    cont = continuous.astype(float)
    group_labels = nominal.astype(str).unique()
    groups = [cont[nominal.astype(str) == g] for g in group_labels]
    f, p = stats.f_oneway(*groups)
    grand_mean = cont.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((cont - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return {
        "method": "ANOVA",
        "label": "η²",
        "coefficient": round(float(eta2), 4),
        "p_value": round(float(p), 4),
        "n": len(nominal),
        "f_stat": round(float(f), 4),
        "groups": int(nominal.nunique()),
    }


def strength_label(coef: float, is_cramers: bool = False, is_eta2: bool = False) -> str:
    v = abs(coef)
    if is_cramers:
        if v >= 0.3: return "Strong"
        if v >= 0.1: return "Moderate"
        return "Weak"
    if is_eta2:
        if v >= 0.14: return "Strong"
        if v >= 0.06: return "Moderate"
        return "Weak"
    if v >= 0.7: return "Strong"
    if v >= 0.3: return "Moderate"
    return "Weak"


def build_correlation_prompt(result: dict, col_a: str, type_a: str, col_b: str, type_b: str, max_tokens: int) -> str:
    """Build the GPT prompt for a single correlation pair."""
    method = result["method"]
    is_chi = "Chi" in method
    is_anova = method == "ANOVA"

    stats_lines = [
        f"  - Method: {method}",
        f"  - {result['label']}: {result['coefficient']}",
        f"  - p-value: {result['p_value']}",
        f"  - N: {result['n']}",
    ]
    ct_str = ""
    if is_chi:
        stats_lines.insert(2, f"  - χ²: {result['chi2']}")
        stats_lines.insert(3, f"  - Degrees of freedom: {result['dof']}")
        ct_str = f"\nContingency table:\n{result['contingency_table'].to_string()}"
    elif is_anova:
        stats_lines.insert(2, f"  - F-statistic: {result['f_stat']}")
        stats_lines.insert(3, f"  - Number of groups: {result['groups']}")

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
    """Return a rule-based interpretation string."""
    p, coef, method = result["p_value"], result["coefficient"], result["method"]
    is_chi = "Chi" in method
    is_anova = method == "ANOVA"
    is_pb = method == "Point-biserial"

    if p >= 0.05:
        return (
            f"No statistically significant association between **{col_a}** and **{col_b}** "
            f"(p = {p}). The result may be due to chance."
        )

    if is_chi:
        strength = strength_label(coef, is_cramers=True)
        return (
            f"**{strength} association** between **{col_a}** and **{col_b}** "
            f"(Cramér's V = {coef}, p = {p}). The categories are not independent."
        )

    if is_anova:
        strength = strength_label(coef, is_eta2=True)
        return (
            f"**{strength} group difference** — one-way ANOVA is significant "
            f"(F = {result['f_stat']}, p = {p}, η² = {coef}). "
            f"**{col_a}** group membership explains {coef * 100:.1f}% of variance in **{col_b}**."
        )

    strength = strength_label(coef)
    direction = "positive" if coef > 0 else "negative"

    if is_pb:
        cats = result.get("categories", ["0", "1"])
        return (
            f"**{strength} {direction} point-biserial correlation** (r_pb = {coef}, p = {p}). "
            f"Higher values of **{col_b}** tend to be associated with the '{cats[1]}' group of **{col_a}**."
            if direction == "positive" else
            f"**{strength} {direction} point-biserial correlation** (r_pb = {coef}, p = {p}). "
            f"Higher values of **{col_b}** tend to be associated with the '{cats[0]}' group of **{col_a}**."
        )

    if strength == "Strong":
        verb = "increases substantially" if direction == "positive" else "decreases substantially"
        return f"**Strong {direction} correlation** — as **{col_a}** increases, **{col_b}** tends to {verb}."
    if strength == "Moderate":
        together = "increase together" if direction == "positive" else "move in opposite directions"
        return f"**Moderate {direction} correlation** — a moderate tendency for **{col_a}** and **{col_b}** to {together}."
    return f"**Weak {direction} correlation** — little reliable association between **{col_a}** and **{col_b}**."

"""Descriptive statistics computation using pandas and statsmodels."""
import io
import pandas as pd
import statsmodels.stats.descriptivestats as smds


def load_data(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def detect_corrupt(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Split column into a clean numeric series and a DataFrame of corrupt rows.
    Corrupt = blank, non-numeric, or unparseable values.
    """
    raw = df[col].copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    corrupt_mask = numeric.isna()
    clean_series = numeric[~corrupt_mask].reset_index(drop=True)
    corrupt_df = df[corrupt_mask].copy()
    corrupt_df["_corrupt_value"] = raw[corrupt_mask].values
    return clean_series, corrupt_df


def compute_stats(series: pd.Series, col_name: str) -> pd.DataFrame:
    """Compute descriptive statistics for a clean numeric series."""
    ds = smds.describe(series.rename(col_name))
    desc = series.describe()

    rows = {
        "Count": int(ds.loc["nobs", col_name]),
        "Mean": round(ds.loc["mean", col_name], 4),
        "Std Dev": round(ds.loc["std", col_name], 4),
        "Variance": round(ds.loc["std", col_name] ** 2, 4),
        "Min": round(ds.loc["min", col_name], 4),
        "25th Percentile": round(desc["25%"], 4),
        "Median (50th)": round(ds.loc["median", col_name], 4),
        "75th Percentile": round(desc["75%"], 4),
        "Max": round(ds.loc["max", col_name], 4),
        "Skewness": round(ds.loc["skew", col_name], 4),
        "Kurtosis": round(ds.loc["kurtosis", col_name], 4),
        "CV%": round(ds.loc["coef_var", col_name] * 100, 2),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])


def to_download_bytes(df: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    """Serialize a DataFrame to bytes for download. Returns (bytes, mime_type)."""
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=True)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return df.to_csv(index=True).encode(), "text/csv"


def build_narrative_prompt(stats_df: pd.DataFrame, col_name: str, max_tokens: int) -> str:
    """Build the OpenAI prompt for the AI narrative."""
    return (
        f"The following descriptive statistics are for the variable '{col_name}':\n\n"
        f"{stats_df.to_string()}\n\n"
        f"Provide an interpretation as a numbered list. "
        f"Cover: central tendency, spread/variability, skewness, kurtosis, and any notable observations. "
        f"Be concise — your response must fit within {max_tokens} tokens total."
    )

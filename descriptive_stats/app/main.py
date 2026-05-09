"""Descriptive Statistics API service."""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any

app = FastAPI(title="Descriptive Statistics Service")

class DataInput(BaseModel):
    column: str
    data: list

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/descriptive-stats")
def get_descriptive_stats(input_data: DataInput) -> Dict[str, Any]:
    """Calculate descriptive statistics for a column."""
    data = np.array(input_data.data)

    return {
        "column": input_data.column,
        "count": len(data),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }

@app.post("/correlation-matrix")
def get_correlation_matrix(data: Dict[str, list]) -> Dict[str, Any]:
    """Calculate correlation matrix for dataset."""
    df = pd.DataFrame(data)
    corr_matrix = df.corr()

    return {
        "correlation": corr_matrix.to_dict(),
        "shape": corr_matrix.shape
    }

@app.post("/hypothesis-test")
def hypothesis_test(data1: list, data2: list, test_type: str = "ttest") -> Dict[str, Any]:
    """Perform hypothesis testing (t-test or Mann-Whitney U)."""
    from scipy import stats

    if test_type == "ttest":
        statistic, p_value = stats.ttest_ind(data1, data2)
        return {
            "test": "Independent t-test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0.05": float(p_value) < 0.05
        }
    elif test_type == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(data1, data2)
        return {
            "test": "Mann-Whitney U test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0.05": float(p_value) < 0.05
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

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

class CorrelationInput(BaseModel):
    data: Dict[str, list]

class HypothesisTestInput(BaseModel):
    data1: list
    data2: list
    test_type: str = "ttest"

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
def get_correlation_matrix(input_data: CorrelationInput) -> Dict[str, Any]:
    """Calculate correlation matrix for dataset."""
    df = pd.DataFrame(input_data.data)
    corr_matrix = df.corr()

    return {
        "correlation": corr_matrix.to_dict(),
        "shape": corr_matrix.shape
    }

@app.post("/hypothesis-test")
def hypothesis_test(input_data: HypothesisTestInput) -> Dict[str, Any]:
    """Perform hypothesis testing (t-test or Mann-Whitney U)."""
    from scipy import stats

    if input_data.test_type == "ttest":
        statistic, p_value = stats.ttest_ind(input_data.data1, input_data.data2)
        return {
            "test": "Independent t-test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0.05": float(p_value) < 0.05
        }
    elif input_data.test_type == "mannwhitney":
        statistic, p_value = stats.mannwhitneyu(input_data.data1, input_data.data2)
        return {
            "test": "Mann-Whitney U test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0.05": float(p_value) < 0.05
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

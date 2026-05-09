"""Time Series Analysis API service."""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import statsmodels.api as sm

app = FastAPI(title="Time Series Service")

class TimeSeriesInput(BaseModel):
    values: List[float]
    name: str = "series"

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/acf-pacf")
def get_acf_pacf(input_data: TimeSeriesInput, nlags: int = 20) -> Dict[str, Any]:
    """Calculate ACF and PACF for time series."""
    from statsmodels.graphics.tsaplots import acf, pacf

    series = np.array(input_data.values)
    acf_vals = acf(series, nlags=nlags)
    pacf_vals = pacf(series, nlags=nlags)

    return {
        "series_name": input_data.name,
        "acf": acf_vals.tolist(),
        "pacf": pacf_vals.tolist(),
        "nlags": nlags
    }

@app.post("/seasonal-decompose")
def decompose_series(input_data: TimeSeriesInput, period: int = 12) -> Dict[str, Any]:
    """Decompose time series into trend, seasonal, residual."""
    series = pd.Series(input_data.values)
    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=period)

    return {
        "trend": decomposition.trend.tolist(),
        "seasonal": decomposition.seasonal.tolist(),
        "residual": decomposition.resid.tolist(),
        "observed": decomposition.observed.tolist()
    }

@app.post("/arima-fit")
def fit_arima(input_data: TimeSeriesInput, order: tuple = (1, 1, 1)) -> Dict[str, Any]:
    """Fit ARIMA model to time series."""
    series = pd.Series(input_data.values)

    try:
        model = sm.tsa.ARIMA(series, order=order)
        results = model.fit()

        return {
            "model": f"ARIMA{order}",
            "aic": float(results.aic),
            "bic": float(results.bic),
            "parameters": results.params.to_dict(),
            "summary": results.summary().as_text()
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/adf-test")
def adf_test(input_data: TimeSeriesInput) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    from statsmodels.tsa.stattools import adfuller

    series = np.array(input_data.values)
    result = adfuller(series)

    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "stationary_at_0.05": float(result[1]) < 0.05
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

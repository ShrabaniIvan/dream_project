"""Predictive Modeling API service."""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI(title="Predictive Modeling Service")

class PredictionInput(BaseModel):
    features: Dict[str, List[float]]
    target: List[float]

class PredictInput(BaseModel):
    features: Dict[str, List[float]]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/train-linear-regression")
def train_regression(input_data: PredictionInput) -> Dict[str, Any]:
    """Train linear regression model."""
    X = pd.DataFrame(input_data.features)
    y = np.array(input_data.target)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": "Linear Regression",
        "features": list(X.columns),
        "coefficients": {col: float(coef) for col, coef in zip(X.columns, model.coef_)},
        "intercept": float(model.intercept_),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

@app.post("/feature-importance")
def get_feature_importance(input_data: PredictionInput) -> Dict[str, Any]:
    """Calculate feature importance using correlation."""
    X = pd.DataFrame(input_data.features)
    y = np.array(input_data.target)

    # Calculate correlation with target
    correlation = {}
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        correlation[col] = {
            "correlation": float(corr) if not np.isnan(corr) else 0,
            "abs_correlation": float(abs(corr)) if not np.isnan(corr) else 0
        }

    return {
        "feature_importance": correlation,
        "n_features": len(X.columns),
        "target_samples": len(y)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

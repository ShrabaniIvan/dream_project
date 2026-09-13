"""Regression models — polynomial+regularization and XGBoost, both auto-tuned via cross-validation."""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


def prepare_xy(df: pd.DataFrame, x_cols: list, y_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with missing values in any selected column."""
    work = df[x_cols + [y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    return work[x_cols], work[y_col]


def run_polynomial_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit polynomial features + ElasticNet, auto-selecting degree and regularization via CV.

    l1_ratio=0 behaves as Ridge; ElasticNetCV can also drive alpha to ~0 (plain OLS)
    when that minimizes cross-validated error.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    best = None
    skipped_degrees = []
    for degree in (1, 2, 3):
        # Skip degrees whose polynomial expansion would out-number the training rows —
        # e.g. 9 real features at degree 3 becomes 219 columns from 40 rows, a model
        # regularization can only mask, never justify.
        n_poly_features = PolynomialFeatures(degree=degree, include_bias=False).fit(X_train).n_output_features_
        if n_poly_features >= len(X_train):
            skipped_degrees.append(degree)
            continue

        regressor = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.0, 0.5, 1.0],
                alphas=np.logspace(-3, 2, 20),
                cv=5,
                random_state=RANDOM_STATE,
                max_iter=10000,
            ),
        )
        # Standardizing y too keeps the coordinate-descent solver's convergence
        # tolerance meaningful regardless of the target's raw scale (e.g. kg/ha).
        pipe = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
        # Cross-validated score for degree selection — matches the rigor already used
        # to pick alpha/l1_ratio, instead of comparing training R² (which always favors
        # higher degree regardless of whether it actually generalizes).
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        if best is None or cv_r2 > best["cv_r2"]:
            best = {"degree": degree, "pipe": pipe, "cv_r2": cv_r2}

    if best is None:
        raise ValueError(
            f"Too many predictors ({X.shape[1]}) for {len(X_train)} training rows — "
            "select fewer independent variables."
        )

    pipe = best["pipe"]
    enet = pipe.regressor_.named_steps["elasticnetcv"]
    y_pred = pipe.predict(X_test)

    return {
        "method": "Polynomial + ElasticNet",
        "degree": best["degree"],
        "skipped_degrees": skipped_degrees,
        "alpha": round(float(enet.alpha_), 5),
        "l1_ratio": float(enet.l1_ratio_),
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_test, y_pred), 4),
        "n_test": len(y_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_xgboost_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit XGBoost, auto-selecting hyperparameters via a small grid search with CV."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # min_child_weight/reg_lambda/subsample are searched too — on small datasets
    # unconstrained trees fit noise and produce a jagged, non-monotonic curve.
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1, 10],
        "subsample": [0.7, 1.0],
    }
    base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    search = RandomizedSearchCV(
        base, param_grid, n_iter=30, cv=5, scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        "method": "XGBoost",
        "best_params": search.best_params_,
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_test, y_pred), 4),
        "n_test": len(y_test),
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importance": dict(zip(X.columns, best_model.feature_importances_.round(4))),
    }

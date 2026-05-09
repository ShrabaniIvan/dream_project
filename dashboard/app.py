"""Agricultural Statistics Dashboard using Streamlit."""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# Configure page
st.set_page_config(page_title="Agricultural Statistics", layout="wide")
st.title("🌾 Agricultural Statistics Platform")

# Service URLs
DESCRIPTIVE_STATS_URL = os.getenv("DESCRIPTIVE_STATS_URL", "http://localhost:8001")
TIME_SERIES_URL = os.getenv("TIME_SERIES_URL", "http://localhost:8002")
PREDICTIVE_MODELING_URL = os.getenv("PREDICTIVE_MODELING_URL", "http://localhost:8003")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", [
    "Home",
    "Descriptive Statistics",
    "Time Series Analysis",
    "Predictive Modeling"
])

def load_sample_data(filename: str, folder: str = "") -> pd.DataFrame:
    """Load sample data from CSV with optional folder."""
    try:
        if folder:
            path = f"../data/{folder}/{filename}"
        else:
            path = f"../data/{filename}"
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"Sample data file not found: {filename}")
        return None

# PAGE: Home
if page == "Home":
    st.markdown("""
    ## Welcome to Agricultural Statistics Platform

    This platform provides statistical analysis tools for agricultural data:

    - **Descriptive Statistics**: Summary statistics, correlation, hypothesis testing
    - **Time Series Analysis**: Decomposition, ARIMA, stationarity testing
    - **Predictive Modeling**: Linear regression, feature importance analysis

    Use the sidebar to navigate between different analysis modules.
    """)

    st.info("ℹ️ Services Status: Check individual pages to verify service connectivity")

# PAGE: Descriptive Statistics
elif page == "Descriptive Statistics":
    st.header("Descriptive Statistics")

    tab1, tab2, tab3 = st.tabs(["Summary Stats", "Correlation", "Hypothesis Testing"])

    with tab1:
        st.subheader("Summary Statistics")

        dataset_options = {
            "Crop Yield Comparison": "crop_yield_comparison.csv",
            "Soil Properties": "soil_properties.csv",
            "Weather Data": "weather_data.csv",
            "Crop Varieties": "crop_varieties.csv"
        }
        selected_dataset = st.selectbox("Select dataset", list(dataset_options.keys()))
        df = load_sample_data(dataset_options[selected_dataset], "descriptive_stats")

        if df is not None:
            st.info(f"📊 Dataset: {selected_dataset} ({len(df)} samples, {len(df.columns)} columns)")
            column = st.selectbox("Select column for analysis", df.columns)

            if st.button("Calculate Statistics"):
                payload = {
                    "column": column,
                    "data": df[column].tolist()
                }

                try:
                    response = requests.post(
                        f"{DESCRIPTIVE_STATS_URL}/descriptive-stats",
                        json=payload,
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Mean", f"{result['mean']:.2f}")
                            st.metric("Median", f"{result['median']:.2f}")
                            st.metric("Std Dev", f"{result['std']:.2f}")

                        with col2:
                            st.metric("Min", f"{result['min']:.2f}")
                            st.metric("Max", f"{result['max']:.2f}")
                            st.metric("Count", result['count'])

                        st.success("✓ Statistics calculated successfully")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    with tab2:
        st.subheader("Correlation Matrix")

        corr_dataset_options = {
            "Soil Properties": "soil_properties.csv",
            "Weather Data": "weather_data.csv"
        }
        selected_corr_dataset = st.selectbox("Select dataset for correlation", list(corr_dataset_options.keys()), key="corr_select")
        df = load_sample_data(corr_dataset_options[selected_corr_dataset], "descriptive_stats")

        if df is not None:
            st.info(f"📊 Dataset: {selected_corr_dataset} ({len(df)} samples)")

            if st.button("Calculate Correlation", key="calc_corr"):
            payload = {"data": {col: df[col].tolist() for col in df.columns}}

            try:
                response = requests.post(
                    f"{DESCRIPTIVE_STATS_URL}/correlation-matrix",
                    json=payload,
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    corr_df = pd.DataFrame(result['correlation'])
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

    with tab3:
        st.subheader("Hypothesis Testing")

        hyp_dataset_options = {
            "Crop Yield Comparison (Traditional vs Treated)": ("crop_yield_comparison.csv", "yield_traditional", "yield_treated"),
            "Pesticide Effectiveness": ("pesticide_effectiveness.csv", "control_damage_percent", "organic_treatment_damage")
        }
        selected_hyp_dataset = st.selectbox("Select dataset", list(hyp_dataset_options.keys()), key="hyp_select")
        filename, var1_name, var2_name = hyp_dataset_options[selected_hyp_dataset]
        df = load_sample_data(filename, "descriptive_stats")

        if df is not None:
            st.info(f"📊 Dataset: {selected_hyp_dataset} ({len(df)} samples)")
            st.write(f"**Comparing**: {var1_name} vs {var2_name}")

            test_type = st.radio("Test Type", ["ttest", "mannwhitney"], key="test_type")

            if st.button("Run Test", key="run_test"):
                payload = {
                    "data1": df[var1_name].tolist(),
                    "data2": df[var2_name].tolist(),
                    "test_type": test_type
                }

                try:
                    response = requests.post(
                        f"{DESCRIPTIVE_STATS_URL}/hypothesis-test",
                        json=payload,
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.write(f"**Test**: {result['test']}")
                        st.write(f"**Statistic**: {result['statistic']:.4f}")
                        st.write(f"**P-value**: {result['p_value']:.4f}")

                        if result['significant_at_0.05']:
                            st.success("✓ Significant difference (p < 0.05)")
                        else:
                            st.info("ℹ️ No significant difference (p ≥ 0.05)")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

# PAGE: Time Series Analysis
elif page == "Time Series Analysis":
    st.header("Time Series Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["ACF/PACF", "Decomposition", "ARIMA", "Stationarity"])

    with tab1:
        st.subheader("ACF and PACF Analysis")

        acf_dataset_options = {
            "Temperature Daily": ("temperature_daily.csv", "temperature_c"),
            "Soil Moisture Daily": ("soil_moisture_daily.csv", "soil_moisture_percent"),
            "Crop Yield Monthly": ("crop_yield_monthly.csv", "crop_yield_tons")
        }
        selected_acf_dataset = st.selectbox("Select time series", list(acf_dataset_options.keys()), key="acf_select")
        filename, column = acf_dataset_options[selected_acf_dataset]
        df = load_sample_data(filename, "time_series")

        if df is not None:
            st.info(f"📊 Dataset: {selected_acf_dataset} ({len(df)} samples)")
            nlags = st.slider("Number of lags", 5, 40, 20, key="nlags")

            if st.button("Calculate ACF/PACF", key="calc_acf"):
                payload = {
                    "values": df[column].tolist(),
                    "name": column
                }

                try:
                    response = requests.post(
                        f"{TIME_SERIES_URL}/acf-pacf",
                        json=payload,
                        params={"nlags": nlags},
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()
                        col1, col2 = st.columns(2)

                        with col1:
                            st.line_chart(result['acf'], use_container_width=True)
                            st.caption("ACF Plot")

                        with col2:
                            st.line_chart(result['pacf'], use_container_width=True)
                            st.caption("PACF Plot")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    with tab2:
        st.subheader("Seasonal Decomposition")

        decomp_dataset_options = {
            "Daily Rainfall (4 years)": ("daily_rainfall.csv", "rainfall_mm", 365),
            "Crop Yield Monthly (5 years)": ("crop_yield_monthly.csv", "crop_yield_tons", 12),
            "Temperature Daily (2 years)": ("temperature_daily.csv", "temperature_c", 365)
        }
        selected_decomp_dataset = st.selectbox("Select time series for decomposition", list(decomp_dataset_options.keys()), key="decomp_select")
        filename, column, default_period = decomp_dataset_options[selected_decomp_dataset]
        df = load_sample_data(filename, "time_series")

        if df is not None:
            st.info(f"📊 Dataset: {selected_decomp_dataset} ({len(df)} samples)")
            period = st.slider("Seasonal period", 3, 52, default_period, key="decomp_period")

            if st.button("Decompose Series", key="decompose"):
                payload = {
                    "values": df[column].tolist(),
                    "name": column
                }

                try:
                    response = requests.post(
                        f"{TIME_SERIES_URL}/seasonal-decompose",
                        json=payload,
                        params={"period": period},
                        timeout=5
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.line_chart(result['observed'], use_container_width=True)
                        st.caption("Observed")

                        st.line_chart(result['trend'], use_container_width=True)
                        st.caption("Trend")

                        st.line_chart(result['seasonal'], use_container_width=True)
                        st.caption("Seasonal")

                        st.line_chart(result['residual'], use_container_width=True)
                        st.caption("Residual")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    with tab3:
        st.subheader("ARIMA Model Fitting")

        arima_dataset_options = {
            "Crop Yield Monthly": ("crop_yield_monthly.csv", "crop_yield_tons"),
            "Commodity Prices": ("commodity_prices.csv", "crop_price_per_unit")
        }
        selected_arima_dataset = st.selectbox("Select time series for ARIMA", list(arima_dataset_options.keys()), key="arima_select")
        filename, column = arima_dataset_options[selected_arima_dataset]
        df = load_sample_data(filename, "time_series")

        if df is not None:
            st.info(f"📊 Dataset: {selected_arima_dataset} ({len(df)} samples)")

            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR order)", value=1, min_value=0, key="p_val")
            with col2:
                d = st.number_input("d (Differencing)", value=1, min_value=0, key="d_val")
            with col3:
                q = st.number_input("q (MA order)", value=1, min_value=0, key="q_val")

            if st.button("Fit ARIMA", key="fit_arima"):
                payload = {
                    "values": df[column].tolist(),
                    "name": column
                }

                try:
                    response = requests.post(
                        f"{TIME_SERIES_URL}/arima-fit",
                        json=payload,
                        params={"order": (int(p), int(d), int(q))},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()

                        if "error" not in result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("AIC", f"{result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{result['bic']:.2f}")

                            with st.expander("Model Summary"):
                                st.text(result['summary'])
                        else:
                            st.error(f"Model Error: {result['error']}")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    with tab4:
        st.subheader("Stationarity Test (Augmented Dickey-Fuller)")

        stationarity_dataset_options = {
            "White Noise (Stationary)": ("white_noise.csv", "white_noise"),
            "Random Walk (Non-stationary)": ("random_walk.csv", "random_walk"),
            "Commodity Prices": ("commodity_prices.csv", "crop_price_per_unit")
        }
        selected_stat_dataset = st.selectbox("Select time series for ADF test", list(stationarity_dataset_options.keys()), key="stat_select")
        filename, column = stationarity_dataset_options[selected_stat_dataset]
        df = load_sample_data(filename, "time_series")

        if df is not None:
            st.info(f"📊 Dataset: {selected_stat_dataset} ({len(df)} samples)")

            if st.button("Run ADF Test", key="run_adf"):
                payload = {
                    "values": df[column].tolist(),
                    "name": column
                }

            try:
                response = requests.post(
                    f"{TIME_SERIES_URL}/adf-test",
                    json=payload,
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    st.write(f"**Test Statistic**: {result['test_statistic']:.4f}")
                    st.write(f"**P-value**: {result['p_value']:.4f}")
                    st.write(f"**Lags Used**: {result['n_lags']}")

                    if result['stationary_at_0.05']:
                        st.success("✓ Series is stationary (reject null hypothesis)")
                    else:
                        st.warning("⚠️ Series is non-stationary (fail to reject null hypothesis)")
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

# PAGE: Predictive Modeling
elif page == "Predictive Modeling":
    st.header("Predictive Modeling")

    tab1, tab2 = st.tabs(["Linear Regression", "Feature Importance"])

    with tab1:
        st.subheader("Linear Regression Model")

        regression_dataset_options = {
            "Crop Yield Prediction": ("crop_yield_features.csv", "crop_yield_tons_per_ha"),
            "Pest Infestation": ("pest_infestation.csv", "infestation_severity"),
            "Irrigation Requirements": ("irrigation_requirements.csv", "irrigation_hours_needed"),
            "Field Productivity": ("field_productivity.csv", "overall_productivity_score")
        }
        selected_regression_dataset = st.selectbox("Select dataset", list(regression_dataset_options.keys()), key="reg_select")
        filename, target = regression_dataset_options[selected_regression_dataset]
        df = load_sample_data(filename, "predictive_modeling")

        if df is not None:
            features = [col for col in df.columns if col != target]
            st.info(f"📊 Dataset: {selected_regression_dataset} ({len(df)} samples)")
            st.write(f"**Features**: {', '.join(features)}")
            st.write(f"**Target**: {target}")

            if st.button("Train Model", key="train_model"):
                payload = {
                    "features": {col: df[col].tolist() for col in features},
                    "target": df[target].tolist()
                }

                try:
                    response = requests.post(
                        f"{PREDICTIVE_MODELING_URL}/train-linear-regression",
                        json=payload,
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{result['r2_score']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                        with col3:
                            st.metric("MSE", f"{result['mse']:.4f}")

                        st.subheader("Coefficients")
                        coef_df = pd.DataFrame([
                            {"Feature": k, "Coefficient": v}
                            for k, v in result['coefficients'].items()
                        ])
                        st.dataframe(coef_df, use_container_width=True)

                        st.write(f"**Intercept**: {result['intercept']:.4f}")
                        st.write(f"**Train/Test Split**: {result['train_size']}/{result['test_size']}")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    with tab2:
        st.subheader("Feature Importance Analysis")

        importance_dataset_options = {
            "Crop Yield Prediction": ("crop_yield_features.csv", "crop_yield_tons_per_ha"),
            "Field Productivity": ("field_productivity.csv", "overall_productivity_score")
        }
        selected_importance_dataset = st.selectbox("Select dataset for importance", list(importance_dataset_options.keys()), key="imp_select")
        filename, target = importance_dataset_options[selected_importance_dataset]
        df = load_sample_data(filename, "predictive_modeling")

        if df is not None:
            features = [col for col in df.columns if col != target]
            st.info(f"📊 Dataset: {selected_importance_dataset} ({len(df)} samples)")

            if st.button("Calculate Feature Importance", key="calc_importance"):
                payload = {
                    "features": {col: df[col].tolist() for col in features},
                    "target": df[target].tolist()
                }

                try:
                    response = requests.post(
                        f"{PREDICTIVE_MODELING_URL}/feature-importance",
                        json=payload,
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()

                        importance_data = []
                        for feature, scores in result['feature_importance'].items():
                            importance_data.append({
                                'Feature': feature,
                                'Correlation': scores['correlation'],
                                'Abs Correlation': scores['abs_correlation']
                            })

                        importance_df = pd.DataFrame(importance_data).sort_values(
                            'Abs Correlation', ascending=False
                        )

                        st.bar_chart(importance_df.set_index('Feature')['Abs Correlation'])
                        st.dataframe(importance_df, use_container_width=True)
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

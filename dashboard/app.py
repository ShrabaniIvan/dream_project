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

def load_sample_data(filename: str) -> pd.DataFrame:
    """Load sample data from CSV."""
    try:
        return pd.read_csv(f"../data/{filename}")
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
        df = load_sample_data("descriptive_sample.csv")

        if df is not None:
            column = st.selectbox("Select column", df.columns)

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
        df = load_sample_data("descriptive_sample.csv")

        if df is not None and st.button("Calculate Correlation"):
            payload = {col: df[col].tolist() for col in df.columns}

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
        df = load_sample_data("descriptive_sample.csv")

        if df is not None:
            col1, col2 = st.columns(2)

            with col1:
                var1 = st.selectbox("Variable 1", df.columns)

            with col2:
                var2 = st.selectbox("Variable 2", df.columns)

            test_type = st.radio("Test Type", ["ttest", "mannwhitney"])

            if st.button("Run Test"):
                payload = {
                    "data1": df[var1].tolist(),
                    "data2": df[var2].tolist(),
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
        st.subheader("ACF and PACF")
        df = load_sample_data("time_series_sample.csv")

        if df is not None:
            nlags = st.slider("Number of lags", 5, 40, 20)

            if st.button("Calculate ACF/PACF"):
                payload = {
                    "values": df['crop_yield'].tolist(),
                    "name": "crop_yield"
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
        df = load_sample_data("time_series_sample.csv")

        if df is not None:
            period = st.slider("Seasonal period", 3, 52, 12)

            if st.button("Decompose Series"):
                payload = {
                    "values": df['crop_yield'].tolist(),
                    "name": "crop_yield"
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
        st.subheader("ARIMA Model")
        df = load_sample_data("time_series_sample.csv")

        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p", value=1, min_value=0)
            with col2:
                d = st.number_input("d", value=1, min_value=0)
            with col3:
                q = st.number_input("q", value=1, min_value=0)

            if st.button("Fit ARIMA"):
                payload = {
                    "values": df['crop_yield'].tolist(),
                    "name": "crop_yield"
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
        st.subheader("Stationarity Test (ADF)")
        df = load_sample_data("time_series_sample.csv")

        if df is not None and st.button("Run ADF Test"):
            payload = {
                "values": df['crop_yield'].tolist(),
                "name": "crop_yield"
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
        df = load_sample_data("predictive_sample.csv")

        if df is not None:
            features = [col for col in df.columns if col != 'crop_yield']
            target = 'crop_yield'

            st.write(f"Features: {', '.join(features)}")
            st.write(f"Target: {target}")

            if st.button("Train Model"):
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
        st.subheader("Feature Importance")
        df = load_sample_data("predictive_sample.csv")

        if df is not None:
            features = [col for col in df.columns if col != 'crop_yield']
            target = 'crop_yield'

            if st.button("Calculate Feature Importance"):
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

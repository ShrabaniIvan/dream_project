"""Agricultural Statistics Dashboard using Streamlit."""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# Configure page
st.set_page_config(page_title="Agricultural Statistics", layout="wide")

# Add background image with semi-transparency
import base64

@st.cache_data
def get_base64_image(image_path):
    """Convert image to base64 for embedding in CSS."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    img_base64 = get_base64_image("/data/paddy.jpg")
    background_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.75);
        z-index: 0;
    }}

    [data-testid="stHeader"] {{
        z-index: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
        z-index: 1;
    }}

    .main {{
        z-index: 1;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: rgba(255, 255, 255, 0.95);
        z-index: 1;
    }}

    /* Stunning Table Styling - All table elements */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%) !important;
        padding: 0 !important;
    }}

    /* DataFrame container styling */
    .stDataFrame {{
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
    }}

    /* Catch all table-like elements */
    .element-container [role="table"],
    div[data-testid*="Table"],
    div[data-testid*="table"] {{
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
    }}

    /* Wrapper for all data displays */
    .stDataFrameWrapper {{
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
    }}

    /* Element container with tables */
    .element-container {{
        border-radius: 12px;
        overflow: visible;
    }}

    /* Direct dataframe parent */
    .stDataFrame > div {{
        border-radius: 12px !important;
        overflow: hidden !important;
    }}

    /* Table body and cells */
    table {{
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
    }}

    /* Add outline to iframe (if used) */
    iframe {{
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
    }}

    /* Metric cards - Beautiful gradient borders */
    [data-testid="metric-container"] {{
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border: 2px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.08);
        transition: all 0.3s ease;
    }}

    [data-testid="metric-container"]:hover {{
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        border: 2px solid rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }}

    /* Expander styling */
    [data-testid="stExpander"] {{
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.08) !important;
        border: 2px solid rgba(102, 126, 234, 0.15) !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%) !important;
    }}

    /* Tab styling */
    [data-testid="stTabs"] {{
        border-bottom: 3px solid rgba(102, 126, 234, 0.2);
    }}

    .stTabs [aria-selected="true"] {{
        color: #667eea;
        border-bottom: 3px solid #667eea;
        font-weight: 600;
    }}

    /* Button styling */
    .stButton > button {{
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }}

    /* Info/Warning/Success boxes */
    .stAlert {{
        border-radius: 10px;
        border-left: 5px solid rgba(102, 126, 234, 0.5);
        padding: 15px 20px;
        backdrop-filter: blur(10px);
    }}

    /* Heading styling */
    h1 {{
        color: white;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 20px;
    }}

    h2, h3 {{
        color: #667eea;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }}

    /* Code blocks styling */
    code {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 6px;
        padding: 2px 6px;
        color: #764ba2;
        font-family: 'Monaco', 'Menlo', monospace;
    }}

    /* Divider styling */
    hr {{
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 25px 0;
    }}

    /* Select box and input styling */
    .stSelectbox, .stNumberInput, .stSlider {{
        border-radius: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
    }}

    .stSelectbox > div > div, .stNumberInput > div > div {{
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        transition: all 0.3s ease;
    }}

    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {{
        border: 2px solid rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ Background image not found")

# Custom title with styling
st.markdown("""
    <div style="text-align: center; padding: 30px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px; color: white;">
        <h1 style="margin: 0; color: white; font-size: 3em; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">🌾 Agricultural Statistics Platform</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">Advanced Analytics for Agricultural Data</p>
    </div>
""", unsafe_allow_html=True)

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
            path = f"/data/{folder}/{filename}"
        else:
            path = f"/data/{filename}"
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"Sample data file not found: {filename}")
        return None

def display_beautiful_table(df: pd.DataFrame, title: str = ""):
    """Display dataframe with beautiful styling."""
    if title:
        st.markdown(f'<h4 style="color: #667eea; margin-bottom: 15px;">{title}</h4>', unsafe_allow_html=True)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=False
    )

# PAGE: Home
if page == "Home":
    st.markdown("""
    ## 🌾 Welcome to Agricultural Statistics Platform

    A comprehensive platform for statistical analysis of agricultural data using Python statsmodels and machine learning.

    ### 📊 Available Modules

    **1. Descriptive Statistics** - Summarize and compare agricultural data
    - Summary Statistics: Mean, median, std, quartiles, min/max
    - Correlation Analysis: Identify relationships between variables
    - Hypothesis Testing: Compare groups (t-test, Mann-Whitney U)

    **2. Time Series Analysis** - Analyze temporal patterns in agricultural data
    - ACF/PACF Analysis: Autocorrelation patterns
    - Seasonal Decomposition: Trend, seasonal, residual components
    - ARIMA Modeling: Forecasting and model fitting
    - Stationarity Testing: ADF test for data properties

    **3. Predictive Modeling** - Build regression models for forecasting
    - Linear Regression: Feature relationships and predictions
    - Feature Importance: Identify key influencing factors
    - Model Evaluation: R², RMSE, MSE metrics

    ### 🚀 Quick Start
    1. Select a module from the sidebar
    2. Choose a pre-loaded agricultural dataset from the dropdown
    3. Click the analysis button
    4. Review results and visualizations

    ### 📁 Data Format
    All datasets are **CSV files** with:
    - Headers in first row
    - Numeric values (comma-separated)
    - Consistent data types per column
    - No missing values

    ### 📚 Sample Datasets Available
    - **Descriptive Stats**: 5 agricultural datasets (150-320 samples)
    - **Time Series**: 7 temporal datasets (60-1460 samples)
    - **Predictive Modeling**: 4 regression datasets (350-500 samples)
    """)

    st.info("ℹ️ All services are running. Navigate to modules using the sidebar to begin analysis.")

# PAGE: Descriptive Statistics
elif page == "Descriptive Statistics":
    st.header("📊 Descriptive Statistics")

    with st.expander("📖 Methods & Data Format Guide", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 Available Methods

            **1. Summary Statistics**
            - Count, Mean, Median
            - Standard Deviation
            - Min, Max, Q25, Q75

            **2. Correlation Matrix**
            - Pearson correlation coefficients
            - Identifies variable relationships
            - Values range from -1 to +1

            **3. Hypothesis Testing**
            - Independent t-test
            - Mann-Whitney U test
            - P-values for significance
            """)

        with col2:
            st.markdown("""
            ### 📄 Data Format Requirements

            **Input Format**: CSV file (comma-separated)

            **Structure**:
            ```
            column1, column2, column3
            10.5,    20.3,    5.2
            15.2,    18.9,    6.1
            12.8,    22.1,    4.8
            ```

            **Requirements**:
            - Numeric values only
            - Headers in first row
            - No missing values
            - Consistent data types

            **Supported Operations**:
            - Single column analysis
            - Multi-column correlations
            - Two-group comparisons
            """)

    st.markdown("---")
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
                        display_beautiful_table(corr_df, "📊 Correlation Matrix")
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
    st.header("📈 Time Series Analysis")

    with st.expander("📖 Methods & Data Format Guide", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 Available Methods

            **1. ACF/PACF Analysis**
            - Autocorrelation Function
            - Partial Autocorrelation
            - Identifies lag patterns
            - Lags: 5-40 adjustable

            **2. Seasonal Decomposition**
            - Trend component
            - Seasonal component
            - Residual component
            - Additive decomposition

            **3. ARIMA Modeling**
            - AutoRegressive (p)
            - Integrated (d)
            - Moving Average (q)
            - AIC/BIC metrics

            **4. Stationarity Testing**
            - Augmented Dickey-Fuller test
            - Tests null hypothesis
            - P-value interpretation
            """)

        with col2:
            st.markdown("""
            ### 📄 Data Format Requirements

            **Input Format**: CSV with date column

            **Structure**:
            ```
            date,       value
            2023-01-01, 45.2
            2023-01-02, 48.7
            2023-01-03, 52.1
            ```

            **Requirements**:
            - Date column (YYYY-MM-DD)
            - Single numeric value column
            - Chronological order
            - Regular intervals preferred
            - No missing values

            **Typical Data Sizes**:
            - Daily: 365-1460 samples
            - Monthly: 12-240 samples
            - Hourly: 1000+ samples

            **What to Expect**:
            - Trend: Long-term direction
            - Seasonality: Repeating patterns
            - Residuals: Random variation
            """)

    st.markdown("---")
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
    st.header("🔮 Predictive Modeling")

    with st.expander("📖 Methods & Data Format Guide", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 Available Methods

            **1. Linear Regression**
            - Fit model to features
            - Learn coefficient weights
            - Predict continuous values
            - Metrics: R², RMSE, MSE
            - Train/test split: 80/20

            **2. Feature Importance**
            - Correlation-based analysis
            - Absolute correlation ranking
            - Identifies key predictors
            - Supports 1-N features

            ### 📊 Model Evaluation
            - **R² Score**: Explained variance
            - **RMSE**: Root mean squared error
            - **Coefficients**: Feature weights
            - **Train Size**: 80% of data
            - **Test Size**: 20% of data
            """)

        with col2:
            st.markdown("""
            ### 📄 Data Format Requirements

            **Input Format**: CSV with features + target

            **Structure**:
            ```
            rainfall, temperature, fertilizer, yield
            25.3,     22.1,        150,        65.2
            30.5,     25.8,        180,        72.5
            20.1,     18.5,        120,        55.8
            ```

            **Requirements**:
            - Features: All numeric
            - Target: Single numeric column
            - Last column = target variable
            - No missing values
            - Features: 2-N columns

            **Feature Types**:
            - Continuous: Any numeric values
            - Categorical*: Converted to numeric
              (*needs preprocessing)

            **Data Sizes**:
            - Minimum: 30 samples
            - Recommended: 100-500
            - Maximum: 10,000+

            **Typical Applications**:
            - Yield prediction
            - Pest forecasting
            - Irrigation scheduling
            - Resource optimization
            """)

    st.markdown("---")
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

                        st.markdown('---')
                        display_beautiful_table(
                            pd.DataFrame([
                                {"Feature": k, "Coefficient": v}
                                for k, v in result['coefficients'].items()
                            ]),
                            "📈 Model Coefficients"
                        )

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

                        st.markdown('<h4 style="color: #667eea; margin-top: 20px; margin-bottom: 15px;">📊 Feature Importance Distribution</h4>', unsafe_allow_html=True)
                        st.bar_chart(importance_df.set_index('Feature')['Abs Correlation'])
                        st.markdown('---')
                        display_beautiful_table(importance_df, "🎯 Feature Rankings")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

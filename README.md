# Agricultural Statistics Platform

A comprehensive platform for agricultural statistical analysis using Python `statsmodels`, FastAPI services, and Streamlit dashboard.

## 📚 Features

### Descriptive Statistics
- Summary statistics (mean, median, std, min, max, quartiles)
- Correlation matrix analysis
- Hypothesis testing (t-test, Mann-Whitney U)

### Time Series Analysis
- ACF/PACF analysis
- Seasonal decomposition
- ARIMA modeling
- Stationarity testing (ADF test)

### Predictive Modeling
- Linear regression
- Feature importance analysis
- Model performance metrics (R², RMSE, MSE)

## 🏗️ Architecture

```
descriptive_stats/     → FastAPI service (port 8001)
time_series/           → FastAPI service (port 8002)
predictive_modeling/   → FastAPI service (port 8003)
dashboard/             → Streamlit app (port 8501)
data/                  → Synthetic datasets & generation script
docker-compose.yml     → Orchestrate all services
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop running

### 1. Navigate to Project
```bash
cd /Users/abhijitghosh/projects/dream_project
```

### 2. Build & Run All Services
```bash
docker-compose up --build
```

This will:
- Build images for all 4 services
- Start containers (ports 8001-8003 for APIs, 8501 for dashboard)
- Wait for health checks before starting dashboard

### 3. Access Dashboard
Open browser: **http://localhost:8501**

## 📊 Usage

### Descriptive Statistics
1. Select "Descriptive Statistics" from sidebar
2. Choose from three tabs:
   - **Summary Stats**: Calculate statistics for any column
   - **Correlation**: Compute correlation matrix
   - **Hypothesis Testing**: Compare two variables

### Time Series Analysis
1. Select "Time Series Analysis"
2. Choose from four tabs:
   - **ACF/PACF**: View autocorrelation patterns
   - **Decomposition**: Break down trend, seasonal, residual
   - **ARIMA**: Fit ARIMA(p,d,q) models
   - **Stationarity**: Test with ADF test

### Predictive Modeling
1. Select "Predictive Modeling"
2. Choose from two tabs:
   - **Linear Regression**: Train and view coefficients
   - **Feature Importance**: Analyze correlation-based importance

## 📁 Sample Datasets

Auto-generated via `data/generate_data.py`:
- `descriptive_sample.csv` → 200 samples, 5 features
- `time_series_sample.csv` → 365 daily observations
- `predictive_sample.csv` → 300 samples with target

## 🛑 Stop Services
```bash
docker-compose down
```

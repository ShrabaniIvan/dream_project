# 🌾 Agricultural Statistics Platform - Data Catalog

Complete guide to all available synthetic datasets for testing each module.

## 📊 Descriptive Statistics Datasets

Located in `data/descriptive_stats/`

### 1. **crop_yield_comparison.csv** (150 samples × 2 columns)
- **Purpose**: Compare crop yields between two farming methods using hypothesis testing
- **Columns**: 
  - `yield_traditional` - Traditional farming method yield
  - `yield_treated` - New fertilizer treatment yield
- **Use Cases**: t-test, Mann-Whitney U test, effect size analysis
- **Example**: Test if new fertilizer method significantly improves yield

### 2. **soil_properties.csv** (250 samples × 6 columns)
- **Purpose**: Analyze soil properties across different fields
- **Columns**:
  - `soil_ph` - Soil pH (6-7 range)
  - `nitrogen_ppm` - Nitrogen content
  - `phosphorus_ppm` - Phosphorus content
  - `potassium_ppm` - Potassium content
  - `organic_matter_percent` - Organic matter %
  - `field_id` - Field identifier (1-5)
- **Use Cases**: Summary statistics, correlation matrix, multivariate analysis
- **Example**: Check correlation between pH and nutrient levels

### 3. **weather_data.csv** (365 samples × 5 columns)
- **Purpose**: Weather variables for correlation and relationship analysis
- **Columns**:
  - `date` - Daily date
  - `temperature_c` - Temperature in Celsius
  - `rainfall_mm` - Rainfall in mm
  - `humidity_percent` - Humidity %
  - `solar_radiation` - Solar radiation index
- **Use Cases**: Correlation analysis, distribution analysis
- **Example**: Analyze correlation between rainfall and humidity

### 4. **pesticide_effectiveness.csv** (120 samples × 3 columns)
- **Purpose**: Compare pesticide treatment effectiveness
- **Columns**:
  - `control_damage_percent` - No treatment damage %
  - `organic_treatment_damage` - Organic pesticide damage %
  - `chemical_treatment_damage` - Chemical pesticide damage %
- **Use Cases**: ANOVA, multiple comparison tests, mean comparisons
- **Example**: Determine which pesticide type is most effective

### 5. **crop_varieties.csv** (320 samples × 2 columns)
- **Purpose**: Compare yields across different crop varieties
- **Columns**:
  - `variety` - Crop variety (A, B, C, D)
  - `yield_kg_per_hectare` - Yield measurement
- **Use Cases**: Group statistics, ANOVA, categorical analysis
- **Example**: Which crop variety has highest average yield?

---

## 📈 Time Series Datasets

Located in `data/time_series/`

### 1. **daily_rainfall.csv** (1460 samples, 4 years)
- **Purpose**: Rainfall data with strong seasonal pattern
- **Columns**: `date`, `rainfall_mm`
- **Characteristics**: 
  - Seasonal peaks (monsoon season)
  - Positive trend
  - Typical variance
- **Use Cases**: Decomposition, seasonal analysis, forecasting
- **Example**: Decompose rainfall into trend, seasonal, and residual components

### 2. **crop_yield_monthly.csv** (60 samples, 5 years)
- **Purpose**: Monthly crop yield with multiple patterns
- **Columns**: `date`, `crop_yield_tons`
- **Characteristics**:
  - Yearly seasonal pattern
  - Positive trend
  - Business cycle (2-year)
- **Use Cases**: ARIMA modeling, trend analysis, seasonal adjustment
- **Example**: Fit ARIMA(1,1,1) model and analyze ACF/PACF

### 3. **temperature_daily.csv** (730 samples, 2 years)
- **Purpose**: Daily temperature with clear seasonality
- **Columns**: `date`, `temperature_c`
- **Characteristics**:
  - Strong yearly seasonality
  - Slight warming trend
  - Low noise
- **Use Cases**: Decomposition, ACF analysis, seasonal pattern identification
- **Example**: Extract seasonal pattern for growing season analysis

### 4. **soil_moisture_daily.csv** (365 samples, 1 year)
- **Purpose**: Soil moisture with irrigation events
- **Columns**: `date`, `soil_moisture_percent`
- **Characteristics**:
  - Irregular pattern from irrigation
  - Seasonal trend
  - Clear spikes from irrigation events
- **Use Cases**: Anomaly detection, pattern recognition, intervention analysis
- **Example**: Identify irrigation timing patterns using ACF

### 5. **commodity_prices.csv** (240 samples, 20 years)
- **Purpose**: Agricultural commodity price history
- **Columns**: `date`, `crop_price_per_unit`
- **Characteristics**:
  - Long-term trend
  - Seasonal variation
  - Market cycles
  - Random shocks
- **Use Cases**: Long-range forecasting, trend analysis, cycle decomposition
- **Example**: Perform ARIMA analysis on commodity prices

### 6. **white_noise.csv** (200 samples)
- **Purpose**: Stationary time series (baseline)
- **Columns**: `date`, `white_noise`
- **Characteristics**:
  - Pure white noise
  - Stationary (mean=0, constant variance)
  - No autocorrelation
- **Use Cases**: Testing stationarity (ADF test should pass)
- **Example**: Run ADF test - should NOT reject stationarity

### 7. **random_walk.csv** (200 samples)
- **Purpose**: Non-stationary time series
- **Columns**: `date`, `random_walk`
- **Characteristics**:
  - Random walk with drift
  - Non-stationary (mean changes over time)
  - Strong autocorrelation
- **Use Cases**: Testing stationarity (ADF test should fail)
- **Example**: Run ADF test - should reject stationarity, then difference to make stationary

---

## 🔮 Predictive Modeling Datasets

Located in `data/predictive_modeling/`

### 1. **crop_yield_features.csv** (500 samples × 7 columns)
- **Purpose**: Predict crop yield from environmental and management features
- **Columns**:
  - `rainfall_mm` - Annual rainfall
  - `temperature_c` - Average temperature
  - `fertilizer_kg_per_ha` - Fertilizer application
  - `soil_ph` - Soil pH
  - `irrigation_hours` - Irrigation applied
  - `pesticide_cost` - Pesticide investment
  - `crop_yield_tons_per_ha` - **TARGET**
- **Target**: `crop_yield_tons_per_ha` (continuous, range: 10-120)
- **Use Cases**: Linear regression, feature importance, predictive modeling
- **Example**: Train model to predict yield from weather and management variables
- **Expected Correlation**: 
  - Positive: rainfall, temperature, fertilizer, irrigation, pesticide cost
  - Non-linear: temperature (optimum ~25°C)

### 2. **pest_infestation.csv** (400 samples × 6 columns)
- **Purpose**: Predict pest infestation severity
- **Columns**:
  - `humidity_percent` - Environmental humidity
  - `temperature_c` - Temperature
  - `rainfall_days` - Rainy days in period
  - `crop_age_days` - Crop age
  - `field_proximity_km` - Distance from infected fields
  - `infestation_severity` - **TARGET** (0-100 scale)
- **Target**: `infestation_severity` (continuous, range: 0-100)
- **Use Cases**: Regression, prediction, risk assessment
- **Example**: Predict pest infestation from environmental conditions
- **Expected Correlation**:
  - Positive: humidity, rainfall_days, crop_age
  - Negative: temperature distance from infected fields

### 3. **irrigation_requirements.csv** (450 samples × 6 columns)
- **Purpose**: Predict irrigation needs from environmental factors
- **Columns**:
  - `soil_moisture_percent` - Current soil moisture
  - `evapotranspiration_mm_per_day` - Water demand
  - `rainfall_forecast_mm` - Predicted rainfall
  - `temperature_c` - Temperature
  - `crop_stage` - Growth stage (categorical)
  - `irrigation_hours_needed` - **TARGET** (0-100)
- **Target**: `irrigation_hours_needed` (continuous, range: 0-100)
- **Use Cases**: Regression with categorical variables, feature importance
- **Example**: Determine optimal irrigation scheduling
- **Features**: 
  - Mix of continuous and categorical variables
  - Good for demonstrating dummy variable encoding

### 4. **field_productivity.csv** (350 samples × 6 columns)
- **Purpose**: Predict overall field productivity
- **Columns**:
  - `field_size_ha` - Field size in hectares
  - `soil_quality_score` - Soil quality rating (1-10)
  - `years_managed` - Years under current management
  - `mechanization_level` - Mechanization % (0-100)
  - `crop_rotation_practiced` - Whether crop rotation is used (0/1)
  - `overall_productivity_score` - **TARGET** (20-100)
- **Target**: `overall_productivity_score` (continuous, range: 20-100)
- **Use Cases**: Feature importance analysis, regression, predictability assessment
- **Example**: Rank features by their impact on productivity
- **Expected Correlation**:
  - All positive with moderate-to-strong correlation
  - Good dataset for comparing feature importance methods

---

## 📁 Directory Structure

```
data/
├── descriptive_stats/
│   ├── crop_yield_comparison.csv
│   ├── soil_properties.csv
│   ├── weather_data.csv
│   ├── pesticide_effectiveness.csv
│   └── crop_varieties.csv
│
├── time_series/
│   ├── daily_rainfall.csv
│   ├── crop_yield_monthly.csv
│   ├── temperature_daily.csv
│   ├── soil_moisture_daily.csv
│   ├── commodity_prices.csv
│   ├── white_noise.csv
│   └── random_walk.csv
│
├── predictive_modeling/
│   ├── crop_yield_features.csv
│   ├── pest_infestation.csv
│   ├── irrigation_requirements.csv
│   └── field_productivity.csv
│
├── generate_datasets.py       # Script to regenerate all datasets
└── DATA_CATALOG.md           # This file
```

---

## 🎯 Quick Reference: Which Dataset to Use When?

### Testing Hypothesis Testing
→ Use `crop_yield_comparison.csv` or `pesticide_effectiveness.csv`

### Testing Descriptive Statistics
→ Use `soil_properties.csv` or `weather_data.csv`

### Testing Correlation Analysis
→ Use `weather_data.csv` or `soil_properties.csv`

### Testing Time Series Decomposition
→ Use `daily_rainfall.csv` or `crop_yield_monthly.csv`

### Testing ARIMA Modeling
→ Use `crop_yield_monthly.csv` or `commodity_prices.csv`

### Testing ACF/PACF Analysis
→ Use `temperature_daily.csv` or `soil_moisture_daily.csv`

### Testing Stationarity (ADF Test)
→ Use `white_noise.csv` (stationary) or `random_walk.csv` (non-stationary)

### Testing Regression Models
→ Use `crop_yield_features.csv` (best for simple regression)

### Testing Feature Importance
→ Use `field_productivity.csv` (all features positively correlated)

### Testing Categorical Variables in Regression
→ Use `irrigation_requirements.csv` (includes crop_stage)

---

## 📊 Data Characteristics Summary

| Dataset | Samples | Variables | Type | Trend | Seasonality | Notes |
|---------|---------|-----------|------|-------|-------------|-------|
| crop_yield_comparison | 150 | 2 | Hypothesis | No | No | Two groups for comparison |
| soil_properties | 250 | 6 | Descriptive | No | No | Strong correlations |
| weather_data | 365 | 5 | Descriptive | Yes | Yes | One year daily data |
| pesticide_effectiveness | 120 | 3 | Hypothesis | No | No | Three treatment groups |
| crop_varieties | 320 | 2 | Descriptive | No | No | Categorical + continuous |
| daily_rainfall | 1460 | 2 | Time Series | Yes | Yes | 4 years, monsoon pattern |
| crop_yield_monthly | 60 | 2 | Time Series | Yes | Yes | 5 years, business cycle |
| temperature_daily | 730 | 2 | Time Series | Yes | Yes | 2 years, strong seasonality |
| soil_moisture_daily | 365 | 2 | Time Series | Yes | No | Irrigation patterns |
| commodity_prices | 240 | 2 | Time Series | Yes | Yes | 20 years, market cycles |
| white_noise | 200 | 2 | Time Series | No | No | **Stationary** |
| random_walk | 200 | 2 | Time Series | Yes | No | **Non-stationary** |
| crop_yield_features | 500 | 7 | Regression | No | No | 1 continuous target |
| pest_infestation | 400 | 6 | Regression | No | No | Complex relationships |
| irrigation_requirements | 450 | 6 | Regression | No | No | Categorical variable |
| field_productivity | 350 | 6 | Regression | No | No | All positive correlations |

---

## 🔄 Regenerating Datasets

To regenerate all datasets with new random seeds:

```bash
python data/generate_datasets.py
```

This will overwrite all existing CSV files with new random data maintaining the same distributions and patterns.

---

## 📝 Notes

- All datasets are **synthetic agricultural data** based on realistic distributions
- Random seed is set to 42 for reproducibility
- All numeric values are within realistic ranges for actual agricultural data
- Time series data includes realistic patterns (trends, seasonality, cycles, noise)
- Predictive modeling datasets have engineered target variables based on input features

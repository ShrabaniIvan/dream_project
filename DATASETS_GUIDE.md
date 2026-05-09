# 📊 Agricultural Statistics Platform - Datasets Guide

Quick reference for all available test datasets organized by module.

## 🚀 Quick Start

1. **Ensure containers are running**:
   ```bash
   docker-compose ps
   ```

2. **Open Dashboard**: http://localhost:8501

3. **Select datasets from dropdowns** in each module

---

## 📁 Dataset Organization

```
data/
├── descriptive_stats/          ← Use for Summary Stats, Correlation, Hypothesis Testing
│   ├── crop_yield_comparison.csv       (150 samples)
│   ├── soil_properties.csv            (250 samples)
│   ├── weather_data.csv               (365 samples)
│   ├── pesticide_effectiveness.csv    (120 samples)
│   └── crop_varieties.csv             (320 samples)
│
├── time_series/                ← Use for ACF/PACF, Decomposition, ARIMA, ADF Tests
│   ├── daily_rainfall.csv              (1460 samples)
│   ├── crop_yield_monthly.csv          (60 samples)
│   ├── temperature_daily.csv           (730 samples)
│   ├── soil_moisture_daily.csv         (365 samples)
│   ├── commodity_prices.csv            (240 samples)
│   ├── white_noise.csv                 (200 samples) ← Stationary
│   └── random_walk.csv                 (200 samples) ← Non-stationary
│
├── predictive_modeling/        ← Use for Regression, Feature Importance
│   ├── crop_yield_features.csv         (500 samples, 7 columns)
│   ├── pest_infestation.csv            (400 samples, 6 columns)
│   ├── irrigation_requirements.csv     (450 samples, 6 columns)
│   └── field_productivity.csv          (350 samples, 6 columns)
│
├── generate_datasets.py        ← Regenerate all datasets
└── DATA_CATALOG.md            ← Detailed documentation
```

---

## 📊 Descriptive Statistics Datasets

### Summary Statistics Examples

**1. crop_yield_comparison.csv**
- Compare traditional vs. new farming method
- Use: t-test or Mann-Whitney U
- Expected: New method should show higher yield

**2. soil_properties.csv**
- 5 nutrients + pH across 5 fields
- Use: Correlation analysis, descriptive stats
- Expected: pH correlated with nutrient levels

**3. weather_data.csv**
- Temperature, rainfall, humidity, solar radiation
- Use: Correlation matrix, distributions
- Expected: Rainfall ↔ Humidity correlation

**4. pesticide_effectiveness.csv**
- 3 treatment groups: control, organic, chemical
- Use: ANOVA, hypothesis testing
- Expected: Chemical treatment most effective

**5. crop_varieties.csv**
- 4 varieties × 80 fields each
- Use: Group comparisons, ANOVA
- Expected: Variety B & C highest yields

---

## 📈 Time Series Datasets

### Characteristics

| Dataset | Samples | Period | Seasonality | Trend | Use Case |
|---------|---------|--------|-------------|-------|----------|
| daily_rainfall | 1460 | 4 years | Strong | Yes | Decomposition, forecasting |
| crop_yield_monthly | 60 | 5 years | Yearly | Yes | ARIMA(1,1,1) |
| temperature_daily | 730 | 2 years | Strong | Slight | ACF/PACF, seasonality |
| soil_moisture | 365 | 1 year | Yes | No | Pattern recognition |
| commodity_prices | 240 | 20 years | Yes | Yes | Long-range forecasting |
| white_noise | 200 | - | No | No | ADF test (PASS) |
| random_walk | 200 | - | No | Yes | ADF test (FAIL) |

### Example Tests

**Stationarity Testing (ADF)**:
- Run on `white_noise.csv` → Should **PASS** stationarity test
- Run on `random_walk.csv` → Should **FAIL** stationarity test

**Decomposition**:
- Best for: `daily_rainfall.csv`, `crop_yield_monthly.csv`
- Shows: Trend + Seasonal + Residual components

**ARIMA Modeling**:
- Use: `crop_yield_monthly.csv` (try ARIMA(1,1,1))
- Use: `commodity_prices.csv` (for long-range trends)

---

## 🔮 Predictive Modeling Datasets

### Model Performance Expectations

| Dataset | Target | Range | Expected R² | Key Features |
|---------|--------|-------|-------------|--------------|
| crop_yield_features | crop_yield_tons_per_ha | 10-120 | 0.7-0.8 | Rainfall, Temperature, Irrigation |
| pest_infestation | infestation_severity | 0-100 | 0.6-0.7 | Humidity, Temperature, Distance |
| irrigation_requirements | irrigation_hours | 0-100 | 0.7-0.8 | Moisture, Evapotranspiration, Forecast |
| field_productivity | productivity_score | 20-100 | 0.8-0.85 | Size, Soil Quality, Mechanization |

### Dataset Details

**1. crop_yield_features.csv** (500 samples)
- Features: rainfall, temperature, fertilizer, soil_ph, irrigation_hours, pesticide_cost
- Target: crop_yield_tons_per_ha
- Use: Standard regression example

**2. pest_infestation.csv** (400 samples)
- Features: humidity, temperature, rainfall_days, crop_age, field_proximity
- Target: infestation_severity (0-100)
- Use: Complex relationships, predictions

**3. irrigation_requirements.csv** (450 samples)
- Features: soil_moisture, evapotranspiration, rainfall_forecast, temperature, **crop_stage** (categorical)
- Target: irrigation_hours_needed
- Use: Categorical variable handling, real-world scenario

**4. field_productivity.csv** (350 samples)
- Features: field_size, soil_quality, years_managed, mechanization_level, crop_rotation
- Target: overall_productivity_score
- Use: Feature importance analysis (all positive correlations)

---

## 🧪 Testing Guide

### Module: Descriptive Statistics
1. **Summary Stats**: 
   - Load: crop_yield_comparison
   - Select: yield_treated
   - Expected: Mean ~58, Std ~11

2. **Correlation**:
   - Load: weather_data
   - Expected: High correlation between rainfall & humidity

3. **Hypothesis Testing**:
   - Load: crop_yield_comparison
   - Test: t-test (Traditional vs Treated)
   - Expected: p-value < 0.05 (significant)

### Module: Time Series Analysis
1. **ACF/PACF**:
   - Load: temperature_daily with 20 lags
   - Expected: Strong seasonal pattern peaks at lag 365

2. **Decomposition**:
   - Load: daily_rainfall, period=365
   - Expected: Clear monsoon seasonal component

3. **ARIMA**:
   - Load: crop_yield_monthly, order=(1,1,1)
   - Expected: AIC ~180-200

4. **Stationarity (ADF)**:
   - Load: white_noise → Should show p-value > 0.05
   - Load: random_walk → Should show p-value < 0.05

### Module: Predictive Modeling
1. **Linear Regression**:
   - Load: crop_yield_features
   - Expected: R² ~0.75

2. **Feature Importance**:
   - Load: field_productivity
   - Expected: soil_quality highest importance

---

## 🔄 Regenerating Datasets

All datasets are generated synthetically with seed=42 for reproducibility.

To regenerate with new random data:
```bash
python data/generate_datasets.py
```

The script will:
- Create new datasets with same structure
- Maintain realistic distributions
- Preserve seasonal patterns and trends
- Update all CSV files in their respective folders

---

## 📝 Notes

- All datasets are **synthetic** but based on **realistic agricultural data**
- Data formats match expected API inputs
- Multiple datasets per module allow testing different scenarios
- Time series data includes clear patterns for teaching purposes
- Predictive datasets have engineered relationships for learning

---

## ❓ Troubleshooting

**Dashboard not loading data?**
- Check: `http://localhost:8501` opens without errors
- Services running: `docker-compose ps` shows all containers UP

**API endpoints not responding?**
- Test: `curl http://localhost:8001/health`
- Check logs: `docker logs descriptive-stats-service`

**Data files not found?**
- Ensure datasets generated: `ls data/descriptive_stats/`
- Regenerate: `python data/generate_datasets.py`

---

## 📚 Full Documentation

See `data/DATA_CATALOG.md` for comprehensive details on each dataset including:
- Complete column descriptions
- Data ranges and distributions
- Expected relationships
- Use case examples
- Statistical characteristics

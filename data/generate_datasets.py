"""Generate comprehensive synthetic agricultural datasets for testing all modules."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

np.random.seed(42)

# ==============================================================================
# DESCRIPTIVE STATISTICS DATASETS
# ==============================================================================

def create_crop_yield_comparison():
    """Compare crop yields between two farming methods."""
    control = np.random.normal(50, 12, 150)  # Traditional method
    treated = np.random.normal(58, 11, 150)  # New fertilizer method

    df = pd.DataFrame({
        'yield_traditional': control,
        'yield_treated': treated
    })
    df.to_csv('data/descriptive_stats/crop_yield_comparison.csv', index=False)
    print(f"✓ crop_yield_comparison.csv ({len(df)} samples)")
    return df

def create_soil_properties():
    """Soil properties across different fields."""
    n = 250
    df = pd.DataFrame({
        'soil_ph': np.random.normal(6.5, 0.6, n),
        'nitrogen_ppm': np.random.gamma(3, 20, n),
        'phosphorus_ppm': np.random.gamma(2, 15, n),
        'potassium_ppm': np.random.gamma(4, 30, n),
        'organic_matter_percent': np.random.gamma(2, 1.5, n),
        'field_id': np.repeat(range(1, 6), n // 5)
    })
    df.to_csv('data/descriptive_stats/soil_properties.csv', index=False)
    print(f"✓ soil_properties.csv ({len(df)} samples)")
    return df

def create_weather_correlation():
    """Weather variables for correlation analysis."""
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    # Weather variables with some correlation
    temp = np.random.normal(25, 8, n)
    rainfall = 50 + 2 * (temp - 25) + np.random.normal(0, 20, n)
    rainfall = np.clip(rainfall, 0, 300)
    humidity = 60 + 0.5 * rainfall + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 100)

    df = pd.DataFrame({
        'date': dates,
        'temperature_c': temp,
        'rainfall_mm': rainfall,
        'humidity_percent': humidity,
        'solar_radiation': np.random.gamma(2, 5, n)
    })
    df.to_csv('data/descriptive_stats/weather_data.csv', index=False)
    print(f"✓ weather_data.csv ({len(df)} samples)")
    return df

def create_pesticide_effectiveness():
    """Pesticide effectiveness data for hypothesis testing."""
    # Control group: minimal pest damage
    control_damage = np.random.beta(3, 8, 120) * 100  # 0-100% damage scale

    # Treatment groups
    organic_damage = np.random.beta(4, 6, 120) * 100
    chemical_damage = np.random.beta(5, 15, 120) * 100

    df = pd.DataFrame({
        'control_damage_percent': control_damage,
        'organic_treatment_damage': organic_damage,
        'chemical_treatment_damage': chemical_damage
    })
    df.to_csv('data/descriptive_stats/pesticide_effectiveness.csv', index=False)
    print(f"✓ pesticide_effectiveness.csv ({len(df)} samples)")
    return df

def create_crop_varieties():
    """Yield comparison across crop varieties."""
    varieties = ['Variety_A', 'Variety_B', 'Variety_C', 'Variety_D']
    data = []

    # Each variety planted in 80 fields
    for i, variety in enumerate(varieties):
        yields = np.random.normal(50 + i*5, 8, 80)
        data.extend(zip([variety]*80, yields))

    df = pd.DataFrame(data, columns=['variety', 'yield_kg_per_hectare'])
    df.to_csv('data/descriptive_stats/crop_varieties.csv', index=False)
    print(f"✓ crop_varieties.csv ({len(df)} samples)")
    return df

# ==============================================================================
# TIME SERIES DATASETS
# ==============================================================================

def create_daily_rainfall():
    """Daily rainfall with seasonal pattern."""
    n = 1460  # 4 years
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    t = np.arange(n)
    # Strong seasonal pattern (monsoon season)
    seasonal = 40 * np.sin(2 * np.pi * t / 365)
    trend = 0.02 * t
    noise = np.random.normal(0, 15, n)

    rainfall = 50 + seasonal + trend + noise
    rainfall = np.clip(rainfall, 0, 200)

    df = pd.DataFrame({
        'date': dates,
        'rainfall_mm': rainfall
    })
    df.to_csv('data/time_series/daily_rainfall.csv', index=False)
    print(f"✓ daily_rainfall.csv ({len(df)} samples)")
    return df

def create_crop_yield_timeseries():
    """Monthly crop yield over 5 years."""
    n = 60  # 5 years
    dates = pd.date_range('2019-01-01', periods=n, freq='MS')

    t = np.arange(n)
    seasonal = 8 * np.sin(2 * np.pi * t / 12)  # Yearly cycle
    trend = 0.5 * t  # Improving yields
    cycle = 5 * np.sin(2 * np.pi * t / 24)  # 2-year cycle
    noise = np.random.normal(0, 3, n)

    yield_values = 50 + seasonal + trend + cycle + noise

    df = pd.DataFrame({
        'date': dates,
        'crop_yield_tons': yield_values
    })
    df.to_csv('data/time_series/crop_yield_monthly.csv', index=False)
    print(f"✓ crop_yield_monthly.csv ({len(df)} samples)")
    return df

def create_temperature_timeseries():
    """Daily temperature with clear seasonality."""
    n = 730  # 2 years
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    t = np.arange(n)
    seasonal = 15 * np.sin(2 * np.pi * t / 365)  # Strong yearly seasonality
    trend = 0.02 * t  # Slight warming trend
    noise = np.random.normal(0, 2, n)

    temp = 20 + seasonal + trend + noise

    df = pd.DataFrame({
        'date': dates,
        'temperature_c': temp
    })
    df.to_csv('data/time_series/temperature_daily.csv', index=False)
    print(f"✓ temperature_daily.csv ({len(df)} samples)")
    return df

def create_soil_moisture_timeseries():
    """Soil moisture with irrigation pattern."""
    n = 365
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    t = np.arange(n)

    # Base moisture decreasing trend during dry season
    seasonal = 30 * np.sin(2 * np.pi * t / 365)

    # Irrigation pulses (every 7-10 days in growing season)
    irrigation_effect = np.zeros(n)
    for i in range(0, n, 8):
        if i < n:
            irrigation_effect[i:min(i+3, n)] += 20

    noise = np.random.normal(0, 3, n)
    moisture = 40 + seasonal + irrigation_effect + noise
    moisture = np.clip(moisture, 20, 80)

    df = pd.DataFrame({
        'date': dates,
        'soil_moisture_percent': moisture
    })
    df.to_csv('data/time_series/soil_moisture_daily.csv', index=False)
    print(f"✓ soil_moisture_daily.csv ({len(df)} samples)")
    return df

def create_agricultural_price_timeseries():
    """Agricultural commodity prices over time."""
    n = 240  # 20 years of monthly data
    dates = pd.date_range('2004-01-01', periods=n, freq='MS')

    t = np.arange(n)

    # Base price with trend
    base_price = 200 + 0.5 * t

    # Seasonal price variation
    seasonal = 30 * np.sin(2 * np.pi * t / 12)

    # Market cycles
    cycle = 50 * np.sin(2 * np.pi * t / 48)

    # Random shocks
    shocks = np.random.normal(0, 20, n)

    price = base_price + seasonal + cycle + shocks
    price = np.clip(price, 50, 500)

    df = pd.DataFrame({
        'date': dates,
        'crop_price_per_unit': price
    })
    df.to_csv('data/time_series/commodity_prices.csv', index=False)
    print(f"✓ commodity_prices.csv ({len(df)} samples)")
    return df

def create_stationary_series():
    """Purely stationary time series (white noise)."""
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    # Pure white noise (stationary)
    values = np.random.normal(0, 5, n)

    df = pd.DataFrame({
        'date': dates,
        'white_noise': values
    })
    df.to_csv('data/time_series/white_noise.csv', index=False)
    print(f"✓ white_noise.csv ({len(df)} samples)")
    return df

def create_nonstationary_series():
    """Non-stationary time series with trend and drift."""
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    # Random walk with drift (non-stationary)
    t = np.arange(n)
    drift = 0.1 * t
    random_walk = np.cumsum(np.random.normal(0, 2, n))
    values = drift + random_walk

    df = pd.DataFrame({
        'date': dates,
        'random_walk': values
    })
    df.to_csv('data/time_series/random_walk.csv', index=False)
    print(f"✓ random_walk.csv ({len(df)} samples)")
    return df

# ==============================================================================
# PREDICTIVE MODELING DATASETS
# ==============================================================================

def create_crop_yield_prediction():
    """Features and target for crop yield prediction."""
    n = 500

    rainfall = np.random.gamma(2, 25, n)  # mm
    temperature = np.random.normal(25, 6, n)  # Celsius
    fertilizer = np.random.exponential(80, n)  # kg/ha
    soil_ph = np.random.normal(6.5, 0.7, n)
    irrigation_hours = np.random.uniform(0, 200, n)
    pesticide_cost = np.random.uniform(0, 500, n)

    # Yield is function of inputs
    yield_target = (
        0.8 * rainfall +
        4 * temperature -
        0.15 * (temperature - 25)**2 +
        0.15 * fertilizer +
        8 * soil_ph +
        0.25 * irrigation_hours +
        0.01 * pesticide_cost +
        np.random.normal(0, 15, n)
    )
    yield_target = np.clip(yield_target, 10, 120)

    df = pd.DataFrame({
        'rainfall_mm': rainfall,
        'temperature_c': temperature,
        'fertilizer_kg_per_ha': fertilizer,
        'soil_ph': soil_ph,
        'irrigation_hours': irrigation_hours,
        'pesticide_cost': pesticide_cost,
        'crop_yield_tons_per_ha': yield_target
    })
    df.to_csv('data/predictive_modeling/crop_yield_features.csv', index=False)
    print(f"✓ crop_yield_features.csv ({len(df)} samples)")
    return df

def create_pest_prediction():
    """Features for predicting pest infestation."""
    n = 400

    humidity = np.random.uniform(30, 95, n)
    temperature = np.random.normal(25, 7, n)
    rainfall_days = np.random.poisson(5, n)
    crop_age_days = np.random.uniform(30, 150, n)
    field_proximity_km = np.random.uniform(0, 50, n)

    # Pest infestation likelihood (0-100)
    infestation = (
        0.5 * humidity +
        -0.3 * np.abs(temperature - 28) +
        2 * rainfall_days +
        0.1 * crop_age_days +
        -0.2 * field_proximity_km +
        np.random.normal(0, 8, n)
    )
    infestation = np.clip(infestation, 0, 100)

    df = pd.DataFrame({
        'humidity_percent': humidity,
        'temperature_c': temperature,
        'rainfall_days': rainfall_days,
        'crop_age_days': crop_age_days,
        'field_proximity_km': field_proximity_km,
        'infestation_severity': infestation
    })
    df.to_csv('data/predictive_modeling/pest_infestation.csv', index=False)
    print(f"✓ pest_infestation.csv ({len(df)} samples)")
    return df

def create_irrigation_prediction():
    """Features for predicting irrigation needs."""
    n = 450

    soil_moisture = np.random.uniform(20, 80, n)
    evapotranspiration = np.random.gamma(2, 3, n)  # mm/day
    rainfall_forecast = np.random.exponential(5, n)  # mm
    temperature = np.random.normal(25, 8, n)
    crop_stage = np.random.choice(['seedling', 'vegetative', 'flowering', 'maturity'], n)

    # Irrigation need in hours
    stage_multiplier = {'seedling': 1, 'vegetative': 1.5, 'flowering': 2, 'maturity': 0.8}
    stage_mult_values = np.array([stage_multiplier[s] for s in crop_stage])

    irrigation_need = (
        30 * (1 - soil_moisture/100) +
        10 * evapotranspiration +
        -0.5 * rainfall_forecast +
        2 * (temperature - 20) +
        20 * stage_mult_values +
        np.random.normal(0, 5, n)
    )
    irrigation_need = np.clip(irrigation_need, 0, 100)

    df = pd.DataFrame({
        'soil_moisture_percent': soil_moisture,
        'evapotranspiration_mm_per_day': evapotranspiration,
        'rainfall_forecast_mm': rainfall_forecast,
        'temperature_c': temperature,
        'crop_stage': crop_stage,
        'irrigation_hours_needed': irrigation_need
    })
    df.to_csv('data/predictive_modeling/irrigation_requirements.csv', index=False)
    print(f"✓ irrigation_requirements.csv ({len(df)} samples)")
    return df

def create_field_productivity():
    """Features affecting overall field productivity."""
    n = 350

    field_size_ha = np.random.uniform(1, 50, n)
    soil_quality_score = np.random.uniform(1, 10, n)
    years_managed = np.random.uniform(1, 30, n)
    mechanization_level = np.random.uniform(0, 100, n)  # 0=manual, 100=fully automated
    crop_rotation_practiced = np.random.choice([0, 1], n)  # 0=no, 1=yes

    # Productivity score
    productivity = (
        30 +
        1.5 * field_size_ha +
        8 * soil_quality_score +
        0.5 * years_managed +
        0.3 * mechanization_level +
        15 * crop_rotation_practiced +
        np.random.normal(0, 10, n)
    )
    productivity = np.clip(productivity, 20, 100)

    df = pd.DataFrame({
        'field_size_ha': field_size_ha,
        'soil_quality_score': soil_quality_score,
        'years_managed': years_managed,
        'mechanization_level': mechanization_level,
        'crop_rotation_practiced': crop_rotation_practiced,
        'overall_productivity_score': productivity
    })
    df.to_csv('data/predictive_modeling/field_productivity.csv', index=False)
    print(f"✓ field_productivity.csv ({len(df)} samples)")
    return df

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("GENERATING AGRICULTURAL DATASETS FOR ALL MODULES")
    print("="*70)

    print("\n📊 DESCRIPTIVE STATISTICS DATASETS:")
    print("-" * 70)
    create_crop_yield_comparison()
    create_soil_properties()
    create_weather_correlation()
    create_pesticide_effectiveness()
    create_crop_varieties()

    print("\n📈 TIME SERIES DATASETS:")
    print("-" * 70)
    create_daily_rainfall()
    create_crop_yield_timeseries()
    create_temperature_timeseries()
    create_soil_moisture_timeseries()
    create_agricultural_price_timeseries()
    create_stationary_series()
    create_nonstationary_series()

    print("\n🔮 PREDICTIVE MODELING DATASETS:")
    print("-" * 70)
    create_crop_yield_prediction()
    create_pest_prediction()
    create_irrigation_prediction()
    create_field_productivity()

    print("\n" + "="*70)
    print("✓ ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

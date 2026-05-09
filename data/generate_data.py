"""Generate synthetic agricultural datasets for testing."""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_descriptive_data():
    """Generate dataset for descriptive statistics examples."""
    np.random.seed(42)
    n_samples = 200

    data = {
        'crop_yield': np.random.normal(50, 15, n_samples),  # kg/ha
        'rainfall': np.random.gamma(2, 20, n_samples),  # mm
        'temperature': np.random.normal(25, 5, n_samples),  # Celsius
        'fertilizer': np.random.exponential(100, n_samples),  # kg/ha
        'soil_ph': np.random.normal(6.5, 0.5, n_samples)
    }

    df = pd.DataFrame(data)
    df.to_csv('data/descriptive_sample.csv', index=False)
    return df

def generate_time_series_data():
    """Generate time series dataset with seasonal patterns."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    # Seasonal pattern + trend + noise
    t = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    trend = 0.02 * t
    noise = np.random.normal(0, 2, len(dates))
    crop_yield = 50 + seasonal + trend + noise

    df = pd.DataFrame({
        'date': dates,
        'crop_yield': crop_yield
    })

    df.to_csv('data/time_series_sample.csv', index=False)
    return df

def generate_predictive_data():
    """Generate dataset for predictive modeling."""
    np.random.seed(42)
    n_samples = 300

    features = {
        'rainfall': np.random.gamma(2, 20, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'fertilizer': np.random.exponential(100, n_samples),
        'soil_ph': np.random.normal(6.5, 0.5, n_samples),
        'irrigation': np.random.uniform(0, 200, n_samples)
    }

    # Target: crop yield with some relationship to features
    yield_target = (
        0.5 * features['rainfall'] +
        2 * features['temperature'] -
        5 * (features['temperature'] - 25)**2 +
        0.1 * features['fertilizer'] +
        10 * features['soil_ph'] +
        0.3 * features['irrigation'] +
        np.random.normal(0, 10, n_samples)
    )

    features['crop_yield'] = np.clip(yield_target, 0, 100)
    df = pd.DataFrame(features)
    df.to_csv('data/predictive_sample.csv', index=False)
    return df

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    print("Generating descriptive statistics data...")
    generate_descriptive_data()
    print("Generating time series data...")
    generate_time_series_data()
    print("Generating predictive modeling data...")
    generate_predictive_data()
    print("✓ All datasets generated")

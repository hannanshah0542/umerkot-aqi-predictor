"""
Step 4: Feature Pipeline

Loads backfilled historical data, engineers features (lags, rolling averages,
time-based features), creates prediction targets (AQI 24h/48h/72h ahead),
and pushes everything to HuggingFace Hub as a feature store.

Features computed:
  - Lag features: AQI from 6h, 12h, 24h ago
  - Rolling averages: 6h, 12h, 24h rolling means (AQI and pollutants)
  - Time features: hour of day, day of week, month
  - Derived: AQI change rate (current vs 24h ago)

Targets (what the model predicts):
  - aqi_target_24h: AQI 24 hours in the future
  - aqi_target_48h: AQI 48 hours in the future
  - aqi_target_72h: AQI 72 hours in the future

Run:
    python step4_feature_pipeline.py
"""

import os
import glob
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi

# ---- CONFIG ----
DATA_FOLDER = "Data"
CITY_NAME = "Umerkot"
USERNAME = "ALIkjjnskdjc"  # your HuggingFace username
REPO_NAME = "umerkot-aqi-features"  # dataset name in HF
# -----------------


#Find and load the most recent historical_*.csv file from the backfill step.
def load_latest_backfill(folder):

    pattern = f"{folder}/historical_*.csv"
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No backfilled CSV found in {folder}. Run step2_backfill_data.py first."
        )

    latest_file = max(files, key=lambda f: f)
    print(f"Loading: {latest_file}")
    df = pd.read_csv(latest_file, parse_dates=["time"])
    return df


#Compute lag features, rolling averages, and time-based features.
def engineer_features(data):

    data = data.sort_values("time").reset_index(drop=True)

    print("Engineering features...")

    #Time-based features
    data["hour"] = data["time"].dt.hour
    data["day_of_week"] = data["time"].dt.dayofweek
    data["month"] = data["time"].dt.month

    #Lag features for AQI, "What was AQI 6/12/24 hours ago?"
    data["aqi_lag_6h"] = data["us_aqi"].shift(6)
    data["aqi_lag_12h"] = data["us_aqi"].shift(12)
    data["aqi_lag_24h"] = data["us_aqi"].shift(24)

    #Rolling averages, Smooth out hourly noise to see real trend
    data["aqi_rolling_6h"] = data["us_aqi"].rolling(6, min_periods=1).mean()
    data["aqi_rolling_12h"] = data["us_aqi"].rolling(12, min_periods=1).mean()
    data["aqi_rolling_24h"] = data["us_aqi"].rolling(24, min_periods=1).mean()

    # Pollutant rolling averages (useful for training)
    data["pm25_rolling_6h"] = data["pm2_5"].rolling(6, min_periods=1).mean()
    data["pm25_rolling_24h"] = data["pm2_5"].rolling(24, min_periods=1).mean()
    data["pm10_rolling_6h"] = data["pm10"].rolling(6, min_periods=1).mean()
    data["pm10_rolling_24h"] = data["pm10"].rolling(24, min_periods=1).mean()

    # ---- Derived features ----
    # AQI change: how much did it go up/down in the last 24h?
    data["aqi_change_24h"] = data["us_aqi"] - data["aqi_lag_24h"]

    print(f"✓ Engineered {len(data.columns)} total columns")
    return data


#prediction targets: AQI values 24h/48h/72h in the future.
def create_targets(data):

    print("Creating prediction targets...")

    # Shift forward (negative shift = future values)
    data["aqi_target_24h"] = data["us_aqi"].shift(-24)
    data["aqi_target_48h"] = data["us_aqi"].shift(-48)
    data["aqi_target_72h"] = data["us_aqi"].shift(-72)

    print("✓ Targets created (24h, 48h, 72h ahead)")
    return data


#Drop rows with missing values
def clean_features(data):
 
    print(f"Rows before cleaning: {len(data)}")

    # Most NaNs come from shifts and rolling windows at the start/end of timeseries
    data_clean = data.dropna()

    print(f"Rows after cleaning: {len(data_clean)}")
    print(f"Dropped {len(data) - len(data_clean)} incomplete rows")

    return data_clean.reset_index(drop=True)


#Save features to HuggingFace Hub as a dataset.
def upload_to_huggingface(data, username, repo_name, api_key):
    """
    Save features to HuggingFace Hub as a dataset.
    Creates the repo if it doesn't exist.

    Args:
        data (pd.DataFrame): engineered features + targets.
        username (str): your HuggingFace username.
        repo_name (str): name of the dataset repo.
        api_key (str): HuggingFace API token.
    """
    print("Uploading to HuggingFace Hub...")

    # Save to CSV locally first
    local_file = "aqi_features.csv"
    data.to_csv(local_file, index=False)
    print(f"✓ Saved locally: {local_file}")

    # Create API connection
    api = HfApi(token=api_key)
    repo_id = f"{username}/{repo_name}"

    try:
        print(f"Uploading to {repo_id}...")

        # Now upload the file
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo="features.csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload features: {len(data)} rows, {len(data.columns)} columns"
        )
        print(f"✓ Uploaded to {repo_id}/features.csv")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise


#Print a summary of the engineered features and targets.
def summary_stats(data):
 
    print("\n--- Feature Engineering Summary ---")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    print(f"\nAQI stats (current):")
    print(f"  Mean: {data['us_aqi'].mean():.1f}")
    print(f"  Std: {data['us_aqi'].std():.1f}")
    print(f"  Min: {data['us_aqi'].min():.1f}")
    print(f"  Max: {data['us_aqi'].max():.1f}")
    print(f"\nPrediction targets:")
    print(f"  aqi_target_24h: mean={data['aqi_target_24h'].mean():.1f}")
    print(f"  aqi_target_48h: mean={data['aqi_target_48h'].mean():.1f}")
    print(f"  aqi_target_72h: mean={data['aqi_target_72h'].mean():.1f}")
    print(f"\nFeatures engineered:")
    feature_cols = [c for c in data.columns if c not in [
        'time', 'city', 'fetched_at', 'us_aqi',
        'aqi_target_24h', 'aqi_target_48h', 'aqi_target_72h'
    ]]
    for col in sorted(feature_cols):
        print(f"  - {col}")


if __name__ == "__main__":
    # Load environment
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")

    if not api_key:
        print("❌ ERROR: HUGGINGFACE_API_KEY not found in .env file")
        exit(1)

    # Load, engineer, and prepare
    df = load_latest_backfill(DATA_FOLDER)
    df = engineer_features(df)
    df = create_targets(df)
    df = clean_features(df)

    # Show what we built
    summary_stats(df)

    # Upload to HuggingFace
    upload_to_huggingface(df, USERNAME, REPO_NAME, api_key)

    print("\n✓ Feature pipeline complete!")
    print(f"✓ Features ready for training in {USERNAME}/{REPO_NAME}")
"""
Step 5: Training Pipeline

Fetches engineered features from HuggingFace Hub, trains multiple ML models,
evaluates them, and saves the best one back to HF Hub as a model artifact.

Models trained:
  1. Random Forest (fast, good for time-series)
  2. Ridge Regression (linear baseline)
  3. XGBoost (gradient boosting, often best)

Evaluation metrics:
  - RMSE: root mean square error (penalizes large mistakes)
  - MAE: mean absolute error (average mistake in AQI points)
  - R²: coefficient of determination (0-1, 1 is perfect)

Run:
    python step5_training_pipeline.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost")

# ---- CONFIG ----
USERNAME = "ALIkjjnskdjc"  # your HuggingFace username
REPO_NAME = "umerkot-aqi-features"  # where features are stored
TARGET_COLUMN = "aqi_target_24h"  # predicting AQI 24h ahead
TEST_SIZE = 0.2  # 80% train, 20% test
RANDOM_STATE = 42
# -----------------


def download_features_from_hf(username, repo_name, api_key):
    """
    Download the engineered features from HuggingFace Hub.

    Args:
        username (str): HF username where dataset is stored.
        repo_name (str): HF dataset name.
        api_key (str): HF API token.

    Returns:
        pd.DataFrame: the features + targets.
    """
    print("Downloading features from HuggingFace...")

    repo_id = f"{username}/{repo_name}"

    try:
        # Download the CSV file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="features.csv",
            repo_type="dataset",
            token=api_key
        )
        print(f"✓ Downloaded from {repo_id}")

        # Load into pandas
        df = pd.read_csv(file_path, parse_dates=["time"])
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print(f"  Make sure {repo_id} exists on HuggingFace")
        raise


def prepare_training_data(df, target_col):
    """
    Split features and targets, handle missing values, create train/test sets.

    Args:
        df (pd.DataFrame): full dataset with features and targets.
        target_col (str): name of the target column to predict.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("\nPreparing training data...")

    # Drop non-feature columns
    non_feature_cols = [
        'time', 'city', 'fetched_at', 'us_aqi',
        'aqi_target_24h', 'aqi_target_48h', 'aqi_target_72h'
    ]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    print(f"Using {len(feature_cols)} features for training")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop any remaining NaNs in features or target
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"After cleanup: {len(X)} rows for training")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate them.

    Args:
        X_train, X_test, y_train, y_test: train/test data.

    Returns:
        dict: results = {model_name: {model, metrics}}
    """
    print("\nTraining models...")

    results = {}

    # ---- Model 1: Random Forest ----
    print("\n1. Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results["Random Forest"] = {
        "model": rf,
        "predictions": y_pred_rf,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "mae": mean_absolute_error(y_test, y_pred_rf),
        "r2": r2_score(y_test, y_pred_rf)
    }
    print(f"  RMSE: {results['Random Forest']['rmse']:.2f}")
    print(f"  MAE: {results['Random Forest']['mae']:.2f}")
    print(f"  R²: {results['Random Forest']['r2']:.3f}")

    # ---- Model 2: Ridge Regression ----
    print("\n2. Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    results["Ridge"] = {
        "model": ridge,
        "predictions": y_pred_ridge,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        "mae": mean_absolute_error(y_test, y_pred_ridge),
        "r2": r2_score(y_test, y_pred_ridge)
    }
    print(f"  RMSE: {results['Ridge']['rmse']:.2f}")
    print(f"  MAE: {results['Ridge']['mae']:.2f}")
    print(f"  R²: {results['Ridge']['r2']:.3f}")

    # ---- Model 3: XGBoost (if available) ----
    if HAS_XGBOOST:
        print("\n3. XGBoost...")
        xgb = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        results["XGBoost"] = {
            "model": xgb,
            "predictions": y_pred_xgb,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "mae": mean_absolute_error(y_test, y_pred_xgb),
            "r2": r2_score(y_test, y_pred_xgb)
        }
        print(f"  RMSE: {results['XGBoost']['rmse']:.2f}")
        print(f"  MAE: {results['XGBoost']['mae']:.2f}")
        print(f"  R²: {results['XGBoost']['r2']:.3f}")
    else:
        print("\n3. XGBoost - skipped (not installed)")

    return results


def compare_models(results):
    """
    Compare all trained models and pick the best.

    Args:
        results (dict): trained models and their metrics.

    Returns:
        tuple: (best_model_name, best_model_obj, best_results_dict)
    """
    print("\n--- Model Comparison ---")

    # Create comparison table
    comparison = []
    for name, data in results.items():
        comparison.append({
            "Model": name,
            "RMSE": f"{data['rmse']:.2f}",
            "MAE": f"{data['mae']:.2f}",
            "R²": f"{data['r2']:.3f}"
        })

    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))

    # Find best (lowest RMSE)
    best_model_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_result = results[best_model_name]

    print(f"\n✓ Best model: {best_model_name} (RMSE: {best_result['rmse']:.2f})")

    # Save comparison to CSV for the report
    comparison_df.to_csv("model_comparison.csv", index=False)
    print("✓ Saved model comparison to model_comparison.csv")

    return best_model_name, best_result["model"], best_result


def save_model_to_hf(model, model_name, username, repo_name, api_key, metrics):
    """
    Save the trained model to HuggingFace Hub as a pickle file.

    Args:
        model: the trained sklearn/xgb model object.
        model_name (str): name of the model (for filename).
        username (str): HF username.
        repo_name (str): HF repo name (will add "-models" suffix).
        api_key (str): HF API token.
        metrics (dict): model performance metrics.
    """
    print(f"\nSaving {model_name} to HuggingFace...")

    # Save model locally as pickle
    local_file = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(local_file, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Saved locally: {local_file}")

    # Upload to HF
    api = HfApi(token=api_key)
    model_repo = f"{username}/{repo_name}-models"

    try:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=local_file,
            repo_id=model_repo,
            repo_type="dataset",
            commit_message=f"{model_name} - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )
        print(f"✓ Uploaded to {model_repo}/{local_file}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        print(f"  Model saved locally as {local_file}")
        print(f"  Can upload manually later")


if __name__ == "__main__":
    # Load environment
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")

    if not api_key:
        print("❌ ERROR: HUGGINGFACE_API_KEY not found in .env")
        exit(1)

    # Download features from HF Hub
    df = download_features_from_hf(USERNAME, REPO_NAME, api_key)

    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(df, TARGET_COLUMN)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Compare and pick best
    best_name, best_model, best_metrics = compare_models(results)

    # Save best model to HF
    save_model_to_hf(best_model, best_name, USERNAME, REPO_NAME, api_key, best_metrics)

    print("\n✓ Training pipeline complete!")
    print(f"✓ Best model: {best_name}")
    print(f"  RMSE: {best_metrics['rmse']:.2f}")
    print(f"  MAE: {best_metrics['mae']:.2f}")
    print(f"  R²: {best_metrics['r2']:.3f}")

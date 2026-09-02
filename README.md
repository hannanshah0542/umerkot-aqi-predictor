# Umerkot AQI Predictor

An end-to-end MLOps pipeline that fetches live weather and air-quality data for Umerkot, Pakistan, engineers predictive features, trains a machine learning model, and serves 24h/48h/72h AQI forecasts through an interactive dashboard.

## Live Demo
Run locally with `streamlit run step_6_dashborad.py` (see Setup below).

## Architecture

```
Open-Meteo API
      |
      v
[1] Data Fetch  ---->  [2] Historical Backfill  ---->  [3] EDA
                                                             |
                                                             v
                                              [4] Feature Engineering
                                                             |
                                                             v
                                              HuggingFace Hub (Feature Store)
                                                             |
                                                             v
                                              [5] Model Training (RF, Ridge, XGBoost)
                                                             |
                                                             v
                                              HuggingFace Hub (Model Registry)
                                                             |
                                                             v
                                              [6] Streamlit Dashboard
```

## Project Structure

| File | Purpose |
|---|---|
| `part_1_data_Fetch.py` | Fetches live weather + air quality data from Open-Meteo API |
| `part_2_backfill_data.py` | Backfills 90 days of historical hourly data |
| `part_3_EDA.py` | Exploratory data analysis: AQI categories, trends, correlations |
| `step_4_pipeline.py` | Feature engineering (lags, rolling averages, time features) + upload to HuggingFace Hub |
| `step_5_pipeline_traning.py` | Trains Random Forest, Ridge, and XGBoost; selects best model |
| `step_6_dashborad.py` | Streamlit dashboard: live AQI, 3-day forecast, pollutants, weather, trends |

## Data Source
[Open-Meteo](https://open-meteo.com/) — free weather and air quality API, no key required.

**Location:** Umerkot, Pakistan (25.3614°N, 69.7436°E)

## Features Engineered
- Time-based: hour, day of week, month
- Lag features: AQI 6h / 12h / 24h ago
- Rolling averages: 6h / 12h / 24h (AQI, PM2.5, PM10)
- Derived: 24h AQI change rate

## Model Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 4.18 | 2.83 | 0.943 |
| Ridge Regression | 10.83 | 8.42 | 0.618 |
| **XGBoost (selected)** | **3.88** | **2.58** | **0.951** |

Trained on 2,064 hours of historical data, 26 engineered features, 80/20 train-test split.

## Dashboard Features
- Real-time AQI gauge with EPA color-coded categories
- 3-day AQI forecast (24h / 48h / 72h) with confidence intervals
- Live pollutant concentrations (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- Current weather conditions (temperature, humidity, pressure, wind)
- 24-hour AQI trend chart
- Model performance metrics
- Health guidance based on current air quality

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/hannanshah0542/umerkot-aqi-predictor.git
cd umerkot-aqi-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly seaborn matplotlib huggingface_hub python-dotenv
```

### 3. Add your HuggingFace API key
Create a `.env` file in the project root:
```
HUGGINGFACE_API_KEY=your_token_here
```

### 4. Run the pipeline
```bash
python part_1_data_Fetch.py
python part_2_backfill_data.py
python part_3_EDA.py
python step_4_pipeline.py
python step_5_pipeline_traning.py
streamlit run step_6_dashborad.py
```

## Feature Store & Model Registry
Engineered features and the trained model are stored on [HuggingFace Hub](https://huggingface.co/datasets/ALIkjjnskdjc/umerkot-aqi-features), serving as the project's feature store.

## Automation
Feature pipeline runs automatically via GitHub Actions (see `.github/workflows/`).

## Author
Built as part of an internship MLOps project — full pipeline from raw API data to a deployed prediction dashboard.
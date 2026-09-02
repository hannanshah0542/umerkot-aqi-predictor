"""
Step:2 Backfill historical weather and air quality data.

"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone


# Configration Block

CITY_NAME = "Umerkot"
LATITUDE = "25.3614"
LONGITUDE = "69.7436"
BACKFILL_DAYS = 90
DATA_FOLDER = "E:\AQI\Data"


# URL for Weather and Air Data

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# Function to Get data range

def get_data_range(days_back):
    end_data = datetime.now().date() - timedelta(days=2)
    start_date = end_data - timedelta(days=days_back)
    return start_date.isoformat(), end_data.isoformat()


# Function to Fetch weather data

def weather_history_data(lat, long, start_date, end_date):
    history_params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ],
        "timezone": "auto",
    }

# Fetch data and crash the system if can't get it in 40 sec due to some issue

    fetch_range = requests.get(WEATHER_ARCHIVE_URL, params=history_params, timeout= 40)
    fetch_range.raise_for_status()
    return fetch_range.json()

# Function to Fetch Air Quality data

def air_quality_history(lat,long,start_date,end_date):
    aq_history_params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
        ],
        timezone: "auto"
    }


    aq_data = requests.get(AIR_QUALITY_URL, params=aq_history_params, timeout= 40)
    aq_data.raise_for_status()
    return aq_data.json()

#

def to_dataframe(weather_json, aq_json):
    weather_df = pd.DataFrame(weather_json["hourly"])
    aq_df = pd.DataFrame(aq_json["hourly"])

    df = pd.merge(weather_df, aq_df, on="time", how= "inner")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = CITY_NAME
    df["fetched_at"] = datetime.now(timezone.utc)
    return df

#..........................

if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)

    start_date, end_date = get_data_range(BACKFILL_DAYS)
    print(f"Backfilling {CITY_NAME} from {start_date} to {end_date}...")

    weather_json = weather_history_data(LATITUDE,LATITUDE, start_date, end_date)
    aq_json = air_quality_history(LATITUDE, LATITUDE, start_date, end_date)

    df = to_dataframe(weather_json, aq_json)

    print(df.head(10))
    print(f"\nFetched {len(df)} hourly rows spanning {start_date} to {end_date}.")


    out_path = f"{DATA_FOLDER}/historical_{CITY_NAME.lower()}_{start_date}_to_{end_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


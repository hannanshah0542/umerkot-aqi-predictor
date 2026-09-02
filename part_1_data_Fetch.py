"""
Part 1: Fetch raw weather + air quality data.

"""

import pandas as pd
import requests
from datetime import datetime, timezone

# Location Info 

CITY_NAME = "Umerkot"
LATITUDE = "25.3614"
LONGITUDE = "69.7436"

# URL to fetch data

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# Function to get weather data

def fetch_weather_data(lati,longi, forecast_days=4):
    weather_params={
        "latitude":lati,
        "longitude":longi,
        "hourly":[
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ],
        "timezone":"auto",
        "forecast_days":forecast_days
    }

# Fetch data and crash after 20 sec in case of any error

    fetch_weather = requests.get(WEATHER_URL, params=weather_params, timeout=20)
    fetch_weather.raise_for_status()
    return fetch_weather.json()

# Function to get Air Quality Data

def air_quality(lati, longi, forecast_days=4):
    aq_params={
        "latitude":lati,
        "longitude":longi,
        "hourly":[
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
        ],
        "timezone":"auto",
        "forecast_days":forecast_days
    }
# Fetch data and crash after 20 sec in case of any error

    fetch_aq = requests.get(AIR_QUALITY_URL,params=aq_params,timeout=20)
    fetch_aq.raise_for_status()
    return fetch_aq.json()

# Create Data Frame from 2 functions above

def to_dataframe(weather_json, air_json):
    weather_df = pd.DataFrame(weather_json["hourly"])
    aq_df = pd.DataFrame(air_json["hourly"])

# merging both weather df and air df based on time column
    df = pd.merge(weather_df, aq_df, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = CITY_NAME
    df["fetched_at"] = datetime.now(timezone.utc)
    return df

#.........................................................
if __name__ == "__main__":
    print(f"Fetching data for {CITY_NAME} ({LATITUDE}, {LONGITUDE})...")

    weather_json = fetch_weather_data(LATITUDE, LONGITUDE)
    aq_json = air_quality(LATITUDE, LONGITUDE)

    df = to_dataframe(weather_json, aq_json)

    print(df.head(10))
    print(f"\nFetched {len(df)} hourly rows, columns: {list(df.columns)}")

    datapath = f"E:/AQI/Data/raw_data_{CITY_NAME.lower()}_{datetime.now().date()}.csv"
    df.to_csv(datapath, index=False)
    print(f"Saved to {datapath}")


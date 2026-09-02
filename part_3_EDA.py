#Step 3: Exploratory Data Analysis (EDA) on the backfilled AQI data.

import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns 


DATA_FLODER = "E:/AQI/Data"
CITY_NAME = "Umerkot"
ALERT_VALUE = 150

# Search for the file

def latest_backfill_data(folder):
    pattren = f"{folder}/historical_*.csv"
    files = glob.glob(pattren)

#Shows error if file isn't found

    if not files:
        raise FileNotFoundError(
            f"No Historical CSV found in {folder}. Run step2_backfill_data.py first.",
        )

# max seaches for max number in file and key=lambda f: f tells its a text file
    latest_data = max(files, key=lambda f: f)
    print(f"loading: {latest_data}")
    return pd.read_csv(latest_data, parse_dates=["time"])

# organize date and time into columns
def time_features(data):
    data["hour"] = data["time"].dt.hour
    data["day_of_week"] = data["time"].dt.day_name()
    data["month"] = data["time"].dt.month
    return data

#classifies AQI levels
data = latest_backfill_data(DATA_FLODER)
data.head()

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealhty for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealty"
    elif aqi <= 300:
        return "Very Unhealty"
    else:
        return "Hazardous"


# basic summary of data
def basic_overview(df):
    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Date range ---")
    print(f"{df['time'].min()}  to  {df['time'].max()}")

    print("\n--- Missing values per column ---")
    print(df.isna().sum())

    print("\n--- AQI summary stats ---")
    print(df["us_aqi"].describe())


# visul
def aqi_category_summary(data):

    order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
             "Unhealthy", "Very Unhealthy", "Hazardous"]

    count = data["aqi_category"].value_counts().reindex(order, fill_value=0)

    pct = (count / len(data) * 100).round(1)
    print("\n--- AQI Category Breakdown ---")

    for cat in order:
        print(f"{cat:35s}: {count[cat]:5d} hours  ({pct[cat]}%)")

    plt.figure(figsize=(8,4))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad", "#7b241c"]
    count.plot(kind="bar", color=colors)
    plt.title("Hours Spend in each AQI categroy")
    plt.ylabel("Hours")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("eda_aqi_category_breakdown.png")
    plt.close

    return count, pct

def alert_hours(data, threshold= ALERT_VALUE):

    alert_hours = data[data["us_aqi"]> threshold]
    pcts = len(alert_hours)/len(data) * 100
    print(f"\n--- Hazardous Alert Check (AQI > {threshold}) ---")
    print(f"{len(alert_hours)} out of {len(data)} hours ({pcts:.1f}%) exceeded the alert threshold.")

    return alert_hours, pcts

def worst_hours(data, top_n=5):

    worst_hours = data.nlargest(top_n, "us_aqi")[["time", "us_aqi", "aqi_category", "pm2_5", "pm10"]]
    print(f"\n--- Top {top_n} worst hours by AQI ---")
    print(worst_hours.to_string(index=False))

    return worst_hours

def aqi_over_time(data):

    plt.figure(figsize=(12,4))
    plt.plot(data["time"], data["us_aqi"])
    plt.title("AQI Over Time")
    plt.ylabel("AQI")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("eda_aqi_over_time.png")
    plt.close()


def rolling_trends(data, window_time=24):

    data_sorted = data.sort_values("time")
    rolling_trends = data_sorted["us_aqi"].rolling(window_time).mean()

    plt.figure(figsize=(12,4))
    plt.plot(data_sorted["time"], data_sorted["us_aqi"], alpha=0.3, label="Hourly")
    plt.plot(data_sorted["time"], rolling_trends, color='red', label=f"{window_time}h rolling avg")
    plt.legend()
    plt.title("AQI Trend with Rolling Average")
    plt.xlabel("Date")
    plt.ylabel("US AQI")
    plt.tight_layout()
    plt.savefig("eda_aqi_rolling_trend.png")
    plt.close()


def aqi_by_hour(data):

    hourly_avg = data.groupby("hour")["us_aqi"].mean()

    plt.figure(figsize=(8,4))
    hourly_avg.plot(kind="bar")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hours")
    plt.ylabel("Average US AQI")
    plt.tight_layout()
    plt.savefig("eda_aqi_by_hour.png")
    plt.close()



def aqi_by_day(data):

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_avg = data.groupby("day_of_week")["us_aqi"].mean().reindex(day_order)

    plt.figure(figsize=(8, 4))
    daily_avg.plot(kind="bar")
    plt.title("Average AQI by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average US AQI")
    plt.tight_layout()
    plt.savefig("eda_aqi_by_day.png")
    plt.close()


def plot_correlations(data):

    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "pressure_msl",
        "wind_speed_10m", "wind_direction_10m", "precipitation",
        "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "us_aqi",

    ]

    corr = data[numeric_cols].corr()


    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt= ".2f", cmap= "coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("eda_correlation_matrix.png")
    plt.close

    return corr


def overall_summary(data, count, pct, alert_hours, worst_hours, corr):


    top_cor = corr["us_aqi"].drop("us_aqi").abs().sort_values(ascending=False)


    lines = []

    lines.append(f"#AQI Exploratory Data Analysis — {CITY_NAME}")
    lines.append("")

    lines.append(f"Data Range: {data['time'].min()} to {data['time'].max()}")
    lines.append(f"Total hourly records: {len(data)}")
    lines.append("")

    lines.append(f"## AQI Overview")
    lines.append(f"Mean AQI: {data['us_aqi'].mean():.1f}")
    lines.append(f"Min AQI: {data['us_aqi'].min():.1f}")
    lines.append(f"Max AQI: {data['us_aqi'].max():.1f}")
    lines.append("")

    lines.append("Time Spend in Each AQI Categroy")
    for cat in count.index:
        lines.append(f"- {cat}: {count[cat]} hours ({pct[cat]}%)")
        lines.append("")

    lines.append(f"##Alert Threshold Check(AQI>{ALERT_VALUE})")
    lines.append(f"-{alert_hours:.1f}% of recorded hours exceeded the alert threshold.")
    lines.append("")

    lines.append("Worst Hours")
    lines.append("''")
    lines.append(worst_hours.to_string(index=False))
    lines.append("''")
    lines.append("")


    lines.append("## Strongest Correlations with AQI")
    for var, val in top_cor.head(5).items():
        lines.append(f"- {var}: {val:.2f}")
        lines.append("")


    lines.append("## Generated Files")
    for i in [
        "eda_aqi_over_time.png", "eda_aqi_rolling_trend.png",
        "eda_aqi_by_hour.png", "eda_aqi_by_day.png",
        "eda_aqi_category_breakdown.png",
        "eda_correlation_matrix.png",
    ]:
        lines.append(f"- {i}")



    with open("eda_summary_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved summary report to eda_summary_report.md")




if __name__ == "__main__":
    df = latest_backfill_data(DATA_FLODER)
    df = time_features(df)
    df["aqi_category"] = df["us_aqi"].apply(aqi_category)

    basic_overview(df)
    counts, percentages = aqi_category_summary(df)
    geet_alert_hours, alert_pct = alert_hours(df)
    worst = worst_hours(df)

    aqi_over_time(df)
    rolling_trends(df)
    aqi_by_hour(df)
    aqi_by_day(df)
    corr = plot_correlations(df)

    overall_summary(df, counts, percentages, alert_pct, worst, corr)

    print("\nDone. Check the .png files and eda_summary_report.md in this folder.")






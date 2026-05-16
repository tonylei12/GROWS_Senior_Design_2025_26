# Pulls historical weather for Las Vegas and generates training labels
import requests
import pandas as pd
import numpy as np
from datetime import date

#  CONFIG 
LATITUDE = 36.17
LONGITUDE = -115.14
START_DATE = "2020-01-01"   
END_DATE = "2025-12-31"
OUTPUT_CSV = "training_data.csv"

# Water balance threshold: water if cumulative deficit exceeds this (mm)
DEFICIT_THRESHOLD_MM = 10.0
# Crop coefficient — 0.7 is a reasonable default for mixed garden plants
CROP_COEFFICIENT = 0.7
# Window for "recent" rain features (days)
LOOKBACK_DAYS = 7
# 


def fetch_historical_weather(lat, lon, start, end):
    """Pull daily historical weather including ET0 from Open-Meteo's archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        "relative_humidity_2m_mean,wind_speed_10m_max,"
        "shortwave_radiation_sum,precipitation_sum,et0_fao_evapotranspiration"
        "&timezone=auto"
    )
    print(f"Fetching {start} to {end}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    print(f"Got {len(df)} days of data.")
    return df


def add_engineered_features(df, lookback=LOOKBACK_DAYS):
    """Add features the model can compute at inference time from forecast data."""
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling rainfall sum over lookback window
    df["rain_last_7d"] = df["precipitation_sum"].rolling(lookback, min_periods=1).sum()
    df["rain_last_3d"] = df["precipitation_sum"].rolling(3, min_periods=1).sum()
    df["rain_last_24h"] = df["precipitation_sum"]  # today's rain

    # Days since meaningful rain (>1mm)
    days_since = []
    counter = 999  # large number until first rain
    for r in df["precipitation_sum"]:
        if r > 1.0:
            counter = 0
        else:
            counter += 1
        days_since.append(counter)
    df["days_since_rain"] = days_since

    # Rolling ET0 (water demand pressure)
    df["et0_last_7d"] = df["et0_fao_evapotranspiration"].rolling(lookback, min_periods=1).sum()

    return df


def generate_labels(df, threshold=DEFICIT_THRESHOLD_MM, kc=CROP_COEFFICIENT):
    """
    Cumulative water deficit method:
    deficit accumulates as ET0*Kc - rainfall each day.
    When deficit > threshold, label = 1 (water needed) and reset to 0.
    This simulates a virtual gardener watering when the soil dries out.
    """
    deficit = 0.0
    labels = []
    deficits = []
    for _, row in df.iterrows():
        et_demand = row["et0_fao_evapotranspiration"] * kc
        rain = row["precipitation_sum"]
        deficit += et_demand - rain
        deficit = max(deficit, 0.0)  # rain can't make soil "more than full"

        if deficit >= threshold:
            labels.append(1)
            deficit = 0.0  # virtual watering event resets the deficit
        else:
            labels.append(0)
        deficits.append(deficit)

    df["water_deficit_mm"] = deficits
    df["label"] = labels
    return df


def main():
    df = fetch_historical_weather(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    # Drop rows with missing critical data
    df = df.dropna(subset=["et0_fao_evapotranspiration", "precipitation_sum",
                            "temperature_2m_mean", "relative_humidity_2m_mean"])

    df = add_engineered_features(df)
    df = generate_labels(df)

    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"Watering frequency: ~once every "
          f"{len(df) / max(df['label'].sum(), 1):.1f} days on average")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
    print(f"\nSample rows:")
    print(df[["date", "temperature_2m_mean", "precipitation_sum",
              "et0_fao_evapotranspiration", "rain_last_7d",
              "days_since_rain", "water_deficit_mm", "label"]].tail(10))


if __name__ == "__main__":
    main()

import pandas as pd
df = pd.read_csv("training_data.csv")
print("Total rows:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())
print(f"Watering events: {df['label'].sum()} out of {len(df)} days")
print(f"That's roughly once every {len(df)/max(df['label'].sum(),1):.1f} days")

print("\nLabel distribution by month:")
df["month"] = pd.to_datetime(df["date"]).dt.month
print(df.groupby("month")["label"].agg(["sum", "count"]))

print("\nSample of WATER days (label=1):")
print(df[df["label"] == 1][["date", "temperature_2m_mean", "et0_fao_evapotranspiration",
                              "rain_last_7d", "days_since_rain",
                              "water_deficit_mm"]].head(10))



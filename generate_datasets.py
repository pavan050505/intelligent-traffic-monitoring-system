import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

LOCATIONS = [
    "Kranti Chowk", "CIDCO", "Jalna Road", "Seven Hills", "Aurangpura", "Railway Station"
]

ROADS = [
    {"road_name": "Kranti Chowk Road", "road_type": "Main", "lanes": 4, "length_km": 3, "speed_limit": 50, "has_signal": True, "nearby_hospitals": "XYZ Hospital"},
    {"road_name": "CIDCO Road", "road_type": "Secondary", "lanes": 2, "length_km": 2.5, "speed_limit": 40, "has_signal": False, "nearby_hospitals": "ABC Hospital"},
    # Add 8+ more roads as needed
]

def simulate_traffic_data():
    dates = pd.date_range("2024-10-28", "2025-10-27", freq="H")
    rows = []
    for dt in dates:
        for loc in LOCATIONS:
            vehicle_count = np.random.poisson(60 if dt.hour in range(8, 20) else 25)
            avg_speed = np.clip(np.random.normal(40, 8) - vehicle_count/80, 15, 60)
            weather = random.choice(["Clear", "Rain", "Foggy", "Cloudy"])
            congestion_level = "High" if vehicle_count > 100 else ("Medium" if vehicle_count > 60 else "Low")
            rows.append([
                dt, loc, vehicle_count, avg_speed, congestion_level, weather
            ])
    df = pd.DataFrame(rows, columns=["date_time", "location_name", "vehicle_count", "avg_speed", "congestion_level", "weather_condition"])
    df.to_csv("data/traffic_data.csv", index=False)

def simulate_accident_data():
    dates = pd.date_range("2024-10-28", "2025-10-27", freq="D")
    rows = []
    for dt in dates:
        n_accidents = np.random.poisson(0.4)
        for _ in range(n_accidents):
            loc = random.choice(LOCATIONS)
            severity = random.choices(["Minor", "Moderate", "Severe"], [0.6,0.3,0.1])[0]
            cause = random.choice(["Overspeed", "Drunk Driving", "Signal Jump", "Weather"])
            casualties = np.random.binomial(3, 0.2)
            rows.append([
                f"{dt.strftime('%Y%m%d')}-{random.randint(1000,9999)}", dt, loc, severity, cause, casualties
            ])
    df = pd.DataFrame(rows, columns=["accident_id", "date_time", "location_name", "severity", "cause", "casualties"])
    df.to_csv("data/accident_data.csv", index=False)

def simulate_road_data():
    df = pd.DataFrame(ROADS)
    df.to_csv("data/road_infra_data.csv", index=False)

def simulate_weather_data():
    dates = pd.date_range("2024-10-28", "2025-10-27", freq="D")
    rows = []
    for dt in dates:
        temp = np.random.normal(25 + 8*np.sin(dt.dayofyear/365.0*2*np.pi), 2)
        rainfall = max(0, np.random.normal(3 if dt.month in [6,7,8,9] else 0, 2))
        visibility = max(100, 1000 - rainfall*20 - np.random.normal(0, 50))
        humidity = np.random.uniform(35, 90)
        wind_speed = np.random.normal(7, 2)
        rows.append([dt.date(), temp, rainfall, visibility, humidity, wind_speed])
    df = pd.DataFrame(rows, columns=["date", "temperature", "rainfall_mm", "visibility_m", "humidity", "wind_speed"])
    df.to_csv("data/weather_data.csv", index=False)

if __name__ == "__main__":
    simulate_traffic_data()
    simulate_accident_data()
    simulate_road_data()
    simulate_weather_data()

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess():
    df_traffic = pd.read_csv("data/traffic_data.csv")
    df_weather = pd.read_csv("data/weather_data.csv")
    df_accident = pd.read_csv("data/accident_data.csv")
    df_road = pd.read_csv("data/road_infra_data.csv")

    # Example derived: traffic density (vehicle_count / lanes)
    df_traffic["hour"] = pd.to_datetime(df_traffic["date_time"]).dt.hour
    df_traffic["traffic_density"] = df_traffic["vehicle_count"] / 3  # avg lanes
    # Merge weather
    df_weather["date_time"] = pd.to_datetime(df_weather["date"])
    df_traffic["date_only"] = pd.to_datetime(df_traffic["date_time"]).dt.date
    df_merged = pd.merge(df_traffic, df_weather, left_on="date_only", right_on="date")
    # Encode categorical
    for col in ["location_name", "congestion_level", "weather_condition"]:
        df_merged[col] = LabelEncoder().fit_transform(df_merged[col])
    df_merged.to_csv("data/traffic_merged.csv", index=False)
    print("Preprocessing Done. Saved as data/traffic_merged.csv.")

if __name__ == "__main__":
    preprocess()


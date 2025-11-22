import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import pickle
import matplotlib.pyplot as plt

st.title("Smart City Traffic & Accident Analytics: Aurangabad")

# --- Load all dataframes ---
df_traffic = pd.read_csv("data/traffic_data.csv")
df_accident = pd.read_csv("data/accident_data.csv")
df_weather = pd.read_csv("data/weather_data.csv")

# ----------- Map and Dashboard Sections -----------

# 1. Live Simulated Congestion Map
st.header("Live Simulated Congestion Map")
location = st.selectbox("Location", df_traffic["location_name"].unique())
hour = st.slider("Hour of day", 0, 23)
day = st.date_input("Date", pd.to_datetime(df_traffic["date_time"].min()))
input_df = df_traffic[
    (pd.to_datetime(df_traffic['date_time']).dt.date == day) & 
    (pd.to_datetime(df_traffic['date_time']).dt.hour == hour) &
    (df_traffic['location_name'] == location)
]
st.write(input_df)

# 2. Accident Hotspot Heatmap
st.header("Accident Hotspot Heatmap")
# (Simple placeholder: just showing mapped coords for one location)
coords = pd.DataFrame([
    {"lat": 19.8779, "lon": 75.3425},  # Add all coords as needed for each location
])
st.map(coords)

# 3. Accident Prediction - placeholder for model
st.header("Accident Prediction")
# Use pickled accident model to predict for selected parameters (to be added)

# 4. Weather vs Traffic Visualization (FULL WORKING SECTION)
st.header("Weather vs Traffic Visualization")

df_traffic['date'] = pd.to_datetime(df_traffic['date_time']).dt.date
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.date

traffic_weather = pd.merge(df_traffic, df_weather, on='date', how='left')

traffic_daily = traffic_weather.groupby('date').agg({
    'vehicle_count': 'mean',
    'avg_speed': 'mean',
    'temperature': 'mean',
    'rainfall_mm': 'mean'
}).reset_index()

fig, ax = plt.subplots()
ax.scatter(traffic_daily["temperature"], traffic_daily["vehicle_count"], color='blue', alpha=0.6)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Avg Vehicle Count")
ax.set_title("Avg Vehicle Count vs Temperature (Daily)")
st.pyplot(fig)

# Optionally, add more correlations:
fig2, ax2 = plt.subplots()
ax2.scatter(traffic_daily["rainfall_mm"], traffic_daily["avg_speed"], color='green', alpha=0.6)
ax2.set_xlabel("Rainfall (mm)")
ax2.set_ylabel("Avg Speed (km/h)")
ax2.set_title("Avg Speed vs Rainfall (Daily)")
st.pyplot(fig2)

# 5. Summary Reports (placeholder)
st.header("Summary Reports")
# Generate insights/report summary per code below
# Example: display basic summary stats
st.write("Total records in traffic data:", len(df_traffic))
st.write("Total accidents:", len(df_accident))
st.write("Total weather records:", len(df_weather))

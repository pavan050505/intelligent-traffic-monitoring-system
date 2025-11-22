import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/traffic_merged.csv")
X = df[["hour", "location_name", "vehicle_count", "avg_speed", "temperature", "rainfall_mm", "humidity", "wind_speed"]]
y = df["congestion_level"]

rf = RandomForestClassifier(n_estimators=60)
rf.fit(X, y)
with open("models/traffic_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("Traffic Congestion Model Saved.")

# Accident Model Example
df_road = pd.read_csv("data/road_infra_data.csv")
df_acc = pd.read_csv("data/accident_data.csv")
# Join, engineer features as needed (not shown for brevity)
# Assume df_acc_proc is generated appropriately

from sklearn.linear_model import LogisticRegression

# X_accident = <engineered features dataframe>
# y_accident = <accident binary column>
# For demonstration
# model_acc = LogisticRegression().fit(X_accident, y_accident)
# with open("models/accident_model.pkl", "wb") as f:
#     pickle.dump(model_acc, f)

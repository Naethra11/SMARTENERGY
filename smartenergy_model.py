import pyexasol
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# 1. CONNECT TO EXASOL
# --------------------------------------------------

password = input("Enter your Exasol SYS password: ")

con = pyexasol.connect(
    dsn="127.0.0.1:8563",
    user="sys",
    password=password,
    websocket_sslopt={"cert_reqs": 0}
)

# --------------------------------------------------
# 2. LOAD DATA FROM EXASOL
# --------------------------------------------------

query = """
SELECT
    "datetime",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
    "is_weekend",
    "date"
FROM SMARTENERGY.HOURLY_CONSUMPTION
ORDER BY "datetime"
"""

df = con.export_to_pandas(query)
con.close()

print("\nData loaded from Exasol!")
print("Shape:", df.shape)

# --------------------------------------------------
# 3. PREPARE DATA
# --------------------------------------------------

df["datetime"] = pd.to_datetime(df["datetime"])

# Previous-hour consumption
df["previous_power"] = df["Global_active_power"].shift(1)

# Previous-day same-hour consumption
df["previous_day_power"] = df["Global_active_power"].shift(24)

# Remove rows created by lag features
df = df.dropna().reset_index(drop=True)

features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "previous_power",
    "previous_day_power"
]

target = "Global_active_power"

X = df[features]
y = df[target]

# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining prediction model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== PREDICTION RESULTS ==========")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# --------------------------------------------------
# 7. SHOW SAMPLE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "datetime": df.iloc[split_index:]["datetime"].values,
    "actual_power": y_test.values,
    "predicted_power": predictions
})

print("\nSample predictions:")
print(results.head(10))

# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

import joblib

joblib.dump(model, "smartenergy_prediction_model.pkl")

print("\nModel saved as: smartenergy_prediction_model.pkl")
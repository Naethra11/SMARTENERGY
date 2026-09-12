import pandas as pd
import pyexasol
import joblib

# --------------------------------------------------
# 1. CONNECT TO EXASOL
# --------------------------------------------------

dsn = "127.0.0.1:8563"

password = input("Enter your Exasol SYS password: ")

con = pyexasol.connect(
    dsn=dsn,
    user="sys",
    password=password,
    websocket_sslopt={"cert_reqs": 0}
)

print("Connected to Exasol.")

# --------------------------------------------------
# 2. LOAD HOURLY DATA FROM EXASOL
# --------------------------------------------------

query = """
SELECT *
FROM SMARTENERGY.HOURLY_CONSUMPTION
ORDER BY "datetime"
"""

df = con.export_to_pandas(query)

print("Rows loaded:", len(df))

con.close()

# --------------------------------------------------
# 3. PREPARE DATA
# --------------------------------------------------

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

# --------------------------------------------------
# 4. CREATE LAG FEATURES
# --------------------------------------------------

# Previous hour's consumption
df["previous_power"] = (
    df["Global_active_power"].shift(1)
)

# Consumption 24 hours earlier
df["previous_day_power"] = (
    df["Global_active_power"].shift(24)
)

# --------------------------------------------------
# 5. DEFINE MODEL FEATURES
# --------------------------------------------------

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

# --------------------------------------------------
# 6. REMOVE ROWS WITHOUT LAG VALUES
# --------------------------------------------------

prediction_df = df.dropna(
    subset=features
).copy()

print(
    "Rows available for prediction:",
    len(prediction_df)
)

# --------------------------------------------------
# 7. LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load(
    "smartenergy_prediction_model.pkl"
)

print("Prediction model loaded.")

# --------------------------------------------------
# 8. GENERATE PREDICTIONS
# --------------------------------------------------

prediction_df["predicted_power"] = model.predict(
    prediction_df[features]
)

# --------------------------------------------------
# 9. CALCULATE PREDICTION ERROR
# --------------------------------------------------

prediction_df["prediction_error"] = (
    prediction_df["Global_active_power"]
    - prediction_df["predicted_power"]
)

prediction_df["absolute_error"] = (
    prediction_df["prediction_error"].abs()
)

# --------------------------------------------------
# 10. CREATE FINAL RESULTS TABLE
# --------------------------------------------------

final_df = prediction_df[
    [
        "datetime",
        "Global_active_power",
        "predicted_power",
        "prediction_error",
        "absolute_error"
    ]
].copy()

# Rename columns for clean Exasol table
final_df.columns = [
    "datetime",
    "actual_power",
    "predicted_power",
    "prediction_error",
    "absolute_error"
]

# --------------------------------------------------
# 11. SAVE CSV
# --------------------------------------------------

output_file = "smartenergy_prediction_results.csv"

final_df.to_csv(
    output_file,
    index=False,
    lineterminator="\n"
)

# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print()
print("====================================")
print("PREDICTION RESULTS CREATED")
print("====================================")
print("Rows:", len(final_df))
print("Output:", output_file)
print()

print(final_df.head())

print()
print("Prediction statistics:")
print(
    final_df[
        [
            "actual_power",
            "predicted_power",
            "prediction_error",
            "absolute_error"
        ]
    ].describe()
)
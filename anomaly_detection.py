import pyexasol
import pandas as pd
import numpy as np

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
    "hour",
    "day_of_week",
    "is_weekend"
FROM SMARTENERGY.HOURLY_CONSUMPTION
ORDER BY "datetime"
"""

df = con.export_to_pandas(query)
con.close()

print("\nData loaded from Exasol!")
print("Rows:", len(df))

# --------------------------------------------------
# 3. PREPARE TIME SERIES
# --------------------------------------------------

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

power = df["Global_active_power"]

# --------------------------------------------------
# 4. ROLLING BASELINE
# --------------------------------------------------

# Previous 24 hours
df["rolling_mean"] = power.shift(1).rolling(24).mean()
df["rolling_std"] = power.shift(1).rolling(24).std()

# --------------------------------------------------
# 5. CALCULATE ANOMALY SCORE
# --------------------------------------------------

df["z_score"] = (
    (power - df["rolling_mean"]) /
    df["rolling_std"].replace(0, np.nan)
)

df["z_score"] = df["z_score"].fillna(0)

# --------------------------------------------------
# 6. DETECT ANOMALIES
# --------------------------------------------------

# |Z| >= 3 means unusually far from recent behavior
df["anomaly"] = df["z_score"].abs() >= 3

anomalies = df[df["anomaly"]].copy()

# --------------------------------------------------
# 7. CLASSIFY ANOMALY
# --------------------------------------------------

anomalies["anomaly_type"] = np.where(
    anomalies["Global_active_power"] >
    anomalies["rolling_mean"],
    "High Consumption",
    "Low Consumption"
)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== ANOMALY RESULTS ==========")

print("Total records analyzed:", len(df))
print("Anomalies detected:", len(anomalies))

if len(anomalies) > 0:

    print("\nTop 10 anomalies:")

    display_columns = [
        "datetime",
        "Global_active_power",
        "rolling_mean",
        "z_score",
        "anomaly_type"
    ]

    print(
        anomalies
        .sort_values("z_score", key=abs, ascending=False)
        [display_columns]
        .head(10)
        .to_string(index=False)
    )

else:

    print("\nNo anomalies detected.")

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

anomalies.to_csv(
    "smartenergy_anomalies.csv",
    index=False
)

print("\nAnomaly results saved as: smartenergy_anomalies.csv")
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

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# --------------------------------------------------
# 3. CREATE BASELINES
# --------------------------------------------------

df["previous_hour_power"] = (
    df["Global_active_power"].shift(1)
)

df["previous_day_power"] = (
    df["Global_active_power"].shift(24)
)

df["rolling_mean"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(24)
    .mean()
)

# --------------------------------------------------
# 4. DETECT ANOMALIES
# --------------------------------------------------

df["rolling_std"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(24)
    .std()
)

df["z_score"] = (
    (df["Global_active_power"] - df["rolling_mean"])
    / df["rolling_std"].replace(0, np.nan)
)

df["z_score"] = df["z_score"].fillna(0)

df["anomaly"] = df["z_score"].abs() >= 3

anomalies = df[df["anomaly"]].copy()

# --------------------------------------------------
# 5. CALCULATE DEVIATIONS
# --------------------------------------------------

anomalies["deviation_percent"] = (
    (
        anomalies["Global_active_power"]
        - anomalies["rolling_mean"]
    )
    / anomalies["rolling_mean"].replace(0, np.nan)
) * 100

# --------------------------------------------------
# 6. GENERATE EXPLANATIONS
# --------------------------------------------------

def generate_explanation(row):

    actual = row["Global_active_power"]
    baseline = row["rolling_mean"]

    previous = row["previous_hour_power"]
    previous_day = row["previous_day_power"]

    reasons = []

    # Compare with recent baseline
    if baseline > 0:
        increase = ((actual - baseline) / baseline) * 100

        if increase >= 100:
            reasons.append(
                f"consumption was {increase:.0f}% above the recent 24-hour baseline"
            )
        elif increase >= 30:
            reasons.append(
                f"consumption was {increase:.0f}% above the recent baseline"
            )

    # Compare with previous hour
    if previous > 0:
        change = ((actual - previous) / previous) * 100

        if change >= 50:
            reasons.append(
                f"consumption increased {change:.0f}% compared with the previous hour"
            )

    # Compare with same hour previous day
    if previous_day > 0:
        day_change = ((actual - previous_day) / previous_day) * 100

        if day_change >= 50:
            reasons.append(
                f"consumption was {day_change:.0f}% higher than the same hour on the previous day"
            )

    # Weekend context
    if row["is_weekend"] == 1:
        reasons.append("the event occurred during a weekend period")

    # Sub-metering context
    sub_total = (
        row["Sub_metering_1"]
        + row["Sub_metering_2"]
        + row["Sub_metering_3"]
    )

    if sub_total > 0:
        reasons.append(
            f"sub-metering activity was {sub_total:.2f}"
        )

    if not reasons:
        reasons.append(
            "consumption significantly differed from recent historical behavior"
        )

    explanation = (
        "High consumption anomaly detected at "
        f"{row['datetime'].strftime('%Y-%m-%d %H:%M')}. "
        + ". ".join(reasons)
        + "."
    )

    return explanation


anomalies["explanation"] = anomalies.apply(
    generate_explanation,
    axis=1
)

# --------------------------------------------------
# 7. SEVERITY
# --------------------------------------------------

def severity(z):

    z = abs(z)

    if z >= 10:
        return "Critical"
    elif z >= 6:
        return "High"
    else:
        return "Moderate"


anomalies["severity"] = anomalies["z_score"].apply(severity)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== EXPLANATION RESULTS ==========")

print("Anomalies explained:", len(anomalies))

columns = [
    "datetime",
    "Global_active_power",
    "rolling_mean",
    "deviation_percent",
    "z_score",
    "severity",
    "explanation"
]

print(
    anomalies
    .sort_values("z_score", key=abs, ascending=False)
    [columns]
    .head(10)
    .to_string(index=False)
)

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

anomalies.to_csv(
    "smartenergy_explained_anomalies.csv",
    index=False
)

print(
    "\nSaved as: smartenergy_explained_anomalies.csv"
)
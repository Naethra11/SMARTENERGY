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
# 2. LOAD EXPLAINED ANOMALIES FROM EXASOL
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
    "is_weekend"
FROM SMARTENERGY.HOURLY_CONSUMPTION
ORDER BY "datetime"
"""

df = con.export_to_pandas(query)
con.close()

df["datetime"] = pd.to_datetime(df["datetime"])

# --------------------------------------------------
# 3. CALCULATE HISTORICAL BASELINE
# --------------------------------------------------

df["rolling_mean"] = (
    df["Global_active_power"]
    .shift(1)
    .rolling(24)
    .mean()
)

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

# --------------------------------------------------
# 4. IDENTIFY ANOMALIES
# --------------------------------------------------

anomalies = df[df["z_score"].abs() >= 3].copy()

# --------------------------------------------------
# 5. CALCULATE EXCESS CONSUMPTION
# --------------------------------------------------

anomalies["excess_power"] = (
    anomalies["Global_active_power"]
    - anomalies["rolling_mean"]
)

# Only positive excess is considered avoidable consumption
anomalies["avoidable_power"] = (
    anomalies["excess_power"].clip(lower=0)
)

# --------------------------------------------------
# 6. ESTIMATE POTENTIAL SAVINGS
# --------------------------------------------------

# Assume 30% of avoidable excess can potentially be reduced
REDUCTION_RATE = 0.30

anomalies["potential_saving_kwh"] = (
    anomalies["avoidable_power"] * REDUCTION_RATE
)

# Example electricity cost assumption
COST_PER_KWH = 8.0

anomalies["potential_cost_saving"] = (
    anomalies["potential_saving_kwh"] * COST_PER_KWH
)

# --------------------------------------------------
# 7. GENERATE RECOMMENDATIONS
# --------------------------------------------------

def recommendation(row):

    z = abs(row["z_score"])
    hour = int(row["hour"])

    if z >= 10:
        action = (
            "Investigate the high-load source immediately and "
            "switch off or postpone non-essential equipment."
        )

    elif z >= 6:
        action = (
            "Reduce non-essential loads and shift flexible "
            "equipment usage to a lower-consumption period."
        )

    else:
        action = (
            "Review the active loads during this period and "
            "consider shifting flexible usage."
        )

    # Time-based suggestion
    if 17 <= hour <= 22:
        time_advice = (
            " This occurred during an evening period, so "
            "peak-load shifting may help reduce consumption."
        )
    else:
        time_advice = (
            " Consider scheduling flexible loads outside "
            "high-consumption periods."
        )

    return action + time_advice


anomalies["recommendation"] = anomalies.apply(
    recommendation,
    axis=1
)

# --------------------------------------------------
# 8. SEVERITY
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
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== OPTIMIZATION RESULTS ==========")

print("Anomalies analyzed:", len(anomalies))

display_columns = [
    "datetime",
    "Global_active_power",
    "rolling_mean",
    "avoidable_power",
    "potential_saving_kwh",
    "potential_cost_saving",
    "severity",
    "recommendation"
]

print(
    anomalies
    .sort_values("z_score", key=abs, ascending=False)
    [display_columns]
    .head(10)
    .to_string(index=False)
)

# --------------------------------------------------
# 10. TOTAL POTENTIAL IMPACT
# --------------------------------------------------

total_energy_saving = anomalies["potential_saving_kwh"].sum()
total_cost_saving = anomalies["potential_cost_saving"].sum()

print("\n========== POTENTIAL IMPACT ==========")
print(f"Potential energy saving: {total_energy_saving:.2f} kWh")
print(f"Potential cost saving: ₹{total_cost_saving:.2f}")

# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

anomalies.to_csv(
    "smartenergy_optimization_results.csv",
    index=False
)

print(
    "\nSaved as: smartenergy_optimization_results.csv"
)

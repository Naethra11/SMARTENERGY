import pandas as pd
import pyexasol

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
# 2. LOAD DATA FROM EXASOL
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

power = df["Global_active_power"]

# --------------------------------------------------
# 4. ROLLING BASELINE
# --------------------------------------------------

df["rolling_mean"] = (
    power
    .shift(1)
    .rolling(window=24, min_periods=24)
    .mean()
)

df["rolling_std"] = (
    power
    .shift(1)
    .rolling(window=24, min_periods=24)
    .std()
)

# --------------------------------------------------
# 5. Z-SCORE
# --------------------------------------------------

df["z_score"] = (
    (power - df["rolling_mean"])
    / df["rolling_std"]
)

# --------------------------------------------------
# 6. ANOMALY DETECTION
# --------------------------------------------------

df["is_anomaly"] = df["z_score"].abs() >= 3

anomalies = df[df["is_anomaly"]].copy()

print("Anomalies detected:", len(anomalies))

# --------------------------------------------------
# 7. DEVIATION %
# --------------------------------------------------

anomalies["deviation_pct"] = (
    (anomalies["Global_active_power"]
     - anomalies["rolling_mean"])
    / anomalies["rolling_mean"]
) * 100

# --------------------------------------------------
# 8. SEVERITY
# --------------------------------------------------

def get_severity(z):

    z = abs(z)

    if z >= 10:
        return "Critical"

    elif z >= 6:
        return "High"

    else:
        return "Moderate"


anomalies["severity"] = (
    anomalies["z_score"]
    .apply(get_severity)
)

# --------------------------------------------------
# 9. EXPLANATION
# --------------------------------------------------

anomalies["previous_hour_power"] = (
    anomalies["Global_active_power"]
    .shift(1)
)

anomalies["same_hour_previous_day"] = (
    anomalies["Global_active_power"]
    .shift(24)
)

anomalies["submeter_total"] = (
    anomalies["Sub_metering_1"]
    + anomalies["Sub_metering_2"]
    + anomalies["Sub_metering_3"]
)


def generate_explanation(row):

    actual = row["Global_active_power"]
    baseline = row["rolling_mean"]

    if baseline > 0:
        deviation = ((actual - baseline) / baseline) * 100
    else:
        deviation = 0

    explanation = (
        f"Consumption is {deviation:.1f}% "
        f"above the recent 24-hour baseline."
    )

    if row["is_weekend"] == 1:
        explanation += " The event occurred during a weekend."

    if row["submeter_total"] > 10:
        explanation += (
            " High sub-metering activity indicates "
            "significant appliance/load usage."
        )

    return explanation


anomalies["explanation"] = (
    anomalies.apply(generate_explanation, axis=1)
)

# --------------------------------------------------
# 10. RECOMMENDATION
# --------------------------------------------------

def generate_recommendation(row):

    z = abs(row["z_score"])
    hour = row["hour"]

    if z >= 10:

        recommendation = (
            "Investigate high-load sources immediately. "
            "Switch off or postpone non-essential equipment."
        )

    elif z >= 6:

        recommendation = (
            "Reduce non-essential loads and shift "
            "flexible equipment usage."
        )

    else:

        recommendation = (
            "Review active loads and consider shifting "
            "flexible electricity usage."
        )

    # Peak-hour recommendation

    if 17 <= hour <= 22:

        recommendation += (
            " Consider shifting flexible loads away "
            "from evening peak hours."
        )

    return recommendation


anomalies["recommendation"] = (
    anomalies.apply(generate_recommendation, axis=1)
)

# --------------------------------------------------
# 11. ESTIMATED SAVINGS
# --------------------------------------------------

REDUCTION_RATE = 0.30
COST_PER_KWH = 8.0

anomalies["excess_power"] = (
    anomalies["Global_active_power"]
    - anomalies["rolling_mean"]
)

anomalies["avoidable_power"] = (
    anomalies["excess_power"]
    .clip(lower=0)
)

anomalies["potential_saving_kwh"] = (
    anomalies["avoidable_power"]
    * REDUCTION_RATE
)

anomalies["potential_cost_saving"] = (
    anomalies["potential_saving_kwh"]
    * COST_PER_KWH
)

# --------------------------------------------------
# 12. FINAL TABLE
# --------------------------------------------------

final_df = anomalies[
    [
        "datetime",
        "Global_active_power",
        "rolling_mean",
        "z_score",
        "deviation_pct",
        "severity",
        "explanation",
        "recommendation",
        "potential_saving_kwh",
        "potential_cost_saving"
    ]
].copy()

final_df.columns = [
    "datetime",
    "actual_power",
    "baseline_power",
    "z_score",
    "deviation_pct",
    "severity",
    "explanation",
    "recommendation",
    "potential_saving_kwh",
    "potential_cost_saving"
]

# --------------------------------------------------
# 13. SAVE CSV
# --------------------------------------------------

output_file = "smartenergy_energy_anomalies.csv"

final_df.to_csv(
    output_file,
    index=False
)

print()
print("====================================")
print("ENERGY ANOMALIES CREATED")
print("====================================")
print("Rows:", len(final_df))
print("Output:", output_file)
print()
print(final_df.head())

import ssl

import pandas as pd
import pyexasol
import streamlit as st


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartEnergy Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ SmartEnergy")
st.caption(
    "Energy consumption prediction, anomaly detection, and optimization"
)


# ---------------------------------------------------------
# Exasol connection
# ---------------------------------------------------------
@st.cache_resource
def get_connection():
    return pyexasol.connect(
        dsn="127.0.0.1:8563",
        user="sys",
        password=st.secrets["exasol"]["password"],
        encryption=True,
        websocket_sslopt={"cert_reqs": ssl.CERT_NONE},
    )


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_hourly_consumption():
    connection = get_connection()

    query = """
        SELECT
            "hour" AS "HOUR",
            AVG("Global_active_power") AS "AVG_POWER"
        FROM SMARTENERGY.HOURLY_CONSUMPTION
        GROUP BY "hour"
        ORDER BY "hour"
    """

    df = connection.export_to_pandas(query)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


@st.cache_data(ttl=300)
def load_daily_predictions():
    connection = get_connection()

    query = """
        SELECT
            CAST("datetime" AS DATE) AS "PREDICTION_DATE",
            AVG("actual_power") AS "ACTUAL_POWER",
            AVG("predicted_power") AS "PREDICTED_POWER"
        FROM SMARTENERGY.ENERGY_PREDICTIONS
        GROUP BY CAST("datetime" AS DATE)
        ORDER BY CAST("datetime" AS DATE)
    """

    df = connection.export_to_pandas(query)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


@st.cache_data(ttl=300)
def load_prediction_metrics():
    connection = get_connection()

    query = """
        SELECT
            COUNT(*) AS "TOTAL_ROWS",
            AVG(ABS("actual_power" - "predicted_power")) AS "MAE",
            SQRT(
                AVG(
                    ("actual_power" - "predicted_power")
                    * ("actual_power" - "predicted_power")
                )
            ) AS "RMSE"
        FROM SMARTENERGY.ENERGY_PREDICTIONS
    """

    df = connection.export_to_pandas(query)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


@st.cache_data(ttl=300)
def load_anomalies():
    connection = get_connection()

    query = """
        SELECT
            "datetime" AS "DATETIME",
            "actual_power" AS "ACTUAL_POWER",
            "baseline_power" AS "BASELINE_POWER",
            "z_score" AS "Z_SCORE",
            "deviation_pct" AS "DEVIATION_PCT",
            "severity" AS "SEVERITY",
            "explanation" AS "EXPLANATION",
            "recommendation" AS "RECOMMENDATION",
            "potential_saving_kwh" AS "POTENTIAL_SAVING_KWH",
            "potential_cost_saving" AS "POTENTIAL_COST_SAVING"
        FROM SMARTENERGY.ENERGY_ANOMALIES
        ORDER BY "datetime" DESC
    """

    df = connection.export_to_pandas(query)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


@st.cache_data(ttl=300)
def load_prediction_sample():
    connection = get_connection()

    query = """
        SELECT
            "datetime" AS "DATETIME",
            "actual_power" AS "ACTUAL_POWER",
            "predicted_power" AS "PREDICTED_POWER",
            "prediction_error" AS "PREDICTION_ERROR",
            "absolute_error" AS "ABSOLUTE_ERROR"
        FROM SMARTENERGY.ENERGY_PREDICTIONS
        ORDER BY "datetime" DESC
        LIMIT 500
    """

    df = connection.export_to_pandas(query)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def show_error(section_name, error):
    st.error(f"Could not load {section_name}.")
    with st.expander("Technical details"):
        st.code(str(error))


def format_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.{decimals}f}"


# ---------------------------------------------------------
# 1. Energy consumption by hour
# ---------------------------------------------------------
st.header("1. Energy Consumption by Hour")

try:
    df_hourly = load_hourly_consumption()

    if df_hourly.empty:
        st.info("No hourly consumption data found.")
    else:
        df_hourly["hour"] = pd.to_numeric(
            df_hourly["hour"], errors="coerce"
        )
        df_hourly["avg_power"] = pd.to_numeric(
            df_hourly["avg_power"], errors="coerce"
        )

        df_hourly = df_hourly.dropna(subset=["hour", "avg_power"])

        if not df_hourly.empty:
            peak_row = df_hourly.loc[df_hourly["avg_power"].idxmax()]
            low_row = df_hourly.loc[df_hourly["avg_power"].idxmin()]

            col1, col2 = st.columns(2)

            col1.metric(
                "Peak average hour",
                f"{int(peak_row['hour']):02d}:00",
                f"{peak_row['avg_power']:.3f} kW",
            )

            col2.metric(
                "Lowest average hour",
                f"{int(low_row['hour']):02d}:00",
                f"{low_row['avg_power']:.3f} kW",
            )

            chart_data = df_hourly.set_index("hour")[["avg_power"]]
            st.bar_chart(chart_data)

            st.caption("Average Global Active Power by hour.")

except Exception as error:
    show_error("hourly consumption", error)


# ---------------------------------------------------------
# 2. Energy consumption predictions
# ---------------------------------------------------------
st.header("2. Energy Consumption Predictions")

try:
    df_daily = load_daily_predictions()
    df_metrics = load_prediction_metrics()

    if not df_daily.empty:
        df_daily["prediction_date"] = pd.to_datetime(
            df_daily["prediction_date"], errors="coerce"
        )

        df_daily = df_daily.dropna(subset=["prediction_date"])

        if not df_daily.empty:
            chart_data = df_daily.set_index("prediction_date")[
                ["actual_power", "predicted_power"]
            ]

            st.line_chart(chart_data)

            st.caption(
                "Daily averages from the stored prediction rows. "
                "The prediction table contains repeated timestamps and "
                "does not identify individual model runs, so these "
                "metrics are preliminary."
            )
    else:
        st.info("No prediction records found.")

    if not df_metrics.empty:
        metrics = df_metrics.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction rows",
            format_number(metrics["total_rows"], 0),
        )
        col2.metric(
            "MAE",
            format_number(metrics["mae"], 5),
        )
        col3.metric(
            "RMSE",
            format_number(metrics["rmse"], 5),
        )

    with st.expander("View a sample of prediction records"):
        df_sample = load_prediction_sample()
        st.dataframe(
            df_sample,
            use_container_width=True,
            hide_index=True,
        )

except Exception as error:
    show_error("predictions", error)


# ---------------------------------------------------------
# 3. Anomaly detection
# ---------------------------------------------------------
st.header("3. Anomaly Detection")

df_anomalies = pd.DataFrame()

try:
    df_anomalies = load_anomalies()

    if df_anomalies.empty:
        st.info("No anomaly records found.")
    else:
        df_anomalies["datetime"] = pd.to_datetime(
            df_anomalies["datetime"], errors="coerce"
        )
        df_anomalies = df_anomalies.dropna(subset=["datetime"])

        # Summary metrics
        total_anomalies = len(df_anomalies)

        severity_counts = (
            df_anomalies["severity"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total anomaly records", f"{total_anomalies:,}")
        col2.metric(
            "High severity",
            int(severity_counts.get("High", 0)),
        )
        col3.metric(
            "Moderate severity",
            int(severity_counts.get("Moderate", 0)),
        )

        # Filters
        st.subheader("Filter anomalies")

        filter_col1, filter_col2 = st.columns(2)

        available_severities = sorted(
            df_anomalies["severity"]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )

        with filter_col1:
            selected_severities = st.multiselect(
                "Severity",
                options=available_severities,
                default=available_severities,
            )

        min_date = df_anomalies["datetime"].min().date()
        max_date = df_anomalies["datetime"].max().date()

        with filter_col2:
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        filtered_anomalies = df_anomalies.copy()

        if selected_severities:
            filtered_anomalies = filtered_anomalies[
                filtered_anomalies["severity"]
                .fillna("Unknown")
                .astype(str)
                .isin(selected_severities)
            ]
        else:
            filtered_anomalies = filtered_anomalies.iloc[0:0]

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range

            filtered_anomalies = filtered_anomalies[
                filtered_anomalies["datetime"].dt.date.between(
                    start_date, end_date
                )
            ]
        elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
            selected_date = date_range[0]

            filtered_anomalies = filtered_anomalies[
                filtered_anomalies["datetime"].dt.date == selected_date
            ]

        st.caption(
            f"Showing {len(filtered_anomalies):,} matching records."
        )

        display_columns = [
            "datetime",
            "actual_power",
            "baseline_power",
            "severity",
            "deviation_pct",
            "explanation",
            "recommendation",
            "potential_saving_kwh",
            "potential_cost_saving",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in filtered_anomalies.columns
        ]

        st.dataframe(
            filtered_anomalies[display_columns],
            use_container_width=True,
            hide_index=True,
        )

except Exception as error:
    show_error("anomaly detection", error)


# ---------------------------------------------------------
# 4. Optimization recommendations
# ---------------------------------------------------------
st.header("4. Optimization Recommendations")

try:
    if df_anomalies.empty:
        st.info("Load anomaly data to view recommendations.")
    else:
        recommendation_columns = [
            "datetime",
            "severity",
            "actual_power",
            "baseline_power",
            "explanation",
            "recommendation",
            "potential_saving_kwh",
            "potential_cost_saving",
        ]

        recommendation_columns = [
            column
            for column in recommendation_columns
            if column in df_anomalies.columns
        ]

        recommendations = df_anomalies[
            recommendation_columns
        ].head(10)

        if recommendations.empty:
            st.info("No recommendations available.")
        else:
            st.caption(
                "These are stored recommendations and estimated savings, "
                "not confirmed energy reductions."
            )

            for _, row in recommendations.iterrows():
                timestamp = row.get("datetime", "Unknown time")
                severity = row.get("severity", "Unknown")
                recommendation = row.get(
                    "recommendation",
                    "No recommendation provided.",
                )
                explanation = row.get("explanation", "")

                actual = row.get("actual_power")
                baseline = row.get("baseline_power")
                saving_kwh = row.get("potential_saving_kwh")
                saving_cost = row.get("potential_cost_saving")

                with st.container(border=True):
                    st.markdown(f"**{timestamp}** · {severity}")

                    if pd.notna(explanation):
                        st.write(explanation)

                    st.write(f"**Suggested action:** {recommendation}")

                    info_col1, info_col2 = st.columns(2)

                    info_col1.metric(
                        "Actual power",
                        f"{format_number(actual, 3)} kW",
                    )
                    info_col2.metric(
                        "Baseline power",
                        f"{format_number(baseline, 3)} kW",
                    )

                    saving_col1, saving_col2 = st.columns(2)

                    saving_col1.metric(
                        "Potential energy saving",
                        f"{format_number(saving_kwh, 3)} kWh",
                    )
                    saving_col2.metric(
                        "Potential cost saving",
                        f"₹{format_number(saving_cost, 2)}",
                    )

except Exception as error:
    show_error("optimization recommendations", error)


# ---------------------------------------------------------
# 5. What-if savings calculator
# ---------------------------------------------------------
st.header("5. What-if Savings Calculator")

st.write(
    "Estimate possible savings by choosing an energy-use amount "
    "and a reduction percentage."
)

with st.container(border=True):
    energy_use = st.number_input(
        "Energy use for the selected period (kWh)",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    reduction_pct = st.slider(
        "Target reduction",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        format="%d%%",
    )

    tariff = st.number_input(
        "Electricity tariff (₹ per kWh)",
        min_value=0.0,
        value=8.0,
        step=0.5,
    )

    period = st.selectbox(
        "Selected period",
        ["Per day", "Per week", "Per month"],
    )

    estimated_energy_saving = energy_use * reduction_pct / 100
    estimated_cost_saving = estimated_energy_saving * tariff

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Estimated energy saving",
        f"{estimated_energy_saving:,.2f} kWh",
    )
    result_col2.metric(
        "Estimated cost saving",
        f"₹{estimated_cost_saving:,.2f}",
    )

    st.caption(
        f"Estimate for {period.lower()}. The ₹8/kWh starting tariff is "
        "illustrative—replace it with your actual electricity tariff. "
        "Savings are hypothetical, not measured results."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "SmartEnergy · Exasol Personal is used as the primary data platform. "
    "Charts and interactive presentation are handled by Streamlit."
)
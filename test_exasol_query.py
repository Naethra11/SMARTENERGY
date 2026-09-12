import pyexasol

dsn = "127.0.0.1:8563"
password = input("Enter your Exasol SYS password: ")

con = pyexasol.connect(
    dsn=dsn,
    user="sys",
    password=password,
    encryption=True,
    websocket_sslopt={"cert_reqs": 0},
)

print("Connected to Exasol.")

query = """
SELECT
    rows_per_timestamp,
    COUNT(*) AS number_of_timestamps
FROM (
    SELECT
        "datetime",
        COUNT(*) AS rows_per_timestamp
    FROM SMARTENERGY.ENERGY_PREDICTIONS
    GROUP BY "datetime"
) AS timestamp_counts
GROUP BY rows_per_timestamp
ORDER BY rows_per_timestamp;
"""
result = con.execute(query).fetchall()

for row in result:
    print(row)

con.close()
import pyexasol

dsn = "127.0.0.1:8563"

password = input("Enter your Exasol SYS password: ")

con = pyexasol.connect(
    dsn=dsn,
    user="sys",
    password=password,
    websocket_sslopt={"cert_reqs": 0}
)

print(con.execute("SELECT 1").fetchone())

con.close()
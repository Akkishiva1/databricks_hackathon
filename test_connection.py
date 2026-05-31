import os
from databricks import sql

with sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN"),
) as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 AS test")
        print(cursor.fetchall())

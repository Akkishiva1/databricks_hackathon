"""
Databricks SQL client and query utilities.
"""
import os
import streamlit as st
from databricks import sql


def get_sql_connection():
    """Create and return a Databricks SQL connection."""
    token = os.getenv("DATABRICKS_TOKEN")

    if not token:
        raise ValueError("DATABRICKS_TOKEN is not set in environment variables.")

    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=token,
    )


def escape_sql(value) -> str:
    """Escape SQL string values (single quotes and backslashes)."""
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("'", "''")


@st.cache_data(ttl=300, show_spinner=False)
def run_query_cached(query: str):
    """Execute read-only query with 5-minute cache."""
    import pandas as pd
    
    with get_sql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)


def run_query(query: str):
    """Execute query using cached implementation."""
    return run_query_cached(query)


def run_statement(query: str) -> None:
    """Execute non-SELECT statement."""
    with get_sql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)


def clear_query_cache() -> None:
    """Clear Streamlit read cache."""
    st.cache_data.clear()

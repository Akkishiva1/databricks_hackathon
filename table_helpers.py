"""
Table schema and column utilities for Databricks tables.
"""
import streamlit as st
from databricks_client import get_sql_connection


_ALLOWED_TABLE_PATTERN = __import__('re').compile(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$')


@st.cache_data(ttl=300, show_spinner=False)
def get_table_columns_cached(table_name: str) -> list:
    """Cache DESCRIBE TABLE results for 5 minutes."""
    if not _ALLOWED_TABLE_PATTERN.match(table_name):
        raise ValueError(f"Invalid table name format: {table_name!r}")
    query = f"DESCRIBE TABLE {table_name}"

    with get_sql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    columns = []

    for row in rows:
        col_name = row[0]

        if not col_name:
            continue

        col_name = str(col_name).strip()

        if col_name.startswith("#"):
            continue

        columns.append(col_name)

    return columns


def get_table_columns(table_name: str) -> list:
    """Get columns from table (cached)."""
    return get_table_columns_cached(table_name)

import os
import re
import ast
import json
import uuid
import time
import smtplib
import requests
import pandas as pd
import streamlit as st

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from databricks import sql
from langfuse import get_client, propagate_attributes


# -------------------------------------------------
# Langfuse client
# -------------------------------------------------
langfuse = get_client()


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Recovery Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("Loan Recovery Assistant")
st.caption(
    "Hybrid Loan Recovery multi-agent POC using Databricks Agent Bricks Supervisor, "
    "Custom Dynamic Supervisor, Databricks SQL Warehouse, Databricks GPT OSS 120B, "
    "Audit Logging, and Langfuse Observability."
)


# -------------------------------------------------
# Databricks configs
# -------------------------------------------------
DATABRICKS_HOST = "https://adb-7405606631030160.0.azuredatabricks.net"

LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
LLM_ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT_NAME}/invocations"

SUPERVISOR_AGENT_ENDPOINT_NAME = "mas-05b00371-endpoint"
SUPERVISOR_AGENT_ENDPOINT_URL = (
    f"{DATABRICKS_HOST}/serving-endpoints/"
    f"{SUPERVISOR_AGENT_ENDPOINT_NAME}/invocations"
)

CUSTOMER_360_TABLE = os.getenv(
    "CUSTOMER_360_TABLE",
    "loan_recovery.gold.loan_recovery_customer_360"
)

AUDIT_TABLE = os.getenv(
    "AUDIT_TABLE",
    "loan_recovery.gold.loan_recovery_agent_audit_log"
)


# -------------------------------------------------
# Pricing helpers for Langfuse backend only
# -------------------------------------------------
INPUT_DBUS_PER_1M_TOKENS = 2.143
OUTPUT_DBUS_PER_1M_TOKENS = 8.571


def calculate_dbu_cost(input_tokens: int, output_tokens: int) -> dict:
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    input_dbus = (input_tokens / 1_000_000) * INPUT_DBUS_PER_1M_TOKENS
    output_dbus = (output_tokens / 1_000_000) * OUTPUT_DBUS_PER_1M_TOKENS
    total_dbus = input_dbus + output_dbus

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_dbus": input_dbus,
        "output_dbus": output_dbus,
        "total_dbus": total_dbus,
    }


def extract_usage(result: dict) -> dict:
    usage = result.get("usage", {}) or {}

    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input_token_count")
        or 0
    )

    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("output_token_count")
        or 0
    )

    return calculate_dbu_cost(input_tokens, output_tokens)


# -------------------------------------------------
# Generic helpers
# -------------------------------------------------
def get_sql_connection():
    token = os.getenv("DATABRICKS_TOKEN")

    if not token:
        raise ValueError("DATABRICKS_TOKEN is not set in environment variables.")

    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=token,
    )


def escape_sql(value) -> str:
    if value is None:
        return ""
    return str(value).replace("'", "''")


@st.cache_data(ttl=300, show_spinner=False)
def run_query_cached(query: str) -> pd.DataFrame:
    """
    Read-only Databricks SQL cache.
    Same query text will be served from cache for 5 minutes.
    Do not use this for INSERT/UPDATE/DELETE statements.
    """
    with get_sql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)


def run_query(query: str) -> pd.DataFrame:
    return run_query_cached(query)


def clear_query_cache() -> None:
    """Clear Streamlit read cache."""
    st.cache_data.clear()


# -------------------------------------------------
# 5-minute recovery question discovery cache
# -------------------------------------------------
DISCOVERY_CACHE_TTL_SECONDS = 300
DISCOVERY_CACHE_VERSION = "v5-intent-cache-fix"


def normalize_question(question: str) -> str:
    """Normalize natural-language question for cache matching."""
    return " ".join((question or "").strip().lower().split())


def get_discovery_cache_key(discovery_mode: str, question: str) -> str:
    return f"{DISCOVERY_CACHE_VERSION}::{discovery_mode}::{normalize_question(question)}"


def is_low_risk_no_escalation_question(question: str) -> bool:
    """Detect common business wording for safe/non-critical/no-escalation customers."""
    q = normalize_question(question)

    low_risk_phrases = [
        "normal customer",
        "normal customers",
        "regular customer",
        "regular customers",
        "safe customer",
        "safe customers",
        "low risk customer",
        "low risk customers",
        "low-risk customer",
        "low-risk customers",
        "non critical",
        "non-critical",
        "not critical",
        "good customer",
        "good customers",
        "healthy customer",
        "healthy customers",
    ]

    no_escalation_phrases = [
        "no escalation",
        "without escalation",
        "not need escalation",
        "not needs escalation",
        "do not need escalation",
        "does not need escalation",
        "dont need escalation",
        "don't need escalation",
        "not require escalation",
        "does not require escalation",
        "do not require escalation",
    ]

    return any(p in q for p in low_risk_phrases) or any(p in q for p in no_escalation_phrases)


def get_cached_discovery_result(discovery_mode: str, question: str):
    """Return cached customer discovery result if present and not expired."""
    cache_key = get_discovery_cache_key(discovery_mode, question)
    cached = st.session_state.discovery_cache.get(cache_key)

    if not cached:
        return None

    age_seconds = time.time() - cached["created_at"]

    if age_seconds > DISCOVERY_CACHE_TTL_SECONDS:
        del st.session_state.discovery_cache[cache_key]
        return None

    return cached


def set_cached_discovery_result(
    discovery_mode: str,
    question: str,
    df: pd.DataFrame,
    query_params: dict,
    discovery_summary: str,
) -> None:
    """Cache the full recovery-question discovery result for 5 minutes."""
    cache_key = get_discovery_cache_key(discovery_mode, question)

    # Do not cache empty results. This avoids reusing a temporary/no-match answer
    # when the user is testing different phrasings during the demo.
    if df is None or df.empty:
        return

    st.session_state.discovery_cache[cache_key] = {
        "created_at": time.time(),
        "df": df.copy(),
        "query_params": query_params,
        "discovery_summary": discovery_summary or "",
    }


def run_statement(query: str) -> None:
    with get_sql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)


def get_default_customer_email(customer: dict) -> str:
    """
    Returns the first available customer email field from the selected row.
    If the Gold table does not have an email field, the UI will allow manual entry.
    """
    email_fields = [
        "email",
        "customer_email",
        "email_address",
        "customer_email_address",
        "primary_email",
        "contact_email",
    ]

    for field in email_fields:
        value = customer.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value and value.lower() not in ["nan", "none", "null"]:
            return value

    return ""


def get_default_customer_name(customer: dict) -> str:
    """
    Returns the first available customer name field from the selected row.
    This is used only for display/approval before sending email.
    """
    name_fields = [
        "name",
        "customer_name",
        "full_name",
        "member_name",
        "borrower_name",
        "first_name",
    ]

    for field in name_fields:
        value = customer.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value and value.lower() not in ["nan", "none", "null"]:
            return value

    return ""


def contains_kannada(text: str) -> bool:
    """Return True if text contains Kannada script characters."""
    return bool(re.search(r"[\u0C80-\u0CFF]", text or ""))


def contains_devanagari(text: str) -> bool:
    """Return True if text contains Devanagari script characters, commonly used for Hindi."""
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def send_email_notification(to_email: str, subject: str, body: str) -> dict:
    """
    Sends the final human-approved recovery email through SMTP.

    Required environment variables:
    - SMTP_HOST
    - SMTP_PORT
    - SMTP_USERNAME
    - SMTP_PASSWORD
    - SENDER_EMAIL

    For demo, use a test mailbox instead of a real customer email.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SENDER_EMAIL", smtp_username)

    if not smtp_host or not smtp_username or not smtp_password or not sender_email:
        raise ValueError(
            "SMTP configuration missing. Please set SMTP_HOST, SMTP_PORT, "
            "SMTP_USERNAME, SMTP_PASSWORD, and SENDER_EMAIL."
        )

    to_email = (to_email or "").strip()
    subject = (subject or "").strip()
    body = (body or "").strip()

    if not to_email:
        raise ValueError("Recipient email is required before sending.")

    if not subject:
        raise ValueError("Email subject is required before sending.")

    if not body:
        raise ValueError("Email body cannot be empty.")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(message)

    return {
        "to_email": to_email,
        "subject": subject,
        "status": "sent",
    }


def parse_possible_dict(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def extract_json_from_text(text: str) -> dict:
    """
    Extracts JSON even if model returns ```json ... ```.
    """
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "")
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from LLM response: {text}")


def clean_agent_text(text: str) -> str:
    """
    Removes raw tool call dictionaries and internal tool tags from final answer.
    """
    if not text:
        return ""

    clean_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("{'type': 'function_call'"):
            continue
        if stripped.startswith('{"type": "function_call"'):
            continue
        if stripped.startswith("{'type': 'function_call_output'"):
            continue
        if stripped.startswith('{"type": "function_call_output"'):
            continue
        if stripped.startswith("<name>") and stripped.endswith("</name>"):
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def extract_tool_traces_from_text(text: str) -> list:
    """
    Extracts tool traces for Langfuse metadata only.
    Not shown in Streamlit UI.
    """
    traces = []

    if not text:
        return traces

    for line in text.splitlines():
        stripped = line.strip()

        is_tool_line = (
            stripped.startswith("{'type': 'function_call'")
            or stripped.startswith('{"type": "function_call"')
            or stripped.startswith("{'type': 'function_call_output'")
            or stripped.startswith('{"type": "function_call_output"')
        )

        if is_tool_line:
            parsed = parse_possible_dict(stripped)
            traces.append(parsed if parsed else stripped)

    return traces


def safe_customer_context(customer: dict) -> dict:
    """
    Keep only useful demo fields in Langfuse trace.
    Avoid logging unnecessary sensitive PII.
    """
    return {
        "member_id": customer.get("member_id"),
        "loan_id": customer.get("loan_id"),
        "name": get_default_customer_name(customer),
        "email": get_default_customer_email(customer),
        "dpd": customer.get("dpd"),
        "risk_score": customer.get("risk_score"),
        "risk_band": customer.get("risk_band"),
        "loan_amount": customer.get("loan_amount"),
        "funded_amount": customer.get("funded_amount"),
        "loan_status": customer.get("loan_status"),
        "loan_purpose": customer.get("loan_purpose"),
        "recommended_action": customer.get("recommended_action"),
        "escalation_flag": customer.get("escalation_flag"),
        "broken_ptp": customer.get("broken_ptp"),
        "refused_count": customer.get("refused_count"),
    }


# -------------------------------------------------
# Langfuse scoring helpers
# -------------------------------------------------
def add_success_score(span, name: str, comment: str = ""):
    try:
        span.score(
            name=name,
            value=1,
            data_type="BOOLEAN",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add boolean score {name}: {e}")


def add_trace_quality_score(span, name: str, value: float, comment: str = ""):
    try:
        span.score_trace(
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add numeric trace score {name}: {e}")


def add_categorical_score(span, name: str, value: str, comment: str = ""):
    try:
        span.score_trace(
            name=name,
            value=value,
            data_type="CATEGORICAL",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add categorical score {name}: {e}")


# -------------------------------------------------
# Table schema helpers
# -------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_table_columns_cached(table_name: str) -> list:
    """
    Cache DESCRIBE TABLE results for 5 minutes.
    This avoids repeated schema lookups for every customer discovery query.
    """
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
    return get_table_columns_cached(table_name)


# -------------------------------------------------
# LLM response parsing fix
# -------------------------------------------------
def extract_text_from_content_blocks(content_blocks) -> str:
    """
    Extract only final text blocks and ignore GPT OSS reasoning blocks.
    """
    text_parts = []

    if not isinstance(content_blocks, list):
        return str(content_blocks).strip()

    for item in content_blocks:
        if not isinstance(item, dict):
            continue

        block_type = item.get("type")

        # Ignore reasoning blocks
        if block_type == "reasoning":
            continue

        if block_type in ["text", "output_text"]:
            text_value = item.get("text") or item.get("content") or ""
            if text_value:
                text_parts.append(str(text_value))

        elif "text" in item and block_type != "reasoning":
            text_parts.append(str(item["text"]))

        elif "content" in item and block_type != "reasoning":
            text_parts.append(str(item["content"]))

    final_text = "\n".join(text_parts).strip()

    if final_text:
        return final_text

    return "No final message was returned by the model. Please retry with a shorter instruction."


def extract_llm_text(result: dict) -> str:
    """
    Handles:
    - normal string content
    - list content with reasoning/text blocks
    - stringified list content
    """
    try:
        content = result["choices"][0]["message"]["content"]

        if isinstance(content, str):
            cleaned = content.strip()

            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = ast.literal_eval(cleaned)
                    return extract_text_from_content_blocks(parsed)
                except Exception:
                    return cleaned

            return cleaned

        if isinstance(content, list):
            return extract_text_from_content_blocks(content)

        return str(content).strip()

    except Exception as e:
        return f"Unable to parse LLM response: {e}"


# -------------------------------------------------
# LLM call
# -------------------------------------------------
def call_databricks_llm(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 700,
    agent_name: str = "llm_agent"
) -> str:
    token = os.getenv("DATABRICKS_TOKEN")

    if not token:
        return "DATABRICKS_TOKEN is not set in environment variables."

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    with langfuse.start_as_current_observation(
        as_type="generation",
        name=agent_name,
        model=LLM_ENDPOINT_NAME,
        input={
            "messages": payload["messages"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        metadata={
            "provider": "databricks",
            "endpoint": LLM_ENDPOINT_NAME,
        }
    ) as generation:
        try:
            response = requests.post(
                LLM_ENDPOINT_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                error_text = f"Error from Databricks LLM endpoint: {response.text}"

                generation.update(
                    output=error_text,
                    level="ERROR",
                    metadata={
                        "status_code": response.status_code,
                        "endpoint": LLM_ENDPOINT_NAME,
                    }
                )

                return error_text

            result = response.json()
            text = extract_llm_text(result)
            usage = result.get("usage", {}) or {}
            cost = extract_usage(result)

            generation.update(
                output=text,
                usage_details={
                    "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
                    "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                cost_details={
                    "input": cost["input_dbus"],
                    "output": cost["output_dbus"],
                    "total": cost["total_dbus"],
                    "unit": "DBU",
                },
                metadata={
                    "status_code": response.status_code,
                    "endpoint": LLM_ENDPOINT_NAME,
                    "pricing": {
                        "input_dbus_per_1m_tokens": INPUT_DBUS_PER_1M_TOKENS,
                        "output_dbus_per_1m_tokens": OUTPUT_DBUS_PER_1M_TOKENS,
                    }
                }
            )

            return text

        except Exception as e:
            error_text = f"Error calling Databricks LLM endpoint: {e}"

            generation.update(
                output=error_text,
                level="ERROR",
                metadata={
                    "endpoint": LLM_ENDPOINT_NAME,
                    "error": str(e),
                }
            )

            return error_text


# -------------------------------------------------
# Agent Bricks Supervisor Discovery
# -------------------------------------------------
def build_agent_bricks_supervisor_question(user_question: str) -> str:
    return f"""
User question:
{user_question}

You are a Loan Recovery Supervisor Agent.

Use available Databricks tools such as Genie Space and Unity Catalog functions to answer the user's question.

Return valid JSON only. Do not return markdown. Do not include tool call details in the final answer.

Use exactly this JSON structure:
{{
  "summary": "short business summary of the result",
  "customers": [
    {{
      "member_id": "string",
      "loan_id": "string",
      "name": "string",
      "email": "string",
      "dpd": 0,
      "risk_score": 0,
      "risk_band": "Low | Medium | High | Critical",
      "loan_amount": 0,
      "funded_amount": 0,
      "loan_status": "string",
      "loan_purpose": "string",
      "recommended_action": "string",
      "escalation_flag": "Y or N",
      "broken_ptp": 0,
      "refused_count": 0
    }}
  ]
}}

Rules:
- The user question can be broad or specific.
- Retrieve only customers relevant to the user question.
- If user asks for normal customers, regular customers, safe customers, low risk customers, good customers, healthy customers, or non-critical customers, return only Low risk customers with no escalation if available.
- If user asks customers who do not need escalation, not need escalation, without escalation, no escalation, or does not require escalation, return only customers where escalation_flag is N. Prefer Low risk customers if available.
- If no matching customers are found, return an empty customers array.
- If escalation_flag is null, treat it as N.
- Include customer name and email only if available in the source data.
- Do not invent customer name, email, loan, DPD, risk score, or amount values.
- Limit to maximum 10 customers.
"""


def parse_agent_bricks_response(result: dict) -> dict:
    parsed = {
        "summary": "",
        "customers": [],
        "final_answer": "",
        "tool_trace": [],
        "raw_response": result,
        "source": "Agent Bricks Supervisor",
    }

    output = result.get("output")

    if isinstance(output, str):
        parsed["tool_trace"] = extract_tool_traces_from_text(output)
        cleaned_text = clean_agent_text(output)

        try:
            json_obj = extract_json_from_text(cleaned_text)
            parsed["summary"] = json_obj.get("summary", "")
            parsed["customers"] = json_obj.get("customers", []) or []
            parsed["final_answer"] = parsed["summary"]
        except Exception:
            parsed["final_answer"] = cleaned_text

        return parsed

    if isinstance(output, list):
        text_parts = []

        for item in output:
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue

            item_type = item.get("type")

            if item_type in ["function_call", "tool_call", "function_call_output", "tool_result"]:
                parsed["tool_trace"].append(item)
                continue

            content = item.get("content")

            if isinstance(content, str):
                parsed["tool_trace"].extend(extract_tool_traces_from_text(content))
                text_parts.append(clean_agent_text(content))

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type in ["function_call", "tool_call", "function_call_output", "tool_result"]:
                            parsed["tool_trace"].append(block)
                        elif "text" in block:
                            block_text = str(block["text"])
                            parsed["tool_trace"].extend(extract_tool_traces_from_text(block_text))
                            text_parts.append(clean_agent_text(block_text))
                        elif "content" in block:
                            block_text = str(block["content"])
                            parsed["tool_trace"].extend(extract_tool_traces_from_text(block_text))
                            text_parts.append(clean_agent_text(block_text))
                    else:
                        text_parts.append(str(block))

            elif "text" in item:
                item_text = str(item["text"])
                parsed["tool_trace"].extend(extract_tool_traces_from_text(item_text))
                text_parts.append(clean_agent_text(item_text))

        full_text = "\n\n".join([t for t in text_parts if t.strip()]).strip()

        try:
            json_obj = extract_json_from_text(full_text)
            parsed["summary"] = json_obj.get("summary", "")
            parsed["customers"] = json_obj.get("customers", []) or []
            parsed["final_answer"] = parsed["summary"]
        except Exception:
            parsed["final_answer"] = full_text

        return parsed

    parsed["final_answer"] = json.dumps(result, indent=2)
    return parsed


def agent_bricks_supervisor_discovery(user_question: str) -> pd.DataFrame:
    token = os.getenv("DATABRICKS_TOKEN")

    if not token:
        st.error("DATABRICKS_TOKEN is not set in environment variables.")
        return pd.DataFrame()

    supervisor_question = build_agent_bricks_supervisor_question(user_question)

    payload = {
        "input": [
            {
                "role": "user",
                "content": supervisor_question
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="agent_bricks_supervisor_discovery",
        model=SUPERVISOR_AGENT_ENDPOINT_NAME,
        input=payload,
        metadata={
            "provider": "databricks",
            "endpoint_type": "agent_bricks_supervisor",
            "endpoint_url": SUPERVISOR_AGENT_ENDPOINT_URL,
        }
    ) as generation:
        try:
            response = requests.post(
                SUPERVISOR_AGENT_ENDPOINT_URL,
                headers=headers,
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                error_text = f"Supervisor Agent error: {response.status_code} - {response.text}"

                generation.update(
                    output=error_text,
                    level="ERROR",
                    metadata={"status_code": response.status_code}
                )

                st.error(error_text)
                return pd.DataFrame()

            result = response.json()
            parsed = parse_agent_bricks_response(result)
            cost = extract_usage(result)

            generation.update(
                output=parsed["final_answer"],
                usage_details={
                    "input_tokens": cost["input_tokens"],
                    "output_tokens": cost["output_tokens"],
                    "total_tokens": cost["total_tokens"],
                },
                cost_details={
                    "input": cost["input_dbus"],
                    "output": cost["output_dbus"],
                    "total": cost["total_dbus"],
                    "unit": "DBU",
                },
                metadata={
                    "status_code": response.status_code,
                    "tool_trace": parsed["tool_trace"],
                    "raw_response": result,
                    "pricing": {
                        "input_dbus_per_1m_tokens": INPUT_DBUS_PER_1M_TOKENS,
                        "output_dbus_per_1m_tokens": OUTPUT_DBUS_PER_1M_TOKENS,
                    }
                }
            )

            customers = parsed.get("customers", []) or []

            st.session_state.discovery_summary = parsed.get(
                "summary",
                f"Agent Bricks returned {len(customers)} customer(s)."
            )

            return pd.DataFrame(customers)

        except Exception as e:
            error_text = f"Error calling Agent Bricks Supervisor endpoint: {e}"

            generation.update(
                output=error_text,
                level="ERROR",
                metadata={"error": str(e)}
            )

            st.error(error_text)
            return pd.DataFrame()


# -------------------------------------------------
# Custom Dynamic Supervisor - Agent 1: Query Understanding
# -------------------------------------------------
def query_understanding_agent(user_question: str) -> dict:
    prompt = f"""
You are a Query Understanding Agent for a Loan Recovery Assistant.

Convert the user question into SQL filter parameters.

User question:
{user_question}

Available fields:
- dpd
- risk_score
- risk_band: Low, Medium, High, Critical
- escalation_flag: Y or N
- loan_status
- loan_purpose
- broken_ptp
- refused_count

Return only valid JSON with this schema:
{{
  "risk_band": "All | Low | Medium | High | Critical",
  "min_dpd": integer,
  "max_dpd": integer_or_null,
  "escalation_only": true_or_false,
  "no_escalation_only": true_or_false,
  "broken_ptp_only": true_or_false,
  "refused_only": true_or_false,
  "limit": integer,
  "sort_by": "risk_score | dpd"
}}

Rules:
- If user says critical, set risk_band to Critical.
- If user says high-risk, set risk_band to High unless they clearly mean all risky customers.
- If user asks escalation, set escalation_only to true.
- If user says "need escalation", "needs escalation", "require escalation", "requires escalation", or "escalation required", set escalation_only to true.
- If user says "no escalation", "without escalation", "not need escalation", "do not need escalation", "does not need escalation", "not require escalation", or "does not require escalation", set no_escalation_only to true and escalation_only to false.
- If user mentions broken PTP, broken promise to pay, or promise to pay broken, set broken_ptp_only to true.
- If user mentions refused, refusal, or refused count, set refused_only to true.
- If user asks for normal customers, regular customers, safe customers, low risk customers, good customers, or non-critical customers:
  set risk_band to Low, min_dpd to 0, max_dpd to 29, escalation_only to false, no_escalation_only to true.
- If user asks for normal customers, do not return High or Critical customers.
- If DPD is not mentioned, use min_dpd 0.
- If max DPD is not needed, use null.
- If user asks top overdue or highest DPD, sort_by should be dpd.
- If limit is not mentioned, use limit 5.
- Use sort_by risk_score by default.
- Return JSON only.
"""

    response = call_databricks_llm(
        prompt,
        temperature=0.1,
        max_tokens=300,
        agent_name="query_understanding_agent"
    )

    try:
        params = extract_json_from_text(response)
    except Exception:
        params = {
            "risk_band": "All",
            "min_dpd": 0,
            "max_dpd": None,
            "escalation_only": False,
            "no_escalation_only": False,
            "broken_ptp_only": False,
            "refused_only": False,
            "limit": 5,
            "sort_by": "risk_score"
        }

    params["risk_band"] = params.get("risk_band", "All")
    params["min_dpd"] = int(params.get("min_dpd", 0))
    params["max_dpd"] = params.get("max_dpd")
    params["escalation_only"] = bool(params.get("escalation_only", False))
    params["no_escalation_only"] = bool(params.get("no_escalation_only", False))
    params["broken_ptp_only"] = bool(params.get("broken_ptp_only", False))
    params["refused_only"] = bool(params.get("refused_only", False))
    params["limit"] = int(params.get("limit", 5))
    params["sort_by"] = params.get("sort_by", "risk_score")

    question_lower = user_question.lower()

    normal_customer_phrases = [
        "normal customer",
        "normal customers",
        "regular customer",
        "regular customers",
        "safe customer",
        "safe customers",
        "low risk customer",
        "low risk customers",
        "low-risk customer",
        "low-risk customers",
        "non critical",
        "non-critical",
        "not critical",
        "good customer",
        "good customers",
        "healthy customer",
        "healthy customers",
    ]

    if any(phrase in question_lower for phrase in normal_customer_phrases):
        params["risk_band"] = "Low"
        params["min_dpd"] = 0
        params["max_dpd"] = 29
        params["escalation_only"] = False
        params["no_escalation_only"] = True
        params["broken_ptp_only"] = False
        params["refused_only"] = False
        params["sort_by"] = "dpd"

    escalation_phrases = [
        "need escalation",
        "needs escalation",
        "require escalation",
        "requires escalation",
        "escalation required",
        "who need escalation",
        "who needs escalation",
        "customers need escalation",
        "customers needing escalation",
    ]

    no_escalation_phrases = [
        "no escalation",
        "without escalation",
        "not need escalation",
        "not needs escalation",
        "do not need escalation",
        "does not need escalation",
        "dont need escalation",
        "don't need escalation",
        "not require escalation",
        "does not require escalation",
        "do not require escalation",
    ]

    if any(phrase in question_lower for phrase in no_escalation_phrases):
        params["escalation_only"] = False
        params["no_escalation_only"] = True

        if any(phrase in question_lower for phrase in normal_customer_phrases) or "non critical" in question_lower or "non-critical" in question_lower:
            params["risk_band"] = "Low"
            params["min_dpd"] = 0
            params["max_dpd"] = 29

    elif any(phrase in question_lower for phrase in escalation_phrases):
        params["escalation_only"] = True
        params["no_escalation_only"] = False

    if "critical" in question_lower and not any(phrase in question_lower for phrase in normal_customer_phrases) and not any(phrase in question_lower for phrase in no_escalation_phrases):
        params["risk_band"] = "Critical"

    if "broken ptp" in question_lower or "broken promise" in question_lower or "promise to pay" in question_lower:
        params["broken_ptp_only"] = True

    if "refused" in question_lower or "refusal" in question_lower:
        params["refused_only"] = True

    if "highest dpd" in question_lower or "top overdue" in question_lower or "most overdue" in question_lower:
        params["sort_by"] = "dpd"

    if params["risk_band"] not in ["All", "Low", "Medium", "High", "Critical"]:
        params["risk_band"] = "All"

    if params["sort_by"] not in ["risk_score", "dpd"]:
        params["sort_by"] = "risk_score"

    params["limit"] = max(1, min(params["limit"], 20))

    return params


# -------------------------------------------------
# Custom Dynamic Supervisor - Agent 2: Data Retrieval
# -------------------------------------------------
def data_retrieval_agent(params: dict) -> pd.DataFrame:
    risk_band = params["risk_band"]
    min_dpd = params["min_dpd"]
    max_dpd = params.get("max_dpd")
    escalation_only = params["escalation_only"]
    no_escalation_only = params.get("no_escalation_only", False)
    broken_ptp_only = params.get("broken_ptp_only", False)
    refused_only = params.get("refused_only", False)
    limit = params["limit"]
    sort_by = params["sort_by"]

    available_columns = get_table_columns(CUSTOMER_360_TABLE)
    available_columns_lower = {c.lower(): c for c in available_columns}

    preferred_columns = [
        "member_id",
        "loan_id",
        "name",
        "customer_name",
        "full_name",
        "member_name",
        "borrower_name",
        "first_name",
        "email",
        "customer_email",
        "email_address",
        "customer_email_address",
        "primary_email",
        "contact_email",
        "emp_title",
        "annual_income",
        "loan_amount",
        "funded_amount",
        "loan_status",
        "loan_purpose",
        "interest_rate",
        "monthly_installment",
        "dpd",
        "risk_score",
        "risk_band",
        "recommended_action",
        "escalation_flag",
        "broken_ptp",
        "refused_count",
    ]

    select_columns = []

    for col in preferred_columns:
        if col.lower() in available_columns_lower:
            actual_col = available_columns_lower[col.lower()]

            if col.lower() == "recommended_action":
                select_columns.append(
                    f"COALESCE({actual_col}, 'Review account and follow standard recovery process') AS recommended_action"
                )
            elif col.lower() == "escalation_flag":
                select_columns.append(
                    f"COALESCE({actual_col}, 'N') AS escalation_flag"
                )
            else:
                select_columns.append(actual_col)

    if not select_columns:
        select_columns = ["*"]

    where_clauses = ["1 = 1"]

    if "dpd" in available_columns_lower:
        dpd_col = available_columns_lower["dpd"]
        where_clauses.append(f"{dpd_col} >= {int(min_dpd)}")

        if max_dpd is not None:
            where_clauses.append(f"{dpd_col} <= {int(max_dpd)}")

    if risk_band != "All" and "risk_band" in available_columns_lower:
        risk_band_col = available_columns_lower["risk_band"]
        where_clauses.append(f"lower({risk_band_col}) = lower('{escape_sql(risk_band)}')")

    if escalation_only and "escalation_flag" in available_columns_lower:
        escalation_col = available_columns_lower["escalation_flag"]
        where_clauses.append(f"upper({escalation_col}) = 'Y'")

    if no_escalation_only and "escalation_flag" in available_columns_lower:
        escalation_col = available_columns_lower["escalation_flag"]
        where_clauses.append(f"upper(COALESCE({escalation_col}, 'N')) = 'N'")

    if broken_ptp_only and "broken_ptp" in available_columns_lower:
        broken_ptp_col = available_columns_lower["broken_ptp"]
        where_clauses.append(f"{broken_ptp_col} > 0")

    if refused_only and "refused_count" in available_columns_lower:
        refused_col = available_columns_lower["refused_count"]
        where_clauses.append(f"{refused_col} > 0")

    where_sql = "\n      AND ".join(where_clauses)

    order_by_parts = []

    if sort_by == "dpd" and "dpd" in available_columns_lower:
        order_by_parts.append(f"{available_columns_lower['dpd']} DESC")

    if "risk_score" in available_columns_lower:
        order_by_parts.append(f"{available_columns_lower['risk_score']} DESC")

    if "dpd" in available_columns_lower and sort_by != "dpd":
        order_by_parts.append(f"{available_columns_lower['dpd']} DESC")

    order_by_sql = ""
    if order_by_parts:
        order_by_sql = "ORDER BY " + ", ".join(order_by_parts)

    query = f"""
    SELECT
      {", ".join(select_columns)}
    FROM {CUSTOMER_360_TABLE}
    WHERE {where_sql}
    {order_by_sql}
    LIMIT {int(limit)}
    """

    with langfuse.start_as_current_observation(
        as_type="span",
        name="data_retrieval_agent",
        input={
            "params": params,
            "table": CUSTOMER_360_TABLE,
            "query": query,
            "available_columns": available_columns,
        },
        metadata={
            "warehouse_http_path": os.getenv("DATABRICKS_HTTP_PATH"),
        }
    ) as span:
        df = run_query(query)

        span.update(
            output={
                "records_returned": len(df),
                "columns": list(df.columns),
            }
        )

        add_success_score(
            span,
            name="data_retrieval_completed",
            comment="Data Retrieval Agent successfully queried Databricks Gold table"
        )

        add_categorical_score(
            span,
            name="data_retrieval_status",
            value="success",
            comment="Data Retrieval Agent completed successfully"
        )

        return df


def custom_dynamic_supervisor_discovery(user_question: str) -> tuple[pd.DataFrame, dict]:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="custom_dynamic_supervisor_agent",
        input={
            "user_question": user_question,
        },
        metadata={
            "app": "loan_recovery_assistant",
            "source_table": CUSTOMER_360_TABLE,
            "llm_endpoint": LLM_ENDPOINT_NAME,
        }
    ) as root_span:
        with propagate_attributes(
            user_id="demo_user",
            session_id="loan_recovery_demo_session",
            tags=["loan-recovery", "databricks", "custom-supervisor"]
        ):
            with st.spinner("Query Understanding Agent is interpreting your question..."):
                params = query_understanding_agent(user_question)

            with st.spinner("Data Retrieval Agent is querying Databricks Gold table..."):
                df = data_retrieval_agent(params)

            root_span.update(
                output={
                    "query_params": params,
                    "records_returned": 0 if df is None else len(df),
                }
            )

            add_success_score(
                root_span,
                name="custom_supervisor_agent_completed",
                comment="Custom Supervisor completed query understanding and data retrieval"
            )

            add_trace_quality_score(
                root_span,
                name="custom_supervisor_workflow_success",
                value=1.0,
                comment="Custom Supervisor workflow completed successfully"
            )

            add_categorical_score(
                root_span,
                name="custom_supervisor_status",
                value="success",
                comment="Custom Supervisor completed successfully"
            )

    return df, params


# -------------------------------------------------
# Common downstream agents
# -------------------------------------------------
def risk_analysis_agent(customer: dict, user_question: str) -> str:
    prompt = f"""
You are a Risk Analysis Agent for a Loan Recovery Assistant.

User question:
{user_question}

Customer/Loan data:
{json.dumps(customer, default=str, indent=2)}

Generate a clear risk explanation.

Rules:
- Use only the given data.
- Explain why this customer is risky.
- Mention DPD, risk score, risk band, loan amount/funded amount, loan status, and loan purpose if available.
- If escalation_flag is Y, mention escalation is required.
- If broken_ptp or refused_count is available, use it as recovery behavior context.
- Do not invent details.
- Keep it concise and demo-friendly.
"""

    return call_databricks_llm(
        prompt,
        temperature=0.2,
        max_tokens=500,
        agent_name="risk_analysis_agent"
    )


def recommendation_agent(customer: dict, risk_explanation: str) -> str:
    prompt = f"""
You are a Recommendation Agent for a Loan Recovery Assistant.

Customer/Loan data:
{json.dumps(customer, default=str, indent=2)}

Risk explanation:
{risk_explanation}

Recommend the next-best recovery action.

Rules:
- Use the recommended_action field if available.
- If escalation_flag is Y, clearly say escalation is required.
- If broken_ptp or refused_count is available, use it as recovery behavior context.
- Explain the reason in 2-3 bullet points.
- Do not invent details.
"""

    return call_databricks_llm(
        prompt,
        temperature=0.2,
        max_tokens=500,
        agent_name="recommendation_agent"
    )


def communication_agent(customer: dict, recommendation: str) -> str:
    prompt = f"""
You are a Communication Agent for a Loan Recovery Assistant.

Customer/Loan data:
{json.dumps(customer, default=str, indent=2)}

Recommendation:
{recommendation}

Draft a professional follow-up message for the customer.

Rules:
- Use only the given context.
- Mention loan ID if available.
- Mention days past due if available.
- Keep tone polite and professional.
- If escalation_flag is Y or risk_band is Critical, mention urgency carefully.
- Do not mention internal risk score unless needed.
- Return only the message draft.
"""

    return call_databricks_llm(
        prompt,
        temperature=0.3,
        max_tokens=500,
        agent_name="communication_agent"
    )


def rephrase_agent(original_message: str, instruction: str, customer: dict) -> str:
    prompt = f"""
You are a Dynamic Rephrase Agent for a Loan Recovery Assistant.

Customer/Loan data:
{json.dumps(customer, default=str, indent=2)}

Original message:
{original_message}

User instruction:
{instruction}

Rewrite the message according to the user instruction.

Very important language rules:
- If the user asks for Kannada, return the entire customer-facing message only in Kannada script.
- If the user asks for Hindi, return the entire customer-facing message only in Hindi script.
- If the user asks for English, return the entire customer-facing message only in English.
- Do not mix Hindi and Kannada.
- Do not mix multiple languages unless the user explicitly asks.
- Preserve important values exactly, such as loan ID, DPD, amount, phone number, and dates.
- If a placeholder like [phone number] is present, keep it as [phone number].
- Do not translate loan ID, amount, or numeric values.

Rules:
- Return only the rewritten customer-facing message.
- Do not include reasoning.
- Do not include explanation.
- Do not include JSON.
- Do not include markdown.
- Do not invent new customer or loan details.
- Use only the given context.
- Keep it suitable for loan recovery.
"""

    return call_databricks_llm(
        prompt,
        temperature=0.1,
        max_tokens=700,
        agent_name="rephrase_agent"
    )


def audit_logger_agent(
    user_query: str,
    customer: dict,
    recommended_action: str,
    reason: str,
    message_draft: str,
    discovery_mode: str
) -> str:
    agent_request_id = str(uuid.uuid4())

    insert_sql = f"""
    INSERT INTO {AUDIT_TABLE} (
      agent_request_id,
      user_query,
      customer_id,
      loan_id,
      recommended_action,
      reason,
      message_draft,
      source_tables_used,
      created_by,
      created_at
    )
    SELECT
      '{agent_request_id}' AS agent_request_id,
      '{escape_sql(user_query)}' AS user_query,
      '{escape_sql(customer.get("member_id"))}' AS customer_id,
      '{escape_sql(customer.get("loan_id"))}' AS loan_id,
      '{escape_sql(recommended_action)}' AS recommended_action,
      '{escape_sql(reason)}' AS reason,
      '{escape_sql(message_draft)}' AS message_draft,
      array('{CUSTOMER_360_TABLE}') AS source_tables_used,
      'hybrid_agent_bricks_custom_supervisor_{escape_sql(discovery_mode)}' AS created_by,
      current_timestamp() AS created_at
    """

    with langfuse.start_as_current_observation(
        as_type="span",
        name="audit_logger_agent",
        input={
            "user_query": user_query,
            "customer_id": customer.get("member_id"),
            "loan_id": customer.get("loan_id"),
            "discovery_mode": discovery_mode,
        },
        metadata={
            "audit_table": AUDIT_TABLE,
        }
    ) as span:
        run_statement(insert_sql)

        span.update(
            output={
                "agent_request_id": agent_request_id,
                "created_by": f"hybrid_agent_bricks_custom_supervisor_{discovery_mode}",
            }
        )

        add_success_score(
            span,
            name="audit_log_written",
            comment="Audit Logger Agent wrote the trace to Delta audit table"
        )

        add_categorical_score(
            span,
            name="audit_logger_status",
            value="success",
            comment="Audit Logger Agent completed successfully"
        )

    return agent_request_id


# -------------------------------------------------
# Session state
# -------------------------------------------------
if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "query_params" not in st.session_state:
    st.session_state.query_params = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = {}

if "rephrased_message" not in st.session_state:
    st.session_state.rephrased_message = ""

if "final_messages" not in st.session_state:
    st.session_state.final_messages = {}

if "rephrase_versions" not in st.session_state:
    st.session_state.rephrase_versions = {}

if "discovery_mode" not in st.session_state:
    st.session_state.discovery_mode = None

if "discovery_summary" not in st.session_state:
    st.session_state.discovery_summary = ""

if "discovery_cache" not in st.session_state:
    st.session_state.discovery_cache = {}


# -------------------------------------------------
# Main input
# -------------------------------------------------
with st.form("agent_form"):
    discovery_mode = st.radio(
        "Choose customer discovery mode",
        [
            "Agent Bricks Supervisor",
            "Custom Dynamic Supervisor"
        ],
        horizontal=True
    )

    question = st.text_input(
        "Ask your recovery question",
        value="Show me critical customers who need escalation"
    )

    submitted = st.form_submit_button("Run Customer Discovery")


# -------------------------------------------------
# Run discovery
# -------------------------------------------------
if submitted:
    st.session_state.agent_outputs = {}
    st.session_state.rephrased_message = ""
    st.session_state.final_messages = {}
    st.session_state.rephrase_versions = {}
    st.session_state.discovery_mode = discovery_mode
    st.session_state.last_question = question
    st.session_state.discovery_summary = ""

    cached_result = get_cached_discovery_result(discovery_mode, question)

    if cached_result:
        # Cache is internal; do not show a UI banner during demo.
        st.session_state.result_df = cached_result["df"].copy()
        st.session_state.query_params = cached_result["query_params"]
        st.session_state.discovery_summary = cached_result["discovery_summary"]

    else:
        if discovery_mode == "Agent Bricks Supervisor":
            with st.spinner("Agent Bricks Supervisor is discovering customers..."):
                df = agent_bricks_supervisor_discovery(question)

            query_params = {
                "mode": "Agent Bricks Supervisor",
                "note": "Customer discovery handled by Databricks Agent Bricks Supervisor."
            }
            discovery_summary = st.session_state.discovery_summary

            # Demo safety: for common low-risk/no-escalation questions, if the
            # Agent Bricks path returns no rows, fall back to the deterministic
            # custom supervisor path. This avoids no-match results caused only
            # by natural-language phrasing such as "who not need escalation".
            if (df is None or df.empty) and is_low_risk_no_escalation_question(question):
                with st.spinner("No rows from Agent Bricks; using Custom Dynamic Supervisor fallback..."):
                    df, params = custom_dynamic_supervisor_discovery(question)

                query_params = {
                    "mode": "Agent Bricks Supervisor with Custom Dynamic Supervisor fallback",
                    "agent_bricks_note": "Agent Bricks returned no rows for this phrasing.",
                    "fallback_params": params,
                }
                discovery_summary = (
                    f"Custom fallback returned {0 if df is None else len(df)} customer(s)."
                )

        else:
            df, params = custom_dynamic_supervisor_discovery(question)

            query_params = params
            discovery_summary = (
                f"Custom Dynamic Supervisor returned {0 if df is None else len(df)} customer(s)."
            )

        st.session_state.result_df = df
        st.session_state.query_params = query_params
        st.session_state.discovery_summary = discovery_summary

        set_cached_discovery_result(
            discovery_mode=discovery_mode,
            question=question,
            df=df,
            query_params=query_params,
            discovery_summary=discovery_summary,
        )

        langfuse.flush()


# -------------------------------------------------
# Display results
# -------------------------------------------------
df = st.session_state.result_df

if df is None:
    st.info("Enter a question and click **Run Customer Discovery**.")

elif df.empty:
    st.warning("No matching customers found.")

else:
    discovery_mode = st.session_state.discovery_mode or "Unknown"
    st.success(f"{discovery_mode} found {len(df)} matching customer(s).")

    if st.session_state.discovery_summary:
        st.markdown(st.session_state.discovery_summary)

    total_customers = len(df)
    avg_dpd = round(df["dpd"].mean(), 1) if "dpd" in df.columns else "NA"
    max_risk_score = round(df["risk_score"].max(), 1) if "risk_score" in df.columns else "NA"
    escalation_count = (
        len(df[df["escalation_flag"] == "Y"])
        if "escalation_flag" in df.columns
        else "NA"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Found", total_customers)
    c2.metric("Average DPD", avg_dpd)
    c3.metric("Max Risk Score", max_risk_score)
    c4.metric("Escalations", escalation_count)

    st.markdown("---")
    st.subheader("Customers Returned")

    display_columns = [
        "member_id",
        "loan_id",
        "name",
        "customer_name",
        "full_name",
        "member_name",
        "borrower_name",
        "email",
        "customer_email",
        "email_address",
        "customer_email_address",
        "primary_email",
        "contact_email",
        "dpd",
        "risk_score",
        "risk_band",
        "loan_amount",
        "funded_amount",
        "loan_status",
        "loan_purpose",
        "recommended_action",
        "escalation_flag",
        "broken_ptp",
        "refused_count",
    ]

    existing_display_columns = [
        col for col in display_columns if col in df.columns
    ]

    if existing_display_columns:
        display_df = df[existing_display_columns]
    else:
        display_df = df

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    df = df.copy()

    loan_col = (
        df["loan_id"].astype(str)
        if "loan_id" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    dpd_col = (
        df["dpd"].astype(str)
        if "dpd" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    risk_col = (
        df["risk_band"].astype(str)
        if "risk_band" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    score_col = (
        df["risk_score"].astype(str)
        if "risk_score" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    def _first_existing_series(frame, candidate_columns, default="NA"):
        for candidate_col in candidate_columns:
            if candidate_col in frame.columns:
                return frame[candidate_col].fillna(default).astype(str)
        return pd.Series([default] * len(frame), index=frame.index)

    name_col = _first_existing_series(
        df,
        ["name", "customer_name", "full_name", "member_name", "borrower_name", "first_name"],
        default="NA"
    )

    email_col = _first_existing_series(
        df,
        ["email", "customer_email", "email_address", "customer_email_address", "primary_email", "contact_email"],
        default="NA"
    )

    df["customer_label"] = (
        "Name " + name_col
        + " | Email " + email_col
        + " | Loan " + loan_col
        + " | DPD " + dpd_col
        + " | Risk " + risk_col
        + " | Score " + score_col
    )

    selected_label = st.selectbox(
        "Select one customer/loan for detailed agent analysis",
        df["customer_label"].tolist(),
        key="selected_customer"
    )

    selected = df[df["customer_label"] == selected_label].iloc[0]
    customer = selected.drop(labels=["customer_label"], errors="ignore").to_dict()

    loan_id = str(customer.get("loan_id", "unknown_loan"))

    if loan_id not in st.session_state.agent_outputs:
        st.session_state.agent_outputs[loan_id] = {}

    outputs = st.session_state.agent_outputs[loan_id]

    if st.button("Generate Dynamic Agent Analysis", key=f"generate_{loan_id}"):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="selected_customer_agent_analysis",
            input={
                "loan_id": loan_id,
                "customer": safe_customer_context(customer),
                "discovery_mode": discovery_mode,
            },
            metadata={
                "source_table": CUSTOMER_360_TABLE,
                "llm_endpoint": LLM_ENDPOINT_NAME,
            }
        ) as analysis_span:
            with propagate_attributes(
                user_id="demo_user",
                session_id=f"loan_recovery_{loan_id}",
                tags=["loan-recovery", "customer-analysis", discovery_mode]
            ):
                with st.spinner("Risk Analysis Agent is generating explanation..."):
                    outputs["risk_explanation"] = risk_analysis_agent(
                        customer=customer,
                        user_question=st.session_state.last_question
                    )

                with st.spinner("Recommendation Agent is generating next-best action..."):
                    outputs["recommendation"] = recommendation_agent(
                        customer=customer,
                        risk_explanation=outputs["risk_explanation"]
                    )

                with st.spinner("Communication Agent is drafting message..."):
                    outputs["draft_message"] = communication_agent(
                        customer=customer,
                        recommendation=outputs["recommendation"]
                    )

                with st.spinner("Audit Logger Agent is writing audit trail..."):
                    outputs["agent_request_id"] = audit_logger_agent(
                        user_query=st.session_state.last_question,
                        customer=customer,
                        recommended_action=outputs["recommendation"],
                        reason=outputs["risk_explanation"],
                        message_draft=outputs["draft_message"],
                        discovery_mode=discovery_mode.replace(" ", "_").lower()
                    )

                add_success_score(
                    analysis_span,
                    name="risk_explanation_generated",
                    comment="Risk Analysis Agent generated an explanation"
                )

                add_success_score(
                    analysis_span,
                    name="recommendation_generated",
                    comment="Recommendation Agent generated next-best action"
                )

                add_success_score(
                    analysis_span,
                    name="communication_draft_generated",
                    comment="Communication Agent generated follow-up message"
                )

                add_success_score(
                    analysis_span,
                    name="selected_customer_analysis_completed",
                    comment="Selected customer analysis completed successfully"
                )

                add_trace_quality_score(
                    analysis_span,
                    name="agent_workflow_success",
                    value=1.0,
                    comment="Full selected customer agent workflow completed successfully"
                )

                add_categorical_score(
                    analysis_span,
                    name="agent_analysis_status",
                    value="success",
                    comment="Risk, recommendation, communication, and audit agents completed successfully"
                )

                analysis_span.update(
                    output={
                        "agent_request_id": outputs["agent_request_id"],
                        "loan_id": loan_id,
                        "discovery_mode": discovery_mode,
                    }
                )

        st.session_state.agent_outputs[loan_id] = outputs
        st.session_state.rephrased_message = ""
        st.session_state.final_messages[loan_id] = outputs["draft_message"]
        st.session_state.rephrase_versions[loan_id] = 0

        langfuse.flush()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Analysis Agent")

        if "risk_explanation" in outputs:
            st.markdown(outputs["risk_explanation"])
        else:
            st.info("Click **Generate Dynamic Agent Analysis** to generate risk explanation.")

    with col2:
        st.subheader("Recommendation Agent")

        if "recommendation" in outputs:
            st.markdown(outputs["recommendation"])
        else:
            st.info("Recommendation will appear after dynamic analysis.")

    st.markdown("---")
    st.subheader("Communication Agent")

    if "draft_message" in outputs:
        # Initialize per-loan final message state only once.
        # This avoids the Streamlit stale-widget issue where the second rephrase
        # does not visually update the text area.
        if loan_id not in st.session_state.final_messages:
            st.session_state.final_messages[loan_id] = outputs["draft_message"]

        if loan_id not in st.session_state.rephrase_versions:
            st.session_state.rephrase_versions[loan_id] = 0

        draft_message = st.text_area(
            "Generated Draft Message - You can edit before rephrasing",
            outputs["draft_message"],
            height=220,
            key=f"draft_{loan_id}"
        )

        # If the user edits the original generated draft before any rephrase,
        # use that edited text as the final message base.
        if st.session_state.rephrase_versions[loan_id] == 0:
            st.session_state.final_messages[loan_id] = draft_message

        st.caption(f"Audit Request ID: {outputs.get('agent_request_id', 'Not logged yet')}")

        st.markdown("### Ask the Communication Agent to Rephrase")

        rephrase_instruction = st.text_input(
            "Rephrase instruction",
            placeholder="Example: Make it softer / Make it shorter / Translate fully to Kannada / Translate fully to Hindi",
            key=f"rephrase_instruction_{loan_id}"
        )

        if st.button("Rephrase Message Dynamically", key=f"rephrase_{loan_id}"):
            if rephrase_instruction.strip():
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="rephrase_request",
                    input={
                        "loan_id": loan_id,
                        "instruction": rephrase_instruction,
                        "current_message_preview": st.session_state.final_messages[loan_id][:300],
                        "discovery_mode": discovery_mode,
                    },
                    metadata={
                        "llm_endpoint": LLM_ENDPOINT_NAME,
                    }
                ) as rephrase_span:
                    with propagate_attributes(
                        user_id="demo_user",
                        session_id=f"loan_recovery_{loan_id}",
                        tags=["loan-recovery", "rephrase", discovery_mode]
                    ):
                        with st.spinner("Rephrase Agent is rewriting the message..."):
                            rephrased_message = rephrase_agent(
                                original_message=st.session_state.final_messages[loan_id],
                                instruction=rephrase_instruction,
                                customer=customer
                            )

                            # IMPORTANT: update the per-loan final message and force a new widget key.
                            st.session_state.final_messages[loan_id] = rephrased_message
                            st.session_state.rephrase_versions[loan_id] += 1
                            st.session_state.rephrased_message = rephrased_message

                        rephrase_span.update(
                            output={
                                "rephrased": True,
                                "loan_id": loan_id,
                                "rephrase_version": st.session_state.rephrase_versions[loan_id],
                                "discovery_mode": discovery_mode,
                            }
                        )

                        add_success_score(
                            rephrase_span,
                            name="rephrase_completed",
                            comment="Rephrase Agent successfully generated rewritten message"
                        )

                        add_trace_quality_score(
                            rephrase_span,
                            name="rephrase_workflow_success",
                            value=1.0,
                            comment="Rephrase workflow completed successfully"
                        )

                        add_categorical_score(
                            rephrase_span,
                            name="rephrase_status",
                            value="success",
                            comment="Rephrase Agent completed successfully"
                        )

                langfuse.flush()

                instruction_lower = rephrase_instruction.lower()
                if "kannada" in instruction_lower and not contains_kannada(st.session_state.final_messages[loan_id]):
                    st.warning(
                        "The model response does not look like Kannada. Try again with: Translate fully to Kannada."
                    )
                elif "hindi" in instruction_lower and not contains_devanagari(st.session_state.final_messages[loan_id]):
                    st.warning(
                        "The model response does not look like Hindi. Try again with: Translate fully to Hindi."
                    )

                # Rerun so the final-message text area is recreated with the latest rephrased output.
                st.rerun()
            else:
                st.warning("Please enter a rephrase instruction.")

        final_message_to_send = st.text_area(
            "Final Message to Send - You can edit before sending",
            value=st.session_state.final_messages[loan_id],
            height=220,
            key=f"final_message_{loan_id}_{st.session_state.rephrase_versions[loan_id]}"
        )

        # Keep manual edits in session state so Send Email uses exactly what the user sees.
        st.session_state.final_messages[loan_id] = final_message_to_send

        st.markdown("---")
        st.markdown("### Final Email Approval")

        default_name = get_default_customer_name(customer)
        default_email = get_default_customer_email(customer)

        st.caption("Selected customer details used for email approval")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            recipient_name = st.text_input(
                "Customer name",
                value=default_name,
                placeholder="Selected customer name",
                key=f"recipient_name_{loan_id}"
            )

        with detail_col2:
            recipient_email = st.text_input(
                "Customer email",
                value=default_email,
                placeholder="Selected customer email or test inbox",
                key=f"recipient_email_{loan_id}"
            )

        if default_email:
            st.success(f"Selected customer email loaded: {default_email}")
        else:
            st.warning("No email field found for the selected customer. Please enter a test/customer email before sending.")

        email_subject = st.text_input(
            "Email subject",
            value=f"Follow-up regarding loan {loan_id}",
            key=f"email_subject_{loan_id}"
        )

        send_confirmation = st.checkbox(
            "I have reviewed the final message and approve sending this email",
            key=f"send_confirm_{loan_id}"
        )

        if st.button("Send Email", key=f"send_email_{loan_id}"):
            final_message_to_send = st.session_state.final_messages[loan_id]

            if not send_confirmation:
                st.warning("Please review and approve the final message before sending.")
            elif not recipient_email.strip():
                st.warning("Please enter recipient email before sending.")
            elif not final_message_to_send.strip():
                st.warning("Email message cannot be empty.")
            else:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="email_send_agent",
                    input={
                        "loan_id": loan_id,
                        "recipient_name": recipient_name,
                        "recipient_email": recipient_email,
                        "subject": email_subject,
                        "message_preview": final_message_to_send[:300],
                        "discovery_mode": discovery_mode,
                    },
                    metadata={
                        "channel": "email",
                        "app": "loan_recovery_assistant",
                        "human_approval_required": True,
                    }
                ) as email_span:
                    try:
                        email_result = send_email_notification(
                            to_email=recipient_email.strip(),
                            subject=email_subject.strip(),
                            body=final_message_to_send.strip()
                        )

                        outputs["email_send_status"] = "sent"
                        outputs["last_sent_to_name"] = recipient_name.strip()
                        outputs["last_sent_to"] = recipient_email.strip()
                        outputs["last_sent_subject"] = email_subject.strip()
                        outputs["last_sent_message"] = final_message_to_send.strip()

                        email_span.update(
                            output={
                                "status": "sent",
                                "to_name": recipient_name.strip(),
                                "to_email": recipient_email.strip(),
                                "subject": email_subject.strip(),
                            }
                        )

                        add_success_score(
                            email_span,
                            name="email_sent",
                            comment="Final edited recovery email was sent successfully after human approval"
                        )

                        add_categorical_score(
                            email_span,
                            name="email_send_status",
                            value="success",
                            comment="Email sent successfully"
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        display_recipient = recipient_email.strip()
                        if recipient_name.strip():
                            display_recipient = f"{recipient_name.strip()} <{recipient_email.strip()}>"
                        st.success(f"Email sent successfully to {display_recipient}.")

                    except Exception as e:
                        outputs["email_send_status"] = "failed"
                        outputs["email_send_error"] = str(e)

                        email_span.update(
                            output={
                                "status": "failed",
                                "error": str(e),
                            },
                            level="ERROR",
                        )

                        add_categorical_score(
                            email_span,
                            name="email_send_status",
                            value="failed",
                            comment=str(e)
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        st.error(f"Failed to send email: {e}")

                langfuse.flush()

        if outputs.get("email_send_status") == "sent":
            st.info(
                f"Last email sent to {outputs.get('last_sent_to')} "
                f"with subject: {outputs.get('last_sent_subject')}"
            )
    else:
        st.info("Draft message will appear after dynamic analysis.")


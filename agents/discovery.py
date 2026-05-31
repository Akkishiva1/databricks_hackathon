"""
Discovery agents for customer discovery: Agent Bricks Supervisor and Custom Dynamic Supervisor.
"""
import os
import json
import requests
import pandas as pd
import streamlit as st
from langfuse import propagate_attributes
from config import (
    SUPERVISOR_AGENT_ENDPOINT_URL, SUPERVISOR_AGENT_ENDPOINT_NAME,
    CUSTOMER_360_TABLE, LLM_ENDPOINT_NAME, INPUT_DBUS_PER_1M_TOKENS,
    OUTPUT_DBUS_PER_1M_TOKENS, NORMAL_CUSTOMER_PHRASES, ESCALATION_PHRASES,
    NO_ESCALATION_PHRASES
)
from core.text_processors import extract_json_from_text, clean_agent_text, extract_tool_traces_from_text
from core.pricing import extract_usage
from services.langfuse_service import langfuse, add_success_score, add_trace_quality_score, add_categorical_score
from services.llm_service import call_databricks_llm
from services.databricks_client import run_query, get_sql_connection, escape_sql
from core.table_helpers import get_table_columns
from config import PREFERRED_DISPLAY_COLUMNS


# -------------------------------------------------
# Agent Bricks Supervisor Discovery
# -------------------------------------------------

def build_agent_bricks_supervisor_question(user_question: str) -> str:
    """Build prompt for Agent Bricks Supervisor discovery."""
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
    """Parse Agent Bricks Supervisor response."""
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
    """Execute Agent Bricks Supervisor discovery."""
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

            if "discovery_summary" not in st.session_state:
                st.session_state.discovery_summary = ""

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
# Custom Dynamic Supervisor - Query Understanding Agent
# -------------------------------------------------

def query_understanding_agent(user_question: str) -> dict:
    """Agent 1: Understand and convert user question into query parameters."""
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

    if any(phrase in question_lower for phrase in NORMAL_CUSTOMER_PHRASES):
        params["risk_band"] = "Low"
        params["min_dpd"] = 0
        params["max_dpd"] = 29
        params["escalation_only"] = False
        params["no_escalation_only"] = True
        params["broken_ptp_only"] = False
        params["refused_only"] = False
        params["sort_by"] = "dpd"

    if any(phrase in question_lower for phrase in NO_ESCALATION_PHRASES):
        params["escalation_only"] = False
        params["no_escalation_only"] = True

        if any(phrase in question_lower for phrase in NORMAL_CUSTOMER_PHRASES) or "non critical" in question_lower or "non-critical" in question_lower:
            params["risk_band"] = "Low"
            params["min_dpd"] = 0
            params["max_dpd"] = 29

    elif any(phrase in question_lower for phrase in ESCALATION_PHRASES):
        params["escalation_only"] = True
        params["no_escalation_only"] = False

    if "critical" in question_lower and not any(phrase in question_lower for phrase in NORMAL_CUSTOMER_PHRASES) and not any(phrase in question_lower for phrase in NO_ESCALATION_PHRASES):
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
    """Agent 2: Retrieve customers based on query parameters."""
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

    select_columns = []

    for col in PREFERRED_DISPLAY_COLUMNS:
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


def custom_dynamic_supervisor_discovery(user_question: str):
    """Execute custom dynamic supervisor discovery with query understanding and data retrieval."""
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

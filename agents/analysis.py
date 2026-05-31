"""
Analysis agents for risk, recommendation, communication, and audit logging.
"""
import os
import json
import uuid
from config import AUDIT_TABLE, LLM_ENDPOINT_NAME, INPUT_DBUS_PER_1M_TOKENS, OUTPUT_DBUS_PER_1M_TOKENS
from services.llm_service import call_databricks_llm
from services.databricks_client import run_statement, escape_sql
from services.langfuse_service import langfuse, add_success_score, add_trace_quality_score, add_categorical_score


def risk_analysis_agent(customer: dict, user_question: str) -> str:
    """Analyze customer risk profile."""
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
    """Generate next-best recovery action recommendation."""
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
    """Draft a professional follow-up message for the customer."""
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
    """Rephrase message according to user instruction."""
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
    """Log audit trail for compliance and tracking."""
    agent_request_id = str(uuid.uuid4())

    catalog, schema = AUDIT_TABLE.split(".")[:2]
    source_table_ref = f"{catalog}.{schema}.loan_recovery_customer_360"

    valid_discovery_modes = {"agent_bricks", "custom_supervisor", "hybrid", "unknown"}
    safe_discovery_mode = discovery_mode if discovery_mode in valid_discovery_modes else "unknown"

    insert_sql = """
    INSERT INTO {audit_table} (
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
      '{req_id}' AS agent_request_id,
      '{user_query}' AS user_query,
      '{customer_id}' AS customer_id,
      '{loan_id}' AS loan_id,
      '{recommended_action}' AS recommended_action,
      '{reason}' AS reason,
      '{message_draft}' AS message_draft,
      array('{source_table}') AS source_tables_used,
      '{created_by}' AS created_by,
      current_timestamp() AS created_at
    """.format(
        audit_table=AUDIT_TABLE,
        req_id=escape_sql(agent_request_id),
        user_query=escape_sql(user_query),
        customer_id=escape_sql(customer.get("member_id")),
        loan_id=escape_sql(customer.get("loan_id")),
        recommended_action=escape_sql(recommended_action),
        reason=escape_sql(reason),
        message_draft=escape_sql(message_draft),
        source_table=escape_sql(source_table_ref),
        created_by=escape_sql(f"hybrid_agent_bricks_custom_supervisor_{safe_discovery_mode}"),
    )

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
        try:
            run_statement(insert_sql)
        except Exception as e:
            span.update(output={"error": str(e)}, level="ERROR")
            return agent_request_id

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

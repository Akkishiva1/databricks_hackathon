"""
LLM service for Databricks GPT-OSS endpoint calls.
"""
import os
import requests
from config import LLM_ENDPOINT_URL, LLM_ENDPOINT_NAME, INPUT_DBUS_PER_1M_TOKENS, OUTPUT_DBUS_PER_1M_TOKENS
from text_processors import extract_llm_text
from pricing import extract_usage
from langfuse_service import langfuse


def call_databricks_llm(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 700,
    agent_name: str = "llm_agent"
) -> str:
    """Call Databricks LLM endpoint with observability."""
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

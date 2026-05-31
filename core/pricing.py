"""
Pricing utilities for DBU cost calculation.
"""
from config import INPUT_DBUS_PER_1M_TOKENS, OUTPUT_DBUS_PER_1M_TOKENS


def calculate_dbu_cost(input_tokens: int, output_tokens: int) -> dict:
    """Calculate DBU cost based on token counts."""
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
    """Extract usage information from LLM result and calculate cost."""
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

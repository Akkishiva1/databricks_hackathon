"""
Customer data utilities for extraction and formatting.
"""
from config import CUSTOMER_EMAIL_FIELDS, CUSTOMER_NAME_FIELDS


def get_default_customer_email(customer: dict) -> str:
    """Return the first available customer email field from the selected row."""
    for field in CUSTOMER_EMAIL_FIELDS:
        value = customer.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value and value.lower() not in ["nan", "none", "null"]:
            return value

    return ""


def get_default_customer_name(customer: dict) -> str:
    """Return the first available customer name field from the selected row."""
    for field in CUSTOMER_NAME_FIELDS:
        value = customer.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value and value.lower() not in ["nan", "none", "null"]:
            return value

    return ""


def safe_customer_context(customer: dict) -> dict:
    """Keep only useful demo fields in Langfuse trace."""
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

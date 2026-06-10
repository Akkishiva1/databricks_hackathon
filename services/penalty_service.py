"""
Penalty and fine management service.
Simulates applying financial penalties to a customer's linked bank account or card
for loan default or non-compliance.
"""
import uuid
import datetime


# Penalty rate schedule keyed by DPD bucket
DPD_PENALTY_SCHEDULE = {
    (30, 60):   {"rate_pct": 1.0, "label": "Early Default Penalty"},
    (60, 90):   {"rate_pct": 2.0, "label": "Moderate Default Penalty"},
    (90, 180):  {"rate_pct": 3.5, "label": "Severe Default Penalty"},
    (180, 365): {"rate_pct": 5.0, "label": "Critical Default Penalty"},
    (365, 9999):{"rate_pct": 7.5, "label": "Legal Default Penalty"},
}

MINIMUM_PENALTY_AMOUNT = 500.0   # INR


def _get_penalty_rate(dpd: int) -> dict:
    for (low, high), schedule in DPD_PENALTY_SCHEDULE.items():
        if low <= dpd < high:
            return schedule
    return {"rate_pct": 0.0, "label": "No Penalty"}


def calculate_penalty(outstanding_amount: float, dpd: int) -> dict:
    """Calculate penalty amount based on outstanding balance and DPD."""
    schedule = _get_penalty_rate(dpd)
    rate = schedule["rate_pct"] / 100.0
    raw_penalty = outstanding_amount * rate
    penalty_amount = max(raw_penalty, MINIMUM_PENALTY_AMOUNT) if raw_penalty > 0 else 0.0
    return {
        "penalty_label": schedule["label"],
        "rate_pct": schedule["rate_pct"],
        "outstanding_amount": outstanding_amount,
        "penalty_amount": round(penalty_amount, 2),
    }


def apply_penalty(
    customer: dict,
    loan_id: str,
    outstanding_amount: float,
    dpd: int,
    applied_by: str = "system",
) -> dict:
    """
    Simulate applying a penalty to the customer's account.
    Returns a structured penalty record.
    """
    penalty_info = calculate_penalty(outstanding_amount, dpd)

    if penalty_info["penalty_amount"] == 0:
        return {
            "status": "skipped",
            "reason": "DPD below penalty threshold (< 30 days).",
            "loan_id": loan_id,
        }

    penalty_id = f"PEN-{uuid.uuid4().hex[:10].upper()}"
    applied_at = datetime.datetime.utcnow().isoformat()

    customer_name = (
        customer.get("name")
        or customer.get("customer_name")
        or customer.get("full_name")
        or "Unknown"
    )

    record = {
        "penalty_id": penalty_id,
        "loan_id": loan_id,
        "customer_id": customer.get("member_id", ""),
        "customer_name": customer_name,
        "dpd": dpd,
        "outstanding_amount": outstanding_amount,
        "penalty_label": penalty_info["penalty_label"],
        "penalty_rate_pct": penalty_info["rate_pct"],
        "penalty_amount": penalty_info["penalty_amount"],
        "total_due": round(outstanding_amount + penalty_info["penalty_amount"], 2),
        "applied_by": applied_by,
        "applied_at": applied_at,
        "status": "applied",
        "channel": "bank_account_debit",
    }

    return record


def get_penalty_summary(record: dict) -> str:
    """Return a human-readable penalty summary."""
    if record.get("status") == "skipped":
        return f"No penalty applied: {record.get('reason', '')}"

    return (
        f"Penalty Applied — ID: {record['penalty_id']}\n"
        f"  Type    : {record['penalty_label']} ({record['penalty_rate_pct']}%)\n"
        f"  Principal Outstanding : ₹{record['outstanding_amount']:,.2f}\n"
        f"  Penalty Amount        : ₹{record['penalty_amount']:,.2f}\n"
        f"  Total Due After Penalty: ₹{record['total_due']:,.2f}\n"
        f"  Applied At: {record['applied_at']}"
    )

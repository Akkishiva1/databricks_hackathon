"""
Legal complaint filing service.
Simulates the process of escalating a defaulted loan case to formal legal proceedings.
"""
import uuid
import datetime


LEGAL_NOTICE_TYPES = {
    "DEMAND_NOTICE":    "Formal demand notice under Section 13(2) of SARFAESI Act",
    "LEGAL_NOTICE":     "Pre-litigation legal notice via registered advocate",
    "ARBITRATION":      "Arbitration claim filing under Arbitration & Conciliation Act",
    "CIVIL_SUIT":       "Civil recovery suit filed in appropriate court",
    "CRIMINAL_COMPLAINT": "Criminal complaint under Section 138 NI Act (cheque dishonour)",
}


def _choose_legal_action(dpd: int, outstanding_amount: float, refused_count: int) -> str:
    """Choose the appropriate legal action based on case severity."""
    if dpd >= 365 or outstanding_amount >= 2_000_000:
        return "CIVIL_SUIT"
    if refused_count >= 3:
        return "ARBITRATION"
    if dpd >= 270 or outstanding_amount >= 500_000:
        return "LEGAL_NOTICE"
    return "DEMAND_NOTICE"


def file_legal_complaint(
    customer: dict,
    loan_id: str,
    outstanding_amount: float,
    dpd: int,
    refused_count: int = 0,
    reason: str = "",
    filed_by: str = "legal_department",
) -> dict:
    """
    Simulate filing an official legal complaint for a defaulted loan.
    Returns a structured legal case record.
    """
    case_id = f"LEGAL-{uuid.uuid4().hex[:10].upper()}"
    filed_at = datetime.datetime.utcnow().isoformat()

    action_type = _choose_legal_action(dpd, outstanding_amount, refused_count)
    action_description = LEGAL_NOTICE_TYPES[action_type]

    customer_name = (
        customer.get("name")
        or customer.get("customer_name")
        or customer.get("full_name")
        or "Unknown"
    )

    case = {
        "case_id": case_id,
        "loan_id": loan_id,
        "customer_id": customer.get("member_id", ""),
        "customer_name": customer_name,
        "dpd": dpd,
        "outstanding_amount": outstanding_amount,
        "refused_count": refused_count,
        "action_type": action_type,
        "action_description": action_description,
        "reason": reason or (
            f"Loan {loan_id} remains unpaid for {dpd} days. "
            f"Outstanding: ₹{outstanding_amount:,.2f}. "
            f"Customer refused payment {refused_count} time(s)."
        ),
        "filed_by": filed_by,
        "filed_at": filed_at,
        "status": "filed",
        "next_hearing_date": (
            datetime.datetime.utcnow() + datetime.timedelta(days=30)
        ).strftime("%Y-%m-%d"),
    }

    return case


def get_legal_case_summary(case: dict) -> str:
    """Return a human-readable legal case summary."""
    return (
        f"Legal Case Filed — ID: {case['case_id']}\n"
        f"  Action Type : {case['action_type']}\n"
        f"  Description : {case['action_description']}\n"
        f"  Loan ID     : {case['loan_id']}\n"
        f"  DPD         : {case['dpd']} days\n"
        f"  Outstanding : ₹{case['outstanding_amount']:,.2f}\n"
        f"  Filed At    : {case['filed_at']}\n"
        f"  Next Hearing: {case['next_hearing_date']}\n"
        f"  Status      : {case['status'].upper()}"
    )

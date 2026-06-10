"""
Dynamic Recovery Strategy Agent.

Determines the appropriate recovery actions based on customer risk profile,
loan amount, DPD, repayment history, and escalation flags.

Recovery Tiers:
  Tier 1 — Soft (DPD < 30, low loan amount, first-time):
      Email only

  Tier 2 — Standard (DPD 30–90 or medium loan):
      Email + SMS + Voice call (simple TTS)

  Tier 3 — Intensive (DPD 90–180, large loan, broken PTP):
      Email + SMS + Conversational voice + CIBIL reporting + Penalty

  Tier 4 — Legal (DPD > 180, very large loan, escalation required, multiple refusals):
      All of Tier 3 + Legal complaint filing
"""

# DPD boundaries
_DPD_SOFT        = 30
_DPD_STANDARD    = 90
_DPD_INTENSIVE   = 180

# Loan amount thresholds (INR)
_AMOUNT_LOW      = 50_000
_AMOUNT_MEDIUM   = 300_000
_AMOUNT_HIGH     = 1_000_000


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def determine_recovery_tier(customer: dict) -> int:
    """
    Compute a recovery tier (1–4) from the customer record.
    Higher tier = more severe actions required.
    """
    dpd             = _safe_int(customer.get("dpd", 0))
    loan_amount     = _safe_float(customer.get("loan_amount") or customer.get("funded_amount", 0))
    risk_score      = _safe_float(customer.get("risk_score", 0))
    escalation_flag = str(customer.get("escalation_flag", "N")).strip().upper()
    broken_ptp      = _safe_int(customer.get("broken_ptp", 0))
    refused_count   = _safe_int(customer.get("refused_count", 0))
    risk_band       = str(customer.get("risk_band", "")).strip().upper()

    # Start from base tier derived from DPD
    if dpd < _DPD_SOFT:
        base_tier = 1
    elif dpd < _DPD_STANDARD:
        base_tier = 2
    elif dpd < _DPD_INTENSIVE:
        base_tier = 3
    else:
        base_tier = 4

    # Bump tier up based on additional risk signals
    if loan_amount >= _AMOUNT_HIGH:
        base_tier = max(base_tier, 3)
    elif loan_amount >= _AMOUNT_MEDIUM:
        base_tier = max(base_tier, 2)

    if escalation_flag == "Y":
        base_tier = max(base_tier, 3)

    if broken_ptp >= 2 or refused_count >= 2:
        base_tier = max(base_tier, 3)

    if broken_ptp >= 3 or refused_count >= 3:
        base_tier = max(base_tier, 4)

    if risk_band in ("CRITICAL", "HIGH") or risk_score >= 80:
        base_tier = max(base_tier, 3)

    return min(base_tier, 4)


def get_recovery_actions(tier: int) -> list[str]:
    """Return ordered list of recovery actions for the given tier."""
    if tier == 1:
        return ["email"]
    if tier == 2:
        return ["email", "sms", "voice_simple"]
    if tier == 3:
        return ["email", "sms", "voice_conversational", "cibil_report", "penalty"]
    # tier 4
    return ["email", "sms", "voice_conversational", "cibil_report", "penalty", "legal_complaint"]


def get_tier_label(tier: int) -> str:
    labels = {
        1: "Tier 1 — Soft Recovery",
        2: "Tier 2 — Standard Recovery",
        3: "Tier 3 — Intensive Recovery",
        4: "Tier 4 — Legal Escalation",
    }
    return labels.get(tier, f"Tier {tier}")


def get_tier_description(tier: int) -> str:
    descs = {
        1: "Low-risk or first-time default. Email communication only.",
        2: "Moderate risk or small overdue. Email, SMS, and a simple voice reminder.",
        3: (
            "High risk: large overdue amount, broken promises, or escalation required. "
            "Full multi-channel outreach + CIBIL reporting + financial penalty applied."
        ),
        4: (
            "Critical: severely overdue, very large outstanding, repeated refusals. "
            "All channels + CIBIL + penalty + formal legal complaint filed."
        ),
    }
    return descs.get(tier, "")


def get_tier_color(tier: int) -> str:
    return {1: "green", 2: "orange", 3: "red", 4: "darkred"}.get(tier, "grey")


ACTION_LABELS = {
    "email":               "Email Notification",
    "sms":                 "SMS Notification",
    "voice_simple":        "Voice Call (Simple TTS)",
    "voice_conversational":"Voice Call (Conversational IVR)",
    "cibil_report":        "CIBIL Adverse Report",
    "penalty":             "Financial Penalty Applied",
    "legal_complaint":     "Legal Complaint Filed",
}

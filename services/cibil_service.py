"""
CIBIL (Credit Information Bureau India Limited) reporting service.
Simulates reporting a defaulter to CIBIL and notifies the customer via SMS and email.
"""
import uuid
import datetime
from services.email_service import send_email_notification
from services.sms_service import send_sms_notification


def report_to_cibil(
    customer: dict,
    loan_id: str,
    outstanding_amount: float,
    dpd: int,
    reason: str = "",
) -> dict:
    """
    Simulate filing a CIBIL adverse report for a defaulting customer.
    Returns a structured report record.
    """
    report_id = f"CIBIL-{uuid.uuid4().hex[:10].upper()}"
    report_date = datetime.datetime.utcnow().isoformat()

    report = {
        "report_id": report_id,
        "report_date": report_date,
        "loan_id": loan_id,
        "customer_id": customer.get("member_id", ""),
        "customer_name": (
            customer.get("name")
            or customer.get("customer_name")
            or customer.get("full_name")
            or "Unknown"
        ),
        "outstanding_amount": outstanding_amount,
        "dpd": dpd,
        "report_type": "NPA_DEFAULT",
        "status": "submitted",
        "reason": reason or f"Loan {loan_id} overdue by {dpd} days with outstanding ₹{outstanding_amount:,.2f}",
    }

    return report


def notify_cibil_report(
    customer: dict,
    report: dict,
    to_email: str = "",
    to_phone: str = "",
) -> dict:
    """
    Notify customer (via email and/or SMS) that an adverse CIBIL report has been filed.
    Returns dict with notification results.
    """
    name = report.get("customer_name", "Valued Customer")
    loan_id = report.get("loan_id", "")
    report_id = report.get("report_id", "")
    dpd = report.get("dpd", 0)
    amount = report.get("outstanding_amount", 0)

    subject = f"IMPORTANT: Adverse CIBIL Report Filed — Loan {loan_id}"
    email_body = (
        f"Dear {name},\n\n"
        f"We regret to inform you that due to non-payment of your loan (Loan ID: {loan_id}), "
        f"an adverse report has been submitted to CIBIL (Credit Reference ID: {report_id}).\n\n"
        f"Outstanding Amount: ₹{amount:,.2f}\n"
        f"Days Past Due: {dpd}\n\n"
        f"This report may significantly impact your credit score and your ability to obtain "
        f"credit in the future. To prevent further action, please clear the outstanding dues "
        f"immediately or contact your loan officer to discuss a repayment plan.\n\n"
        f"Time is of the essence. Please act now.\n\n"
        f"Regards,\nLoan Recovery Team"
    )

    sms_body = (
        f"CIBIL report filed for Loan {loan_id} (Ref: {report_id}). "
        f"DPD: {dpd} days. Clear dues to avoid credit damage."
    )

    results = {"report_id": report_id}

    if to_email:
        try:
            send_email_notification(to_email=to_email, subject=subject, body=email_body)
            results["email_status"] = "sent"
            results["email_to"] = to_email
        except Exception as e:
            results["email_status"] = f"failed: {e}"

    if to_phone:
        try:
            send_sms_notification(to_phone=to_phone, message=sms_body)
            results["sms_status"] = "sent"
            results["sms_to"] = to_phone
        except Exception as e:
            results["sms_status"] = f"failed: {e}"

    return results

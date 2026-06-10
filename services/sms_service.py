"""
Twilio SMS notification service for loan recovery communications.
"""
import os
from twilio.rest import Client


def _get_twilio_client():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not account_sid or not auth_token or not from_number:
        raise ValueError(
            "Twilio configuration missing. Please set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER in app.yaml."
        )
    return Client(account_sid, auth_token), from_number


def send_sms_notification(to_phone: str, message: str) -> dict:
    """Send an SMS notification via Twilio."""
    to_phone = (to_phone or "").strip()
    message = (message or "").strip()

    if not to_phone:
        raise ValueError("Recipient phone number is required for SMS.")
    if not message:
        raise ValueError("SMS message cannot be empty.")

    client, from_number = _get_twilio_client()

    try:
        msg = client.messages.create(
            to=to_phone,
            from_=from_number,
            body=message[:1600],
        )
    except Exception as e:
        error_msg = str(e)
        if "21608" in error_msg or "unverified" in error_msg.lower():
            raise ValueError(
                f"Twilio trial account restriction: {to_phone} is not a verified number. "
                "Add it at twilio.com/console → Verified Caller IDs."
            )
        raise

    return {"message_sid": msg.sid, "to_phone": to_phone, "status": msg.status}

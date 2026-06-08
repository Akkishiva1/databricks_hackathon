"""
Twilio Programmable Voice service for outbound customer calls.
"""
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse


def make_voice_call(to_phone: str, message: str, customer_name: str = "") -> dict:
    """
    Place an outbound TTS voice call to the customer via Twilio.

    Required environment variables:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_FROM_NUMBER

    Returns a dict with call SID and status.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not account_sid or not auth_token or not from_number:
        raise ValueError(
            "Twilio configuration missing. Please set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER in app.yaml."
        )

    to_phone = (to_phone or "").strip()
    message = (message or "").strip()

    if not to_phone:
        raise ValueError("Customer phone number is required before placing a call.")

    if not message:
        raise ValueError("Voice message cannot be empty.")

    twiml = build_twiml(message, customer_name)
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=to_phone,
        from_=from_number,
        twiml=twiml,
    )

    return {
        "call_sid": call.sid,
        "to_phone": to_phone,
        "status": call.status,
    }


def build_twiml(message: str, customer_name: str = "") -> str:
    """Build TwiML for TTS voice message."""
    response = VoiceResponse()

    greeting = f"Hello {customer_name}, " if customer_name else "Hello, "
    full_text = greeting + message + " Thank you."

    response.say(full_text, voice="Polly.Raveena", language="en-IN")
    response.pause(length=1)
    response.say(
        "This message was sent by your loan recovery officer. "
        "Please contact us at your earliest convenience.",
        voice="Polly.Raveena",
        language="en-IN"
    )

    return str(response)


def get_customer_phone(customer: dict) -> str:
    """Extract phone number from customer record."""
    for field in ["phone", "phone_number", "mobile", "mobile_number", "contact_number", "cell"]:
        value = customer.get(field) or customer.get(field.upper())
        if value:
            return str(value).strip()
    return ""

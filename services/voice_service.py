"""
Twilio Programmable Voice service for outbound customer calls.
Supports Indian languages via Amazon Polly TTS.
"""
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# Amazon Polly voices supported by Twilio for Indian languages.
# Kannada, Tamil, Telugu, Malayalam are NOT supported by Polly —
# for those languages the message should be kept in English or Hindi.
LANGUAGE_VOICE_MAP = {
    "English (Indian)":  {"polly_voice": "Polly.Kajal",  "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Hindi":             {"polly_voice": "Polly.Kajal",  "language_code": "hi-IN", "closing": "कृपया जल्द से जल्द हमसे संपर्क करें।"},
    "Bengali":           {"polly_voice": "Polly.Kajal",  "language_code": "bn-IN", "closing": "Please contact us at your earliest convenience."},
    "Gujarati":          {"polly_voice": "Polly.Kajal",  "language_code": "gu-IN", "closing": "Please contact us at your earliest convenience."},
    "Marathi":           {"polly_voice": "Polly.Kajal",  "language_code": "mr-IN", "closing": "Please contact us at your earliest convenience."},
    "Punjabi":           {"polly_voice": "Polly.Kajal",  "language_code": "pa-IN", "closing": "Please contact us at your earliest convenience."},
    "Kannada (English voice)": {"polly_voice": "Polly.Kajal", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Tamil (English voice)":   {"polly_voice": "Polly.Kajal", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Telugu (English voice)":  {"polly_voice": "Polly.Kajal", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
}

SUPPORTED_LANGUAGES = list(LANGUAGE_VOICE_MAP.keys())


def make_voice_call(to_phone: str, message: str, customer_name: str = "", language: str = "English (Indian)") -> dict:
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

    twiml = build_twiml(message, customer_name, language)
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
        "language": language,
    }


def build_twiml(message: str, customer_name: str = "", language: str = "English (Indian)") -> str:
    """Build TwiML with the correct Polly voice for the selected Indian language."""
    config = LANGUAGE_VOICE_MAP.get(language, LANGUAGE_VOICE_MAP["English (Indian)"])
    polly_voice = config["polly_voice"]
    language_code = config["language_code"]
    closing = config["closing"]

    response = VoiceResponse()

    greeting = f"Hello {customer_name}, " if customer_name else "Hello, "
    full_text = greeting + message + " Thank you."

    response.say(full_text, voice=polly_voice, language=language_code)
    response.pause(length=1)
    response.say(closing, voice=polly_voice, language=language_code)

    return str(response)


def get_customer_phone(customer: dict) -> str:
    """Extract phone number from customer record."""
    for field in ["phone", "phone_number", "mobile", "mobile_number", "contact_number", "cell"]:
        value = customer.get(field) or customer.get(field.upper())
        if value:
            return str(value).strip()
    return ""

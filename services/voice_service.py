"""
Twilio Programmable Voice service for outbound customer calls.
Supports Indian languages via Amazon Polly TTS.
"""
import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# Polly.Kajal is a Neural voice requiring a paid Twilio plan.
# Polly.Raveena (en-IN) and Polly.Aditi (en-IN/hi-IN bilingual)
# are standard voices that work on all Twilio accounts including trial.
LANGUAGE_VOICE_MAP = {
    "English (Indian)":        {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Hindi":                   {"polly_voice": "Polly.Aditi",   "language_code": "hi-IN", "closing": "कृपया जल्द से जल्द हमसे संपर्क करें।"},
    "Kannada (English voice)": {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Tamil (English voice)":   {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Telugu (English voice)":  {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Marathi (English voice)": {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
    "Bengali (English voice)": {"polly_voice": "Polly.Raveena", "language_code": "en-IN", "closing": "Please contact us at your earliest convenience."},
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

    try:
        call = client.calls.create(
            to=to_phone,
            from_=from_number,
            twiml=twiml,
        )
    except Exception as e:
        error_msg = str(e)
        if "21608" in error_msg or "unverified" in error_msg.lower():
            raise ValueError(
                f"Twilio trial account restriction: {to_phone} is not a verified number. "
                "Go to twilio.com/console → Verified Caller IDs and add this number first."
            )
        if "21211" in error_msg or "invalid" in error_msg.lower():
            raise ValueError(
                f"Invalid phone number format: {to_phone}. "
                "Use E.164 format, e.g. +919876543210"
            )
        raise

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

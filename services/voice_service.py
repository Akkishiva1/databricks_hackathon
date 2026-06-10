"""
Twilio Programmable Voice service for outbound customer calls.
Supports Indian languages via Amazon Polly TTS.
Two modes:
  - Simple TTS: plays the message directly (no interaction)
  - Conversational IVR: asks customer's name, plays message, collects confirmation
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


def _handle_twilio_error(e, to_phone):
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
    raise e


def make_voice_call(to_phone: str, message: str, customer_name: str = "", language: str = "English (Indian)") -> dict:
    """Place a simple outbound TTS call — plays message directly, no interaction."""
    client, from_number = _get_twilio_client()
    to_phone = (to_phone or "").strip()
    message = (message or "").strip()

    if not to_phone:
        raise ValueError("Customer phone number is required before placing a call.")
    if not message:
        raise ValueError("Voice message cannot be empty.")

    twiml = build_twiml(message, customer_name, language)

    try:
        call = client.calls.create(to=to_phone, from_=from_number, twiml=twiml)
    except Exception as e:
        _handle_twilio_error(e, to_phone)

    return {"call_sid": call.sid, "to_phone": to_phone, "status": call.status, "language": language, "mode": "simple"}


def make_conversational_call(
    to_phone: str,
    loan_id: str,
    message: str,
    customer_name: str = "",
    language: str = "English (Indian)",
    webhook_base_url: str = "",
) -> dict:
    """
    Place a conversational IVR call.
    Flow: greet → ask name → customer speaks name → play message → press 1 to confirm.
    Requires webhook_base_url pointing to the running voice_webhook FastAPI server.
    """
    from services.voice_webhook import register_call, start_webhook_server

    client, from_number = _get_twilio_client()
    to_phone = (to_phone or "").strip()
    message = (message or "").strip()

    if not to_phone:
        raise ValueError("Customer phone number is required before placing a call.")
    if not webhook_base_url:
        raise ValueError("Webhook base URL is required for conversational calls.")

    # Start the webhook server if not already running
    start_webhook_server(port=8502)

    try:
        call = client.calls.create(
            to=to_phone,
            from_=from_number,
            url=f"{webhook_base_url}/twilio/voice",
            method="POST",
        )
    except Exception as e:
        _handle_twilio_error(e, to_phone)

    # Register context so webhook knows what to say
    register_call(
        call_sid=call.sid,
        loan_id=loan_id,
        customer_name=customer_name,
        message=message,
        language=language,
    )

    return {"call_sid": call.sid, "to_phone": to_phone, "status": call.status, "language": language, "mode": "conversational"}


def spell_loan_ids(text: str) -> str:
    """
    Find patterns that look like loan IDs (e.g. L12345, LN-00123, 8-digit+ numbers)
    and insert spaces between characters so Polly reads them digit by digit.
    """
    import re
    def _spell(m):
        return " ".join(m.group(0))
    # Letters + digits (loan IDs like L12345, LN00456) — 5+ digits
    text = re.sub(r'\b[A-Z]+\d{5,}\b', _spell, text)
    # Pure digit strings 8+ digits (loan IDs, not amounts like 50000)
    text = re.sub(r'\b\d{8,}\b', _spell, text)
    return text


def build_twiml(message: str, customer_name: str = "", language: str = "English (Indian)") -> str:
    """Build TwiML with the correct Polly voice for the selected Indian language."""
    config = LANGUAGE_VOICE_MAP.get(language, LANGUAGE_VOICE_MAP["English (Indian)"])
    polly_voice = config["polly_voice"]
    language_code = config["language_code"]
    closing = config["closing"]

    response = VoiceResponse()

    greeting = f"Hello {customer_name}, " if customer_name else "Hello, "
    full_text = greeting + spell_loan_ids(message) + " Thank you."

    response.say(full_text, voice=polly_voice, language=language_code)
    response.pause(length=1)
    response.say(closing, voice=polly_voice, language=language_code)

    return str(response)


def get_call_transcript(call_sid: str) -> dict:
    """Return structured conversation output for a completed conversational call."""
    from services.voice_webhook import get_call_transcript as _get
    return _get(call_sid)


def get_customer_phone(customer: dict) -> str:
    """Extract phone number from customer record."""
    candidates = [
        "phone", "phone_number", "mobile", "mobile_number",
        "contact_number", "cell", "cell_number", "telephone",
        "tel", "contact", "customer_phone", "customer_mobile",
        "primary_phone", "primary_mobile", "alt_phone", "alternate_phone",
        "whatsapp", "whatsapp_number",
    ]
    for field in candidates:
        value = customer.get(field) or customer.get(field.upper())
        if value and str(value).strip() not in ("", "nan", "None", "null"):
            return str(value).strip()

    # Last resort: scan all fields whose name contains 'phone' or 'mobile'
    for key, value in customer.items():
        if any(kw in str(key).lower() for kw in ("phone", "mobile", "contact", "cell")):
            if value and str(value).strip() not in ("", "nan", "None", "null"):
                return str(value).strip()
    return ""



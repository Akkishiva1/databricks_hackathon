"""
Twilio webhook server for conversational IVR voice calls.
Runs as a FastAPI app in a background thread alongside Streamlit.
"""
import os
import threading
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Gather

app = FastAPI()

VOICE = "Polly.Raveena"
LANGUAGE = "en-IN"


def spell_out(value: str) -> str:
    """Insert spaces between every character so Polly reads each digit/letter individually."""
    return " ".join(str(value).strip())
HINDI_VOICE = "Polly.Aditi"
HINDI_LANGUAGE = "hi-IN"

# In-memory store: call_sid → {loan_id, customer_name, message, language}
call_context: dict = {}


def register_call(call_sid: str, loan_id: str, customer_name: str, message: str, language: str = "English (Indian)"):
    """Register call context before placing the call."""
    call_context[call_sid] = {
        "loan_id": loan_id,
        "customer_name": customer_name,
        "message": message,
        "language": language,
    }


def _voice_config(language: str) -> tuple:
    if language == "Hindi":
        return HINDI_VOICE, HINDI_LANGUAGE
    return VOICE, LANGUAGE


@app.post("/twilio/voice")
async def voice_webhook(
    CallSid: str = Form(default=""),
    SpeechResult: str = Form(default=""),
):
    """Initial webhook — greet and ask customer's name."""
    ctx = call_context.get(CallSid, {})
    voice, lang = _voice_config(ctx.get("language", "English (Indian)"))

    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"/twilio/gather?call_sid={CallSid}&step=name",
        timeout=5,
        language=lang,
        speech_timeout="auto",
    )
    gather.say(
        "Hello. This is an automated loan recovery notice from your lender. "
        "May I know your name please?",
        voice=voice,
        language=lang,
    )
    response.append(gather)
    response.say("We did not receive your response. Goodbye.", voice=voice, language=lang)

    return Response(content=str(response), media_type="application/xml")


@app.post("/twilio/gather")
async def gather_webhook(
    request: Request,
    CallSid: str = Form(default=""),
    SpeechResult: str = Form(default=""),
    Digits: str = Form(default=""),
):
    step = request.query_params.get("step", "name")
    call_sid = request.query_params.get("call_sid", CallSid)
    ctx = call_context.get(call_sid, {})
    voice, lang = _voice_config(ctx.get("language", "English (Indian)"))
    loan_id = ctx.get("loan_id", "your loan")
    message = ctx.get("message", "Please contact us regarding your overdue loan.")

    response = VoiceResponse()

    if step == "name":
        spoken_name = SpeechResult.strip() if SpeechResult.strip() else ctx.get("customer_name", "valued customer")
        call_context[call_sid]["spoken_name"] = spoken_name

        gather = Gather(
            input="dtmf",
            action=f"/twilio/gather?call_sid={call_sid}&step=confirm",
            timeout=8,
            num_digits=1,
        )
        gather.say(
            f"Thank you, {spoken_name}. "
            f"This message is regarding loan {spell_out(loan_id)}. "
            f"{message} "
            f"Press 1 to confirm you have received this message, or press 2 to repeat.",
            voice=voice,
            language=lang,
        )
        response.append(gather)
        response.say("We did not receive your input. Goodbye.", voice=voice, language=lang)

    elif step == "confirm":
        spoken_name = ctx.get("spoken_name", "valued customer")

        if Digits == "1":
            response.say(
                f"Thank you, {spoken_name}. Your confirmation has been recorded. "
                "Please contact your loan officer at the earliest. Goodbye.",
                voice=voice,
                language=lang,
            )
        elif Digits == "2":
            gather = Gather(
                input="dtmf",
                action=f"/twilio/gather?call_sid={call_sid}&step=confirm",
                timeout=8,
                num_digits=1,
            )
            gather.say(
                f"Repeating the message. "
                f"This message is regarding loan {spell_out(loan_id)}. "
                f"{message} "
                f"Press 1 to confirm, or press 2 to repeat.",
                voice=voice,
                language=lang,
            )
            response.append(gather)
        else:
            response.say(
                "We did not receive a valid response. Please call your loan officer directly. Goodbye.",
                voice=voice,
                language=lang,
            )

    return Response(content=str(response), media_type="application/xml")


_server_started = False
_server_lock = threading.Lock()


def start_webhook_server(port: int = 8501):
    """Start FastAPI webhook server in a background thread. Safe to call multiple times."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    def run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

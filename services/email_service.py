"""
Email service for sending notifications.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email_notification(to_email: str, subject: str, body: str) -> dict:
    """
    Send email notification through SMTP.
    
    Required environment variables:
    - SMTP_HOST
    - SMTP_PORT
    - SMTP_USERNAME
    - SMTP_PASSWORD
    - SENDER_EMAIL
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SENDER_EMAIL", smtp_username)

    if not smtp_host or not smtp_username or not smtp_password or not sender_email:
        raise ValueError(
            "SMTP configuration missing. Please set SMTP_HOST, SMTP_PORT, "
            "SMTP_USERNAME, SMTP_PASSWORD, and SENDER_EMAIL."
        )

    to_email = (to_email or "").strip()
    subject = (subject or "").strip()
    body = (body or "").strip()

    if not to_email:
        raise ValueError("Recipient email is required before sending.")

    if not subject:
        raise ValueError("Email subject is required before sending.")

    if not body:
        raise ValueError("Email body cannot be empty.")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(message)

    return {
        "to_email": to_email,
        "subject": subject,
        "status": "sent",
    }

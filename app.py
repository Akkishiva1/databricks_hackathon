"""
Loan Recovery Assistant - Main Streamlit Application

A hybrid loan recovery multi-agent POC using:
- Databricks Agent Bricks Supervisor
- Custom Dynamic Supervisor
- Databricks SQL Warehouse
- Databricks GPT-OSS 120B
- Audit Logging
- Langfuse Observability
"""
import streamlit as st
import pandas as pd
from langfuse import propagate_attributes

# Import from modular components
from config import (
    CUSTOMER_360_TABLE, LLM_ENDPOINT_NAME, DISPLAY_COLUMNS,
    INPUT_DBUS_PER_1M_TOKENS, OUTPUT_DBUS_PER_1M_TOKENS
)
from core.discovery_cache import (
    get_cached_discovery_result, set_cached_discovery_result,
    is_low_risk_no_escalation_question
)
from core.customer_helpers import (
    get_default_customer_name, get_default_customer_email,
    safe_customer_context
)
from core.text_processors import contains_kannada, contains_devanagari
from services.email_service import send_email_notification
from services.sms_service import send_sms_notification
from services.cibil_service import report_to_cibil, notify_cibil_report
from services.penalty_service import apply_penalty, get_penalty_summary
from services.legal_service import file_legal_complaint, get_legal_case_summary
from services.voice_service import (
    make_voice_call, make_conversational_call, get_customer_phone,
    get_call_transcript, SUPPORTED_LANGUAGES,
)
from services.langfuse_service import langfuse, add_success_score, add_trace_quality_score, add_categorical_score
from agents.discovery import (
    agent_bricks_supervisor_discovery, custom_dynamic_supervisor_discovery
)
from agents.analysis import (
    risk_analysis_agent, recommendation_agent, communication_agent,
    rephrase_agent, audit_logger_agent
)
from agents.recovery_strategy import (
    determine_recovery_tier, get_recovery_actions,
    get_tier_label, get_tier_description, ACTION_LABELS,
)


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Recovery Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("Loan Recovery Assistant")
st.caption(
    "Hybrid Loan Recovery multi-agent POC using Databricks Agent Bricks Supervisor, "
    "Custom Dynamic Supervisor, Databricks SQL Warehouse, Databricks GPT OSS 120B, "
    "Audit Logging, and Langfuse Observability."
)


# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "result_df": None,
        "query_params": None,
        "last_question": None,
        "agent_outputs": {},
        "rephrased_message": "",
        "final_messages": {},
        "rephrase_versions": {},
        "discovery_mode": None,
        "discovery_summary": "",
        "discovery_cache": {},
        "recovery_actions": {},
        "voice_transcripts": {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# -------------------------------------------------
# UI Helper Functions
# -------------------------------------------------
def _first_existing_series(frame, candidate_columns, default="NA"):
    """Get first existing column as series."""
    for candidate_col in candidate_columns:
        if candidate_col in frame.columns:
            return frame[candidate_col].fillna(default).astype(str)
    return pd.Series([default] * len(frame), index=frame.index)


def display_discovery_metrics(df):
    """Display discovery result metrics."""
    total_customers = len(df)
    avg_dpd = round(df["dpd"].mean(), 1) if "dpd" in df.columns else "NA"
    max_risk_score = round(df["risk_score"].max(), 1) if "risk_score" in df.columns else "NA"
    escalation_count = (
        len(df[df["escalation_flag"] == "Y"])
        if "escalation_flag" in df.columns
        else "NA"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers Found", total_customers)
    c2.metric("Average DPD", avg_dpd)
    c3.metric("Max Risk Score", max_risk_score)
    c4.metric("Escalations", escalation_count)


def display_customer_table(df):
    """Display customer discovery results table."""
    display_df_columns = [
        col for col in DISPLAY_COLUMNS if col in df.columns
    ]

    if display_df_columns:
        display_df = df[display_df_columns]
    else:
        display_df = df

    st.dataframe(display_df, use_container_width=True)


def create_customer_labels(df):
    """Create customer selection labels."""
    loan_col = (
        df["loan_id"].astype(str)
        if "loan_id" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    dpd_col = (
        df["dpd"].astype(str)
        if "dpd" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    risk_col = (
        df["risk_band"].astype(str)
        if "risk_band" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    score_col = (
        df["risk_score"].astype(str)
        if "risk_score" in df.columns
        else pd.Series(["NA"] * len(df), index=df.index)
    )

    name_col = _first_existing_series(
        df,
        ["name", "customer_name", "full_name", "member_name", "borrower_name", "first_name"],
        default="NA"
    )

    email_col = _first_existing_series(
        df,
        ["email", "customer_email", "email_address", "customer_email_address", "primary_email", "contact_email"],
        default="NA"
    )

    return (
        "Name " + name_col
        + " | Email " + email_col
        + " | Loan " + loan_col
        + " | DPD " + dpd_col
        + " | Risk " + risk_col
        + " | Score " + score_col
    )


# -------------------------------------------------
# Main Form
# -------------------------------------------------
with st.form("agent_form"):
    discovery_mode = st.radio(
        "Choose customer discovery mode",
        [
            "Agent Bricks Supervisor",
            "Custom Dynamic Supervisor"
        ],
        horizontal=True
    )

    question = st.text_input(
        "Ask your recovery question",
        value="Show me critical customers who need escalation"
    )

    submitted = st.form_submit_button("Run Customer Discovery")


# -------------------------------------------------
# Run Discovery
# -------------------------------------------------
if submitted:
    st.session_state.agent_outputs = {}
    st.session_state.rephrased_message = ""
    st.session_state.final_messages = {}
    st.session_state.rephrase_versions = {}
    st.session_state.discovery_mode = discovery_mode
    st.session_state.last_question = question
    st.session_state.discovery_summary = ""

    cached_result = get_cached_discovery_result(discovery_mode, question)

    if cached_result:
        st.session_state.result_df = cached_result["df"].copy()
        st.session_state.query_params = cached_result["query_params"]
        st.session_state.discovery_summary = cached_result["discovery_summary"]

    else:
        if discovery_mode == "Agent Bricks Supervisor":
            with st.spinner("Agent Bricks Supervisor is discovering customers..."):
                df = agent_bricks_supervisor_discovery(question)

            query_params = {
                "mode": "Agent Bricks Supervisor",
                "note": "Customer discovery handled by Databricks Agent Bricks Supervisor."
            }
            discovery_summary = st.session_state.discovery_summary

            if (df is None or df.empty) and is_low_risk_no_escalation_question(question):
                with st.spinner("No rows from Agent Bricks; using Custom Dynamic Supervisor fallback..."):
                    df, params = custom_dynamic_supervisor_discovery(question)

                query_params = {
                    "mode": "Agent Bricks Supervisor with Custom Dynamic Supervisor fallback",
                    "agent_bricks_note": "Agent Bricks returned no rows for this phrasing.",
                    "fallback_params": params,
                }
                discovery_summary = (
                    f"Custom fallback returned {0 if df is None else len(df)} customer(s)."
                )

        else:
            df, params = custom_dynamic_supervisor_discovery(question)

            query_params = params
            discovery_summary = (
                f"Custom Dynamic Supervisor returned {0 if df is None else len(df)} customer(s)."
            )

        st.session_state.result_df = df
        st.session_state.query_params = query_params
        st.session_state.discovery_summary = discovery_summary

        set_cached_discovery_result(
            discovery_mode=discovery_mode,
            question=question,
            df=df,
            query_params=query_params,
            discovery_summary=discovery_summary,
        )

        langfuse.flush()


# -------------------------------------------------
# Display Results
# -------------------------------------------------
df = st.session_state.result_df

if df is None:
    st.info("Enter a question and click **Run Customer Discovery**.")

elif df.empty:
    st.warning("No matching customers found.")

else:
    discovery_mode = st.session_state.discovery_mode or "Unknown"
    st.success(f"{discovery_mode} found {len(df)} matching customer(s).")

    if st.session_state.discovery_summary:
        st.markdown(st.session_state.discovery_summary)

    display_discovery_metrics(df)

    st.markdown("---")
    st.subheader("Customers Returned")

    display_customer_table(df)

    st.markdown("---")

    df = df.copy()
    df["customer_label"] = create_customer_labels(df)

    selected_label = st.selectbox(
        "Select one customer/loan for detailed agent analysis",
        df["customer_label"].tolist(),
        key="selected_customer"
    )

    selected = df[df["customer_label"] == selected_label].iloc[0]
    customer = selected.drop(labels=["customer_label"], errors="ignore").to_dict()

    loan_id = str(customer.get("loan_id", "unknown_loan"))

    if loan_id not in st.session_state.agent_outputs:
        st.session_state.agent_outputs[loan_id] = {}

    outputs = st.session_state.agent_outputs[loan_id]

    if st.button("Generate Dynamic Agent Analysis", key=f"generate_{loan_id}"):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="selected_customer_agent_analysis",
            input={
                "loan_id": loan_id,
                "customer": safe_customer_context(customer),
                "discovery_mode": discovery_mode,
            },
            metadata={
                "source_table": CUSTOMER_360_TABLE,
                "llm_endpoint": LLM_ENDPOINT_NAME,
            }
        ) as analysis_span:
            with propagate_attributes(
                user_id="demo_user",
                session_id=f"loan_recovery_{loan_id}",
                tags=["loan-recovery", "customer-analysis", discovery_mode]
            ):
                with st.spinner("Risk Analysis Agent is generating explanation..."):
                    outputs["risk_explanation"] = risk_analysis_agent(
                        customer=customer,
                        user_question=st.session_state.last_question
                    )

                with st.spinner("Recommendation Agent is generating next-best action..."):
                    outputs["recommendation"] = recommendation_agent(
                        customer=customer,
                        risk_explanation=outputs["risk_explanation"]
                    )

                with st.spinner("Communication Agent is drafting message..."):
                    outputs["draft_message"] = communication_agent(
                        customer=customer,
                        recommendation=outputs["recommendation"]
                    )

                with st.spinner("Audit Logger Agent is writing audit trail..."):
                    outputs["agent_request_id"] = audit_logger_agent(
                        user_query=st.session_state.last_question,
                        customer=customer,
                        recommended_action=outputs["recommendation"],
                        reason=outputs["risk_explanation"],
                        message_draft=outputs["draft_message"],
                        discovery_mode=discovery_mode.replace(" ", "_").lower()
                    )

                add_success_score(
                    analysis_span,
                    name="risk_explanation_generated",
                    comment="Risk Analysis Agent generated an explanation"
                )

                add_success_score(
                    analysis_span,
                    name="recommendation_generated",
                    comment="Recommendation Agent generated next-best action"
                )

                add_success_score(
                    analysis_span,
                    name="communication_draft_generated",
                    comment="Communication Agent generated follow-up message"
                )

                add_success_score(
                    analysis_span,
                    name="selected_customer_analysis_completed",
                    comment="Selected customer analysis completed successfully"
                )

                add_trace_quality_score(
                    analysis_span,
                    name="agent_workflow_success",
                    value=1.0,
                    comment="Full selected customer agent workflow completed successfully"
                )

                add_categorical_score(
                    analysis_span,
                    name="agent_analysis_status",
                    value="success",
                    comment="Risk, recommendation, communication, and audit agents completed successfully"
                )

                analysis_span.update(
                    output={
                        "agent_request_id": outputs["agent_request_id"],
                        "loan_id": loan_id,
                        "discovery_mode": discovery_mode,
                    }
                )

        st.session_state.agent_outputs[loan_id] = outputs
        st.session_state.rephrased_message = ""
        st.session_state.final_messages[loan_id] = outputs["draft_message"]
        st.session_state.rephrase_versions[loan_id] = 0

        langfuse.flush()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Analysis Agent")

        if "risk_explanation" in outputs:
            st.markdown(outputs["risk_explanation"])
        else:
            st.info("Click **Generate Dynamic Agent Analysis** to generate risk explanation.")

    with col2:
        st.subheader("Recommendation Agent")

        if "recommendation" in outputs:
            st.markdown(outputs["recommendation"])
        else:
            st.info("Recommendation will appear after dynamic analysis.")

    st.markdown("---")
    st.subheader("Communication Agent")

    if "draft_message" in outputs:
        if loan_id not in st.session_state.final_messages:
            st.session_state.final_messages[loan_id] = outputs["draft_message"]

        if loan_id not in st.session_state.rephrase_versions:
            st.session_state.rephrase_versions[loan_id] = 0

        draft_message = st.text_area(
            "Generated Draft Message - You can edit before rephrasing",
            outputs["draft_message"],
            height=220,
            key=f"draft_{loan_id}"
        )

        if st.session_state.rephrase_versions[loan_id] == 0:
            st.session_state.final_messages[loan_id] = draft_message

        st.caption(f"Audit Request ID: {outputs.get('agent_request_id', 'Not logged yet')}")

        st.markdown("### Ask the Communication Agent to Rephrase")

        rephrase_instruction = st.text_input(
            "Rephrase instruction",
            placeholder="Example: Make it softer / Make it shorter / Translate fully to Kannada / Translate fully to Hindi",
            key=f"rephrase_instruction_{loan_id}"
        )

        if st.button("Rephrase Message Dynamically", key=f"rephrase_{loan_id}"):
            if rephrase_instruction.strip():
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="rephrase_request",
                    input={
                        "loan_id": loan_id,
                        "instruction": rephrase_instruction,
                        "current_message_preview": st.session_state.final_messages[loan_id][:300],
                        "discovery_mode": discovery_mode,
                    },
                    metadata={
                        "llm_endpoint": LLM_ENDPOINT_NAME,
                    }
                ) as rephrase_span:
                    with propagate_attributes(
                        user_id="demo_user",
                        session_id=f"loan_recovery_{loan_id}",
                        tags=["loan-recovery", "rephrase", discovery_mode]
                    ):
                        with st.spinner("Rephrase Agent is rewriting the message..."):
                            rephrased_message = rephrase_agent(
                                original_message=st.session_state.final_messages[loan_id],
                                instruction=rephrase_instruction,
                                customer=customer
                            )

                            st.session_state.final_messages[loan_id] = rephrased_message
                            st.session_state.rephrase_versions[loan_id] += 1
                            st.session_state.rephrased_message = rephrased_message

                        rephrase_span.update(
                            output={
                                "rephrased": True,
                                "loan_id": loan_id,
                                "rephrase_version": st.session_state.rephrase_versions[loan_id],
                                "discovery_mode": discovery_mode,
                            }
                        )

                        add_success_score(
                            rephrase_span,
                            name="rephrase_completed",
                            comment="Rephrase Agent successfully generated rewritten message"
                        )

                        add_trace_quality_score(
                            rephrase_span,
                            name="rephrase_workflow_success",
                            value=1.0,
                            comment="Rephrase workflow completed successfully"
                        )

                        add_categorical_score(
                            rephrase_span,
                            name="rephrase_status",
                            value="success",
                            comment="Rephrase Agent completed successfully"
                        )

                langfuse.flush()

                instruction_lower = rephrase_instruction.lower()
                if "kannada" in instruction_lower and not contains_kannada(st.session_state.final_messages[loan_id]):
                    st.warning(
                        "The model response does not look like Kannada. Try again with: Translate fully to Kannada."
                    )
                elif "hindi" in instruction_lower and not contains_devanagari(st.session_state.final_messages[loan_id]):
                    st.warning(
                        "The model response does not look like Hindi. Try again with: Translate fully to Hindi."
                    )

                st.rerun()
            else:
                st.warning("Please enter a rephrase instruction.")

        final_message_to_send = st.text_area(
            "Final Message to Send - You can edit before sending",
            value=st.session_state.final_messages[loan_id],
            height=220,
            key=f"final_message_{loan_id}_{st.session_state.rephrase_versions[loan_id]}"
        )

        st.session_state.final_messages[loan_id] = final_message_to_send

        st.markdown("---")
        st.markdown("### Final Email Approval")

        default_name = get_default_customer_name(customer)
        default_email = get_default_customer_email(customer)

        st.caption("Selected customer details used for email approval")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            recipient_name = st.text_input(
                "Customer name",
                value=default_name,
                placeholder="Selected customer name",
                key=f"recipient_name_{loan_id}"
            )

        with detail_col2:
            recipient_email = st.text_input(
                "Customer email",
                value=default_email,
                placeholder="Selected customer email or test inbox",
                key=f"recipient_email_{loan_id}"
            )

        if default_email:
            st.success(f"Selected customer email loaded: {default_email}")
        else:
            st.warning("No email field found for the selected customer. Please enter a test/customer email before sending.")

        email_subject = st.text_input(
            "Email subject",
            value=f"Follow-up regarding loan {loan_id}",
            key=f"email_subject_{loan_id}"
        )

        send_confirmation = st.checkbox(
            "I have reviewed the final message and approve sending this email",
            key=f"send_confirm_{loan_id}"
        )

        if st.button("Send Email", key=f"send_email_{loan_id}"):
            final_message_to_send = st.session_state.final_messages[loan_id]

            if not send_confirmation:
                st.warning("Please review and approve the final message before sending.")
            elif not recipient_email.strip():
                st.warning("Please enter recipient email before sending.")
            elif not final_message_to_send.strip():
                st.warning("Email message cannot be empty.")
            else:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="email_send_agent",
                    input={
                        "loan_id": loan_id,
                        "recipient_name": recipient_name,
                        "recipient_email": recipient_email,
                        "subject": email_subject,
                        "message_preview": final_message_to_send[:300],
                        "discovery_mode": discovery_mode,
                    },
                    metadata={
                        "channel": "email",
                        "app": "loan_recovery_assistant",
                        "human_approval_required": True,
                    }
                ) as email_span:
                    try:
                        email_result = send_email_notification(
                            to_email=recipient_email.strip(),
                            subject=email_subject.strip(),
                            body=final_message_to_send.strip()
                        )

                        outputs["email_send_status"] = "sent"
                        outputs["last_sent_to_name"] = recipient_name.strip()
                        outputs["last_sent_to"] = recipient_email.strip()
                        outputs["last_sent_subject"] = email_subject.strip()
                        outputs["last_sent_message"] = final_message_to_send.strip()

                        email_span.update(
                            output={
                                "status": "sent",
                                "to_name": recipient_name.strip(),
                                "to_email": recipient_email.strip(),
                                "subject": email_subject.strip(),
                            }
                        )

                        add_success_score(
                            email_span,
                            name="email_sent",
                            comment="Final edited recovery email was sent successfully after human approval"
                        )

                        add_categorical_score(
                            email_span,
                            name="email_send_status",
                            value="success",
                            comment="Email sent successfully"
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        display_recipient = recipient_email.strip()
                        if recipient_name.strip():
                            display_recipient = f"{recipient_name.strip()} <{recipient_email.strip()}>"
                        st.success(f"Email sent successfully to {display_recipient}.")

                    except Exception as e:
                        outputs["email_send_status"] = "failed"
                        outputs["email_send_error"] = str(e)

                        email_span.update(
                            output={
                                "status": "failed",
                                "error": str(e),
                            },
                            level="ERROR",
                        )

                        add_categorical_score(
                            email_span,
                            name="email_send_status",
                            value="failed",
                            comment=str(e)
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        st.error(f"Failed to send email: {e}")

                langfuse.flush()

        if outputs.get("email_send_status") == "sent":
            st.info(
                f"Last email sent to {outputs.get('last_sent_to')} "
                f"with subject: {outputs.get('last_sent_subject')}"
            )

        # ---------------------------------------------------
        # Outbound Voice Call — Twilio Programmable Voice TTS
        # ---------------------------------------------------
        st.markdown("---")
        st.markdown("### Outbound Voice Call")

        default_phone = get_customer_phone(customer)

        call_col1, call_col2, call_col3 = st.columns(3)

        with call_col1:
            recipient_phone = st.text_input(
                "Customer phone number",
                value=default_phone,
                placeholder="+919876543210",
                key=f"recipient_phone_{loan_id}"
            )

        with call_col2:
            call_name = st.text_input(
                "Customer name for greeting",
                value=get_default_customer_name(customer),
                key=f"call_name_{loan_id}"
            )

        with call_col3:
            call_language = st.selectbox(
                "Voice language",
                options=SUPPORTED_LANGUAGES,
                index=0,
                key=f"call_language_{loan_id}"
            )

        if default_phone:
            st.success(f"Customer phone loaded: {default_phone}")
        else:
            st.warning("No phone field found for this customer. Enter a number manually.")

        call_mode = st.radio(
            "Call mode",
            ["Simple TTS", "Conversational IVR (asks customer name + confirmation)"],
            key=f"call_mode_{loan_id}",
            horizontal=True,
        )

        webhook_url = ""
        if "Conversational" in call_mode:
            webhook_url = st.text_input(
                "Webhook base URL (your public app URL)",
                placeholder="https://your-databricks-app.azuredatabricks.net",
                key=f"webhook_url_{loan_id}",
                help="Twilio needs a public URL to call back during the conversation. Use your Databricks App URL.",
            )

        call_confirmation = st.checkbox(
            "I have reviewed the voice message and approve placing this call",
            key=f"call_confirm_{loan_id}"
        )

        if st.button("Place Voice Call", key=f"place_call_{loan_id}"):
            final_message_to_send = st.session_state.final_messages.get(loan_id, "")

            if not call_confirmation:
                st.warning("Please review and approve before placing the call.")
            elif not recipient_phone.strip():
                st.warning("Please enter a phone number before placing the call.")
            elif not final_message_to_send.strip():
                st.warning("Voice message cannot be empty.")
            else:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="voice_call_agent",
                    input={
                        "loan_id": loan_id,
                        "to_phone": recipient_phone.strip(),
                        "customer_name": call_name.strip(),
                        "language": call_language,
                        "message_preview": final_message_to_send[:300],
                        "discovery_mode": discovery_mode,
                    },
                    metadata={
                        "channel": "voice",
                        "app": "loan_recovery_assistant",
                        "human_approval_required": True,
                    }
                ) as call_span:
                    try:
                        if "Conversational" in call_mode:
                            if not webhook_url.strip():
                                st.warning("Please enter the webhook base URL for conversational calls.")
                                langfuse.flush()
                                st.stop()
                            call_result = make_conversational_call(
                                to_phone=recipient_phone.strip(),
                                loan_id=str(loan_id),
                                message=final_message_to_send.strip(),
                                customer_name=call_name.strip(),
                                language=call_language,
                                webhook_base_url=webhook_url.strip(),
                            )
                        else:
                            call_result = make_voice_call(
                                to_phone=recipient_phone.strip(),
                                message=final_message_to_send.strip(),
                                customer_name=call_name.strip(),
                                language=call_language,
                            )

                        outputs["voice_call_status"] = "initiated"
                        outputs["voice_call_sid"] = call_result.get("call_sid")
                        outputs["voice_call_to"] = recipient_phone.strip()

                        call_span.update(
                            output={
                                "status": "initiated",
                                "call_sid": call_result.get("call_sid"),
                                "to_phone": recipient_phone.strip(),
                            }
                        )

                        add_success_score(
                            call_span,
                            name="voice_call_placed",
                            comment="Outbound voice call placed successfully after human approval"
                        )

                        add_categorical_score(
                            call_span,
                            name="voice_call_status",
                            value="success",
                            comment="Voice call initiated successfully"
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        st.success(
                            f"Voice call initiated to {recipient_phone.strip()}. "
                            f"Call SID: {call_result.get('call_sid')}"
                        )

                    except Exception as e:
                        outputs["voice_call_status"] = "failed"
                        outputs["voice_call_error"] = str(e)

                        call_span.update(
                            output={"status": "failed", "error": str(e)},
                            level="ERROR",
                        )

                        add_categorical_score(
                            call_span,
                            name="voice_call_status",
                            value="failed",
                            comment=str(e)
                        )

                        st.session_state.agent_outputs[loan_id] = outputs
                        st.error(f"Failed to place voice call: {e}")

                    langfuse.flush()

        if outputs.get("voice_call_status") == "initiated":
            st.info(
                f"Last call placed to {outputs.get('voice_call_to')} | "
                f"Call SID: {outputs.get('voice_call_sid')}"
            )

            # Show voice transcript if available (conversational calls)
            call_sid = outputs.get("voice_call_sid", "")
            if call_sid:
                transcript_data = get_call_transcript(call_sid)
                if transcript_data and transcript_data.get("transcript"):
                    st.session_state.voice_transcripts[loan_id] = transcript_data
                    with st.expander("Voice Conversation Transcript", expanded=False):
                        for turn in transcript_data["transcript"]:
                            role = turn.get("role", "")
                            text = turn.get("text", "")
                            ts   = turn.get("ts", "")
                            label = "Agent" if role == "agent" else "Customer"
                            st.markdown(f"**{label}** `{ts}`: {text}")
                        outcome = transcript_data.get("outcome", "pending")
                        spoken  = transcript_data.get("spoken_name", "")
                        st.caption(f"Outcome: **{outcome}** | Customer identified as: {spoken or 'unknown'}")

        # --------------------------------------------------------
        # Voice Transcript as Future Context
        # --------------------------------------------------------
        prior_transcript = st.session_state.voice_transcripts.get(loan_id)
        if prior_transcript and prior_transcript.get("transcript"):
            with st.expander("Prior Voice Conversation Context", expanded=False):
                st.markdown(
                    "The following voice conversation transcript was captured and "
                    "can be used as additional context for the next agent interaction."
                )
                st.json(prior_transcript)

        # ======================================================
        # Dynamic Recovery Strategy
        # ======================================================
        st.markdown("---")
        st.subheader("Dynamic Recovery Strategy")

        tier = determine_recovery_tier(customer)
        tier_label = get_tier_label(tier)
        tier_desc  = get_tier_description(tier)
        actions    = get_recovery_actions(tier)

        tier_colors = {1: "#2e7d32", 2: "#e65100", 3: "#b71c1c", 4: "#4a148c"}
        tier_color  = tier_colors.get(tier, "#546e7a")

        st.markdown(
            f"<div style='border-left: 5px solid {tier_color}; padding: 0.5rem 1rem; "
            f"background: #f9f9f9; border-radius: 4px;'>"
            f"<b style='color:{tier_color};'>{tier_label}</b><br/>{tier_desc}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Recovery actions determined for this customer:**")
        for action in actions:
            st.markdown(f"- {ACTION_LABELS.get(action, action)}")

        if loan_id not in st.session_state.recovery_actions:
            st.session_state.recovery_actions[loan_id] = {}
        recovery = st.session_state.recovery_actions[loan_id]

        # --------------------------------------------------------
        # Execute Recovery Actions
        # --------------------------------------------------------
        final_msg = st.session_state.final_messages.get(loan_id, "")
        customer_name_for_recovery = get_default_customer_name(customer)
        customer_email_for_recovery = get_default_customer_email(customer)
        customer_phone_for_recovery = get_customer_phone(customer)
        outstanding_amount = float(
            customer.get("loan_amount") or customer.get("funded_amount") or 0
        )
        dpd_val = int(float(customer.get("dpd", 0) or 0))
        refused_count_val = int(float(customer.get("refused_count", 0) or 0))

        recovery_approved = st.checkbox(
            f"Approve executing all {len(actions)} recovery action(s) for this customer",
            key=f"recovery_approve_{loan_id}",
        )

        if st.button("Execute Recovery Strategy", key=f"execute_recovery_{loan_id}"):
            if not recovery_approved:
                st.warning("Please approve the recovery actions before executing.")
            elif not final_msg.strip():
                st.warning("Generate a draft message first before executing recovery.")
            else:
                recovery_results = {}

                # --- SMS ---
                if "sms" in actions:
                    with st.spinner("Sending SMS notification..."):
                        if customer_phone_for_recovery:
                            try:
                                sms_result = send_sms_notification(
                                    to_phone=customer_phone_for_recovery,
                                    message=f"[Loan Recovery Notice] {final_msg[:400]}",
                                )
                                recovery_results["sms"] = {"status": "sent", **sms_result}
                                st.success(f"SMS sent to {customer_phone_for_recovery} — SID: {sms_result.get('message_sid')}")
                            except Exception as e:
                                recovery_results["sms"] = {"status": "failed", "error": str(e)}
                                st.warning(f"SMS failed: {e}")
                        else:
                            recovery_results["sms"] = {"status": "skipped", "reason": "No phone number available"}
                            st.info("SMS skipped: no phone number for this customer.")

                # --- CIBIL Report ---
                if "cibil_report" in actions:
                    with st.spinner("Filing CIBIL adverse report..."):
                        try:
                            cibil_record = report_to_cibil(
                                customer=customer,
                                loan_id=loan_id,
                                outstanding_amount=outstanding_amount,
                                dpd=dpd_val,
                                reason=outputs.get("risk_explanation", ""),
                            )
                            notify_result = notify_cibil_report(
                                customer=customer,
                                report=cibil_record,
                                to_email=customer_email_for_recovery,
                                to_phone=customer_phone_for_recovery,
                            )
                            recovery_results["cibil_report"] = {
                                "status": "filed",
                                "report_id": cibil_record["report_id"],
                                "notifications": notify_result,
                            }
                            st.success(
                                f"CIBIL report filed — Ref: {cibil_record['report_id']}. "
                                f"Customer notified via email/SMS."
                            )
                        except Exception as e:
                            recovery_results["cibil_report"] = {"status": "failed", "error": str(e)}
                            st.warning(f"CIBIL reporting failed: {e}")

                # --- Penalty ---
                if "penalty" in actions:
                    with st.spinner("Applying financial penalty..."):
                        try:
                            penalty_record = apply_penalty(
                                customer=customer,
                                loan_id=loan_id,
                                outstanding_amount=outstanding_amount,
                                dpd=dpd_val,
                                applied_by="recovery_strategy_agent",
                            )
                            recovery_results["penalty"] = penalty_record
                            penalty_summary = get_penalty_summary(penalty_record)
                            if penalty_record.get("status") == "applied":
                                st.success(
                                    f"Penalty applied — ID: {penalty_record['penalty_id']} | "
                                    f"Amount: ₹{penalty_record['penalty_amount']:,.2f}"
                                )
                            else:
                                st.info(penalty_summary)
                        except Exception as e:
                            recovery_results["penalty"] = {"status": "failed", "error": str(e)}
                            st.warning(f"Penalty application failed: {e}")

                # --- Legal Complaint ---
                if "legal_complaint" in actions:
                    with st.spinner("Filing legal complaint..."):
                        try:
                            legal_case = file_legal_complaint(
                                customer=customer,
                                loan_id=loan_id,
                                outstanding_amount=outstanding_amount,
                                dpd=dpd_val,
                                refused_count=refused_count_val,
                                reason=outputs.get("risk_explanation", ""),
                                filed_by="recovery_strategy_agent",
                            )
                            recovery_results["legal_complaint"] = legal_case
                            st.error(
                                f"Legal complaint filed — Case ID: {legal_case['case_id']} | "
                                f"Type: {legal_case['action_type']} | "
                                f"Next Hearing: {legal_case['next_hearing_date']}"
                            )
                        except Exception as e:
                            recovery_results["legal_complaint"] = {"status": "failed", "error": str(e)}
                            st.warning(f"Legal complaint filing failed: {e}")

                st.session_state.recovery_actions[loan_id] = recovery_results

        # --- Show previous recovery results ---
        if recovery:
            st.markdown("---")
            st.markdown("**Recovery Actions Already Executed:**")

            if "sms" in recovery:
                r = recovery["sms"]
                icon = "✅" if r.get("status") == "sent" else "⚠️"
                st.write(f"{icon} SMS: {r.get('status')} {r.get('message_sid', r.get('reason', r.get('error', '')))}")

            if "cibil_report" in recovery:
                r = recovery["cibil_report"]
                icon = "✅" if r.get("status") == "filed" else "⚠️"
                st.write(f"{icon} CIBIL: {r.get('status')} — Ref: {r.get('report_id', r.get('error', ''))}")

            if "penalty" in recovery:
                r = recovery["penalty"]
                icon = "✅" if r.get("status") in ("applied", "skipped") else "⚠️"
                amt  = f"₹{r['penalty_amount']:,.2f}" if r.get("status") == "applied" else ""
                pen_id = r.get("penalty_id", r.get("error", ""))
                st.write(f"{icon} Penalty: {r.get('status')} {pen_id} {amt}")

            if "legal_complaint" in recovery:
                r = recovery["legal_complaint"]
                icon = "✅" if r.get("status") == "filed" else "⚠️"
                st.write(
                    f"{icon} Legal: {r.get('status')} — Case: {r.get('case_id', r.get('error', ''))} "
                    f"| Type: {r.get('action_type', '')} | Hearing: {r.get('next_hearing_date', '')}"
                )

            with st.expander("Full Recovery Action Details", expanded=False):
                st.json(recovery)

    else:
        st.info("Draft message will appear after dynamic analysis.")

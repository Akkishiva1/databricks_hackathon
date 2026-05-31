"""
Configuration and constants for Loan Recovery Assistant.
"""
import os

# -------------------------------------------------
# Databricks Configuration
# -------------------------------------------------
DATABRICKS_HOST = "https://adb-7405606631030160.0.azuredatabricks.net"

LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
LLM_ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT_NAME}/invocations"

SUPERVISOR_AGENT_ENDPOINT_NAME = "mas-05b00371-endpoint"
SUPERVISOR_AGENT_ENDPOINT_URL = (
    f"{DATABRICKS_HOST}/serving-endpoints/"
    f"{SUPERVISOR_AGENT_ENDPOINT_NAME}/invocations"
)

CUSTOMER_360_TABLE = os.getenv(
    "CUSTOMER_360_TABLE",
    "loan_recovery.gold.loan_recovery_customer_360"
)

AUDIT_TABLE = os.getenv(
    "AUDIT_TABLE",
    "loan_recovery.gold.loan_recovery_agent_audit_log"
)

# -------------------------------------------------
# Pricing Configuration
# -------------------------------------------------
INPUT_DBUS_PER_1M_TOKENS = 2.143
OUTPUT_DBUS_PER_1M_TOKENS = 8.571

# -------------------------------------------------
# Cache Configuration
# -------------------------------------------------
DISCOVERY_CACHE_TTL_SECONDS = 300
DISCOVERY_CACHE_VERSION = "v5-intent-cache-fix"

# -------------------------------------------------
# LLM Parameters
# -------------------------------------------------
LLM_DEFAULT_TEMPERATURE = 0.2
LLM_DEFAULT_MAX_TOKENS = 700

# -------------------------------------------------
# Email Fields (Priority order)
# -------------------------------------------------
CUSTOMER_EMAIL_FIELDS = [
    "email",
    "customer_email",
    "email_address",
    "customer_email_address",
    "primary_email",
    "contact_email",
]

# -------------------------------------------------
# Customer Name Fields (Priority order)
# -------------------------------------------------
CUSTOMER_NAME_FIELDS = [
    "name",
    "customer_name",
    "full_name",
    "member_name",
    "borrower_name",
    "first_name",
]

# -------------------------------------------------
# Preferred Display Columns
# -------------------------------------------------
PREFERRED_DISPLAY_COLUMNS = [
    "member_id",
    "loan_id",
    "name",
    "customer_name",
    "full_name",
    "member_name",
    "borrower_name",
    "first_name",
    "email",
    "customer_email",
    "email_address",
    "customer_email_address",
    "primary_email",
    "contact_email",
    "emp_title",
    "annual_income",
    "loan_amount",
    "funded_amount",
    "loan_status",
    "loan_purpose",
    "interest_rate",
    "monthly_installment",
    "dpd",
    "risk_score",
    "risk_band",
    "recommended_action",
    "escalation_flag",
    "broken_ptp",
    "refused_count",
]

# -------------------------------------------------
# Display Column Subset
# -------------------------------------------------
DISPLAY_COLUMNS = [
    "member_id",
    "loan_id",
    "name",
    "customer_name",
    "full_name",
    "member_name",
    "borrower_name",
    "email",
    "customer_email",
    "email_address",
    "customer_email_address",
    "primary_email",
    "contact_email",
    "dpd",
    "risk_score",
    "risk_band",
    "loan_amount",
    "funded_amount",
    "loan_status",
    "loan_purpose",
    "recommended_action",
    "escalation_flag",
    "broken_ptp",
    "refused_count",
]

# -------------------------------------------------
# Business Logic Phrases
# -------------------------------------------------
NORMAL_CUSTOMER_PHRASES = [
    "normal customer",
    "normal customers",
    "regular customer",
    "regular customers",
    "safe customer",
    "safe customers",
    "low risk customer",
    "low risk customers",
    "low-risk customer",
    "low-risk customers",
    "non critical",
    "non-critical",
    "not critical",
    "good customer",
    "good customers",
    "healthy customer",
    "healthy customers",
]

ESCALATION_PHRASES = [
    "need escalation",
    "needs escalation",
    "require escalation",
    "requires escalation",
    "escalation required",
    "who need escalation",
    "who needs escalation",
    "customers need escalation",
    "customers needing escalation",
]

NO_ESCALATION_PHRASES = [
    "no escalation",
    "without escalation",
    "not need escalation",
    "not needs escalation",
    "do not need escalation",
    "does not need escalation",
    "dont need escalation",
    "don't need escalation",
    "not require escalation",
    "does not require escalation",
    "do not require escalation",
]

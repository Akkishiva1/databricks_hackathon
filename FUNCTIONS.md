# Agent Functions — Explanation & Temperature Guide

## What is Temperature?

Temperature controls how creative or deterministic the LLM output is.

| Range | Behaviour | Use when |
|---|---|---|
| 0.0 – 0.1 | Highly deterministic, consistent output | Structured data extraction, JSON, translation |
| 0.2 – 0.3 | Mostly factual with slight natural variation | Analysis, recommendations, professional writing |
| 0.7 – 1.0 | Creative, varied, unpredictable | Brainstorming, creative content |

All agents in this app use **low temperatures (0.1 – 0.3)** because loan recovery is a compliance-sensitive domain — outputs must be consistent, factual, and grounded in the data provided.

---

## Agent Functions

---

### 1. `query_understanding_agent()`
**File:** `agents/discovery.py` | **Temperature: 0.1**

#### What it does
Converts a plain English question from the recovery officer into a structured JSON of SQL filter parameters. This is Agent 1 of the Custom Dynamic Supervisor flow.

#### Why temperature 0.1
The lowest temperature in the app. The output must be a strict JSON object — any creativity or variation would produce invalid JSON or wrong filter values. We need the same question to always produce the same parameters.

#### Input → Output
```
Input:  "Show me critical customers with broken PTP who need escalation"

Output: {
  "risk_band": "Critical",
  "min_dpd": 0,
  "max_dpd": null,
  "escalation_only": true,
  "no_escalation_only": false,
  "broken_ptp_only": true,
  "refused_only": false,
  "limit": 100,
  "sort_by": "risk_score"
}
```

#### Key design decisions
- Temperature 0.1 so JSON structure is always valid
- LLM output is then passed through a **keyword override guard** — rule-based checks that enforce correctness regardless of what the LLM returned
- `risk_band` and `sort_by` are **whitelisted** — any unexpected value is reset to a safe default
- `limit` is **clamped** between 1 and 100 — the LLM cannot trigger an unbounded query

---

### 2. `data_retrieval_agent()`
**File:** `agents/discovery.py` | **Temperature: N/A (no LLM call)**

#### What it does
Takes the JSON params from `query_understanding_agent` and executes a real SQL query against the Databricks Gold table. This is Agent 2 of the Custom Dynamic Supervisor flow. No LLM is involved — this is pure deterministic code.

#### Steps
1. Calls `DESCRIBE TABLE` to discover the live schema (cached 5 min)
2. Builds a `SELECT` clause using only columns that exist in the table
3. Builds a `WHERE` clause — one condition per active param
4. Wraps in `ROW_NUMBER() OVER (PARTITION BY loan_id)` to deduplicate
5. Executes via `databricks-sql-connector` SDK
6. Returns a pandas DataFrame

#### Why no LLM
SQL construction is deterministic. Using an LLM here would introduce unpredictability into data retrieval, which is unacceptable for a compliance application. The LLM's job ended at parameter extraction.

#### Deduplication logic
The Gold table stores multiple rows per loan — one per recommended action. `ROW_NUMBER()` partitioned by `loan_id` keeps exactly one row per customer, selecting the row with the highest `risk_score`.

```sql
SELECT ...
FROM (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY loan_id
      ORDER BY risk_score DESC
    ) AS _row_num
  FROM loan_recovery_customer_360
  WHERE <filters>
) _deduped
WHERE _row_num = 1
ORDER BY risk_score DESC
LIMIT 100
```

---

### 3. `build_agent_bricks_supervisor_question()`
**File:** `agents/discovery.py` | **Temperature: N/A (prompt builder)**

#### What it does
Constructs the prompt sent to the Databricks Agent Bricks endpoint. Not an LLM call itself — it wraps the user question with instructions telling Agent Bricks to use Genie Space and Unity Catalog tools, and specifying the exact JSON structure to return.

#### Why this exists separately
Agent Bricks is a managed Databricks endpoint — it handles its own tool selection (Genie, UC functions) internally. Your code only controls the question format and the response schema you expect back.

---

### 4. `agent_bricks_supervisor_discovery()`
**File:** `agents/discovery.py` | **Temperature: Managed by Databricks**

#### What it does
Executes the Agent Bricks flow — sends the prompt via plain HTTP POST to the Databricks Agent Bricks serving endpoint, parses the response, and returns a DataFrame of customers.

#### Key behaviour
- Databricks internally decides whether to use Genie Space, Unity Catalog functions, or both
- Your code captures the internal `tool_trace` from the response (which tools Databricks called)
- This trace is logged to Langfuse under `metadata.tool_trace`
- On failure (403, timeout, parse error) → falls back to Custom Dynamic Supervisor

#### Connection method
```python
requests.post(
    SUPERVISOR_AGENT_ENDPOINT_URL,    # Agent Bricks serving endpoint
    headers={"Authorization": f"Bearer {token}"},
    json=payload,
    timeout=180
)
```

---

### 5. `risk_analysis_agent()`
**File:** `agents/analysis.py` | **Temperature: 0.2**

#### What it does
Analyses the selected customer's risk profile and produces a clear written explanation of why this customer is risky. Uses the customer's DPD, risk score, risk band, loan amount, escalation flag, broken PTP count, and refused count.

#### Why temperature 0.2
The explanation must be factual and grounded in the data provided — no invented details. A small amount of variation (0.2 vs 0.1) is acceptable here because the output is natural language prose, not structured data, and slight variation makes the explanation feel less robotic.

#### What it outputs
A short paragraph explaining:
- How many days past due the customer is
- What the risk band and risk score mean
- Whether escalation is required
- Any behavioural signals (broken PTP, refusals)

#### Prompt rules enforced
- Use only the given data — do not invent details
- Mention DPD, risk score, risk band, loan amount, loan status, loan purpose
- If `escalation_flag = Y` → explicitly state escalation is required
- Keep it concise and demo-friendly

---

### 6. `recommendation_agent()`
**File:** `agents/analysis.py` | **Temperature: 0.2**

#### What it does
Takes the risk explanation and customer data and recommends the next-best recovery action — what the officer should do next with this customer.

#### Why temperature 0.2
Same reasoning as `risk_analysis_agent` — factual, grounded, consistent. The recommendation must align with the `recommended_action` field from the data where available. No guessing.

#### What it outputs
2–3 bullet points explaining:
- What action to take (call, escalate, send notice, legal action, etc.)
- Why this action is appropriate for this customer
- Any urgency signals

#### Prompt rules enforced
- Use `recommended_action` field from customer data if available
- If `escalation_flag = Y` → clearly state escalation is required
- If `broken_ptp` or `refused_count` is available → use as recovery behaviour context
- Do not invent details

---

### 7. `communication_agent()`
**File:** `agents/analysis.py` | **Temperature: 0.3**

#### What it does
Drafts a professional follow-up message addressed to the customer. This is the message the recovery officer will review, potentially rephrase, and then send via email.

#### Why temperature 0.3
Slightly higher than the analysis agents. The message is customer-facing and benefits from natural, human-sounding language — a small amount of variation prevents every message sounding identical. Still low enough to stay professional and factual.

#### What it outputs
A complete customer-facing message that:
- Addresses the customer by name
- Mentions the loan ID and days past due
- States what action is expected
- Maintains a polite and professional tone
- Adds urgency if `escalation_flag = Y` or `risk_band = Critical`

#### Prompt rules enforced
- Use only the given context — no invented details
- Mention loan ID and DPD if available
- Do not reveal internal risk score unless necessary
- Return only the message draft — no explanations or reasoning

---

### 8. `rephrase_agent()`
**File:** `agents/analysis.py` | **Temperature: 0.1**

#### What it does
Rewrites the drafted message according to an officer's instruction — change the tone (formal, urgent, empathetic) or translate to a different language (Hindi or Kannada).

#### Why temperature 0.1
The lowest temperature among the analysis agents. Translation especially requires precision — a higher temperature risks mixing languages, mistranslating values, or dropping important loan details. Consistent, accurate output is critical here.

#### What it outputs
A rewritten version of the original message following the instruction exactly. Loan ID, DPD, and amounts are preserved verbatim.

#### Language validation (post-rephrase)
After the LLM returns, `app.py` validates the language:
```python
# Unicode range check — not just keyword matching
if "kannada" in instruction:
    if not contains_kannada(rephrased_text):
        st.warning("Response may not be in Kannada. Please verify.")

if "hindi" in instruction:
    if not contains_devanagari(rephrased_text):
        st.warning("Response may not be in Hindi. Please verify.")
```

#### Prompt rules enforced
- Return only the rewritten message — no reasoning, no JSON, no markdown
- Preserve all numeric values exactly (loan ID, DPD, amount, phone number)
- Do not mix languages unless explicitly asked
- Do not invent new customer or loan details

---

### 9. `audit_logger_agent()`
**File:** `agents/analysis.py` | **Temperature: N/A (no LLM call)**

#### What it does
Writes a compliance record to the Delta audit table after every analysis run. Captures who asked what, about which customer, and what the agents recommended and drafted.

#### Why no LLM
Audit logging must be exact and deterministic. There is no place for LLM involvement — this is a pure database write.

#### What it writes
```sql
INSERT INTO audit_table (
  agent_request_id,      -- UUID generated per run
  user_query,            -- original question from officer
  customer_id,           -- member_id from customer record
  loan_id,               -- loan_id from customer record
  recommended_action,    -- what the recommendation agent suggested
  reason,                -- reasoning from recommendation agent
  message_draft,         -- the communication agent's drafted message
  source_tables_used,    -- which Gold table was queried
  created_by,            -- which discovery mode was used
  created_at             -- timestamp
)
```

#### Security measures applied
- All string values passed through `escape_sql()` (escapes `'` and `\`)
- `discovery_mode` validated against a whitelist before insertion
- `run_statement()` wrapped in try/except — audit failure is logged to Langfuse but does not crash the app

---

## Temperature Summary Table

| Agent | Temperature | Reason |
|---|---|---|
| `query_understanding_agent` | **0.1** | Must produce valid JSON — zero tolerance for variation |
| `rephrase_agent` | **0.1** | Translation requires precision — no language mixing |
| `risk_analysis_agent` | **0.2** | Factual prose — slight variation acceptable |
| `recommendation_agent` | **0.2** | Factual recommendations — must align with data |
| `communication_agent` | **0.3** | Customer-facing — benefits from natural language variation |
| `data_retrieval_agent` | **N/A** | No LLM — pure SQL execution |
| `audit_logger_agent` | **N/A** | No LLM — pure database write |
| Agent Bricks Supervisor | **Managed** | Databricks controls internally |

---

## Design Principle

> **Use the lowest temperature that still produces natural output for the task.**

- **Structured output** (JSON, SQL params, translation) → `0.1`
- **Analytical prose** (risk, recommendation) → `0.2`
- **Human-facing writing** (customer message) → `0.3`
- **No LLM needed** (SQL execution, DB writes) → skip the LLM entirely

This ensures the app is consistent and reliable in production while still producing natural, readable output for the end user.

# Loan Recovery Assistant - Modular Architecture

## Overview

The Loan Recovery Assistant has been refactored from a **2,255-line monolithic** Streamlit application into a **clean, modular, and maintainable** architecture. This document outlines the new structure and how to work with it.

## Project Structure

```
Databricks_poc/
├── app.py                      # Main Streamlit UI (refactored, ~800 lines)
├── config.py                   # Configuration & constants
├── pricing.py                  # DBU cost calculation
├── langfuse_service.py         # Langfuse observability
├── databricks_client.py        # SQL connection & queries
├── email_service.py            # Email functionality
├── text_processors.py          # Text parsing & processing
├── llm_service.py              # Databricks LLM calls
├── discovery_cache.py          # Cache management
├── customer_helpers.py         # Customer data utilities
├── table_helpers.py            # Database schema utilities
├── agents/
│   ├── __init__.py
│   ├── discovery.py            # Discovery agents (Agent Bricks & Custom Supervisor)
│   └── analysis.py             # Analysis agents (Risk, Recommendation, Communication, Audit)
└── app_old.py                  # Backup of original monolithic code
```

## Module Breakdown

### 1. **config.py** (Configuration)
Centralized configuration and constants.

**Contents:**
- Databricks endpoint URLs
- LLM configuration
- Pricing parameters
- Cache settings
- Business logic phrases (customer types, escalation, etc.)
- Preferred column lists

**Why:** Single source of truth for all configuration values.

```python
from config import DATABRICKS_HOST, LLM_ENDPOINT_NAME, CUSTOMER_360_TABLE
```

### 2. **pricing.py** (Cost Calculation)
DBU pricing and token usage calculations.

**Functions:**
- `calculate_dbu_cost()` - Calculate cost from token counts
- `extract_usage()` - Extract and calculate usage from LLM response

**Why:** Separate cost calculation logic for better testability.

```python
from pricing import calculate_dbu_cost, extract_usage
```

### 3. **langfuse_service.py** (Observability)
Langfuse client initialization and scoring.

**Functions:**
- `add_success_score()` - Add boolean success scores
- `add_trace_quality_score()` - Add numeric quality scores
- `add_categorical_score()` - Add categorical scores

**Why:** Centralized observability, easy to swap providers.

```python
from langfuse_service import langfuse, add_success_score
```

### 4. **databricks_client.py** (Database)
Databricks SQL connection and query execution.

**Functions:**
- `get_sql_connection()` - Create DB connection
- `run_query()` - Execute SELECT query with caching
- `run_statement()` - Execute INSERT/UPDATE/DELETE
- `escape_sql()` - SQL string escaping

**Why:** Centralized DB access layer, consistent connection management.

```python
from databricks_client import run_query, get_sql_connection
```

### 5. **email_service.py** (Email)
Email sending via SMTP.

**Functions:**
- `send_email_notification()` - Send recovery emails

**Why:** Isolated email functionality, easy to mock for testing.

```python
from email_service import send_email_notification
```

### 6. **text_processors.py** (Text Utilities)
Text parsing, cleaning, and extraction.

**Functions:**
- `extract_json_from_text()` - Extract JSON from LLM responses
- `clean_agent_text()` - Remove tool artifacts
- `extract_llm_text()` - Parse various LLM response formats
- `contains_kannada()` / `contains_devanagari()` - Language detection

**Why:** Reusable text processing utilities.

```python
from text_processors import extract_json_from_text, clean_agent_text
```

### 7. **llm_service.py** (LLM Calls)
Databricks GPT-OSS LLM integration with observability.

**Functions:**
- `call_databricks_llm()` - Call LLM with Langfuse tracing

**Why:** Centralized LLM access, consistent error handling.

```python
from llm_service import call_databricks_llm
```

### 8. **discovery_cache.py** (Caching)
Customer discovery result caching.

**Functions:**
- `get_cached_discovery_result()` - Retrieve cached results
- `set_cached_discovery_result()` - Cache discovery results
- `is_low_risk_no_escalation_question()` - Detect low-risk queries

**Why:** Efficient caching strategy, avoids redundant LLM calls.

```python
from discovery_cache import get_cached_discovery_result
```

### 9. **customer_helpers.py** (Customer Data)
Customer information extraction.

**Functions:**
- `get_default_customer_email()` - Extract email
- `get_default_customer_name()` - Extract name
- `safe_customer_context()` - Create audit-safe context

**Why:** Consistent customer data extraction, reduced code duplication.

```python
from customer_helpers import get_default_customer_name, safe_customer_context
```

### 10. **table_helpers.py** (Schema)
Database table schema utilities.

**Functions:**
- `get_table_columns()` - Get column names from table

**Why:** Dynamic schema discovery.

```python
from table_helpers import get_table_columns
```

### 11. **agents/discovery.py** (Discovery Agents)
Customer discovery implementations.

**Functions:**
- `agent_bricks_supervisor_discovery()` - Agent Bricks Supervisor
- `query_understanding_agent()` - Query understanding
- `data_retrieval_agent()` - Data retrieval
- `custom_dynamic_supervisor_discovery()` - Custom supervisor

**Why:** Separated discovery logic, easier to test and modify.

```python
from agents.discovery import agent_bricks_supervisor_discovery, custom_dynamic_supervisor_discovery
```

### 12. **agents/analysis.py** (Analysis Agents)
Customer analysis and communication agents.

**Functions:**
- `risk_analysis_agent()` - Risk explanation
- `recommendation_agent()` - Recovery action recommendation
- `communication_agent()` - Message drafting
- `rephrase_agent()` - Message rephrasing
- `audit_logger_agent()` - Audit trail logging

**Why:** Modular agent functions, reusable across workflows.

```python
from agents.analysis import risk_analysis_agent, recommendation_agent
```

### 13. **app.py** (Main UI)
Streamlit UI and orchestration (~800 lines vs original 2,255).

**Structure:**
- Page config
- Session state initialization
- Helper functions for UI
- Main form & discovery flow
- Results display & analysis
- Email approval & sending

**Why:** Clean, focused UI layer with logic abstracted to modules.

## Benefits of Modular Architecture

### 1. **Maintainability**
- Each module has a single responsibility
- Easier to locate and fix bugs
- Clear dependencies

### 2. **Testability**
- Isolated modules can be tested independently
- Mock external services easily
- Test coverage is simpler

### 3. **Reusability**
- Modules can be imported in other projects
- No tight coupling to Streamlit
- Functions are pure and stateless

### 4. **Scalability**
- Easy to add new agents
- Simple to add new data sources
- Can build CLI/API wrappers on top

### 5. **Collaboration**
- Team members can work on different modules
- Reduced merge conflicts
- Clear module interfaces

## Usage Examples

### Example 1: Running a Query
```python
from databricks_client import run_query

df = run_query("""
    SELECT * FROM loan_recovery.gold.loan_recovery_customer_360
    WHERE dpd > 30
""")
```

### Example 2: Calling the LLM
```python
from llm_service import call_databricks_llm

response = call_databricks_llm(
    prompt="Analyze this customer risk",
    temperature=0.2,
    max_tokens=500,
    agent_name="custom_agent"
)
```

### Example 3: Using Discovery Agents
```python
from agents.discovery import custom_dynamic_supervisor_discovery

df, params = custom_dynamic_supervisor_discovery("Show me critical customers")
print(f"Found {len(df)} customers")
```

### Example 4: Running Analysis
```python
from agents.analysis import risk_analysis_agent

explanation = risk_analysis_agent(
    customer=customer_dict,
    user_question="Should we escalate this customer?"
)
```

## Adding New Features

### Adding a New Agent
1. Create function in `agents/analysis.py` or `agents/discovery.py`
2. Import in `app.py`
3. Add UI component in main form

### Adding a New Data Source
1. Create wrapper in `databricks_client.py`
2. Add constants to `config.py`
3. Import and use in agents

### Adding a New Analysis Tool
1. Create module (e.g., `prediction_service.py`)
2. Add to imports in `app.py`
3. Integrate into workflow

## Dependencies

- `streamlit` - UI framework
- `pandas` - Data handling
- `databricks` - SQL connection
- `langfuse` - Observability
- `requests` - HTTP calls

All specified in `requirements.txt`.

## Configuration

Edit `config.py` to customize:
- Databricks endpoints
- LLM parameters
- Cache TTL
- Business rules (phrases, fields, etc.)
- Display columns

## Deployment

The modular structure is deployment-ready:
- Pure Python functions (no Streamlit-specific logic except in `app.py`)
- Can be packaged as pip module
- Can be containerized easily
- Can be wrapped with FastAPI/Flask for APIs

## Migration Notes

The original 2,255-line code is preserved in `app_old.py` for reference. The new architecture provides:
- **~65% code reduction** in main file (2,255 → 800 lines)
- **11 focused modules** instead of one monolith
- **Clear separation of concerns**
- **Improved testability and reusability**

## Future Improvements

1. Add unit tests for each module
2. Add type hints throughout
3. Create API wrapper with FastAPI
4. Add configuration file support (YAML/JSON)
5. Add logging framework
6. Create plugin system for custom agents

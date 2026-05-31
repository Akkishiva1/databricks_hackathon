"""
Discovery cache management for caching customer discovery results.
"""
import time
import streamlit as st
from config import DISCOVERY_CACHE_TTL_SECONDS, DISCOVERY_CACHE_VERSION, NORMAL_CUSTOMER_PHRASES, NO_ESCALATION_PHRASES


def normalize_question(question: str) -> str:
    """Normalize natural-language question for cache matching."""
    return " ".join((question or "").strip().lower().split())


def get_discovery_cache_key(discovery_mode: str, question: str) -> str:
    """Generate cache key for discovery result."""
    return f"{DISCOVERY_CACHE_VERSION}::{discovery_mode}::{normalize_question(question)}"


def is_low_risk_no_escalation_question(question: str) -> bool:
    """Detect common business wording for safe/non-critical customers."""
    q = normalize_question(question)
    
    return any(p in q for p in NORMAL_CUSTOMER_PHRASES) or any(p in q for p in NO_ESCALATION_PHRASES)


def get_cached_discovery_result(discovery_mode: str, question: str):
    """Return cached discovery result if present and not expired."""
    if "discovery_cache" not in st.session_state:
        st.session_state.discovery_cache = {}
    
    cache_key = get_discovery_cache_key(discovery_mode, question)
    cached = st.session_state.discovery_cache.get(cache_key)

    if not cached:
        return None

    age_seconds = time.time() - cached["created_at"]

    if age_seconds > DISCOVERY_CACHE_TTL_SECONDS:
        del st.session_state.discovery_cache[cache_key]
        return None

    return cached


def set_cached_discovery_result(
    discovery_mode: str,
    question: str,
    df,
    query_params: dict,
    discovery_summary: str,
) -> None:
    """Cache the full recovery-question discovery result."""
    if "discovery_cache" not in st.session_state:
        st.session_state.discovery_cache = {}
    
    cache_key = get_discovery_cache_key(discovery_mode, question)

    # Do not cache empty results
    if df is None or df.empty:
        return

    st.session_state.discovery_cache[cache_key] = {
        "created_at": time.time(),
        "df": df.copy(),
        "query_params": query_params,
        "discovery_summary": discovery_summary or "",
    }

"""
Langfuse observability service for tracing and scoring.
"""
from langfuse import get_client

# Initialize Langfuse client
langfuse = get_client()


def add_success_score(span, name: str, comment: str = ""):
    """Add a boolean success score to a span."""
    try:
        span.score(
            name=name,
            value=1,
            data_type="BOOLEAN",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add boolean score {name}: {e}")


def add_trace_quality_score(span, name: str, value: float, comment: str = ""):
    """Add a numeric quality score to a trace."""
    try:
        span.score_trace(
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add numeric trace score {name}: {e}")


def add_categorical_score(span, name: str, value: str, comment: str = ""):
    """Add a categorical score to a trace."""
    try:
        span.score_trace(
            name=name,
            value=value,
            data_type="CATEGORICAL",
            comment=comment
        )
    except Exception as e:
        print(f"Failed to add categorical score {name}: {e}")

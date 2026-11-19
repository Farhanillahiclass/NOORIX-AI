"""Privacy-minimizing diagnostics for user-facing error summaries."""


def safe_error_summary(error: BaseException, context: str = "Operation failed") -> str:
    """Summarize an error without exposing exception messages or local content."""
    safe_context = " ".join(str(context).split())[:100] or "Operation failed"
    return f"{safe_context} ({type(error).__name__})."

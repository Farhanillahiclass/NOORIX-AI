"""Privacy-minimizing diagnostics for user-facing error summaries."""


def safe_error_summary(error: BaseException, context: str = "Operation failed") -> str:
    """Summarize an error without exposing exception messages or local content."""
    printable_context = "".join(character for character in str(context) if character.isprintable())
    safe_context = " ".join(printable_context.split())[:100] or "Operation failed"
    return f"{safe_context} ({type(error).__name__})."

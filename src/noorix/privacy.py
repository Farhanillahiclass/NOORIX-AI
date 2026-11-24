"""Privacy-minimizing diagnostics for user-facing error summaries."""
from __future__ import annotations

import re

_ANSI_ESCAPE = re.compile(r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\\\)|\[[0-?]*[ -/]*[@-~]|[@-_])")


def safe_error_summary(error: BaseException, context: str = "Operation failed") -> str:
    """Summarize an error without exposing exception messages or local content."""
    plain_context = _ANSI_ESCAPE.sub("", str(context))
    printable_context = "".join(character for character in plain_context if character.isprintable())
    safe_context = " ".join(printable_context.split())[:100] or "Operation failed"
    return f"{safe_context} ({type(error).__name__})."

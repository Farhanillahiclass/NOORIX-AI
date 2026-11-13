"""Capability readiness declarations.

A capability is marked unavailable until a real model/backend is installed,
configured, and tested. Never infer readiness from this scaffold alone.
"""

CAPABILITIES = {
    "camera": "not_configured",
    "pakistani_currency_estimation": "model_not_integrated",
    "everyday_object_detection": "model_not_integrated",
    "printed_english_ocr": "backend_not_integrated",
    "spoken_feedback": "backend_not_integrated",
}


def capability_status() -> dict[str, str]:
    """Return a copy so callers cannot mutate the module defaults."""
    return dict(CAPABILITIES)

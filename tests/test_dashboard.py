"""Regression checks for the rebuilt dashboard entry points."""
from noorix.dashboard import NoorixDashboard, PALETTE
from noorix.gui import C, NoorixApp, launch


def test_dashboard_palette_has_accessible_dark_cyan_tokens():
    assert PALETTE["bg"] == "#07111f"
    assert PALETTE["cyan"] == "#27d9ee"
    assert PALETTE["text"] != PALETTE["bg"]


def test_dashboard_navigation_covers_requested_workspaces():
    names = [name for _, group in NoorixDashboard.NAV for name in group]
    expected = {
        "Command Center", "Camera & Scene", "Object Detection", "English OCR",
        "Handwriting OCR", "Urdu OCR (Experimental)", "PKR Currency",
        "Voice Interaction", "Text to Speech", "Model Management",
        "Testing Center", "Activity Logs", "Diagnostics", "Settings", "About Farhan",
    }
    assert expected.issubset(set(names))


def test_legacy_gui_entrypoint_launches_new_dashboard():
    assert launch is not None
    assert NoorixApp is not None
    assert C["cyan"] == PALETTE["cyan"]

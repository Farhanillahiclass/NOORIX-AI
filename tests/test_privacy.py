from noorix.privacy import safe_error_summary


def test_safe_error_summary_omits_exception_message():
    secret = "private OCR phrase at C:/Users/person/Documents/page.png"
    summary = safe_error_summary(RuntimeError(secret), "OCR unavailable")
    assert summary == "OCR unavailable (RuntimeError)."
    assert secret not in summary


def test_safe_error_summary_bounds_and_normalizes_context():
    summary = safe_error_summary(ValueError("private"), "  " + "context " * 40)
    assert len(summary) < 140
    assert "private" not in summary



def test_safe_error_summary_removes_terminal_control_characters_from_context():
    context = "OCR\x1b[31m unavailable\x1b[0m\nnext-line"
    summary = safe_error_summary(RuntimeError("private"), context)

    assert chr(27) not in summary
    assert "\n" not in summary
    assert "OCR unavailable next-line" in summary

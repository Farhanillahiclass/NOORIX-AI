"""Regression tests for strict specialist adapter output contracts."""

import pytest

from noorix.agents import AgentState, CurrencyAgent


class FixedCurrency:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _frame):
        return self.prediction


@pytest.mark.parametrize(
    "confidence",
    [
        True,
        False,
        "0.9",
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        1.01,
    ],
)
def test_currency_agent_rejects_non_numeric_or_invalid_confidence(confidence):
    agent = CurrencyAgent(
        {},
        adapter=FixedCurrency(
            {"denomination": "500", "confidence": confidence}
        ),
    )

    result = agent.run(object())

    assert result.state == AgentState.ERROR
    assert "Currency prediction failed" in result.summary
    assert "confidence" not in result.summary.lower() or "failed" in result.summary.lower()
    assert any("do not rely" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 0.9, 1.0])
def test_currency_agent_accepts_finite_real_confidence_in_range(confidence):
    agent = CurrencyAgent(
        {},
        adapter=FixedCurrency(
            {"denomination": "500", "confidence": confidence}
        ),
    )

    result = agent.run(object())

    assert result.state == AgentState.COMPLETED
    assert "500" in result.summary

@pytest.mark.parametrize(
    "detections",
    [
        None,
        {"label": "cup", "position": "left", "confidence": 0.8},
        [{"label": "cup", "position": "left", "confidence": True}],
        [{"label": "cup", "position": "left", "confidence": float("nan")}],
        [{"label": "cup", "position": "left", "confidence": 1.2}],
        [{"label": "", "position": "left", "confidence": 0.8}],
    ],
)
def test_object_agent_rejects_malformed_adapter_results(detections):
    from noorix.agents import ObjectDetectionAgent

    class FixedDetector:
        def detect(self, _frame):
            return detections

    result = ObjectDetectionAgent({}, adapter=FixedDetector()).run(object())

    assert result.state == AgentState.ERROR
    assert "Object detection failed" in result.summary
    assert any("do not assume" in warning.lower() for warning in result.warnings)


def test_ocr_agent_rejects_non_text_adapter_result():
    from noorix.agents import OCRAgent

    class InvalidOCR:
        def read(self, _frame):
            return {"recognized": "private text"}

    result = OCRAgent(InvalidOCR()).run(object())

    assert result.state == AgentState.ERROR
    assert result.data is None
    assert any("do not rely" in warning.lower() for warning in result.warnings)

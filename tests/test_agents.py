from pathlib import Path

from noorix.agents import AgentCoordinator, AgentResult, AgentState, CurrencyAgent, ObjectDetectionAgent, OCRAgent, SafetyAgent, SystemHealthAgent

class FakeOCR:
    def read(self, frame): return "HELLO NOORIX"
class FakeObjectDetector:
    def detect(self, frame):
        return [{"label": "cup", "position": "left", "confidence": 0.9, "box": [1, 2, 3, 4]}]
class FakeCurrency:
    def predict(self, frame): return {"denomination": "500", "confidence": 0.8}

def test_ocr_agent_uses_adapter():
    result = OCRAgent(FakeOCR()).run(object())
    assert result.state == AgentState.COMPLETED and result.data == "HELLO NOORIX"

def test_object_agent_requires_model_config():
    result = ObjectDetectionAgent({}).run(object())
    assert result.state == AgentState.NEEDS_SETUP and "not configured" in result.summary

def test_object_agent_returns_relative_detection():
    result = ObjectDetectionAgent({"model_path": "unused", "labels": ["cup"]},
                                  adapter=FakeObjectDetector()).run(object())
    assert result.successful and result.data[0]["position"] == "left" and result.warnings

def test_currency_agent_requires_model_config():
    assert CurrencyAgent({}).run(object()).state == AgentState.NEEDS_SETUP

def test_currency_agent_marks_prediction_as_estimate():
    result = CurrencyAgent({"model_path": "unused", "labels": ["500"]},
                           adapter=FakeCurrency()).run(object())
    assert result.successful and "500" in result.summary
    assert any("estimate" in warning.lower() for warning in result.warnings)

def test_safety_agent_includes_accessibility_warning():
    result = SafetyAgent().run([])
    assert result.successful and any("white cane" in w for w in result.warnings)

def test_coordinator_converts_exceptions_to_structured_errors():
    class Broken:
        name = "broken"
        def run(self, payload=None): raise RuntimeError("expected test error")
    result = AgentCoordinator([Broken()]).run("broken")
    assert result.state == AgentState.ERROR and "expected test error" in result.summary

def test_coordinator_rejects_duplicate_names():
    class Dummy:
        name = "dummy"
        def run(self, payload=None): return AgentResult("dummy", AgentState.COMPLETED, "ok")
    dummy = Dummy()
    coordinator = AgentCoordinator([dummy])
    try:
        coordinator.register(dummy)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate agent should fail")

def test_health_agent_reports_missing_config(tmp_path):
    result = SystemHealthAgent(tmp_path / "missing.json").run()
    assert result.successful
    assert result.data["model_configuration"] == "config.json_missing"


def test_relative_model_path_is_resolved_from_config_folder(tmp_path):
    from noorix.paths import resolve_config_path
    config_file = tmp_path / "settings" / "config.json"
    expected = (config_file.parent / "models" / "object.onnx").resolve()
    assert resolve_config_path("models/object.onnx", config_file) == expected


def test_absolute_model_path_is_preserved(tmp_path):
    from noorix.paths import resolve_config_path
    absolute = (tmp_path / "model.onnx").resolve()
    assert resolve_config_path(absolute, tmp_path / "config.json") == absolute


def test_object_agent_resolves_relative_path_from_config(tmp_path, monkeypatch):
    import noorix.agents.specialists as specialists
    captured = {}
    class DummyDetector:
        def __init__(self, model_path, labels, confidence_threshold):
            captured["path"] = Path(model_path)
        def detect(self, frame):
            return []
    monkeypatch.setattr(specialists, "ObjectDetector", DummyDetector)
    agent = ObjectDetectionAgent({"model_path": "models/objects.onnx", "labels": ["cup"]},
                                 config_path=tmp_path / "config.json")
    result = agent.run(object())
    assert result.successful
    assert captured["path"] == (tmp_path / "models/objects.onnx").resolve()

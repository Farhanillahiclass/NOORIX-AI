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
    assert result.state == AgentState.ERROR\n    assert "RuntimeError" in result.summary\n    assert "expected test error" not in result.summary

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


def test_object_agent_uses_shared_detector_factory(tmp_path, monkeypatch):
    import noorix.agents.specialists as specialists

    captured = {}

    class DummyDetector:
        def detect(self, frame):
            return []

    def fake_factory(config, config_path):
        captured["config"] = config
        captured["config_path"] = Path(config_path)
        return DummyDetector()

    monkeypatch.setattr(specialists, "create_object_detector", fake_factory)

    config = {"model_path": "models/objects.onnx", "labels": ["cup"]}
    config_path = tmp_path / "config.json"
    agent = ObjectDetectionAgent(config, config_path=config_path)

    result = agent.run(object())

    assert result.successful
    assert captured["config"] == config
    assert captured["config_path"] == config_path


def test_coordinator_does_not_expose_sensitive_exception_text():
    secret = "private OCR text and C:/Users/example/private-image.png"

    class Broken:
        name = "broken"
        def run(self, payload=None):
            raise RuntimeError(secret)

    result = AgentCoordinator([Broken()]).run("broken")
    assert result.state == AgentState.ERROR
    assert secret not in result.summary
    assert all(secret not in warning for warning in result.warnings)



def test_object_agent_redacts_model_setup_exception_details(monkeypatch, tmp_path):
    import noorix.agents.specialists as specialists

    secret = "C:/Users/private/model-folder/model.onnx"

    def broken_factory(_config, _config_path):
        raise OSError(secret)

    monkeypatch.setattr(specialists, "create_object_detector", broken_factory)
    agent = ObjectDetectionAgent(
        {"model_path": "models/model.onnx"},
        config_path=tmp_path / "config.json",
    )

    result = agent.run(object())

    assert result.state == AgentState.NEEDS_SETUP
    assert "Object detection setup failed (OSError)." == result.summary
    assert secret not in result.summary


def test_health_agent_redacts_config_read_exception_details(monkeypatch, tmp_path):
    import noorix.agents.specialists as specialists

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    secret = "C:/Users/private/settings.json"

    def broken_loader(_path):
        raise OSError(secret)

    monkeypatch.setattr(specialists, "load_json_document", broken_loader)
    result = SystemHealthAgent(config_path).run()

    detail = result.data["model_files"]["config"]
    assert detail == "Configuration unreadable (OSError)."
    assert secret not in detail



def test_health_agent_marks_symlinked_model_path_unsafe(tmp_path):
    real_model = tmp_path / "real.onnx"
    real_model.write_bytes(b"model")
    linked_model = tmp_path / "linked.onnx"
    try:
        linked_model.symlink_to(real_model)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("file symlinks are unavailable on this platform")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"object_detection": {"model_path": "linked.onnx"}}',
        encoding="utf-8",
    )

    result = SystemHealthAgent(config_path).run()

    assert result.data["model_files"]["object_detection"] == "unsafe_or_invalid_path"



def test_health_agent_marks_symlinked_config_unsafe(tmp_path):
    real_config = tmp_path / "real-config.json"
    real_config.write_text("{}", encoding="utf-8")
    linked_config = tmp_path / "linked-config.json"
    try:
        linked_config.symlink_to(real_config)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("file symlinks are unavailable on this platform")

    result = SystemHealthAgent(linked_config).run()

    assert result.data["model_configuration"] == "unsafe_or_invalid"
    assert result.data["model_files"]["config"] == "Configuration unreadable (ValueError)."



def test_currency_agent_rejects_non_integer_input_size():
    import pytest

    for invalid in (True, "224", 224.5, 0, -1):
        result = CurrencyAgent(
            {"model_path": "unused", "labels": ["500"], "input_size": invalid}
        ).run(object())
        assert result.state == AgentState.NEEDS_SETUP
        assert "positive integer" in result.summary


def test_currency_agent_redacts_model_setup_errors(monkeypatch, tmp_path):
    import noorix.agents.specialists as specialists

    secret = "C:/Users/private/currency-model.onnx"

    def broken_resolve(_path, _config_path):
        raise OSError(secret)

    monkeypatch.setattr(specialists, "resolve_config_path", broken_resolve)
    result = CurrencyAgent(
        {"model_path": "model.onnx", "labels": ["500"]},
        config_path=tmp_path / "config.json",
    ).run(object())

    assert result.state == AgentState.NEEDS_SETUP
    assert result.summary == "Currency model setup failed (OSError)."
    assert secret not in result.summary


def test_currency_agent_passes_validated_input_size_to_estimator(monkeypatch, tmp_path):
    import noorix.agents.specialists as specialists

    captured = {}

    class FakeEstimator:
        def __init__(self, model_path, labels, input_size):
            captured["model_path"] = model_path
            captured["labels"] = labels
            captured["input_size"] = input_size

        def predict(self, _frame):
            return {"denomination": "500", "confidence": 0.8}

    monkeypatch.setattr(specialists, "CurrencyEstimator", FakeEstimator)
    result = CurrencyAgent(
        {"model_path": "model.onnx", "labels": ["500"], "input_size": 320},
        config_path=tmp_path / "config.json",
    ).run(object())

    assert result.successful
    assert captured["input_size"] == 320
    assert captured["model_path"] == tmp_path / "model.onnx"



def test_specialist_runtime_errors_are_structured_and_redacted():
    secret = "private frame text C:/Users/private/frame.png"

    class BrokenOCR:
        def read(self, _frame):
            raise RuntimeError(secret)

    class BrokenDetector:
        def detect(self, _frame):
            raise RuntimeError(secret)

    class BrokenCurrency:
        def predict(self, _frame):
            raise RuntimeError(secret)

    class BrokenSpeech:
        def speak(self, _text):
            raise RuntimeError(secret)

    agents = [
        OCRAgent(BrokenOCR()),
        ObjectDetectionAgent({}, adapter=BrokenDetector()),
        CurrencyAgent({}, adapter=BrokenCurrency()),
    ]
    from noorix.agents import SpeechAgent
    agents.append(SpeechAgent(BrokenSpeech()))

    for agent in agents:
        result = agent.run("sample text" if agent.name == "speech" else object())
        assert result.state == AgentState.ERROR
        assert secret not in result.summary
        assert any("did not" in warning.lower() or "confirmed" in warning.lower()
                   for warning in result.warnings)


def test_currency_agent_rejects_invalid_adapter_prediction():
    class InvalidCurrency:
        def predict(self, _frame):
            return {"denomination": "500", "confidence": float("nan")}

    result = CurrencyAgent({}, adapter=InvalidCurrency()).run(object())

    assert result.state == AgentState.ERROR
    assert "Currency prediction failed" in result.summary
    assert "nan" not in result.summary.lower()

def test_ocr_agent_rejects_non_text_adapter_output():
    class InvalidOCR:
        def read(self, _frame):
            return {"recognized": "private text"}

    result = OCRAgent(InvalidOCR()).run(object())

    assert result.state == AgentState.ERROR
    assert result.summary == "OCR task failed (TypeError)."
    assert result.data is None
    assert any("invalid result" in warning.lower() for warning in result.warnings)


def test_object_agent_rejects_malformed_detection_results():
    import pytest

    class InvalidDetector:
        def __init__(self, detections):
            self.detections = detections

        def detect(self, _frame):
            return self.detections

    invalid_results = (
        {"label": "cup", "position": "left", "confidence": 0.9},
        [{"label": "", "position": "left", "confidence": 0.9}],
        [{"label": "cup", "position": "", "confidence": 0.9}],
        [{"label": "cup", "position": "left", "confidence": True}],
        [{"label": "cup", "position": "left", "confidence": float("nan")}],
        [{"label": "cup", "position": "left", "confidence": float("inf")}],
        [{"label": "cup", "position": "left", "confidence": 1.1}],
    )

    for detections in invalid_results:
        result = ObjectDetectionAgent(
            {}, adapter=InvalidDetector(detections)
        ).run(object())
        assert result.state == AgentState.ERROR
        assert "Object detection failed" in result.summary
        assert any("do not assume" in warning.lower() for warning in result.warnings)


def test_currency_agent_rejects_malformed_prediction_fields():
    import pytest

    class InvalidCurrency:
        def __init__(self, prediction):
            self.prediction = prediction

        def predict(self, _frame):
            return self.prediction

    invalid_predictions = (
        None,
        [],
        {"confidence": 0.8},
        {"denomination": "", "confidence": 0.8},
        {"denomination": "500", "confidence": True},
        {"denomination": "500", "confidence": "0.8"},
        {"denomination": "500", "confidence": float("nan")},
        {"denomination": "500", "confidence": float("inf")},
        {"denomination": "500", "confidence": -0.01},
        {"denomination": "500", "confidence": 1.01},
    )

    for prediction in invalid_predictions:
        result = CurrencyAgent(
            {}, adapter=InvalidCurrency(prediction)
        ).run(object())
        assert result.state == AgentState.ERROR
        assert "Currency prediction failed" in result.summary
        assert result.data is None
        assert any("do not rely" in warning.lower() for warning in result.warnings)


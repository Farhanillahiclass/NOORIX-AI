from pathlib import Path

from noorix.agents import AgentCoordinator, AgentResult, AgentState, CurrencyAgent, ObjectDetectionAgent, OCRAgent, SafetyAgent, SystemHealthAgent, WhisperASRAgent

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

def test_currency_agent_rejects_low_confidence_prediction():
    class LowConfidenceCurrency:
        def predict(self, frame):
            return {"denomination": "10", "side": "front", "confidence": 0.13}

    result = CurrencyAgent(
        {"model_path": "unused", "labels": ["10"], "min_confidence": 0.60},
        adapter=LowConfidenceCurrency(),
    ).run(object())

    assert result.successful
    assert "Low confidence (13%" in result.summary
    assert "No reliable denomination detected" in result.summary
    assert "10 PKR" not in result.summary
    assert result.data == {
        "recognized": False,
        "confidence": 0.13,
        "minimum_confidence": 0.60,
    }


def test_currency_agent_accepts_prediction_at_configured_threshold():
    result = CurrencyAgent(
        {"model_path": "unused", "labels": ["500"], "min_confidence": 0.60},
        adapter=FakeCurrency(),
    ).run(object())

    assert result.successful
    assert "500 PKR" in result.summary
    assert result.data["recognized"] is True


def test_currency_agent_rejects_invalid_confidence_threshold():
    result = CurrencyAgent(
        {"model_path": "unused", "labels": ["500"], "min_confidence": 1.5},
        adapter=FakeCurrency(),
    ).run(object())

    assert result.state == AgentState.NEEDS_SETUP
    assert "between 0.0 and 1.0" in result.summary

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


def test_whisper_asr_agent_transcribes_with_injected_adapter():
    class FakeASR:
        def transcribe(self, payload):
            assert payload == {"audio": [0.1], "sampling_rate": 16000}
            return "سلام"

    result = WhisperASRAgent({"model_path": "unused"}, adapter=FakeASR()).run(
        {"audio": [0.1], "sampling_rate": 16000}
    )
    assert result.successful
    assert result.data == {"transcript": "سلام"}
    assert result.warnings


def test_whisper_asr_agent_requires_model_configuration():
    result = WhisperASRAgent({}).run("sample.wav")
    assert result.state == AgentState.NEEDS_SETUP
    assert "model_path" in result.summary


def test_health_agent_recognizes_whisper_model_directory(tmp_path):
    model_dir = tmp_path / "models" / "whisper"
    processor_dir = tmp_path / "models" / "processor"
    model_dir.mkdir(parents=True)
    processor_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (processor_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (processor_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.json"
    import json
    config.write_text(json.dumps({"asr": {
        "model_path": "models/whisper",
        "processor_path": "models/processor"
    }}), encoding="utf-8")
    result = SystemHealthAgent(config).run()
    assert result.data["model_files"]["asr"] == "available"

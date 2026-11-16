from noorix.app import main


def test_check_mode_reports_readiness(capsys):
    assert main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "NOORIX AI" in output
    assert "pakistani_currency_estimation" in output


def test_negative_camera_index_is_rejected():
    try:
        main(["--camera-preview", "--camera", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject a negative camera index")


def test_modes_are_mutually_exclusive():
    try:
        main(["--gui", "--check"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject conflicting modes")


def test_transcribe_mode_uses_local_asr_agent(tmp_path, monkeypatch, capsys):
    import json
    import noorix.agents

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"placeholder")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"asr": {"model_path": "models/whisper", "language": "urdu"}}), encoding="utf-8")
    captured = {}

    class FakeResult:
        successful = True
        summary = "سلام دنیا"
        warnings = ("Verify important words.",)

    class FakeASR:
        def __init__(self, settings, config_path):
            captured["settings"] = settings
            captured["config_path"] = config_path
        def run(self, payload):
            captured["payload"] = payload
            return FakeResult()

    monkeypatch.setattr(noorix.agents, "WhisperASRAgent", FakeASR)
    assert main(["--transcribe", str(audio), "--config", str(config)]) == 0
    assert captured["settings"]["language"] == "urdu"
    assert captured["payload"] == audio
    output = capsys.readouterr().out
    assert "سلام دنیا" in output
    assert "Verify important words." in output


def test_test_all_mode_runs_sequential_suite(tmp_path, monkeypatch, capsys):
    import noorix.model_testing

    captured = {}
    def fake_run(config):
        captured["config"] = config
        return {
            "results": [
                {"status": "PASS", "name": "PKR", "duration_seconds": 0.1, "message": "ok"},
                {"status": "SKIPPED", "name": "TTS", "duration_seconds": 0.0, "message": "voice missing"},
            ],
            "totals": {"pass": 1, "fail": 0, "skipped": 1, "warn": 0},
            "report_json": "report.json",
            "report_markdown": "report.md",
        }

    monkeypatch.setattr(noorix.model_testing, "run_all_tests", fake_run)
    assert main(["--test-all", "--config", str(tmp_path / "config.json")]) == 0
    assert captured["config"] == tmp_path / "config.json"
    output = capsys.readouterr().out
    assert "[PASS] PKR" in output
    assert "[SKIPPED] TTS" in output
    assert "PRODUCTION READINESS: NOT CERTIFIED" in output

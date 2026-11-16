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

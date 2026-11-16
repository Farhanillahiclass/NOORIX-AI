import json

from noorix.model_testing import TestResult, _run_one, run_all_tests


def test_run_one_converts_exceptions_to_failures():
    result = _run_one("broken", lambda: (_ for _ in ()).throw(RuntimeError("bad model")))
    assert result.status == "FAIL"
    assert "bad model" in result.message


def test_run_all_tests_runs_checks_sequentially_and_writes_reports(tmp_path, monkeypatch):
    import noorix.model_testing as testing

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    order = []

    def fake(name):
        def run(*args):
            order.append(name)
            return "PASS", {"ok": True}
        return run

    monkeypatch.setattr(testing, "_test_package", fake("package"))
    monkeypatch.setattr(testing, "_test_currency", fake("currency"))
    monkeypatch.setattr(testing, "_test_objects", fake("objects"))
    monkeypatch.setattr(testing, "_test_ocr", fake("ocr"))
    monkeypatch.setattr(testing, "_test_asr", fake("asr"))
    monkeypatch.setattr(testing, "_test_tts", fake("tts"))
    progress = []

    report = run_all_tests(config, tmp_path / "reports", progress.append)

    assert order == ["package", "currency", "objects", "ocr", "asr", "tts"]
    assert len(progress) == 6
    assert report["totals"]["pass"] == 6
    assert report["totals"]["fail"] == 0
    assert report["production_readiness"] == "NOT_CERTIFIED"
    assert (tmp_path / "reports" / Path(report["report_json"]).name).is_file()
    assert Path(report["report_markdown"]).is_file()
    parsed = json.loads(Path(report["report_json"]).read_text(encoding="utf-8"))
    assert len(parsed["results"]) == 6


def test_run_all_tests_reports_invalid_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("[]", encoding="utf-8")
    report = run_all_tests(config, tmp_path / "reports")
    assert report["totals"]["fail"] == 1
    assert report["results"][0]["name"] == "Configuration"


def test_test_package_verifies_checksums(tmp_path):
    import hashlib
    from noorix.model_testing import _test_package

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    payload = tmp_path / "models" / "weights.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"noorix-test")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{digest}  models/weights.bin\n", encoding="utf-8")
    status, details = _test_package({}, config)
    assert status == "PASS"
    assert details["checksum_files_verified"] == 1


def test_test_package_detects_checksum_mismatch(tmp_path):
    import pytest
    from noorix.model_testing import _test_package

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    (tmp_path / "models.bin").write_bytes(b"changed")
    (tmp_path / "checksums.sha256").write_text("0" * 64 + "  models.bin\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        _test_package({}, config)

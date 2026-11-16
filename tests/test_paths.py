from noorix.paths import resolve_config_path


def test_config_relative_path_has_precedence(tmp_path, monkeypatch):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    local = config_dir / "models"
    local.mkdir()
    external = tmp_path / "external"
    (external / "models").mkdir(parents=True)
    monkeypatch.setenv("NOORIX_MODELS_DIR", str(external))
    assert resolve_config_path("models", config_dir / "config.json") == local


def test_model_root_fallback_resolves_extracted_package(tmp_path, monkeypatch):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    model_root = tmp_path / "extracted"
    target = model_root / "models" / "pkr_currency" / "model.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model")
    monkeypatch.setenv("NOORIX_MODELS_DIR", str(model_root))
    assert resolve_config_path("models/pkr_currency/model.pth", config_dir / "config.json") == target


def test_model_root_does_not_change_missing_path_result(tmp_path, monkeypatch):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    monkeypatch.delenv("NOORIX_MODELS_DIR", raising=False)
    assert resolve_config_path("models/missing.onnx", config_dir / "config.json") == config_dir / "models/missing.onnx"

from pathlib import Path

import pytest

from noorix.paths import load_json_document, resolve_config_path


def test_load_json_document_reads_nested_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"camera": {"index": 0}, "enabled": true}', encoding="utf-8")

    assert load_json_document(path) == {"camera": {"index": 0}, "enabled": True}


def test_load_json_document_rejects_duplicate_keys_at_any_depth(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"camera": {"index": 0, "index": 1}}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object key"):
        load_json_document(path)


def test_load_json_document_rejects_oversized_input(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_bytes(b" " * 33)

    with pytest.raises(ValueError, match="32-byte"):
        load_json_document(path, max_bytes=32)


def test_load_json_document_rejects_non_regular_input(tmp_path: Path):
    with pytest.raises(ValueError, match="regular file"):
        load_json_document(tmp_path)


def test_load_json_document_rejects_symlinked_file(tmp_path: Path):
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_json_document(linked)


def test_load_json_document_rejects_symlinked_parent(tmp_path: Path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "config.json"
    target.write_text("{}", encoding="utf-8")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_json_document(linked_dir / "config.json")


def test_load_json_document_rejects_invalid_size_limit(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        load_json_document(path, max_bytes=0)
    with pytest.raises(ValueError, match="positive integer"):
        load_json_document(path, max_bytes=True)


def test_resolve_config_path_uses_config_parent(tmp_path: Path):
    config = tmp_path / "settings" / "config.json"
    assert resolve_config_path("models/object.onnx", config) == (
        config.parent.resolve() / "models" / "object.onnx"
    )


def test_resolve_config_path_rejects_symlinked_config_parent(tmp_path: Path):
    real_dir = tmp_path / "real-settings"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked-settings"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="Configuration path contains"):
        resolve_config_path("models/object.onnx", linked_dir / "config.json")


def test_resolve_config_path_rejects_windows_junction_parent(tmp_path: Path, monkeypatch):
    junction_dir = tmp_path / "junction-settings"
    junction_dir.mkdir()
    path_type = type(tmp_path)
    monkeypatch.setattr(
        path_type,
        "is_junction",
        lambda path: path == junction_dir,
        raising=False,
    )

    with pytest.raises(ValueError, match="Configuration path contains"):
        resolve_config_path("models/object.onnx", junction_dir / "config.json")


def test_load_json_document_rejects_windows_junction_component(tmp_path: Path, monkeypatch):
    junction_dir = tmp_path / "junction"
    junction_dir.mkdir()
    config = junction_dir / "config.json"
    config.write_text("{}", encoding="utf-8")
    path_type = type(tmp_path)
    monkeypatch.setattr(
        path_type,
        "is_junction",
        lambda path: path == junction_dir,
        raising=False,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        load_json_document(config)


def test_require_regular_file_accepts_regular_file(tmp_path: Path):
    from noorix.paths import require_regular_file

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    assert require_regular_file(model, label="Model") == model


def test_require_regular_file_preserves_missing_file_error(tmp_path: Path):
    from noorix.paths import require_regular_file

    with pytest.raises(FileNotFoundError, match="Model not found"):
        require_regular_file(tmp_path / "missing.onnx", label="Model")


def test_require_regular_file_rejects_directory(tmp_path: Path):
    from noorix.paths import require_regular_file

    with pytest.raises(ValueError, match="regular file"):
        require_regular_file(tmp_path, label="Model")


def test_require_regular_file_rejects_symlink_file(tmp_path: Path):
    from noorix.paths import require_regular_file

    target = tmp_path / "real.onnx"
    target.write_bytes(b"model")
    linked = tmp_path / "linked.onnx"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link or junction"):
        require_regular_file(linked, label="Model")


def test_require_regular_file_rejects_symlink_parent(tmp_path: Path):
    from noorix.paths import require_regular_file

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "model.onnx"
    target.write_bytes(b"model")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link or junction"):
        require_regular_file(linked_dir / "model.onnx", label="Model")


def test_require_regular_file_rejects_windows_reparse_point_on_python_311(tmp_path: Path, monkeypatch):
    import stat
    from types import SimpleNamespace
    from noorix.paths import require_regular_file

    candidate = tmp_path / "model.onnx"
    original_lstat = Path.lstat

    def fake_lstat(path):
        if path == candidate:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="symbolic link or junction"):
        require_regular_file(candidate, label="Model")


def test_load_json_document_rejects_excessive_nesting(tmp_path: Path):
    path = tmp_path / "deep.json"
    path.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")

    with pytest.raises(ValueError, match="nesting exceeds the parser limit"):
        load_json_document(path)


def test_resolve_config_path_rejects_symlinked_relative_target(tmp_path: Path):
    real_dir = tmp_path / "real-models"
    real_dir.mkdir()
    (real_dir / "model.onnx").write_bytes(b"model")
    linked_dir = tmp_path / "models"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="Configured path contains"):
        resolve_config_path("models/model.onnx", tmp_path / "config.json")


def test_resolve_config_path_rejects_absolute_symlink_target(tmp_path: Path):
    real_model = tmp_path / "real.onnx"
    real_model.write_bytes(b"model")
    linked_model = tmp_path / "linked.onnx"
    try:
        linked_model.symlink_to(real_model)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="Configured path contains"):
        resolve_config_path(linked_model, tmp_path / "config.json")

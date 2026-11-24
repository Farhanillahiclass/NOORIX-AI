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

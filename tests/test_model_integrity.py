import hashlib
import json

from scripts.verify_model_integrity import verify_manifest


def _write_manifest(tmp_path, entries):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"algorithm": "sha256", "files": entries}), encoding="utf-8")
    return manifest


def test_verifies_matching_file(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model-bytes")
    manifest = _write_manifest(tmp_path, [{
        "path": "model.onnx",
        "sha256": hashlib.sha256(b"model-bytes").hexdigest(),
    }])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "PASS"


def test_detects_modified_model(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"changed")
    manifest = _write_manifest(tmp_path, [{"path": "model.onnx", "sha256": "0" * 64}])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "MISMATCH"


def test_rejects_parent_traversal(tmp_path):
    manifest = _write_manifest(tmp_path, [{"path": "../outside.onnx", "sha256": "0" * 64}])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ERROR"


def test_rejects_duplicate_case_insensitive_paths(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    manifest = _write_manifest(tmp_path, [
        {"path": "model.onnx", "sha256": digest},
        {"path": "MODEL.ONNX", "sha256": digest},
    ])
    results = verify_manifest(tmp_path, manifest)
    assert [item["status"] for item in results] == ["PASS", "ERROR"]


def test_reports_missing_files(tmp_path):
    manifest = _write_manifest(tmp_path, [{"path": "missing.onnx", "sha256": "0" * 64}])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ERROR"


def test_rejects_invalid_digest(tmp_path):
    manifest = _write_manifest(tmp_path, [{"path": "model.onnx", "sha256": "not-a-digest"}])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ERROR"


def test_accepts_importer_nested_integrity_object(tmp_path):
    model = tmp_path / "subdir" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")
    manifest = tmp_path / "import_manifest.json"
    manifest.write_text(json.dumps({
        "project": "NOORIX-AI",
        "integrity": {
            "algorithm": "sha256",
            "files": [{
                "path": "subdir/model.onnx",
                "sha256": hashlib.sha256(b"model").hexdigest(),
            }],
        },
    }), encoding="utf-8")
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "PASS"


def test_rejects_symlinked_model_when_supported(tmp_path):
    outside = tmp_path.parent / "outside-model.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "linked-model.bin"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return
    manifest = _write_manifest(tmp_path, [{
        "path": "linked-model.bin",
        "sha256": hashlib.sha256(b"outside").hexdigest(),
    }])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ERROR"


def test_rejects_symlinked_parent_directory_even_when_target_stays_inside_root(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    model = real_dir / "model.onnx"
    model.write_bytes(b"model")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    manifest = _write_manifest(tmp_path, [{
        "path": "alias/model.onnx",
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }])
    assert verify_manifest(tmp_path, manifest)[0]["status"] == "ERROR"


def test_rejects_windows_reserved_and_nonportable_path_components(tmp_path):
    unsafe_paths = [
        "CON",
        "nul.txt",
        "folder/COM1.onnx",
        "folder/LPT9.bin",
        "model.onnx ",
        "model.onnx.",
        f"bad{chr(0)}name.onnx",
        "folder//model.onnx",
        "./model.onnx",
    ]
    manifest = _write_manifest(tmp_path, [
        {"path": path, "sha256": "0" * 64} for path in unsafe_paths
    ])
    results = verify_manifest(tmp_path, manifest)
    assert len(results) == len(unsafe_paths)
    assert all(item["status"] == "ERROR" for item in results)


def test_rejects_manifest_outside_trusted_model_root(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    outside_manifest = tmp_path / "external-manifest.json"
    outside_manifest.write_text(json.dumps({
        "algorithm": "sha256",
        "files": [{"path": "missing.onnx", "sha256": "0" * 64}],
    }), encoding="utf-8")

    result = verify_manifest(model_root, outside_manifest)
    assert result[0]["status"] == "ERROR"
    assert "inside the trusted model root" in result[0]["detail"]


def test_rejects_symlinked_manifest_inside_trusted_root_when_supported(tmp_path):
    outside_manifest = tmp_path / "external-manifest.json"
    outside_manifest.write_text(json.dumps({
        "algorithm": "sha256",
        "files": [{"path": "missing.onnx", "sha256": "0" * 64}],
    }), encoding="utf-8")
    link = tmp_path / "import_manifest.json"
    try:
        link.symlink_to(outside_manifest)
    except (OSError, NotImplementedError):
        return

    result = verify_manifest(tmp_path, link)
    assert result[0]["status"] == "ERROR"
    assert "non-symlink file" in result[0]["detail"]

def test_normalizes_single_windows_separator_in_manifest_path(tmp_path):
    model = tmp_path / "folder" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")
    manifest = _write_manifest(tmp_path, [{
        "path": r"folder\model.onnx",
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }])
    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "PASS"


def test_rejects_actual_nul_character_in_manifest_path(tmp_path):
    manifest = _write_manifest(tmp_path, [{
        "path": f"bad{chr(0)}name.onnx",
        "sha256": "0" * 64,
    }])
    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert result[0]["detail"] == "unsafe or duplicate manifest path"




def test_rejects_symlinked_trusted_model_root_when_supported(tmp_path):
    real_root = tmp_path / "real-models"
    real_root.mkdir()
    manifest = _write_manifest(real_root, [{
        "path": "missing.onnx",
        "sha256": "0" * 64,
    }])
    linked_root = tmp_path / "models-link"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    result = verify_manifest(linked_root, linked_root / manifest.name)
    assert result[0]["status"] == "ERROR"
    assert "trusted model root" in result[0]["detail"]



def test_rejects_manifest_larger_than_size_limit(tmp_path):
    from scripts.verify_model_integrity import MAX_MANIFEST_BYTES

    manifest = tmp_path / "manifest.json"
    with manifest.open("wb") as stream:
        stream.truncate(MAX_MANIFEST_BYTES + 1)

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "size limit" in result[0]["detail"]


def test_rejects_manifest_with_too_many_entries(tmp_path):
    from scripts.verify_model_integrity import MAX_MANIFEST_ENTRIES

    manifest = _write_manifest(tmp_path, [
        {"path": f"model-{index}.onnx", "sha256": "0" * 64}
        for index in range(MAX_MANIFEST_ENTRIES + 1)
    ])

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "entry limit" in result[0]["detail"]


def test_rejects_excessively_long_manifest_path(tmp_path):
    from scripts.verify_model_integrity import MAX_MANIFEST_PATH_LENGTH

    manifest = _write_manifest(tmp_path, [{
        "path": "a" * (MAX_MANIFEST_PATH_LENGTH + 1),
        "sha256": "0" * 64,
    }])

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "bounded relative path" in result[0]["detail"]


def test_rejects_duplicate_json_object_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"algorithm":"sha256","algorithm":"sha1","files":['
        '{"path":"model.onnx","sha256":"'+("0" * 64)+'"}]}',
        encoding="utf-8",
    )

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert result[0]["detail"] == "manifest unreadable"


def test_rejects_duplicate_keys_inside_file_entry(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"algorithm":"sha256","files":['
        '{"path":"missing.onnx","path":"other.onnx","sha256":"'+("0" * 64)+'"}]}',
        encoding="utf-8",
    )

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert result[0]["detail"] == "manifest unreadable"


def test_bounds_manifest_read_if_file_grows_after_size_check(tmp_path, monkeypatch):
    import io
    from scripts.verify_model_integrity import MAX_MANIFEST_BYTES

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    original_open = type(manifest).open

    def oversized_open(path, mode="r", *args, **kwargs):
        if path == manifest and mode == "rb":
            return io.BytesIO(b" " * (MAX_MANIFEST_BYTES + 1))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(type(manifest), "open", oversized_open)
    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "size limit" in result[0]["detail"]



def test_rejects_model_file_changed_during_hashing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.verify_model_integrity as verifier

    model = tmp_path / "model.onnx"
    original_bytes = b"original-model-bytes"
    model.write_bytes(original_bytes)
    manifest = _write_manifest(tmp_path, [{
        "path": "model.onnx",
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }])

    original_fstat = verifier.os.fstat
    calls = 0

    def changing_fstat(fd):
        nonlocal calls
        calls += 1
        metadata = original_fstat(fd)
        if calls == 2:
            return SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_size=metadata.st_size,
                st_mtime_ns=metadata.st_mtime_ns + 1,
                st_ctime_ns=metadata.st_ctime_ns,
                st_mode=metadata.st_mode,
            )
        return metadata

    monkeypatch.setattr(verifier.os, "fstat", changing_fstat)
    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "changed during verification" in result[0]["detail"]

def test_rejects_manifest_with_mixed_nested_and_flat_checksum_fields(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "algorithm": "sha256",
        "files": [{"path": "flat.onnx", "sha256": "0" * 64}],
        "integrity": {
            "algorithm": "sha256",
            "files": [{"path": "nested.onnx", "sha256": "0" * 64}],
        },
    }), encoding="utf-8")

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert "either nested integrity or flat checksum fields" in result[0]["detail"]

def test_stable_hash_rejects_path_replaced_by_symlink_after_read(tmp_path, monkeypatch):
    from pathlib import Path
    import stat
    from types import SimpleNamespace
    from scripts.verify_model_integrity import _hash_file_stably

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    original_lstat = Path.lstat
    calls = 0

    def replaced_path_lstat(path):
        nonlocal calls
        metadata = original_lstat(path)
        if path == model:
            calls += 1
            if calls == 2:
                return SimpleNamespace(
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino,
                    st_size=metadata.st_size,
                    st_mtime_ns=metadata.st_mtime_ns,
                    st_ctime_ns=metadata.st_ctime_ns,
                    st_mode=stat.S_IFLNK,
                )
        return metadata

    monkeypatch.setattr(Path, "lstat", replaced_path_lstat)
    assert _hash_file_stably(model) is None

def test_rejects_excessively_nested_json_manifest_without_crashing(tmp_path):
    manifest = tmp_path / "manifest.json"
    nested = "[" * 2_000 + "0" + "]" * 2_000
    manifest.write_text(
        '{"metadata":' + nested
        + ',"algorithm":"sha256","files":['
        + '{"path":"missing.onnx","sha256":"' + ("0" * 64) + '"}]}',
        encoding="utf-8",
    )

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert result[0]["detail"] == "manifest unreadable"


def test_manifest_errors_do_not_disclose_local_parent_paths_or_control_chars(tmp_path):
    from scripts.verify_model_integrity import verify_manifest

    manifest = tmp_path / "private-area" / "bad\\x1b[31mname.json"
    manifest.parent.mkdir()
    manifest.write_text("not-json", encoding="utf-8")

    result = verify_manifest(tmp_path, manifest)
    assert result[0]["status"] == "ERROR"
    assert str(tmp_path) not in result[0]["path"]
    assert "\\x1b" not in result[0]["path"]
    assert result[0]["path"] == "bad?[31mname.json"


def test_manifest_entry_rejects_and_sanitizes_terminal_control_path(tmp_path):
    unsafe_path = "weights/" + chr(27) + "[31m.onnx"
    manifest = _write_manifest(tmp_path, [{"path": unsafe_path, "sha256": "0" * 64}])

    result = verify_manifest(tmp_path, manifest)[0]

    assert result["status"] == "ERROR"
    assert chr(27) not in result["path"]
    assert "?" in result["path"]



def test_trusted_model_paths_reject_windows_junction_components(tmp_path, monkeypatch):
    from scripts.verify_model_integrity import _has_symlink_component

    junction = tmp_path / "junction"
    junction.mkdir()
    path_type = type(tmp_path)
    monkeypatch.setattr(
        path_type,
        "is_junction",
        lambda path: path == junction,
        raising=False,
    )

    assert _has_symlink_component(junction / "models" / "model.onnx") is True

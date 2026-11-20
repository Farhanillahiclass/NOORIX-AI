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
        "bad\\x00name.onnx",
        "folder//model.onnx",
        "./model.onnx",
    ]
    manifest = _write_manifest(tmp_path, [
        {"path": path, "sha256": "0" * 64} for path in unsafe_paths
    ])
    results = verify_manifest(tmp_path, manifest)
    assert len(results) == len(unsafe_paths)
    assert all(item["status"] == "ERROR" for item in results)

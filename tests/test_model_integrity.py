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

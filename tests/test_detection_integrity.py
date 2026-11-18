import json
import subprocess
import sys
from pathlib import Path

import pytest

from noorix.datasets import analyze_coco_integrity


def _payload(*names):
    return {
        "images": [
            {"id": index, "file_name": name, "width": 10, "height": 10}
            for index, name in enumerate(names, start=1)
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }


def test_integrity_hashes_files_and_detects_cross_split_duplicates(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    train_root.mkdir()
    val_root.mkdir()
    (train_root / "a.jpg").write_bytes(b"same-image-bytes")
    (val_root / "copy.jpg").write_bytes(b"same-image-bytes")
    (val_root / "other.jpg").write_bytes(b"different")
    report = analyze_coco_integrity({
        "train": {"payload": _payload("a.jpg"), "images_dir": train_root},
        "validation": {"payload": _payload("copy.jpg", "other.jpg"), "images_dir": val_root},
    })
    assert report["summary"]["files_hashed"] == 3
    assert report["summary"]["duplicate_groups"] == 1
    assert report["duplicate_groups"][0]["file_count"] == 2
    assert {item["split"] for item in report["duplicate_groups"][0]["locations"]} == {
        "train", "validation"
    }


def test_integrity_reports_missing_and_unsafe_paths(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    report = analyze_coco_integrity({
        "train": {
            "payload": _payload("missing.jpg", "../outside.jpg", "/absolute.jpg", "bad\\\\path.jpg"),
            "images_dir": root,
        }
    })
    assert report["summary"]["missing_files"] == 1
    assert report["summary"]["unsafe_paths"] == 3
    assert report["summary"]["files_hashed"] == 0


def test_integrity_rejects_symlink_escape(tmp_path):
    root = tmp_path / "images"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"not-in-root")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not permitted in this environment")
    report = analyze_coco_integrity({
        "train": {"payload": _payload("linked/secret.jpg"), "images_dir": root}
    })
    assert report["summary"]["unsafe_paths"] == 1
    assert report["summary"]["files_hashed"] == 0


def test_integrity_empty_files_are_valid_hash_inputs(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "empty.jpg").write_bytes(b"")
    report = analyze_coco_integrity({
        "test": {"payload": _payload("empty.jpg"), "images_dir": root}
    })
    assert report["summary"]["files_hashed"] == 1
    assert report["summary"]["missing_files"] == 0


def test_integrity_requires_non_empty_split_mapping():
    with pytest.raises(ValueError, match="non-empty mapping"):
        analyze_coco_integrity({})


def test_integrity_cli_writes_report_and_finds_duplicates(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"duplicate")
    (images / "b.jpg").write_bytes(b"duplicate")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(_payload("a.jpg", "b.jpg")), encoding="utf-8")
    output = tmp_path / "reports" / "integrity.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_detection_integrity.py"
    result = subprocess.run(
        [sys.executable, str(script), "--train-annotations", str(annotations),
         "--train-images-dir", str(images), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["duplicate_groups"] == 1


def test_integrity_cli_does_not_generate_report_for_invalid_annotations(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "invalid.json"
    annotations.write_text("{}", encoding="utf-8")
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_detection_integrity.py"
    result = subprocess.run(
        [sys.executable, str(script), "--train-annotations", str(annotations),
         "--train-images-dir", str(images), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "not generated" in result.stdout
    assert not output.exists()

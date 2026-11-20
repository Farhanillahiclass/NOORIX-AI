from __future__ import annotations

import io
import stat
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_kaggle_models import archive_slug, safe_extract_member


def make_archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            archive.writestr(info, payload)
    stream.seek(0)
    return zipfile.ZipFile(stream)


def test_safe_extract_allows_expected_model_file(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    with make_archive([("weights/model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)
    assert result == target / "weights" / "model.onnx"
    assert result.read_bytes() == b"model"


def test_safe_extract_rejects_parent_traversal_and_windows_paths(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    names = ["../escape.onnx", r"..\..\escape.onnx", r"C:\temp\escape.onnx", "/tmp/escape.onnx"]
    for name in names:
        with make_archive([(name, b"bad", None)]) as archive:
            assert safe_extract_member(archive, archive.infolist()[0], target) is None
    assert not (tmp_path / "escape.onnx").exists()


def test_safe_extract_rejects_symlinks(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    symlink_mode = stat.S_IFLNK | 0o777
    with make_archive([("link.onnx", b"target", symlink_mode)]) as archive:
        assert safe_extract_member(archive, archive.infolist()[0], target) is None
    assert not (target / "link.onnx").exists()


def test_safe_extract_does_not_overwrite_duplicate_archive_entries(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    with make_archive([("weights/model.onnx", b"first", None), ("weights/model.onnx", b"second", None)]) as archive:
        entries = archive.infolist()
        first = safe_extract_member(archive, entries[0], target)
        second = safe_extract_member(archive, entries[1], target)
    assert first is not None
    assert second is None
    assert first.read_bytes() == b"first"

def test_safe_extract_rejects_windows_reserved_names_and_trailing_dots(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    for name in (
        "CON.onnx", "nul.txt", "weights/model.onnx.",
        "COM1.bin", "folder/LPT9.onnx", "COM¹.txt",
        "weights//model.onnx", "./model.onnx", "folder/../model.onnx",
        "trailing /model.onnx", "model.onnx ",
        "bad\x00name.onnx",
    ):
        with make_archive([(name, b"bad", None)]) as archive:
            assert safe_extract_member(archive, archive.infolist()[0], target) is None


def test_safe_extract_rejects_case_insensitive_duplicate_names(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    with make_archive([("Weights/Model.onnx", b"first", None)]) as archive:
        first = safe_extract_member(archive, archive.infolist()[0], target)
    with make_archive([("weights/model.ONNX", b"second", None)]) as archive:
        second = safe_extract_member(archive, archive.infolist()[0], target)
    assert first is not None
    assert second is None
    assert first.read_bytes() == b"first"


def test_safe_extract_normalizes_single_windows_backslashes(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    with make_archive([(r"folder\model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)
    assert result == target / "folder" / "model.onnx"
    assert result.read_bytes() == b"model"


def test_safe_extract_rejects_encrypted_entries(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    with make_archive([("weights/model.onnx", b"model", None)]) as archive:
        info = archive.infolist()[0]
        info.flag_bits |= 0x1
        assert safe_extract_member(archive, info, target) is None



def test_archive_slug_cannot_escape_staging_directory():
    for filename in ("...zip", "..\\\\escape.zip", "folder/name.zip", "   .zip"):
        slug = archive_slug(filename)
        assert slug not in {".", "..", ""}
        assert "/" not in slug
        assert "\\\\" not in slug
        assert len(slug) <= 91

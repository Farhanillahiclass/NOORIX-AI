from __future__ import annotations

import io
import stat
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_kaggle_models import (
    archive_slug,
    copy_dataset_archive,
    has_symlink_component,
    prepare_staging_directory,
    publish_staging_directory,
    safe_display_filename,
    safe_extract_member,
    ensure_directory_without_symlinks,
    main,
)


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
        "COM1.bin", "folder/LPT9.onnx", "COM¹.txt", "folder/LPT².bin",
        "weights//model.onnx", "./model.onnx", "folder/../model.onnx",
        "trailing /model.onnx", "model.onnx ",
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


def test_safe_extract_rejects_nul_in_filename_before_open(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()

    class FakeEntry:
        filename = "bad\x00name.onnx"
        external_attr = 0
        file_size = 3
        compress_size = 3
        flag_bits = 0

        @staticmethod
        def is_dir():
            return False

    class FakeArchive:
        @staticmethod
        def open(_entry):
            raise AssertionError("unsafe entry must be rejected before opening")

    assert safe_extract_member(FakeArchive(), FakeEntry(), target) is None


def test_safe_extract_rejects_casefold_directory_collision(tmp_path):
    target = tmp_path / "stage"
    (target / "Weights").mkdir(parents=True)
    with make_archive([("weights/new-model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)
    assert result is None
    assert not (target / "weights").exists()


def test_safe_extract_allows_exact_case_shared_parent_directory(tmp_path):
    target = tmp_path / "stage"
    (target / "weights").mkdir(parents=True)
    (target / "weights" / "existing.onnx").write_bytes(b"existing")
    with make_archive([("weights/new-model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)
    assert result == target / "weights" / "new-model.onnx"
    assert result.read_bytes() == b"model"


def test_staging_setup_preserves_existing_directory(tmp_path):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    marker = staging / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    assert not prepare_staging_directory(staging, destination)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_staging_setup_refuses_existing_destination(tmp_path):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    assert not prepare_staging_directory(staging, destination)
    assert not staging.exists()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_staging_setup_creates_only_new_directory(tmp_path):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"

    assert prepare_staging_directory(staging, destination)
    assert staging.is_dir()


def test_safe_extract_reuses_path_index_for_case_collision_checks(tmp_path):
    target = tmp_path / "stage"
    target.mkdir()
    path_index = {}
    with make_archive([("Weights/first.onnx", b"first", None)]) as archive:
        first = safe_extract_member(archive, archive.infolist()[0], target, path_index)
    with make_archive([("weights/second.onnx", b"second", None)]) as archive:
        second = safe_extract_member(archive, archive.infolist()[0], target, path_index)
    assert first is not None
    assert second is None
    assert first.read_bytes() == b"first"


def test_safe_extract_rejects_ambiguous_existing_casefold_directories(tmp_path):
    target = tmp_path / "stage"
    (target / "Weights").mkdir(parents=True)
    (target / "weights").mkdir()
    with make_archive([("weights/new-model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)
    assert result is None

def test_safe_extract_rejects_symlinked_target_root(tmp_path):
    import pytest

    real_target = tmp_path / "real-stage"
    real_target.mkdir()
    linked_target = tmp_path / "linked-stage"
    try:
        linked_target.symlink_to(real_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with make_archive([("weights/model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], linked_target)

    assert result is None
    assert not (real_target / "weights").exists()


def test_safe_extract_rejects_symlinked_parent_of_target(tmp_path):
    import pytest

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    target = linked_parent / "stage"
    target.mkdir()
    with make_archive([("weights/model.onnx", b"model", None)]) as archive:
        result = safe_extract_member(archive, archive.infolist()[0], target)

    assert result is None
    assert not (real_parent / "stage" / "weights").exists()

def test_staging_setup_refuses_symlinked_parent(tmp_path):
    import pytest

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    staging = linked_parent / "staging"
    destination = linked_parent / "destination"
    assert not prepare_staging_directory(staging, destination)
    assert not (real_parent / "staging").exists()

def test_publish_staging_refuses_existing_destination_without_nesting(tmp_path):
    import pytest

    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "import_manifest.json").write_text("{}", encoding="utf-8")
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_staging_directory(staging, destination)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert (staging / "import_manifest.json").is_file()
    assert not (destination / "staging").exists()


def test_publish_staging_renames_completed_tree(tmp_path):
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "import_manifest.json").write_text("{}", encoding="utf-8")

    publish_staging_directory(staging, destination)

    assert not staging.exists()
    assert (destination / "import_manifest.json").is_file()




def test_dataset_zip_copy_preserves_bytes_and_source(tmp_path):
    downloads = tmp_path / "downloads"
    inbox = tmp_path / "inbox"
    downloads.mkdir()
    inbox.mkdir()
    source = downloads / "research-data.zip"
    source.write_bytes(b"zip-content")

    assert copy_dataset_archive(source, inbox) is True
    assert (inbox / source.name).read_bytes() == b"zip-content"
    assert source.read_bytes() == b"zip-content"


def test_dataset_zip_copy_never_overwrites_existing_file(tmp_path):
    downloads = tmp_path / "downloads"
    inbox = tmp_path / "inbox"
    downloads.mkdir()
    inbox.mkdir()
    source = downloads / "research-data.zip"
    source.write_bytes(b"new-content")
    existing = inbox / source.name
    existing.write_bytes(b"keep-existing")

    assert copy_dataset_archive(source, inbox) is False
    assert existing.read_bytes() == b"keep-existing"


def test_dataset_zip_copy_rejects_symlinked_source(tmp_path):
    import pytest

    downloads = tmp_path / "downloads"
    inbox = tmp_path / "inbox"
    downloads.mkdir()
    inbox.mkdir()
    real_source = downloads / "real.zip"
    real_source.write_bytes(b"zip-content")
    linked_source = downloads / "linked.zip"
    try:
        linked_source.symlink_to(real_source)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="regular, non-symlink"):
        copy_dataset_archive(linked_source, inbox)
    assert not (inbox / linked_source.name).exists()


def test_model_archive_path_check_detects_symlinked_file_and_parent(tmp_path):
    import pytest

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    archive = real_dir / "model.zip"
    archive.write_bytes(b"zip-content")
    assert has_symlink_component(archive) is False

    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    assert has_symlink_component(linked_dir / "model.zip") is True


def test_dataset_zip_copy_rejects_symlinked_source_parent(tmp_path):
    import pytest

    real_downloads = tmp_path / "real-downloads"
    real_downloads.mkdir()
    linked_downloads = tmp_path / "linked-downloads"
    try:
        linked_downloads.symlink_to(real_downloads, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    source = linked_downloads / "data.zip"
    (real_downloads / "data.zip").write_bytes(b"zip-content")
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    with pytest.raises(ValueError, match="source path has a symlinked component"):
        copy_dataset_archive(source, inbox)
    assert not (inbox / "data.zip").exists()


def test_dataset_zip_copy_rejects_symlinked_inbox(tmp_path):
    import pytest

    real_inbox = tmp_path / "real-inbox"
    real_inbox.mkdir()
    linked_inbox = tmp_path / "linked-inbox"
    try:
        linked_inbox.symlink_to(real_inbox, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    source = tmp_path / "data.zip"
    source.write_bytes(b"zip-content")

    with pytest.raises(ValueError, match="existing, non-symlink"):
        copy_dataset_archive(source, linked_inbox)
    assert not (real_inbox / source.name).exists()


def test_dataset_zip_copy_refuses_symlinked_inbox_parent(tmp_path):
    import pytest

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    inbox = linked_parent / "inbox"
    inbox.mkdir()
    source = tmp_path / "data.zip"
    source.write_bytes(b"zip-content")

    with pytest.raises(ValueError, match="symlinked path component"):
        copy_dataset_archive(source, inbox)
    assert not (real_parent / "inbox" / source.name).exists()


def test_safe_display_filename_removes_terminal_control_characters():
    assert safe_display_filename("data" + chr(10) + ".zip") == "data.zip"
    assert safe_display_filename("bad" + chr(27) + "[31m.zip") == "bad[31m.zip"
    assert safe_display_filename(chr(10) + chr(9)) == "<unnamed>"



def test_importer_returns_nonzero_when_archive_import_fails(tmp_path, monkeypatch):
    import import_kaggle_models as importer

    incoming = tmp_path / "incoming"
    destination = tmp_path / "imported"
    data_incoming = tmp_path / "datasets" / "incoming"
    staging = tmp_path / ".staging"
    incoming.mkdir()
    (incoming / "NOORIX_broken.zip").write_bytes(b"not-a-zip")

    monkeypatch.setattr(importer, "INCOMING", incoming)
    monkeypatch.setattr(importer, "DEST", destination)
    monkeypatch.setattr(importer, "DATA_INCOMING", data_incoming)
    monkeypatch.setattr(importer, "TEMP", staging)
    monkeypatch.setattr(sys, "argv", ["import_kaggle_models.py"])

    assert main() == 1
    assert (incoming / "NOORIX_broken.zip").read_bytes() == b"not-a-zip"
    assert list(destination.iterdir()) == []



def test_ensure_directory_refuses_symlinked_parent(tmp_path):
    import pytest

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlinked component"):
        ensure_directory_without_symlinks(linked_parent / "incoming")
    assert list(real_parent.iterdir()) == []


def test_ensure_directory_refuses_symlink_target(tmp_path):
    import pytest

    real_target = tmp_path / "real-target"
    real_target.mkdir()
    linked_target = tmp_path / "linked-target"
    try:
        linked_target.symlink_to(real_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlinked component"):
        ensure_directory_without_symlinks(linked_target)
    assert list(real_target.iterdir()) == []

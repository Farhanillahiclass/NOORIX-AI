from pathlib import Path

import pytest

from scripts.dependency_audit import audit_project, load_project, parse_requirement


def test_requirement_parser_detects_lower_and_upper_bounds():
    parsed = parse_requirement("Example_Pkg>=1.2,<2")
    assert parsed["normalized_name"] == "example-pkg"
    assert parsed["has_lower_bound"] is True
    assert parsed["has_upper_bound"] is True
    assert parsed["is_exact"] is False


def test_requirement_parser_identifies_unbounded_range():
    parsed = parse_requirement("opencv-python>=4.9")
    assert parsed["has_lower_bound"] is True
    assert parsed["has_upper_bound"] is False
    assert parsed["is_exact"] is False


def test_requirement_parser_rejects_unsupported_marker():
    with pytest.raises(ValueError):
        parse_requirement("demo>=1; python_version<'4'")


def test_audit_includes_base_and_optional_dependencies():
    report = audit_project({
        "name": "demo",
        "version": "1.0",
        "requires_python": ">=3.11",
        "dependencies": [],
        "optional": {"dev": ["pytest>=8,<9"]},
    })
    assert report["project"] == "demo"
    assert report["summary"]["declared"] == 1
    assert report["dependencies"][0]["group"] == "extra:dev"
    assert report["dependencies"][0]["review"] == "upper bound present"


def test_audit_reports_installed_version_mismatch(monkeypatch):
    import scripts.dependency_audit as audit_module

    class FakeSpecifierSet:
        def __init__(self, specifier):
            self.specifier = specifier

        def __contains__(self, version):
            return str(version) == "1.0"

    class FakeVersion:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.value

    monkeypatch.setattr(audit_module, "SpecifierSet", FakeSpecifierSet)
    monkeypatch.setattr(audit_module, "Version", FakeVersion)
    monkeypatch.setattr(audit_module.metadata, "version", lambda name: "2.0")
    report = audit_project({
        "name": "demo", "version": "1.0", "requires_python": ">=3.11",
        "dependencies": ["demo-lib>=1,<2"], "optional": {},
    })
    assert report["dependencies"][0]["installed_satisfies_specifier"] is False
    assert report["summary"]["installed_version_mismatches"] == 1


def test_requirement_parser_rejects_malformed_specifier_list():
    with pytest.raises(ValueError):
        parse_requirement("demo>=1,,<2")


def test_markdown_cell_escapes_table_delimiters_and_newlines():
    from scripts.dependency_audit import _markdown_cell

    assert _markdown_cell("pkg|name\nnext") == r"pkg\|name next"
    assert chr(27) not in _markdown_cell("pkg" + chr(27) + "name")
    assert chr(9) not in _markdown_cell("pkg" + chr(9) + "name")



def test_dependency_report_rejects_symlinked_output_directory(tmp_path: Path):
    from scripts.dependency_audit import write_report

    target = tmp_path / "real-reports"
    target.mkdir()
    linked = tmp_path / "reports-link"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    report = audit_project({
        "name": "demo", "version": "1.0", "requires_python": ">=3.11",
        "dependencies": [], "optional": {},
    })
    with pytest.raises(ValueError, match="symbolic link"):
        write_report(report, linked)
    assert list(target.iterdir()) == []



def test_dependency_report_timestamp_collision_uses_suffix_and_preserves_existing(tmp_path: Path, monkeypatch):
    import datetime as datetime_module
    import scripts.dependency_audit as audit_module

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime_module.datetime(2026, 11, 23, 12, 34, 56).astimezone()

    monkeypatch.setattr(audit_module, "datetime", FixedDateTime)
    output = tmp_path / "reports"
    stamp = "20261123-123456"
    existing = output / stamp
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    report = audit_project({
        "name": "demo", "version": "1.0", "requires_python": ">=3.11",
        "dependencies": [], "optional": {},
    })

    json_path, md_path = audit_module.write_report(report, output)

    assert json_path.parent.name == f"{stamp}-1"
    assert md_path.parent == json_path.parent
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert json_path.is_file()
    assert md_path.is_file()


def test_dependency_report_removes_own_partial_json_when_markdown_write_fails(tmp_path, monkeypatch):
    import pytest
    from pathlib import Path
    from scripts.dependency_audit import write_report

    report = {
        "project": "NOORIX-AI",
        "project_version": "0.2.0",
        "requires_python": ">=3.11",
        "generated_at": "2026-11-23T12:00:00+05:00",
        "dependencies": [],
        "limitations": [],
    }
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.name == "dependency-audit.md":
            raise OSError("simulated report write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="simulated report write failure"):
        write_report(report, tmp_path)

    report_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(report_dirs) == 1
    assert list(report_dirs[0].iterdir()) == []


def test_dependency_audit_cli_does_not_print_local_path_on_read_error(tmp_path, capsys):
    from scripts.dependency_audit import main

    result = main(["--pyproject", str(tmp_path), "--output-dir", str(tmp_path / "reports")])

    captured = capsys.readouterr()
    assert result == 2
    assert "IsADirectoryError" in captured.err
    assert str(tmp_path) not in captured.err



def test_project_metadata_rejects_oversized_pyproject(tmp_path: Path):
    path = tmp_path / "pyproject.toml"
    with path.open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)

    with pytest.raises(ValueError, match="1 MiB"):
        load_project(path)


def test_project_metadata_rejects_symlinked_pyproject(tmp_path: Path):
    target = tmp_path / "real.toml"
    target.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    linked = tmp_path / "pyproject.toml"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_project(linked)


def test_project_metadata_rejects_normalized_extra_name_collision(tmp_path: Path):
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\nrequires-python = ">=3.11"\n'
        '[project.optional-dependencies]\n"test_extra" = []\n"test-extra" = []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collide after normalization"):
        load_project(path)


def test_project_metadata_rejects_invalid_extra_name(tmp_path: Path):
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\nrequires-python = ">=3.11"\n'
        '[project.optional-dependencies]\n"bad extra" = []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid extra names"):
        load_project(path)

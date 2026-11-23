from pathlib import Path

import pytest

from scripts.dependency_audit import audit_project, parse_requirement


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

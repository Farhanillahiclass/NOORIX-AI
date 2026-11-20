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
    assert report["dependencies"][0]["group"] == "extra:dev"\n    assert report["dependencies"][0]["review"] == "upper bound present"

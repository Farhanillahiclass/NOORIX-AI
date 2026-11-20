from pathlib import Path

import pytest

from scripts.verify_offline_bundle import (
    normalize_distribution,
    selected_requirement,
    validate_wheelhouse,
)


def test_distribution_names_are_normalized():
    assert normalize_distribution("NOORIX_AI") == "noorix-ai"


def test_selected_requirement_builds_extras():
    assert selected_requirement("noorix-ai", ["all", "dev"]) == "noorix-ai[all,dev]"
    assert selected_requirement("noorix-ai", []) == "noorix-ai"


def test_selected_requirement_rejects_invalid_names():
    with pytest.raises(ValueError, match="valid distribution"):
        selected_requirement("../noorix", [])
    with pytest.raises(ValueError, match="valid distribution extra"):
        selected_requirement("noorix-ai", ["dev;--index-url"])


def test_wheelhouse_requires_directory_and_project_wheel(tmp_path: Path):
    missing = tmp_path / "missing"
    assert validate_wheelhouse(missing, "noorix-ai") == "wheelhouse directory does not exist"

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    assert validate_wheelhouse(wheelhouse, "noorix-ai") == "wheelhouse contains no wheel (.whl) files"

    (wheelhouse / "numpy-2.0.0-py3-none-any.whl").write_bytes(b"")
    assert validate_wheelhouse(wheelhouse, "noorix-ai") == "wheelhouse does not contain a noorix-ai project wheel"

    (wheelhouse / "noorix_ai-0.2.0-py3-none-any.whl").write_bytes(b"")
    assert validate_wheelhouse(wheelhouse, "noorix-ai") is None

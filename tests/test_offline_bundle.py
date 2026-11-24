from pathlib import Path

import pytest

from scripts.verify_offline_bundle import (
    normalize_distribution,
    pip_supports_dry_run,
    redact_local_paths,
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


def test_wheelhouse_rejects_symlinked_directory(tmp_path: Path):
    actual = tmp_path / "actual-wheelhouse"
    actual.mkdir()
    linked = tmp_path / "wheelhouse-link"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    assert validate_wheelhouse(linked, "noorix-ai") == "wheelhouse path contains a symbolic link"


def test_wheelhouse_rejects_symlinked_wheel(tmp_path: Path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    actual = tmp_path / "noorix_ai-0.2.0-py3-none-any.whl"
    actual.write_bytes(b"")
    linked = wheelhouse / actual.name
    try:
        linked.symlink_to(actual)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    assert validate_wheelhouse(wheelhouse, "noorix-ai") == "wheelhouse contains a symbolic-link wheel path"


def test_pip_dry_run_capability_detection():
    assert pip_supports_dry_run("Usage: pip install [options] --dry-run")
    assert not pip_supports_dry_run("Usage: pip install [options]")


def test_diagnostics_redact_local_path_separators(tmp_path: Path):
    private_root = tmp_path / "private" / "project"
    private_root.mkdir(parents=True)
    message = f"failed at {private_root} and {str(private_root).replace(chr(47), chr(92))}"

    redacted = redact_local_paths(message, [private_root])

    assert str(private_root) not in redacted
    assert str(private_root).replace("/", chr(92)) not in redacted
    assert redacted.count("<local-path>") == 2


def test_offline_bundle_cli_rejects_duplicate_and_invalid_extras(capsys):
    from scripts.verify_offline_bundle import main

    assert main(["--extras", "all,all"]) == 2
    assert "Duplicate extras" in capsys.readouterr().err
    assert main(["--extras", "all,../dev"]) == 2
    assert "Invalid extra name" in capsys.readouterr().err



def test_load_extras_rejects_oversized_pyproject(tmp_path: Path):
    from scripts.verify_offline_bundle import load_extras

    path = tmp_path / "pyproject.toml"
    with path.open("wb") as stream:
        stream.truncate(1024 * 1024 + 1)

    with pytest.raises(ValueError, match="1 MiB"):
        load_extras(path)


def test_load_extras_rejects_symlinked_pyproject(tmp_path: Path):
    from scripts.verify_offline_bundle import load_extras

    target = tmp_path / "real.toml"
    target.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\n'
        '[project.optional-dependencies]\ndev = []\n',
        encoding="utf-8",
    )
    linked = tmp_path / "pyproject.toml"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_extras(linked)


def test_load_extras_rejects_normalized_name_collisions(tmp_path: Path):
    from scripts.verify_offline_bundle import load_extras

    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\n'
        '[project.optional-dependencies]\n"test_extra" = []\n"test-extra" = []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collide after normalization"):
        load_extras(path)


def test_load_extras_rejects_invalid_extra_names(tmp_path: Path):
    from scripts.verify_offline_bundle import load_extras

    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "demo"\nversion = "1.0"\n'
        '[project.optional-dependencies]\n"bad extra" = []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid extra name"):
        load_extras(path)

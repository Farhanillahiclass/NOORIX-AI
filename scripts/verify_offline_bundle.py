"""Verify a prebuilt wheelhouse can satisfy NOORIX extras without network access."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import stat
import subprocess
import sys
import tomllib

MAX_PYPROJECT_BYTES = 1024 * 1024
_EXTRA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def redact_local_paths(text: str, paths: list[Path]) -> str:
    """Hide local filesystem roots from pip and validation diagnostics."""
    redacted = text
    candidates: set[str] = set()
    for path in paths:
        try:
            value = str(Path(path).expanduser().absolute())
        except (OSError, RuntimeError):
            continue
        if value and value not in {".", "/"}:
            candidates.add(value)
            candidates.add(value.replace(chr(92), "/"))
            candidates.add(value.replace("/", chr(92)))
    for value in sorted(candidates, key=len, reverse=True):
        redacted = re.sub(re.escape(value), "<local-path>", redacted, flags=re.IGNORECASE)
    return redacted


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def selected_requirement(package: str, extras: list[str]) -> str:
    if not package or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package):
        raise ValueError("package must be a valid distribution name")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", extra) for extra in extras):
        raise ValueError("extras must be valid distribution extra names")
    return f"{package}[{','.join(extras)}]" if extras else package


def pip_supports_dry_run(help_text: str) -> bool:
    """Return whether this pip exposes the non-installing dry-run option."""
    return "--dry-run" in help_text


def has_symlink_component(path: Path) -> bool:
    """Detect a symlink at the candidate path or any existing ancestor."""
    candidate = Path(path).absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def validate_wheelhouse(wheelhouse: Path, package: str) -> str | None:
    wheelhouse = Path(wheelhouse).expanduser()
    if has_symlink_component(wheelhouse):
        return "wheelhouse path contains a symbolic link"
    if not wheelhouse.exists() or not wheelhouse.is_dir():
        return "wheelhouse directory does not exist"

    try:
        wheels = list(wheelhouse.glob("*.whl"))
    except OSError:
        return "wheelhouse contents could not be inspected"
    if not wheels:
        return "wheelhouse contains no wheel (.whl) files"

    expected = normalize_distribution(package)
    matching = []
    for wheel in wheels:
        # Do not let pip follow a wheel symlink or resolve a non-file entry.
        if has_symlink_component(wheel):
            return "wheelhouse contains a symbolic-link wheel path"
        try:
            if not wheel.is_file() or wheel.resolve().parent != wheelhouse.resolve():
                return "wheelhouse contains an invalid wheel entry"
        except OSError:
            return "wheelhouse contains an unreadable wheel entry"
        parts = wheel.name.split("-")
        if len(parts) >= 5 and normalize_distribution(parts[0]) == expected:
            matching.append(wheel)
    if not matching:
        return f"wheelhouse does not contain a {package} project wheel"
    return None


def load_extras(pyproject: Path) -> set[str]:
    """Read a bounded project metadata file without following symlink paths."""
    pyproject = Path(pyproject).expanduser()
    if has_symlink_component(pyproject):
        raise ValueError("pyproject path contains a symbolic link")
    if not stat.S_ISREG(pyproject.lstat().st_mode):
        raise ValueError("pyproject path is not a regular file")
    with pyproject.open("rb") as stream:
        raw = stream.read(MAX_PYPROJECT_BYTES + 1)
    if len(raw) > MAX_PYPROJECT_BYTES:
        raise ValueError("pyproject.toml exceeds the 1 MiB read limit")
    metadata = tomllib.loads(raw.decode("utf-8"))
    project = metadata.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject must contain a project table")
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("pyproject optional-dependencies must be a table")
    names = list(optional)
    if any(not isinstance(name, str) or not _EXTRA_NAME.fullmatch(name) for name in names):
        raise ValueError("pyproject contains an invalid extra name")
    normalized = [normalize_distribution(name) for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("pyproject extra names collide after normalization")
    return set(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve NOORIX package extras from local wheels only; no packages are installed."
    )
    parser.add_argument("--wheelhouse", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--package", default="noorix-ai")
    parser.add_argument("--extras", default="all,dev",
                        help="Comma-separated pyproject extras (default: all,dev)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    wheelhouse = args.wheelhouse.expanduser()
    if not wheelhouse.is_absolute():
        wheelhouse = Path.cwd() / wheelhouse
    extras = [item.strip() for item in args.extras.split(",") if item.strip()]
    if len(extras) != len(set(extras)):
        print("Duplicate extras were supplied.", file=sys.stderr)
        return 2
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", extra) for extra in extras):
        print("Invalid extra name supplied.", file=sys.stderr)
        return 2
    try:
        available = load_extras(root / "pyproject.toml")
        unknown = sorted(set(extras) - available)
        if unknown:
            raise ValueError("unknown extras: " + ", ".join(unknown))
        requirement = selected_requirement(args.package, extras)
        problem = validate_wheelhouse(wheelhouse, args.package)
        if problem:
            raise ValueError(problem)
        wheelhouse = wheelhouse.resolve(strict=True)
    except (OSError, RuntimeError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"Offline bundle check failed ({type(exc).__name__}).", file=sys.stderr)
        return 2

    try:
        help_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--help"],
            cwd=root, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Could not inspect pip capabilities ({type(exc).__name__}).", file=sys.stderr)
        return 2
    if help_result.returncode or not pip_supports_dry_run(help_result.stdout or ""):
        print("This pip version does not expose install --dry-run.", file=sys.stderr)
        print("Upgrade pip on a connected machine, rebuild or retain the wheelhouse, then retry.", file=sys.stderr)
        return 2

    command = [
        sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--no-index", "--find-links", str(wheelhouse), requirement,
    ]
    print("NOORIX-AI OFFLINE WHEELHOUSE CHECK")
    print("Wheelhouse: <local wheelhouse>")
    print(f"Requirement: {requirement}")
    print("Network access: disabled (--no-index)")
    print("Installation: dry-run only; no packages will be installed")
    try:
        result = subprocess.run(
            command, cwd=root, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Offline dependency resolution could not complete ({type(exc).__name__}).", file=sys.stderr)
        return 2
    if result.stdout:
        safe_output = redact_local_paths(result.stdout[-12000:], [root, Path.home(), wheelhouse])
        print(safe_output)
    if result.returncode:
        print("FAIL: the wheelhouse cannot currently resolve this environment.", file=sys.stderr)
        print("Build a fresh wheelhouse for this OS, architecture and Python version.", file=sys.stderr)
        return 1
    print("PASS: pip resolved the selected package and dependencies from local wheels.")
    print("This verifies dependency availability only; it does not install or test the application.")
    print("For installation, run the documented pip install command after reviewing this report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

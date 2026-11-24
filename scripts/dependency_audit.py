"""Audit declared Python dependencies against local metadata; never contacts the network."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import stat
import sys
import tomllib
from typing import Any

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:
    Requirement = SpecifierSet = Version = None

_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)\s*(?:\[([^]]+)\])?\s*(.*)$"
)
_VERSION_BOUND = re.compile(r"(?:^|,)\s*(===|==|~=|!=|<=|>=|<|>)\s*([^,;\s]+)")
_SPECIFIER_LIST = re.compile(r"^(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[^,;\s]+\s*)(?:,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[^,;\s]+\s*)*$")
MAX_PYPROJECT_BYTES = 1024 * 1024
_EXTRA_NAME = re.compile(r"^[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*$")


def _read_pyproject(path: Path) -> dict[str, Any]:
    """Read a bounded, regular, non-symlink project metadata file."""
    path = Path(path).expanduser()
    if _has_symlink_component(path):
        raise ValueError("pyproject path contains a symbolic link")
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("pyproject path is not a regular file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_PYPROJECT_BYTES + 1)
    except FileNotFoundError:
        raise
    if len(raw) > MAX_PYPROJECT_BYTES:
        raise ValueError("pyproject.toml exceeds the 1 MiB read limit")
    return tomllib.loads(raw.decode("utf-8"))


def parse_requirement(value: str) -> dict[str, Any]:
    """Parse the deliberately simple PEP 508 requirements used by pyproject.toml."""
    if not isinstance(value, str):
        raise ValueError("dependency requirement must be a string")
    match = _REQUIREMENT.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported dependency requirement: {value!r}")
    name, extras, specifier = match.groups()
    if extras and any(not _EXTRA_NAME.fullmatch(item.strip()) for item in extras.split(",")):
        raise ValueError("dependency requirement contains an invalid extra name")
    if ";" in specifier:
        raise ValueError("environment markers are not supported by this local audit")
    if "@" in specifier:
        raise ValueError("direct URL dependencies are not supported by this local audit")
    if specifier.strip() and not _SPECIFIER_LIST.fullmatch(specifier):
        raise ValueError(f"invalid version specifier: {specifier!r}")
    if Requirement is not None:
        try:
            parsed = Requirement(value)
        except Exception as exc:
            raise ValueError(f"invalid PEP 508 requirement: {value!r}") from exc
        if parsed.marker is not None or parsed.url is not None:
            raise ValueError("environment markers and direct URLs are not supported by this local audit")
        name = parsed.name
        extras = ",".join(sorted(parsed.extras)) if parsed.extras else None
        specifier = str(parsed.specifier)
    if specifier and SpecifierSet is not None:
        SpecifierSet(specifier)
    operators = [item[0] for item in _VERSION_BOUND.findall(specifier)]
    return {
        "requirement": value,
        "name": name,
        "normalized_name": re.sub(r"[-_.]+", "-", name).lower(),
        "specifier": specifier.strip(),
        "has_lower_bound": any(op in {">", ">="} for op in operators),
        "has_upper_bound": any(op in {"<", "<="} for op in operators),
        "is_exact": any(op in {"==", "==="} for op in operators),
        "extras": [item.strip() for item in extras.split(",")] if extras else [],
    }


def load_project(path: Path) -> dict[str, Any]:
    data = _read_pyproject(path)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml must contain a [project] table")
    dependencies = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
        raise ValueError("project.dependencies must be a list of strings")
    if not isinstance(optional, dict) or any(
        not isinstance(key, str) or not isinstance(value, list)
        or not all(isinstance(x, str) for x in value)
        for key, value in optional.items()
    ):
        raise ValueError("project.optional-dependencies must map extras to string lists")
    extra_names = list(optional)
    if any(not _EXTRA_NAME.fullmatch(name) for name in extra_names):
        raise ValueError("optional dependency group names must be valid extra names")
    normalized_extras = [re.sub(r"[-_.]+", "-", name).lower() for name in extra_names]
    if len(normalized_extras) != len(set(normalized_extras)):
        raise ValueError("optional dependency group names collide after normalization")
    name = project.get("name", "unknown")
    version = project.get("version", "unknown")
    requires_python = project.get("requires-python", "unspecified")
    if not all(isinstance(value, str) and value.strip() for value in (name, version, requires_python)):
        raise ValueError("project name, version and requires-python must be non-empty strings")
    return {
        "name": name,
        "version": version,
        "requires_python": requires_python,
        "dependencies": dependencies,
        "optional": optional,
    }


def audit_project(project: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"runtime": project["dependencies"]}
    groups.update({f"extra:{name}": values for name, values in project["optional"].items()})
    entries = []
    for group, requirements in groups.items():
        for raw in requirements:
            item = parse_requirement(raw)
            try:
                installed = metadata.version(item["name"])
                item["installed_version"] = installed
                item["installed"] = True
                if not item["specifier"]:
                    item["installed_satisfies_specifier"] = True
                elif SpecifierSet is not None:
                    item["installed_satisfies_specifier"] = Version(installed) in SpecifierSet(item["specifier"])
                else:
                    item["installed_satisfies_specifier"] = None
            except metadata.PackageNotFoundError:
                item["installed_version"] = None
                item["installed"] = False
                item["installed_satisfies_specifier"] = None
            item["group"] = group
            item["review"] = (
                "exact pin" if item["is_exact"] else
                "upper bound present" if item["has_upper_bound"] else
                "no upper bound; review compatibility policy"
            )
            entries.append(item)
    return {
        "project": project["name"],
        "project_version": project["version"],
        "requires_python": project["requires_python"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dependencies": entries,
        "summary": {
            "declared": len(entries),
            "installed": sum(item["installed"] for item in entries),
            "missing_in_current_environment": sum(not item["installed"] for item in entries),
            "without_upper_bound": sum(not item["has_upper_bound"] and not item["is_exact"] for item in entries),
            "installed_version_mismatches": sum(item["installed_satisfies_specifier"] is False for item in entries),
            "version_checks_unavailable": sum(item["installed"] and item["installed_satisfies_specifier"] is None for item in entries),
        },
        "limitations": [
            "Installed-version compatibility is checked with packaging when available; unknown means the optional packaging library is absent.",
            "No network vulnerability database, malware scan, license scan, or transitive lockfile audit is performed.",
            "Optional extras are listed even when they are not installed in the current environment.",
            "A missing optional dependency is not automatically a failure for an environment that did not select that extra.",
        ],
    }


def _markdown_cell(value: Any) -> str:
    """Escape untrusted dependency metadata for a single Markdown table cell."""
    printable = "".join(character if character.isprintable() else " " for character in str(value))
    safe = printable.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in (chr(92), "|", "`", "*", "_", "[", "]", "!"):
        safe = safe.replace(character, chr(92) + character)
    return safe

def _is_symlink_or_junction(path: Path) -> bool:
    """Detect symbolic links and Windows directory junctions."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _has_symlink_component(path: Path) -> bool:
    """Reject symlinks in report output paths before creating files."""
    candidate = Path(path).absolute()
    return any(_is_symlink_or_junction(component) for component in (candidate, *candidate.parents))


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser()
    if _has_symlink_component(output_dir):
        raise ValueError("report output path contains a symbolic link")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    target = output_dir / stamp
    suffix = 1
    while True:
        if _has_symlink_component(target):
            raise ValueError("report directory path contains a symbolic link")
        try:
            target.mkdir(exist_ok=False)
            break
        except FileExistsError:
            target = output_dir / f"{stamp}-{suffix}"
            suffix += 1

    json_path = target / "dependency-audit.json"
    md_path = target / "dependency-audit.md"
    created_files: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8", newline="\n") as stream:
            created_files.append(json_path)
            stream.write(json.dumps(report, indent=2) + "\n")

        lines = [
            "# NOORIX-AI Dependency Audit", "",
            f"- Project: {_markdown_cell(report['project'])} {_markdown_cell(report['project_version'])}",
            f"- Python requirement: {_markdown_cell(report['requires_python'])}",
            f"- Generated: {report['generated_at']}", "",
            "| Group | Requirement | Installed version | Satisfies specifier | Upper bound | Review |",
            "|---|---|---|---|---|---",
        ]
        for item in report["dependencies"]:
            installed = item["installed_version"] or "not installed here"
            upper = "yes" if item["has_upper_bound"] or item["is_exact"] else "no"
            satisfies = (
                "yes" if item["installed_satisfies_specifier"] is True
                else "no" if item["installed_satisfies_specifier"] is False
                else "unknown"
            )
            cells = (
                item["group"], item["requirement"], installed,
                satisfies, upper, item["review"],
            )
            lines.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")
        lines += ["", "## Scope and limitations", ""]
        lines.extend(f"- {line}" for line in report["limitations"])

        with md_path.open("x", encoding="utf-8", newline="\n") as stream:
            created_files.append(md_path)
            stream.write("\n".join(lines) + "\n")
    except Exception:
        # Exclusive creation proves these paths belong to this invocation.
        # Remove only our own partial outputs; never touch pre-existing files.
        for created in created_files:
            try:
                created.unlink()
            except OSError:
                pass
        raise
    return json_path, md_path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report declared dependencies and local installed versions; no network.")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dependencies"))
    parser.add_argument("--strict-upper-bounds", action="store_true",
                        help="exit 1 if any non-exact requirement has no upper bound")
    args = parser.parse_args(argv)
    try:
        project = load_project(args.pyproject)
        report = audit_project(project)
        json_path, md_path = write_report(report, args.output_dir)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        # Exception text can embed local paths or machine-specific metadata.
        print(f"Dependency audit failed ({type(exc).__name__}).", file=sys.stderr)
        return 2
    summary = report["summary"]
    print("NOORIX-AI DEPENDENCY AUDIT (offline metadata only)")
    print(f"Declared: {summary['declared']}; installed in this environment: {summary['installed']}")
    print(f"Missing here: {summary['missing_in_current_environment']}; version mismatches: {summary['installed_version_mismatches']}; without upper bound: {summary['without_upper_bound']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if summary["without_upper_bound"]:
        print("REVIEW: one or more dependencies have no upper version bound.")
    return 1 if args.strict_upper_bounds and summary["without_upper_bound"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

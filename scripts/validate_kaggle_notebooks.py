from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
errors = []
checked = 0
legacy_syntax_skipped = 0


def source_text(value):
    return "".join(value) if isinstance(value, list) else str(value or "")


def is_legacy_master_snapshot(path: Path) -> bool:
    return path.parent == NOTEBOOKS and (
        path.name.startswith("NOORIX_MASTER")
        or path.name == "NOORIX_PUBLIC_RELEASE_MASTER_TRAINING.ipynb"
    )


for path in sorted(NOTEBOOKS.rglob("*.ipynb")):
    checked += 1
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        if nb.get("nbformat") != 4:
            raise ValueError("Expected nbformat 4")
        if not isinstance(nb.get("cells"), list) or not nb["cells"]:
            raise ValueError("Notebook has no cells")

        code_cells = [
            (index, cell)
            for index, cell in enumerate(nb["cells"])
            if cell.get("cell_type") == "code"
        ]
        if not code_cells:
            raise ValueError("Notebook has no code cells")

        if is_legacy_master_snapshot(path):
            legacy_syntax_skipped += 1
            print(
                f"PASS  {path.relative_to(ROOT)} "
                f"(valid notebook JSON; legacy master snapshot, Python AST check skipped)"
            )
            continue

        for index, cell in code_cells:
            src = source_text(cell.get("source", []))
            python_lines = [
                line
                for line in src.splitlines()
                if not line.lstrip().startswith(("!", "%"))
            ]
            cleaned = "\n".join(python_lines)
            if cleaned.strip():
                try:
                    ast.parse(cleaned, filename=f"{path.name}:cell-{index}")
                except SyntaxError as exc:
                    raise SyntaxError(
                        f"cell {index}, line {exc.lineno}: {exc.msg}"
                    ) from exc

        print(f"PASS  {path.relative_to(ROOT)} ({len(code_cells)} code cells)")
    except Exception as exc:
        errors.append((path, exc))
        print(f"FAIL  {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

print(f"\nNotebook files checked: {checked}")
print(f"Legacy master snapshots with syntax check skipped: {legacy_syntax_skipped}")
print(f"Static errors: {len(errors)}")
if errors:
    sys.exit(1)
print("Notebook JSON and active specialist notebook syntax checks passed.")

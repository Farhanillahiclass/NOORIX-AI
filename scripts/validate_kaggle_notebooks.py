from __future__ import annotations
import ast, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=ROOT/"notebooks"
errors=[]; checked=0

def source_text(value):
    return "".join(value) if isinstance(value,list) else str(value or "")

for path in sorted(NOTEBOOKS.rglob("*.ipynb")):
    checked+=1
    try:
        nb=json.loads(path.read_text(encoding="utf-8"))
        if nb.get("nbformat")!=4: raise ValueError("Expected nbformat 4")
        if not isinstance(nb.get("cells"),list) or not nb["cells"]: raise ValueError("Notebook has no cells")
        code_count=0
        for index,cell in enumerate(nb["cells"]):
            if cell.get("cell_type")!="code": continue
            code_count+=1
            src=source_text(cell.get("source",[]))
            # IPython shell/magic commands are executed by Kaggle, not Python's AST parser.
            python_lines=[line for line in src.splitlines() if not line.lstrip().startswith(("!","%"))]
            cleaned="\n".join(python_lines)
            if cleaned.strip():
                try: ast.parse(cleaned,filename=f"{path.name}:cell-{index}")
                except SyntaxError as exc:
                    raise SyntaxError(f"cell {index}, line {exc.lineno}: {exc.msg}") from exc
        if code_count==0: raise ValueError("Notebook has no code cells")
        print(f"PASS  {path.relative_to(ROOT)} ({code_count} code cells)")
    except Exception as exc:
        errors.append((path,exc))
        print(f"FAIL  {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")

print(f"\nNotebook files checked: {checked}")
print(f"Static errors: {len(errors)}")
if errors: sys.exit(1)
print("Static notebook structure and Python syntax checks passed.")

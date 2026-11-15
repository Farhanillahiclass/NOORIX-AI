from __future__ import annotations
import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "models" / "incoming"
DEST = ROOT / "models" / "imported"
TEMP = ROOT / "models" / ".staging"

# Keep model weights, their configs/tokenizers, and labels/metadata.
ALLOWED_SUFFIXES = {
    ".onnx", ".pth", ".pt", ".safetensors", ".bin",
    ".json", ".txt", ".model", ".vocab", ".merges",
    ".tiktoken", ".yaml", ".yml", ".cfg", ".pb",
    ".xml", ".data"
}
SKIP_PARTS = {
    "__pycache__", ".ipynb_checkpoints", "runs", "wandb",
    "logs", "samples", "sample", "images", "datasets",
    "dataset", "cache", "caches", "test_images"
}
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB safety ceiling per archive member

def safe_extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> Path | None:
    raw = PurePosixPath(info.filename)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    if info.is_dir():
        return None
    if info.file_size > MAX_MEMBER_BYTES:
        print(f"  SKIP (over 4 GiB): {info.filename}")
        return None
    if any(part.lower() in SKIP_PARTS for part in raw.parts):
        return None
    if Path(raw.name).suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    out = target.joinpath(*raw.parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description="Safely import model artifacts from Kaggle ZIPs.")
    parser.add_argument("--delete-zips", action="store_true",
                        help="Delete source ZIPs only after successful import (not recommended).")
    args = parser.parse_args()
    INCOMING.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    archives = sorted(INCOMING.glob("*.zip"))
    if not archives:
        print(f"No ZIP files found. Put Kaggle ZIP files in: {INCOMING}")
        return 0
    for archive in archives:
        slug = archive.stem.replace(" ", "_")
        staging = TEMP / slug
        destination = DEST / slug
        if staging.exists():
            shutil.rmtree(staging)
        if destination.exists():
            print(f"\nAlready imported (skipping): {archive.name}")
            continue
        staging.mkdir(parents=True, exist_ok=True)
        kept = []
        try:
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"Corrupt member: {bad}")
                for info in zf.infolist():
                    p = safe_extract_member(zf, info, staging)
                    if p:
                        kept.append(p)
            if not kept:
                print(f"\nNo recognized model/config files in {archive.name}; leaving ZIP untouched.")
                shutil.rmtree(staging, ignore_errors=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(destination))
            manifest = {
                "source_zip": archive.name,
                "project": "NOORIX-AI",
                "imported_files": [str(p.relative_to(destination)).replace("\\", "/")
                                   for p in destination.rglob("*") if p.is_file()],
                "note": "Imported artifacts only. Model adapters and validation are still required."
            }
            (destination / "import_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(f"\nIMPORTED: {archive.name}")
            for name in manifest["imported_files"]:
                print("  +", name)
            print("  Destination:", destination)
            if args.delete_zips:
                archive.unlink()
                print("  Source ZIP deleted after successful import.")
            else:
                print("  Source ZIP preserved for rollback.")
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            print(f"\nFAILED: {archive.name}: {exc}")
            print("  Original ZIP has been preserved.")
    try:
        TEMP.rmdir()
    except OSError:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

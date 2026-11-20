from __future__ import annotations
import argparse, hashlib, json, re, shutil, zipfile
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import stat

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "models" / "incoming"
DEST = ROOT / "models" / "imported"
DATA_INCOMING = ROOT / "data" / "datasets" / "incoming"
TEMP = ROOT / "models" / ".staging"
ALLOWED_SUFFIXES = {".onnx",".pth",".pt",".safetensors",".bin",".json",".txt",".model",".vocab",".merges",".tiktoken",".yaml",".yml",".cfg",".pb",".xml",".data"}
SKIP_PARTS = {"__pycache__",".ipynb_checkpoints","runs","wandb","logs","samples","sample","images","datasets","dataset","cache","caches","test_images"}
MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
MAX_COMPRESSION_RATIO = 1_000
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
    "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
}

def archive_slug(filename: str) -> str:
    """Create a bounded, filesystem-safe destination name from an archive filename."""
    stem = Path(filename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80]
    safe_stem = safe_stem or "archive"
    digest = hashlib.sha256(filename.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{safe_stem}-{digest}"


def _build_path_index(target_root: Path) -> dict[str, tuple[tuple[str, ...], str]]:
    """Index existing paths once for portable case-insensitive collision checks."""
    index: dict[str, tuple[tuple[str, ...], str]] = {}
    for existing in target_root.rglob("*"):
        parts = existing.relative_to(target_root).parts
        key = "/".join(part.casefold() for part in parts)
        kind = "blocked" if existing.is_symlink() else "directory" if existing.is_dir() else "file"
        previous = index.get(key)
        if previous is not None and previous[0] != parts:
            index[key] = (previous[0], "blocked")
        else:
            index[key] = (parts, kind)
    return index


def safe_extract_member(zf, info, target, path_index=None):
    """Extract one approved regular file without escaping or overwriting target."""
    normalized = info.filename.replace("\\", "/")
    raw = PurePosixPath(normalized)
    mode = (info.external_attr >> 16) & 0o170000
    windows_parts = normalized.split("/")
    reserved_names = WINDOWS_RESERVED_NAMES | {
        "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
    }
    invalid_component = any(
        not part
        or part in {".", ".."}
        or part.endswith((" ", "."))
        or part.split(".", 1)[0].rstrip(" .").upper() in reserved_names
        for part in windows_parts
    )
    if (
        not normalized
        or "\x00" in normalized
        or raw.is_absolute()
        or invalid_component
        or any(":" in part for part in windows_parts)
        or mode not in (0, stat.S_IFREG)
        or info.is_dir()
        or info.flag_bits & 0x1
    ):
        return None
    if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
        print(f"  SKIP (member size limit): {info.filename}")
        return None
    if info.file_size and (
        info.compress_size == 0
        or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
    ):
        print(f"  SKIP (suspicious compression ratio): {info.filename}")
        return None
    if any(part.lower() in SKIP_PARTS for part in raw.parts):
        return None
    if Path(raw.name).suffix.lower() not in ALLOWED_SUFFIXES:
        return None

    target_path = Path(target)
    # Do not resolve first: doing so would silently accept a symlinked staging
    # root and redirect extraction outside the caller's intended directory.
    try:
        if target_path.is_symlink() or not target_path.is_dir():
            return None
        for component in (target_path, *target_path.parents):
            if component.is_symlink():
                return None
        target_root = target_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    out = target_root.joinpath(*raw.parts)
    try:
        if not out.resolve().is_relative_to(target_root):
            return None
    except (OSError, RuntimeError):
        return None

    # Reuse one path index for the archive instead of rescanning the complete
    # staging tree for every member. Exact-case shared parent directories are valid.
    if path_index is None:
        path_index = _build_path_index(target_root)
    for position, part in enumerate(raw.parts):
        prefix = tuple(raw.parts[:position + 1])
        key = "/".join(item.casefold() for item in prefix)
        existing = path_index.get(key)
        if existing is None:
            continue
        existing_parts, kind = existing
        if existing_parts != prefix:
            return None
        if kind == "blocked":
            return None
        if position < len(raw.parts) - 1:
            if kind != "directory":
                return None
        else:
            return None

    out.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, out.open("xb") as dst:
        copied = 0
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_MEMBER_BYTES or copied > info.file_size:
                raise zipfile.BadZipFile(f"Member exceeded declared size: {info.filename}")
            dst.write(chunk)
        if copied != info.file_size:
            raise zipfile.BadZipFile(f"Member size mismatch: {info.filename}")
    for position in range(1, len(raw.parts)):
        prefix = tuple(raw.parts[:position])
        key = "/".join(item.casefold() for item in prefix)
        path_index.setdefault(key, (prefix, "directory"))
    leaf_key = "/".join(part.casefold() for part in raw.parts)
    path_index[leaf_key] = (tuple(raw.parts), "file")
    return out


def prepare_staging_directory(staging: Path, destination: Path) -> bool:
    """Create a fresh staging directory without deleting existing user data."""
    if staging.exists() or staging.is_symlink():
        return False
    if destination.exists() or destination.is_symlink():
        return False
    staging.mkdir(parents=True, exist_ok=False)
    return True


def main():
    parser=argparse.ArgumentParser(description="Safely import NOORIX model artifacts from Kaggle ZIPs.")
    parser.add_argument("--scan-downloads",action="store_true",help="Also scan the Windows Downloads folder.")
    parser.add_argument("--downloads-dir",default=str(Path.home()/"Downloads"),help="Downloads folder path.")
    parser.add_argument("--since-hours",type=float,default=2.0,help="Only include Downloads ZIPs modified within this many hours.")
    parser.add_argument("--delete-zips",action="store_true",help="Delete source ZIPs after successful import (not recommended).")
    args=parser.parse_args()
    INCOMING.mkdir(parents=True,exist_ok=True); DEST.mkdir(parents=True,exist_ok=True); DATA_INCOMING.mkdir(parents=True,exist_ok=True); TEMP.mkdir(parents=True,exist_ok=True)
    archives={p.resolve():p for p in INCOMING.glob("*.zip")}
    if args.scan_downloads:
        downloads=Path(args.downloads_dir).expanduser()
        cutoff=datetime.now().timestamp()-max(0,args.since_hours)*3600
        if downloads.is_dir():
            recent=[]
            for p in downloads.glob("*.zip"):
                try:
                    if p.is_file() and p.stat().st_mtime>=cutoff: recent.append(p)
                except OSError: pass
            print(f"Scanning recent ZIP downloads in: {downloads}")
            print(f"Time window: last {args.since_hours:g} hour(s)")
            for p in recent:
                # NOORIX-named archives are trained model packages.
                # Other recent ZIPs are copied intact into the dataset inbox.
                if p.name.upper().startswith("NOORIX_") or p.name.upper().startswith("NOORIX-"):
                    archives[p.resolve()]=p
                else:
                    target=DATA_INCOMING/p.name
                    if target.exists():
                        print(f"Dataset ZIP already present (skip): {target.name}")
                    else:
                        shutil.copy2(p,target)
                        print(f"DATASET ZIP COPIED: {p.name} -> {target}")
        else: print(f"Downloads folder not found: {downloads}")
    if not archives:
        print(f"No NOORIX model ZIPs found in {INCOMING}; recent dataset ZIPs, if any, are copied to {DATA_INCOMING}")
        return 0
    for archive in sorted(archives.values(),key=lambda p:p.stat().st_mtime):
        slug=archive_slug(archive.name)
        staging=TEMP/slug; destination=DEST/slug
        if not prepare_staging_directory(staging, destination):
            print(f"\nExisting staging or destination path found; refusing to overwrite: {archive.name}")
            continue
        kept=[]
        path_index: dict[str, tuple[tuple[str, ...], str]] = {}
        try:
            with zipfile.ZipFile(archive) as zf:
                members = zf.infolist()
                if len(members) > MAX_ARCHIVE_MEMBERS:
                    raise zipfile.BadZipFile(
                        f"Archive has {len(members)} entries; limit is {MAX_ARCHIVE_MEMBERS}"
                    )
                total_size = sum(info.file_size for info in members)
                if total_size > MAX_ARCHIVE_BYTES:
                    raise zipfile.BadZipFile(
                        f"Archive expands to {total_size} bytes; limit is {MAX_ARCHIVE_BYTES}"
                    )
                # CRC is verified while each accepted member is streamed to disk.
                # Avoid eagerly decompressing ignored ZIP entries via testzip().
                for info in members:
                    p=safe_extract_member(zf,info,staging,path_index)
                    if p: kept.append(p)
            if not kept:
                print(f"\nNo recognized model/config files in {archive.name}; source ZIP kept.")
                shutil.rmtree(staging,ignore_errors=True); continue
            files=[p for p in staging.rglob("*") if p.is_file()]
            imported_files=[str(p.relative_to(staging)).replace("\\","/") for p in files]
            checksums=[]
            for path in files:
                digest=hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                checksums.append({
                    "path": str(path.relative_to(staging)).replace("\\","/"),
                    "sha256": digest.hexdigest(),
                })
            manifest={
                "source_zip":archive.name,
                "project":"NOORIX-AI",
                "imported_files":imported_files,
                "integrity":{"algorithm":"sha256","files":checksums},
                "note":"Artifacts imported only. Validate provenance, integrity and a model-specific adapter before activation.",
            }
            (staging/"import_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            # Publish only after extraction, checksums and manifest are complete.
            shutil.move(str(staging),str(destination))
            print(f"\\nIMPORTED: {archive.name}")
            for name in imported_files: print("  +",name)
            print("  + import_manifest.json")
            print("  Destination:",destination)
            if args.delete_zips: archive.unlink(); print("  Source ZIP deleted.")
            else: print("  Source ZIP preserved.")
        except Exception as exc:
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging,ignore_errors=True)
            print(f"\nFAILED: {archive.name}: {type(exc).__name__}: {exc}")
            print("  Original ZIP preserved.")
    try: TEMP.rmdir()
    except OSError: pass
    return 0

if __name__=="__main__": raise SystemExit(main())

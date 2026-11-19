from __future__ import annotations
import argparse, json, shutil, zipfile
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
}

def safe_extract_member(zf, info, target):
    """Extract one approved regular file without escaping or overwriting target."""
    normalized = info.filename.replace("\\\\", "/")
    raw = PurePosixPath(normalized)
    mode = (info.external_attr >> 16) & 0o170000
    windows_parts = normalized.split("/")
    reserved = any(
        part.rstrip(" .").split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        or part.endswith((" ", "."))
        for part in windows_parts
    )
    if (
        not normalized
        or "\\x00" in normalized
        or raw.is_absolute()
        or ".." in raw.parts
        or any(":" in part for part in raw.parts)
        or reserved
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

    target_root = Path(target).resolve()
    out = target_root.joinpath(*raw.parts)
    try:
        if not out.resolve().is_relative_to(target_root):
            return None
    except (OSError, RuntimeError):
        return None

    # Windows filesystems are commonly case-insensitive. Reject collisions on
    # every platform so an import validated on Linux remains portable.
    relative_key = "/".join(raw.parts).casefold()
    for existing in target_root.rglob("*"):
        if existing.is_file():
            if existing.relative_to(target_root).as_posix().casefold() == relative_key:
                return None
    if out.exists():
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
    return out

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
        slug=archive.stem.replace(" ","_")
        staging=TEMP/slug; destination=DEST/slug
        if destination.exists():
            print(f"\nAlready imported (skipping): {archive.name}"); continue
        if staging.exists(): shutil.rmtree(staging)
        staging.mkdir(parents=True,exist_ok=True); kept=[]
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
                bad=zf.testzip()
                if bad: raise zipfile.BadZipFile(f"Corrupt member: {bad}")
                for info in members:
                    p=safe_extract_member(zf,info,staging)
                    if p: kept.append(p)
            if not kept:
                print(f"\nNo recognized model/config files in {archive.name}; source ZIP kept.")
                shutil.rmtree(staging,ignore_errors=True); continue
            shutil.move(str(staging),str(destination))
            files=[str(p.relative_to(destination)).replace("\\","/") for p in destination.rglob("*") if p.is_file()]
            manifest={"source_zip":archive.name,"source_path":str(archive),"project":"NOORIX-AI","imported_files":files,"note":"Artifacts imported only. Validate and implement a model-specific adapter before activation."}
            (destination/"import_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"\nIMPORTED: {archive.name}")
            for name in files: print("  +",name)
            print("  Destination:",destination)
            if args.delete_zips: archive.unlink(); print("  Source ZIP deleted.")
            else: print("  Source ZIP preserved.")
        except Exception as exc:
            shutil.rmtree(staging,ignore_errors=True)
            print(f"\nFAILED: {archive.name}: {type(exc).__name__}: {exc}")
            print("  Original ZIP preserved.")
    try: TEMP.rmdir()
    except OSError: pass
    return 0

if __name__=="__main__": raise SystemExit(main())

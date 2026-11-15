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

def safe_extract_member(zf, info, target):
    # ZIP entry names may use Windows separators even when processed on POSIX.
    normalized=info.filename.replace("\\","/")
    raw=PurePosixPath(normalized)
    mode=(info.external_attr >> 16) & 0o170000
    if (not normalized or "\\x00" in normalized or raw.is_absolute()
        or ".." in raw.parts or any(":" in part for part in raw.parts)
        or mode == stat.S_IFLNK or info.is_dir()):
        return None
    if info.file_size > MAX_MEMBER_BYTES:
        print(f"  SKIP (over 4 GiB): {info.filename}"); return None
    if any(part.lower() in SKIP_PARTS for part in raw.parts): return None
    if Path(raw.name).suffix.lower() not in ALLOWED_SUFFIXES: return None
    out=target.joinpath(*raw.parts)
    # Avoid duplicate ZIP entries silently replacing an earlier extracted file.
    if out.exists(): return None
    out.parent.mkdir(parents=True,exist_ok=True)
    with zf.open(info) as src, out.open("wb") as dst: shutil.copyfileobj(src,dst)
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
                bad=zf.testzip()
                if bad: raise zipfile.BadZipFile(f"Corrupt member: {bad}")
                for info in zf.infolist():
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

from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, zipfile
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



def copy_dataset_archive(source: Path, target_dir: Path) -> bool:
    """Copy a downloaded dataset ZIP without following links or overwriting data."""
    source = Path(source)
    target_dir = Path(target_dir)
    if source.is_symlink() or not source.is_file():
        raise ValueError("source must be a regular, non-symlink file")
    # A regular-looking file can still be reached through a symlinked parent
    # (for example, Downloads itself may be redirected elsewhere).
    if has_symlink_component(source):
        raise ValueError("source path has a symlinked component")
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("dataset inbox must be an existing, non-symlink directory")
    for component in (target_dir, *target_dir.parents):
        if component.is_symlink():
            raise ValueError("dataset inbox has a symlinked path component")

    target = target_dir / source.name
    if target.exists() or target.is_symlink():
        return False

    created = False
    try:
        # Exclusive creation prevents a concurrent file or symlink from being
        # overwritten after the existence check.
        with source.open("rb") as src, target.open("xb") as dst:
            created = True
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        shutil.copystat(source, target, follow_symlinks=False)
    except Exception:
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise
    return True


def has_symlink_component(path: Path) -> bool:
    """Detect a symlink at any existing component of a local path."""
    candidate = Path(path).absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def ensure_directory_without_symlinks(path: Path) -> None:
    """Create a directory only when its existing path components are not symlinks."""
    candidate = Path(path)
    if has_symlink_component(candidate):
        raise ValueError("directory path contains a symlinked component")
    candidate.mkdir(parents=True, exist_ok=True)
    if has_symlink_component(candidate) or not candidate.is_dir():
        raise ValueError("directory path is not a safe regular directory")


def safe_display_filename(filename: str) -> str:
    """Keep untrusted filenames from injecting terminal control characters."""
    cleaned = "".join(
        character for character in filename
        if character.isprintable() and character not in "\r\n\t"
    )
    return cleaned[:120] or "<unnamed>"


def prepare_staging_directory(staging: Path, destination: Path) -> bool:
    """Create a fresh staging directory without following symlinked parents."""
    for candidate in (staging, destination):
        if candidate.exists() or candidate.is_symlink():
            return False
        for component in candidate.parents:
            if component.is_symlink():
                return False
    staging.mkdir(parents=True, exist_ok=False)
    return True


def publish_staging_directory(staging: Path, destination: Path) -> None:
    """Rename a completed staging tree without following symlinked parents."""
    if has_symlink_component(staging) or has_symlink_component(destination.parent):
        raise ValueError("staging or destination path contains a symlinked component")
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("staging path must be an existing regular directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Import destination already exists: {destination.name}")
    # Both paths live under the repository's models directory, so this is an
    # atomic same-filesystem rename in the normal importer layout. The checks
    # are defense in depth, not a cross-process lock against concurrent mutation.
    os.rename(staging, destination)


def main():
    parser=argparse.ArgumentParser(description="Safely import NOORIX model artifacts from Kaggle ZIPs.")
    parser.add_argument("--scan-downloads",action="store_true",help="Also scan the Windows Downloads folder.")
    parser.add_argument("--downloads-dir",default=str(Path.home()/"Downloads"),help="Downloads folder path.")
    parser.add_argument("--since-hours",type=float,default=2.0,help="Only include Downloads ZIPs modified within this many hours.")
    parser.add_argument("--delete-zips",action="store_true",help="Delete source ZIPs after successful import (not recommended).")
    args=parser.parse_args()
    if not math.isfinite(args.since_hours) or args.since_hours < 0:
        print("--since-hours must be a finite, non-negative number.")
        return 2
    try:
        for directory in (INCOMING, DEST, DATA_INCOMING, TEMP):
            ensure_directory_without_symlinks(directory)
    except (OSError, ValueError) as exc:
        print(f"Unsafe importer directory ({type(exc).__name__}); refusing to continue.")
        return 1
    failures = 0
    archives = {}
    for candidate in INCOMING.glob("*.zip"):
        if has_symlink_component(candidate) or not candidate.is_file():
            print(f"Skipping non-regular or symlinked model ZIP: {safe_display_filename(candidate.name)}")
            failures += 1
            continue
        try:
            archives[candidate.resolve()]=candidate
        except (OSError, RuntimeError):
            print(f"Skipping inaccessible model ZIP: {safe_display_filename(candidate.name)}")
            failures += 1
    if args.scan_downloads:
        downloads=Path(args.downloads_dir).expanduser()
        cutoff=datetime.now().timestamp()-args.since_hours*3600
        if has_symlink_component(downloads):
            print("Downloads path contains a symlinked component; refusing to scan.")
            failures += 1
        elif downloads.is_dir():
            recent=[]
            for p in downloads.glob("*.zip"):
                try:
                    if has_symlink_component(p):
                        print(f"Skipping ZIP with symlinked path: {safe_display_filename(p.name)}")
                        failures += 1
                        continue
                    if p.is_file() and p.stat().st_mtime>=cutoff:
                        recent.append(p)
                except OSError as scan_error:
                    print(
                        f"Skipping unavailable ZIP: {safe_display_filename(p.name)} "
                        f"({type(scan_error).__name__})"
                    )
                    failures += 1
            print(f"Scanning recent ZIP downloads in: {downloads}")
            print(f"Time window: last {args.since_hours:g} hour(s)")
            for p in recent:
                # NOORIX-named archives are trained model packages.
                # Other recent ZIPs are copied intact into the dataset inbox.
                if p.name.upper().startswith("NOORIX_") or p.name.upper().startswith("NOORIX-"):
                    if has_symlink_component(p):
                        print(f"Skipping symlinked model ZIP path: {safe_display_filename(p.name)}")
                        failures += 1
                        continue
                    try:
                        archives[p.resolve()]=p
                    except (OSError, RuntimeError):
                        print(f"Skipping inaccessible model ZIP: {safe_display_filename(p.name)}")
                        failures += 1
                else:
                    try:
                        copied = copy_dataset_archive(p, DATA_INCOMING)
                        if copied:
                            print(f"DATASET ZIP COPIED: {safe_display_filename(p.name)}")
                        else:
                            print(f"Dataset ZIP already present (skip): {safe_display_filename(p.name)}")
                    except (OSError, ValueError) as copy_error:
                        print(
                            f"Dataset ZIP copy skipped: {safe_display_filename(p.name)} "
                            f"({type(copy_error).__name__})"
                        )
                        failures += 1
        else:
            print(f"Downloads folder not found: {safe_display_filename(str(downloads))}")
            failures += 1
    if not archives:
        print(f"No NOORIX model ZIPs found in {INCOMING}; recent dataset ZIPs, if any, are copied to {DATA_INCOMING}")
        return 1 if failures else 0
    archive_order = []
    for candidate in archives.values():
        try:
            if has_symlink_component(candidate) or not candidate.is_file():
                raise OSError("archive is no longer a regular file")
            archive_order.append((candidate.stat().st_mtime, candidate))
        except OSError as exc:
            print(
                f"Skipping unavailable model ZIP: {safe_display_filename(candidate.name)} "
                f"({type(exc).__name__})"
            )
            failures += 1
    for _, archive in sorted(archive_order, key=lambda item: item[0]):
        slug=archive_slug(archive.name)
        staging=TEMP/slug; destination=DEST/slug
        if not prepare_staging_directory(staging, destination):
            print(f"\nExisting staging or destination path found; refusing to overwrite: {archive.name}")
            failures += 1
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
            publish_staging_directory(staging, destination)
            print(f"\\nIMPORTED: {archive.name}")
            for name in imported_files: print("  +",name)
            print("  + import_manifest.json")
            print("  Destination:",destination)
            if args.delete_zips:
                try:
                    archive.unlink()
                    print("  Source ZIP deleted.")
                except OSError as cleanup_error:
                    print(
                        "  IMPORT SUCCEEDED; source ZIP could not be deleted "
                        f"({type(cleanup_error).__name__})."
                    )
            else:
                print("  Source ZIP preserved.")
        except Exception as exc:
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging,ignore_errors=True)
            print(
                f"\nFAILED: {safe_display_filename(archive.name)} "
                f"({type(exc).__name__}); details omitted."
            )
            print("  Original ZIP preserved.")
            failures += 1
    try: TEMP.rmdir()
    except OSError: pass
    return 1 if failures else 0

if __name__=="__main__": raise SystemExit(main())

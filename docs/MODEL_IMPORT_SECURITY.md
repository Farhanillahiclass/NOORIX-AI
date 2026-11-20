# Model archive import security

The Kaggle model importer treats downloaded ZIP files as untrusted input. Importing
an archive only copies candidate artifacts into a quarantine-style local folder;
it does not activate, execute, or trust a model.

## Current extraction controls

- ZIP paths are normalized for Windows separators before validation.
- Absolute paths, parent traversal, empty or dot path components,
  drive/alternate-stream colons, NUL bytes, Windows reserved device names
  (including superscript COM/LPT aliases), trailing spaces/dots, symlinks,
  directories, encrypted entries and unsupported file types are rejected.
- Only the configured artifact suffix allowlist is extracted.
- A single member is limited to 4 GiB; the expanded archive is limited to 8 GiB;
  archives are limited to 20,000 entries.
- Members with a compression ratio above 1,000:1 are rejected.
- Extraction streams in 1 MiB chunks, checks actual bytes against the declared
  size and member limit, and relies on ZIP CRC verification while reading.
- Existing paths and case-insensitive path collisions are rejected, making
  imports more portable between Linux and Windows.
- Archive-derived staging names are sanitized and bounded; manifests record the ZIP filename, not the developer's absolute local source path.
- Existing staging and destination paths are never automatically deleted or overwritten; the importer refuses to proceed if either path already exists.
- Failed imports remove only the fresh staging directory created for that import and preserve the original ZIP.

## Important limitations

These checks reduce common path traversal, overwrite and decompression-bomb risks;
they do not prove that a model file is safe, correct, licensed or compatible.
ONNX/PyTorch and other model formats can contain malicious or malformed content.
Do not load untrusted model artifacts in a privileged environment. Keep imported
models disabled until provenance, hashes, model format, output contract, label
order, license and runtime behavior have been reviewed.

The importer has not yet been validated against a comprehensive adversarial ZIP
corpus. Its new regression tests must be run in the planned final local validation
batch before the controls are treated as verified.


## Integrity verification

New imports store an `integrity` object in `import_manifest.json` with
`algorithm: sha256` and a relative `files` inventory. Each extracted file is
hashed after extraction. This detects later modification; it does not establish
the model's publisher, provenance, safety or license.

Verify an imported package from the repository root:

~~~bash
python scripts/verify_model_integrity.py --root models/imported/ARCHIVE_FOLDER --manifest models/imported/ARCHIVE_FOLDER/import_manifest.json
~~~

Replace `ARCHIVE_FOLDER` with the generated folder name. The offline verifier
rejects unsafe paths (including Windows reserved device names, NUL bytes,
empty/dot components and trailing spaces/dots), duplicate case-insensitive
entries, symlinks in any path component, missing files and checksum mismatches. The combined local validation command automatically checks
imported packages containing this inventory. Older imports without checksums are
reported as warnings; re-import them if integrity evidence is required.

The importer and verifier regression tests must be run in the planned final local
validation batch. A comprehensive adversarial ZIP corpus is still outstanding.

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
  imports more portable between Linux and Windows. A per-archive path index
  avoids rescanning all extracted entries for every ZIP member; ambiguous
  pre-existing case-folded directory aliases are treated as blocked.
- Archive-derived staging names are sanitized and bounded; manifests record the ZIP filename, not the developer's absolute local source path.
- Existing staging and destination paths are never automatically deleted or overwritten; the importer refuses to proceed if either path already exists.
- Staging setup and the extraction helper reject symlinked target directories or ancestors before creating or resolving the extraction root, preventing a redirected staging path.
- Completed imports are published with a same-filesystem directory rename rather than `shutil.move`, avoiding its behavior of nesting a source directory inside a destination that appears concurrently. The destination is checked again immediately before publication.

- Model ZIP discovery in both the incoming folder and Downloads rejects a
  symlinked file or any symlinked parent path before opening the archive.
- Recent dataset ZIPs copied from Downloads are accepted only when the source is
  a regular non-symlink file with no symlinked ancestor (including the Downloads
  directory) and the dataset inbox has no symlinked path components. Destination
  creation uses exclusive mode, so a file or symlink
  created after the initial existence check is not overwritten. Partial copies
  created by the current attempt are removed on failure; the original ZIP is
  preserved. Dataset copy errors report only the exception type.
- Console display of archive filenames strips control characters, and importer
  failures omit raw exception messages so hostile ZIP metadata cannot inject
  terminal control sequences or leak local diagnostic paths.

- Failed imports remove only the fresh staging directory created for that import and preserve the original ZIP. The optional `--delete-zips` flag is applied only after publication succeeds; if source cleanup fails, the importer reports that cleanup issue separately and keeps the successful import status.

## Important limitations

These checks reduce common path traversal, overwrite and decompression-bomb risks;
they do not prove that a model file is safe, correct, licensed or compatible.
ONNX/PyTorch and other model formats can contain malicious or malformed content.
Do not load untrusted model artifacts in a privileged environment. Keep imported
models disabled until provenance, hashes, model format, output contract, label
order, license and runtime behavior have been reviewed.

The importer does not provide a cross-process filesystem lock or eliminate every time-of-check/time-of-use race if another process mutates the model directories concurrently. Avoid concurrent manual changes to staging/destination paths during import. The importer has not yet been validated against a comprehensive adversarial ZIP
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
requires the supplied trusted model root itself to be an existing non-symlink
directory. Manifest reads are bounded to 16 MiB even if the file grows after its
initial size check, and duplicate JSON object keys are rejected throughout the
manifest to prevent ambiguous algorithm or file-entry fields. It also caps the manifest at 16 MiB, 20,000 file entries and 4,096
characters per relative path before hashing artifacts, limiting resource use
from malformed or unexpectedly large inventory files. It rejects unsafe paths (including Windows reserved device names,
NUL bytes, empty/dot components and trailing spaces/dots), duplicate
case-insensitive entries, symlinks in manifest/file path components, missing
files and checksum mismatches. Model files are checked for observable identity, size
and timestamp changes during hashing; a file that changes while being read is an error. The combined local validation command automatically checks
imported packages containing this inventory. Older imports without checksums are
reported as warnings; re-import them if integrity evidence is required.

The importer and verifier regression tests must be run in the planned final local
validation batch. A comprehensive adversarial ZIP corpus is still outstanding.

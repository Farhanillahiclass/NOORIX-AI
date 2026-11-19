# Model archive import security

The Kaggle model importer treats downloaded ZIP files as untrusted input. Importing
an archive only copies candidate artifacts into a quarantine-style local folder;
it does not activate, execute, or trust a model.

## Current extraction controls

- ZIP paths are normalized for Windows separators before validation.
- Absolute paths, parent traversal, drive/alternate-stream colons, NUL bytes,
  Windows reserved device names, trailing spaces/dots, symlinks, directories,
  encrypted entries and unsupported file types are rejected.
- Only the configured artifact suffix allowlist is extracted.
- A single member is limited to 4 GiB; the expanded archive is limited to 8 GiB;
  archives are limited to 20,000 entries.
- Members with a compression ratio above 1,000:1 are rejected.
- Extraction streams in 1 MiB chunks, checks actual bytes against the declared
  size and member limit, and relies on ZIP CRC verification while reading.
- Existing paths and case-insensitive path collisions are rejected, making
  imports more portable between Linux and Windows.
- Failed imports remove the staging directory and preserve the original ZIP.

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

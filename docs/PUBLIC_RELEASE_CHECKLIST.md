# Public Release Security Checklist

This checklist is for the NOORIX-AI repository before changing visibility to Public.

## Current snapshot review

- A pattern scan was performed on the current tracked text files for common GitHub, AWS, Hugging Face, and private-key credentials. No obvious credential matches were found.
- The repository tree was checked for obvious credential filenames such as `.env`, `kaggle.json`, private-key files, and local database files.
- This is a snapshot review, **not a guarantee that every historical commit is clean**. Old commits can still contain files or credentials that are absent from the current branch.

## Before changing visibility

1. Review the complete commit history and old branches/tags for accidentally committed credentials or personal files.
2. Review commit author metadata. Configure GitHub's private/noreply commit email for future commits. Changing that setting does not rewrite existing commits.
3. If any real credential was ever committed, revoke/rotate it first. Removing it from the latest file is not enough; clean the affected Git history before publishing.
4. Confirm that no private datasets, model weights, camera captures, logs, exports, or local configuration files are tracked.
5. Keep `config.json`, `.env`, Kaggle credentials, private keys, local datasets, and imported model artifacts out of Git.
6. After publication, review GitHub security alerts and any secret-scanning findings immediately.

## Repository safeguards added

- `.gitignore` now excludes common local credentials, private keys, local databases, datasets, and imported model artifacts.
- The COCO label metadata file is explicitly allowed so CI can test the detector without committing model weights.
- Model ZIP extraction rejects parent traversal, absolute paths, Windows drive paths, symbolic links/junctions, non-regular entries, unsafe names, compression bombs and duplicate output paths.
- Unit tests cover ZIP extraction safeguards, model-manifest integrity checks, bounded JSON inputs and path redirection checks (latest PR #26 additions still await the final local run).

## Important

A `.gitignore` rule does not remove a file that is already tracked, and it does not erase old versions from Git history. Never publish a repository solely because the current working tree looks clean.


## Current branch review additions

- Runtime configuration and external object-label JSON are bounded to 1 MiB, reject duplicate JSON keys, and reject symbolic links or Windows junctions in path components.
- SSD, YOLOv8 and currency model files plus SSD anchor files must be regular non-linked files.
- Model integrity manifests use bounded JSON parsing, reject duplicate keys and unsafe path aliases, and compare file metadata around SHA-256 hashing.
- Validation, dependency-audit, benchmark and offline-wheelhouse report/input paths reject symlinks and Windows junctions.
- These controls reduce path redirection and malformed-input risk; they do not replace a full historical commit/branch/tag scan or eliminate concurrent filesystem races.

On Windows, path guards also inspect `st_file_attributes` for `FILE_ATTRIBUTE_REPARSE_POINT` (0x400). This detects junction/reparse-point paths on Python 3.11, where `Path.is_junction()` is unavailable. The check is intentionally conservative and rejects other reparse-point types too.

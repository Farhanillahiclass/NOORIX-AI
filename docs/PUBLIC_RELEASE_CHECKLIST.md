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
- Model ZIP extraction rejects parent traversal, absolute paths, Windows drive paths, symbolic links, and duplicate output paths.
- Unit tests cover the ZIP extraction safeguards.

## Important

A `.gitignore` rule does not remove a file that is already tracked, and it does not erase old versions from Git history. Never publish a repository solely because the current working tree looks clean.

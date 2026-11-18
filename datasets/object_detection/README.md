# Object-detection datasets

This directory is a registry and annotation workspace. It intentionally does not bundle third-party image datasets.

## Current state

- Dataset registry: datasets/object_detection/splits/manifest.json
- Annotation contract: COCO-style detection JSON
- Annotation workspace: datasets/object_detection/annotations/
- Validation command: python scripts/validate_detection_dataset.py PATH_TO_ANNOTATIONS.json

Every external dataset must be reviewed before use. Record its canonical source URL, exact version/release, license and attribution requirements, permitted use (including redistribution and commercial use), download date, split policy, class mapping, and checksums. Do not treat Kaggle availability as permission to redistribute.

## Adding a dataset

1. Review the dataset's official terms and license; record the exact version and source.
2. Download it outside Git unless its license explicitly permits repository redistribution and the repository owners approve the size.
3. Convert or map annotations to the documented COCO fields without inventing labels.
4. Add a manifest entry with status pending_validation; include the annotation path, image root, split, source and license metadata.
5. Run the validator, first without and then with --check-files.
6. Review class counts, missing files, split leakage, duplicate images and annotation quality before training.
7. Change the manifest status only after those checks have actually passed.

## Dataset status vocabulary

- candidate: source identified; license and suitability not yet approved.
- license_reviewed: terms recorded and approved for the stated use.
- downloaded: files acquired; integrity and layout not yet validated.
- pending_validation: converted annotations awaiting validation.
- validated: structural and file checks passed; this does not certify label quality.
- ready_for_training: human review, split integrity and project-specific acceptance checks completed.
- rejected: unsuitable, inaccessible, or incompatible; explain why in the entry.

A structurally valid annotation file is not evidence of model accuracy or dataset quality.

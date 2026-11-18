# Object-detection datasets

This directory is a registry and annotation workspace. It intentionally does not bundle third-party image datasets.

## Current state

- Dataset registry: datasets/object_detection/splits/manifest.json
- Annotation contract: COCO-style detection JSON
- Annotation workspace: datasets/object_detection/annotations/
- Registry validation: `python scripts/validate_dataset_manifest.py` (defaults to the project manifest)
- Annotation validation: `python scripts/validate_detection_dataset.py PATH_TO_ANNOTATIONS.json`

Every external dataset must be reviewed before use. Record its canonical source URL, exact version/release, license and attribution requirements, permitted use (including redistribution and commercial use), download date, split policy, class mapping, and checksums. Do not treat Kaggle availability as permission to redistribute.

## Adding a dataset

1. Review the dataset's official terms and license; record the exact version and source.
2. Download it outside Git unless its license explicitly permits repository redistribution and the repository owners approve the size.
3. Convert or map annotations to the documented COCO fields without inventing labels.
4. Add a manifest entry with status pending_validation; include the annotation path, image root, split, source and license metadata.
5. Run the validator, first without an image directory and then with --images-dir PATH_TO_IMAGES.
6. Review class counts, missing files, split leakage, duplicate images and annotation quality before training. Use `validate_split_disjointness` in `noorix.datasets` to check sample IDs across train/validation/test partitions.
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


## Initial dataset candidates

The manifest records four research candidates, all with status `candidate`:

- **COCO 2017** — broad everyday-object benchmark. Its annotations are CC BY 4.0, but the image files are governed by original Flickr terms and rights; these must be reviewed separately ([official terms](https://cocodataset.org/dataset/termsofuse.htm)).
- **LVIS 1.0** — useful for long-tail categories and instance segmentation. Keep its federated annotation semantics and verify exact release and image provenance before conversion ([LVIS project](https://www.lvisdataset.org/)).
- **Objects365** — large object-detection benchmark. The official page states academic-purpose availability; annotations are CC BY 4.0, while images remain subject to Flickr terms and must not be redistributed ([official terms](https://www.objects365.org/download.html)).
- **BDD100K** — driving-scene domain. Its dataset terms allow educational, research and not-for-profit use; commercial use is restricted to BDD/BAIR Commons members and affiliates ([dataset license](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)).

These are source candidates, not approved training datasets. The manifest deliberately records restrictions rather than marking any source as license-reviewed, downloaded, validated or ready for training. Confirm intended use, exact release, image-level rights, attribution and redistribution rules before acquisition. COCO and Objects365 terms distinguish annotations from image copyrights; BDD100K has a separate commercial-use restriction. 


## Annotation adapters (metadata only)

The package exposes explicit adapters from `noorix.datasets`:

- `adapt_coco(payload)`: loss-minimising copy of COCO-style collections.
- `adapt_lvis(payload)`: retains LVIS federated fields (including
  `neg_category_ids` and `not_exhaustive_category_ids`), category frequency,
  segmentation and other source metadata.
- `adapt_objects365(payload)`: accepts only an Objects365 export already
  shaped as `images/categories/annotations`; release-specific schemas must be
  inspected and mapped explicitly rather than guessed.
- `adapt_bdd100k(frames, image_dimensions=...)`: converts BDD100K frame labels
  and `box2d` coordinates. Supply verified image dimensions by frame filename;
  non-box labels are skipped rather than fabricated as detections.

Adapters preserve source labels/metadata and add a `noorix_source` provenance
record. Category ID remapping is opt-in. These are annotation-metadata
converters only: they do not fetch, inspect, resize, or redistribute images,
approve dataset licenses, or establish model quality. Run the COCO validator
on converted output before considering any dataset for training.


## Dataset quality analysis

Use `analyze_coco_quality(payload)` or the
`scripts/analyze_detection_dataset.py` command to produce a deterministic JSON
report containing:

- image/category/annotation totals and image annotation coverage;
- per-category instance and distinct-image counts, including zero-instance classes;
- max-to-min class imbalance ratio (zero-instance classes are separately reported);
- bounding-box width, height and area min/max/mean/median;
- exact duplicate box counts (same image, category and XYWH).

This is descriptive analysis, not a correctness audit or model-accuracy estimate.
Run the COCO structural validator first. Near-duplicate detection, visual review,
and split-level distribution analysis are not included in this initial report.

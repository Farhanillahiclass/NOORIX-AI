import json

from noorix.datasets import validate_coco_dataset


def _write_dataset(tmp_path, *, bbox=None, area=200):
    payload = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 80}],
        "categories": [{"id": 3, "name": "person"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 3,
                "bbox": bbox if bbox is not None else [10, 10, 20, 10],
                "area": area,
                "iscrowd": 0,
            }
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_coco_document_passes(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    assert validate_coco_dataset(annotation_file) == []


def test_missing_image_file_is_reported(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    image_root = tmp_path / "images"
    image_root.mkdir()
    errors = validate_coco_dataset(annotation_file, image_root)
    assert any("does not exist" in error for error in errors)


def test_bbox_outside_image_is_rejected(tmp_path):
    annotation_file = _write_dataset(tmp_path, bbox=[90, 10, 20, 10], area=200)
    errors = validate_coco_dataset(annotation_file)
    assert any("extends beyond image" in error for error in errors)


def test_duplicate_image_ids_are_rejected(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    payload["images"].append(payload["images"][0])
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_coco_dataset(annotation_file)
    assert any("duplicates" in error for error in errors)


def test_unknown_category_reference_is_rejected(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    payload["annotations"][0]["category_id"] = 999
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_coco_dataset(annotation_file)
    assert any("category_id" in error for error in errors)


def test_path_traversal_is_rejected(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "../outside.jpg"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_coco_dataset(annotation_file)
    assert any("safe, non-empty relative" in error for error in errors)


def test_segmentation_area_can_differ_from_bbox_area(tmp_path):
    annotation_file = _write_dataset(tmp_path, area=199)
    assert validate_coco_dataset(annotation_file) == []


def test_area_must_be_positive_and_finite(tmp_path):
    annotation_file = _write_dataset(tmp_path, area=0)
    errors = validate_coco_dataset(annotation_file)
    assert any(".area must be a positive finite number" in error for error in errors)



def test_manifest_candidate_registry_passes(tmp_path):
    from noorix.datasets import validate_dataset_manifest

    manifest = {
        "schema_version": 1,
        "format": "coco_detection",
        "datasets": [{
            "id": "sample-set",
            "name": "Sample set",
            "release": "1",
            "task": "object_detection",
            "source_url": "https://example.org/data",
            "terms_url": "https://example.org/license",
            "license_summary": "Research-only pending verification.",
            "status": "candidate",
            "planned_splits": ["train", "validation"],
            "use_restrictions": ["Verify source terms before use."],
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_dataset_manifest(path) == []


def test_manifest_rejects_duplicate_ids_and_unknown_status(tmp_path):
    from noorix.datasets import validate_dataset_manifest

    item = {
        "id": "same", "name": "Set", "release": "1", "task": "detection",
        "source_url": "https://example.org", "terms_url": "https://example.org/license",
        "license_summary": "Terms pending.", "status": "trained",
        "planned_splits": ["train"], "use_restrictions": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1, "format": "coco_detection", "datasets": [item, item]
    }), encoding="utf-8")
    errors = validate_dataset_manifest(path)
    assert any("duplicates" in error for error in errors)
    assert any(".status" in error for error in errors)


def test_split_leakage_is_reported():
    from noorix.datasets import validate_split_disjointness

    errors = validate_split_disjointness({
        "train": ["a", "b"],
        "validation": ["b", "c"],
        "test": ["d"],
    })
    assert any("leaks across splits" in error for error in errors)


def test_split_duplicate_within_same_partition_is_reported():
    from noorix.datasets import validate_split_disjointness

    errors = validate_split_disjointness({"train": ["a", "a"]})
    assert any("duplicated within split" in error for error in errors)


def test_lvis_adapter_preserves_federated_fields():
    from noorix.datasets import adapt_lvis
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10,
                    "neg_category_ids": [4], "not_exhaustive_category_ids": [5]}],
        "categories": [{"id": 4, "name": "rare", "frequency": "r"}],
        "annotations": [{"id": 2, "image_id": 1, "category_id": 4,
                         "bbox": [1, 1, 2, 2], "segmentation": []}],
    }
    converted = adapt_lvis(payload)
    assert converted["images"][0]["neg_category_ids"] == [4]
    assert converted["images"][0]["not_exhaustive_category_ids"] == [5]
    assert converted["categories"][0]["frequency"] == "r"
    assert converted["annotations"][0]["segmentation"] == []


def test_coco_adapter_category_remap_is_explicit_and_records_it():
    from noorix.datasets import adapt_coco
    payload = {
        "images": [], "categories": [{"id": 7, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 1, 1]}],
    }
    converted = adapt_coco(payload, category_id_map={7: 2})
    assert converted["categories"][0]["id"] == 2
    assert converted["annotations"][0]["category_id"] == 2
    assert converted["noorix_source"]["source_category_ids_preserved"] is False


def test_objects365_adapter_requires_coco_collections():
    import pytest
    from noorix.datasets import adapt_objects365
    with pytest.raises(ValueError, match="images"):
        adapt_objects365({"records": []})


def test_bdd100k_adapter_converts_box_and_preserves_source_label():
    from noorix.datasets import adapt_bdd100k
    frames = [{
        "name": "frame-001.jpg", "videoName": "video-01", "frameIndex": 1,
        "labels": [
            {"category": "car", "attributes": {"occluded": False},
             "box2d": {"x1": 1, "y1": 2, "x2": 6, "y2": 9}},
            {"category": "lane", "poly2d": []},
        ],
    }]
    converted = adapt_bdd100k(frames, image_dimensions={"frame-001.jpg": (20, 20)})
    assert converted["images"][0]["source_frame"]["videoName"] == "video-01"
    assert converted["annotations"][0]["bbox"] == [1, 2, 5, 7]
    assert converted["annotations"][0]["source_label"]["attributes"]["occluded"] is False
    assert [item["name"] for item in converted["categories"]] == ["car"]


def test_bdd100k_adapter_requires_image_dimensions():
    import pytest
    from noorix.datasets import adapt_bdd100k
    with pytest.raises(ValueError, match="Missing valid"):
        adapt_bdd100k([{"name": "missing.jpg", "labels": []}], image_dimensions={})

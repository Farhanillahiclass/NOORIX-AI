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


def test_area_must_match_bbox_area(tmp_path):
    annotation_file = _write_dataset(tmp_path, area=999)
    errors = validate_coco_dataset(annotation_file)
    assert any("does not match bbox" in error for error in errors)

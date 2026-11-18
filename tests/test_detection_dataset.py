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


def test_quality_report_counts_classes_coverage_and_bbox_statistics():
    from noorix.datasets import analyze_coco_quality

    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
            {"id": 3, "file_name": "c.jpg", "width": 100, "height": 80},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "empty-class"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 20]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5]},
        ],
    }
    report = analyze_coco_quality(payload)
    assert report["summary"]["images"] == 3
    assert report["summary"]["images_with_annotations"] == 2
    assert report["summary"]["images_without_annotations"] == 1
    assert report["summary"]["annotation_coverage_percent"] == 66.6667
    assert report["summary"]["classes_without_instances"] == 1
    assert report["summary"]["class_imbalance_ratio_max_to_min"] == 2.0
    assert report["summary"]["duplicate_box_annotations"] == 1
    assert report["bbox_statistics"]["width"]["median"] == 10.0
    assert report["bbox_statistics"]["area"]["count"] == 3


def test_quality_report_handles_empty_dataset_without_division_error():
    from noorix.datasets import analyze_coco_quality

    report = analyze_coco_quality({"images": [], "categories": [], "annotations": []})
    assert report["summary"]["annotation_coverage_percent"] is None
    assert report["summary"]["class_imbalance_ratio_max_to_min"] is None
    assert report["bbox_statistics"]["area"]["count"] == 0


def test_quality_report_rejects_unknown_category_reference():
    import pytest
    from noorix.datasets import analyze_coco_quality

    payload = {
        "images": [{"id": 1}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"image_id": 1, "category_id": 999, "bbox": [0, 0, 1, 1]}],
    }
    with pytest.raises(ValueError, match="unknown category_id"):
        analyze_coco_quality(payload)



def test_split_analysis_reports_counts_and_class_distribution():
    from noorix.datasets import analyze_coco_splits

    train = {
        "images": [
            {"id": 1, "file_name": "train/a.jpg"},
            {"id": 2, "file_name": "train/b.jpg"},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 2, 2]},
        ],
    }
    validation = {
        "images": [{"id": 1, "file_name": "validation/c.jpg"}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 2, 2]},
        ],
    }
    report = analyze_coco_splits({"train": train, "validation": validation})
    assert report["splits"]["train"]["images"] == 2
    assert report["splits"]["train"]["class_distribution"] == {"car": 0, "person": 2}
    assert report["splits"]["validation"]["class_distribution"] == {"car": 1, "person": 0}
    assert report["cross_split_filename_overlap_count"] == 0


def test_split_analysis_detects_case_and_separator_normalized_filename_leakage():
    from noorix.datasets import analyze_coco_splits

    base = {
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }
    train = {**base, "images": [{"id": 1, "file_name": "images\\Person.JPG"}]}
    test = {**base, "images": [{"id": 1, "file_name": "images/person.jpg"}]}
    report = analyze_coco_splits({"train": train, "test": test})
    assert report["cross_split_filename_overlaps"] == [
        {"file_name": "images/person.jpg", "splits": ["test", "train"]}
    ]


def test_split_analysis_rejects_empty_splits_and_missing_filenames():
    import pytest
    from noorix.datasets import analyze_coco_splits

    with pytest.raises(ValueError, match="non-empty mapping"):
        analyze_coco_splits({})
    with pytest.raises(ValueError, match="file_name"):
        analyze_coco_splits({
            "train": {
                "images": [{"id": 1}],
                "categories": [],
                "annotations": [],
            }
        })



def _write_split_json(path, *, file_name="sample.jpg", valid=True):
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5],
             "area": 25, "iscrowd": 0}
        ] if valid else [
            {"id": 1, "image_id": 1, "category_id": 99, "bbox": [1, 1, 5, 5],
             "area": 25, "iscrowd": 0}
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_split_analysis_cli_writes_report_and_flags_filename_overlap(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    train = _write_split_json(tmp_path / "train.json", file_name="images/Person.JPG")
    validation = _write_split_json(tmp_path / "validation.json", file_name="images/person.jpg")
    output = tmp_path / "reports" / "split_report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_detection_splits.py"

    result = subprocess.run(
        [sys.executable, str(script), "--train", str(train),
         "--validation", str(validation), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cross_split_filename_overlap_count"] == 1
    assert "overlaps found: 1" in result.stdout


def test_split_analysis_cli_stops_when_a_split_is_invalid(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    train = _write_split_json(tmp_path / "train.json")
    validation = _write_split_json(tmp_path / "validation.json", valid=False)
    output = tmp_path / "must_not_exist" / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_detection_splits.py"

    result = subprocess.run(
        [sys.executable, str(script), "--train", str(train),
         "--validation", str(validation), "--output", str(output)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 1
    assert "analysis not generated" in result.stdout
    assert not output.exists()


def test_split_analysis_compares_class_percentages_and_spread():
    from noorix.datasets import analyze_coco_splits

    def make_split(prefix, person_count, car_count):
        categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
        annotations = []
        images = []
        for index in range(person_count + car_count):
            image_id = index + 1
            category_id = 1 if index < person_count else 2
            images.append({
                "id": image_id, "file_name": f"{prefix}/{image_id}.jpg",
                "width": 10, "height": 10,
            })
            annotations.append({
                "id": image_id, "image_id": image_id, "category_id": category_id,
                "bbox": [0, 0, 1, 1],
            })
        return {"images": images, "categories": categories, "annotations": annotations}

    report = analyze_coco_splits({
        "train": make_split("train", 3, 1),
        "validation": make_split("validation", 1, 3),
        "test": make_split("test", 0, 2),
    })

    assert report["splits"]["train"]["class_distribution_percentages"] == {
        "car": 25.0, "person": 75.0
    }
    assert report["splits"]["validation"]["class_distribution_percentages"] == {
        "car": 75.0, "person": 25.0
    }
    assert report["splits"]["test"]["class_distribution_percentages"] == {
        "car": 100.0, "person": 0.0
    }
    assert report["class_distribution_comparison"]["person"]["spread_percentage_points"] == 75.0
    assert report["class_distribution_comparison"]["car"]["spread_percentage_points"] == 75.0
    assert report["class_distribution_comparison"]["person"]["instances_by_split"] == {
        "test": 0, "train": 3, "validation": 1
    }


def test_split_analysis_reports_none_percentages_for_empty_split():
    from noorix.datasets import analyze_coco_splits

    categories = [{"id": 1, "name": "person"}]
    populated = {
        "images": [{"id": 1, "file_name": "train/person.jpg", "width": 10, "height": 10}],
        "categories": categories,
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
    }
    empty = {"images": [], "categories": categories, "annotations": []}

    report = analyze_coco_splits({"train": populated, "validation": empty})
    assert report["splits"]["validation"]["class_distribution_percentages"] == {
        "person": None
    }
    assert report["class_distribution_comparison"]["person"]["percentage_by_split"] == {
        "train": 100.0, "validation": None
    }
    assert report["class_distribution_comparison"]["person"]["spread_percentage_points"] is None

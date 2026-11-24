from scripts.benchmark_runtime import percentile, summarize_latencies


def test_percentile_interpolates_and_handles_single_value():
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([7.0], 0.95) == 7.0


def test_percentile_rejects_invalid_inputs():
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.1)


def test_latency_summary_contains_repeatable_statistics():
    summary = summarize_latencies([10.0, 20.0, 30.0])
    assert summary["iterations"] == 3
    assert summary["mean_ms"] == 20.0
    assert summary["median_ms"] == 20.0
    assert summary["p95_ms"] == 29.0
    assert summary["throughput_fps"] == 50.0


def test_latency_summary_rejects_empty_samples():
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        summarize_latencies([])



def test_percentile_rejects_non_finite_values():
    import math
    import pytest

    with pytest.raises(ValueError, match="finite"):
        percentile([1.0, math.inf], 0.5)


def test_latency_summary_rejects_non_positive_or_non_finite_samples():
    import math
    import pytest

    with pytest.raises(ValueError, match="finite positive"):
        summarize_latencies([0.0, 1.0])
    with pytest.raises(ValueError, match="finite positive"):
        summarize_latencies([math.nan])



def test_benchmark_report_path_detects_symlinked_directory(tmp_path):
    import pytest
    from pathlib import Path
    from scripts.benchmark_runtime import _has_symlink_component

    actual = tmp_path / "actual-reports"
    actual.mkdir()
    alias = tmp_path / "reports-link"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    assert _has_symlink_component(alias)
    assert not _has_symlink_component(actual / "new-report")


def test_benchmark_cold_start_duration_must_be_finite_and_positive():
    import math
    import pytest
    from scripts.benchmark_runtime import validate_positive_latency

    assert validate_positive_latency(12.5, "cold-start timing") == 12.5
    for invalid in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite positive"):
            validate_positive_latency(invalid, "cold-start timing")


def test_benchmark_report_directory_uses_suffix_on_collision(tmp_path):
    from scripts.benchmark_runtime import create_unique_report_directory

    output = tmp_path / "reports"
    output.mkdir()
    existing = output / "fixed-stamp"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    created = create_unique_report_directory(output, "fixed-stamp")

    assert created == output / "fixed-stamp-1"
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_benchmark_report_directory_rejects_symlinked_output(tmp_path):
    import pytest
    from scripts.benchmark_runtime import create_unique_report_directory

    external = tmp_path / "external"
    external.mkdir()
    output_link = tmp_path / "reports-link"
    try:
        output_link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        create_unique_report_directory(output_link, "fixed-stamp")


def test_benchmark_report_files_are_created_exclusively(tmp_path):
    import pytest
    from scripts.benchmark_runtime import create_unique_report_directory

    report = create_unique_report_directory(tmp_path, "fixed-stamp")
    target = report / "benchmark-report.json"
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        with target.open("x", encoding="utf-8"):
            pass

    assert target.read_text(encoding="utf-8") == "keep"


def test_benchmark_report_pair_cleans_partial_json_on_markdown_failure(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    from scripts.benchmark_runtime import write_report_pair

    json_path = tmp_path / "benchmark.json"
    markdown_path = tmp_path / "benchmark.md"
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path == markdown_path:
            raise OSError("simulated markdown write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="simulated markdown write failure"):
        write_report_pair(json_path, markdown_path, {"ok": True}, "# benchmark")

    assert not json_path.exists()
    assert not markdown_path.exists()


def test_benchmark_config_reader_accepts_object_and_rejects_non_object(tmp_path):
    import json
    import pytest
    from scripts.benchmark_runtime import load_benchmark_config

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"object_detection": {"backend": "yolov8_onnx"}}), encoding="utf-8")
    assert load_benchmark_config(config)["object_detection"]["backend"] == "yolov8_onnx"
    config.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_benchmark_config(config)


def test_benchmark_config_reader_rejects_oversized_file(tmp_path):
    import pytest
    from scripts.benchmark_runtime import MAX_CONFIG_BYTES, load_benchmark_config

    config = tmp_path / "large-config.json"
    config.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        load_benchmark_config(config)


def test_benchmark_metadata_text_is_single_line_and_bounded():
    from scripts.benchmark_runtime import safe_metadata_text

    value = "image" + chr(27) + "\nname|`field`"
    cleaned = safe_metadata_text(value, limit=12)
    assert chr(27) not in cleaned
    assert "\n" not in cleaned
    assert "|" not in cleaned or "\\|" in cleaned
    assert len(cleaned) <= 12



def test_benchmark_output_guard_rejects_windows_junction(tmp_path, monkeypatch):
    from scripts.benchmark_runtime import _has_symlink_component

    junction = tmp_path / "junction"
    junction.mkdir()
    path_type = type(tmp_path)
    monkeypatch.setattr(
        path_type,
        "is_junction",
        lambda path: path == junction,
        raising=False,
    )

    assert _has_symlink_component(junction / "reports") is True



def test_benchmark_config_reader_rejects_symlinked_file(tmp_path):
    import pytest
    from scripts.benchmark_runtime import load_benchmark_config

    target = tmp_path / "real.json"
    target.write_text('{"object_detection": {}}', encoding="utf-8")
    linked = tmp_path / "config.json"
    try:
        linked.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_benchmark_config(linked)


def test_benchmark_config_reader_rejects_symlinked_parent(tmp_path):
    import pytest
    from scripts.benchmark_runtime import load_benchmark_config

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "config.json"
    target.write_text('{"object_detection": {}}', encoding="utf-8")
    linked_dir = tmp_path / "linked"
    try:
        linked_dir.symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        load_benchmark_config(linked_dir / "config.json")


def test_benchmark_config_reader_rejects_non_regular_file(tmp_path):
    import pytest
    from scripts.benchmark_runtime import load_benchmark_config

    directory = tmp_path / "config.json"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        load_benchmark_config(directory)

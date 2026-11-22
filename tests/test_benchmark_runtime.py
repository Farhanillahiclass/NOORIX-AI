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

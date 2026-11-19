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

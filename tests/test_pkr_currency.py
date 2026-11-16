import json
import pytest
from noorix.pkr_currency import PytorchPKRCurrencyEstimator

def test_currency_estimator_rejects_invalid_input_size():
    with pytest.raises(ValueError, match="input_size"):
        PytorchPKRCurrencyEstimator("model.pth", "labels.json", input_size=0)

def test_currency_estimator_requires_labels_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="labels"):
        PytorchPKRCurrencyEstimator._read_labels(tmp_path / "missing.json")

def test_currency_estimator_validates_duplicate_labels(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"classes": ["100_front", "100_front"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        PytorchPKRCurrencyEstimator._read_labels(labels)

def test_currency_estimator_requires_non_empty_classes(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"classes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        PytorchPKRCurrencyEstimator._read_labels(labels)

@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("100_front", ("100", "front")),
        ("5000_back", ("5000", "back")),
        ("50_FRONT", ("50", "front")),
    ],
)
def test_currency_estimator_parses_pkr_class(label, expected):
    assert PytorchPKRCurrencyEstimator._describe_class(label) == expected

def test_currency_estimator_rejects_unsupported_class():
    with pytest.raises(ValueError, match="Unsupported PKR class"):
        PytorchPKRCurrencyEstimator._describe_class("unknown")

def test_currency_estimator_requires_model_file(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"classes": ["100_front"]}), encoding="utf-8")
    estimator = PytorchPKRCurrencyEstimator(tmp_path / "missing.pth", labels)
    with pytest.raises(FileNotFoundError, match="Currency model"):
        estimator.load()

def test_currency_estimator_rejects_none_image():
    estimator = PytorchPKRCurrencyEstimator("model.pth", "labels.json")
    with pytest.raises(ValueError, match="must not be None"):
        estimator.predict(None)

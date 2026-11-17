import json
import sys
import types

import numpy as np
import pytest

from noorix.whisper_asr import WhisperASR, create_whisper_asr


def test_whisper_asr_validates_language_and_token_limit():
    with pytest.raises(ValueError, match="language"):
        WhisperASR("model", language="punjabi")
    with pytest.raises(ValueError, match="max_new_tokens"):
        WhisperASR("model", max_new_tokens=0)


def test_whisper_asr_requires_local_model_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local Whisper model directory"):
        WhisperASR(tmp_path / "missing").load()


def test_audio_input_accepts_array_tuple_and_mapping():
    data, rate = WhisperASR._audio_input((np.ones(1600), 16000))
    assert data.shape == (1600,)
    assert rate == 16000
    data, rate = WhisperASR._audio_input({"audio": np.ones(1600), "sampling_rate": 16000})
    assert data.shape == (1600,)
    assert rate == 16000


def test_audio_input_converts_stereo_to_mono():
    stereo = np.ones((1600, 2), dtype=np.float32)
    data, rate = WhisperASR._audio_input(stereo, 16000)
    assert data.shape == (1600,)
    assert rate == 16000


def test_audio_input_rejects_missing_rate_empty_and_nonfinite():
    with pytest.raises(ValueError, match="sampling_rate is required"):
        WhisperASR._audio_input(np.ones(100), None)
    with pytest.raises(ValueError, match="non-empty"):
        WhisperASR._audio_input(np.array([]), 16000)
    with pytest.raises(ValueError, match="non-finite"):
        WhisperASR._audio_input(np.array([np.nan]), 16000)


def test_audio_input_resamples_to_16khz():
    pytest.importorskip("scipy")
    data, rate = WhisperASR._audio_input(np.ones(32000), 32000)
    assert rate == 16000
    assert abs(len(data) - 16000) <= 1


def test_factory_resolves_model_path_relative_to_config(tmp_path):
    model = tmp_path / "models" / "whisper"
    model.mkdir(parents=True)
    adapter = create_whisper_asr(
        {"model_path": "models/whisper", "language": "urdu"},
        tmp_path / "config.json",
    )
    assert adapter.model_path == model
    assert adapter.language == "urdu"
    assert adapter.processor_path == model


def test_transcribe_uses_cpu_and_handles_transformers_v5_tokenizer_config(monkeypatch, tmp_path):
    model_dir = tmp_path / "whisper"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<|ur|>", "<|transcribe|>"]}),
        encoding="utf-8",
    )
    calls = {}

    class FakeTokenizer:
        def set_prefix_tokens(self, **kwargs):
            calls["prefix"] = kwargs

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["processor_load"] = (path, kwargs)
            return cls()
        def __call__(self, samples, **kwargs):
            calls["sample_count"] = len(samples)
            calls["processor_call"] = kwargs
            return types.SimpleNamespace(input_features=FakeTensor())
        def batch_decode(self, generated, **kwargs):
            return ["  سلام دنیا  "]

    class FakeTensor:
        def to(self, device):
            calls["device"] = device
            return self

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_load"] = (path, kwargs)
            return cls()
        def to(self, device):
            calls["model_device"] = device
            return self
        def eval(self):
            calls["eval"] = True
        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return ["tokens"]

    class FakeTorch:
        float32 = "float32"
        class inference_mode:
            def __enter__(self): pass
            def __exit__(self, *args): pass

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperProcessor = FakeProcessor
    fake_transformers.WhisperForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    asr = WhisperASR(model_dir, language="urdu")
    assert asr.transcribe(np.ones(1600), 16000) == "سلام دنیا"
    processor_kwargs = calls["processor_load"][1]
    assert processor_kwargs["local_files_only"] is True
    assert processor_kwargs["extra_special_tokens"] == {}
    assert calls["model_load"][1]["local_files_only"] is True
    assert calls["model_device"] == "cpu"
    assert calls["prefix"] == {"language": "urdu", "task": "transcribe"}
    assert calls["device"] == "cpu"


def test_factory_supports_separate_processor_directory(tmp_path):
    model = tmp_path / "models" / "model"
    processor = tmp_path / "models" / "processor"
    model.mkdir(parents=True)
    processor.mkdir(parents=True)
    adapter = create_whisper_asr(
        {"model_path": "models/model", "processor_path": "models/processor"},
        tmp_path / "config.json",
    )
    assert adapter.model_path == model
    assert adapter.processor_path == processor


def test_whisper_asr_requires_processor_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    adapter = WhisperASR(model, processor_path=tmp_path / "missing_processor")
    with pytest.raises(FileNotFoundError, match="processor directory"):
        adapter.load()

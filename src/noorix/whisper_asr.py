"""Offline Whisper speech-to-text adapter for local Urdu/English models.

Model and processor files must already exist locally. No network downloads are
permitted by this adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class WhisperASR:
    SAMPLE_RATE = 16000

    def __init__(self, model_path: str | Path, language: str = "urdu", max_new_tokens: int = 128):
        self.model_path = Path(model_path)
        if language not in {"urdu", "english"}:
            raise ValueError("language must be 'urdu' or 'english'")
        if not 1 <= int(max_new_tokens) <= 448:
            raise ValueError("max_new_tokens must be between 1 and 448")
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._processor = None
        self._torch = None

    def load(self) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local Whisper model directory not found: {self.model_path}")
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Whisper ASR requires the optional extra: pip install -e '.[asr-whisper]'"
            ) from exc
        self._processor = WhisperProcessor.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model = WhisperForConditionalGeneration.from_pretrained(
            str(self.model_path), local_files_only=True, torch_dtype=torch.float32
        )
        self._model.to("cpu")
        self._model.eval()
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "set_prefix_tokens"):
            tokenizer.set_prefix_tokens(language=self.language, task="transcribe")
        self._torch = torch

    @staticmethod
    def _audio_input(audio: Any, sampling_rate: int | None):
        if isinstance(audio, Mapping):
            sampling_rate = audio.get("sampling_rate", sampling_rate)
            audio = audio.get("audio")
        if isinstance(audio, (str, Path)):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError("Audio file input requires soundfile from the asr-whisper extra") from exc
            samples, file_rate = sf.read(str(audio), dtype="float32", always_2d=True)
            sampling_rate = int(file_rate)
            audio = samples.mean(axis=1)
        elif isinstance(audio, tuple) and len(audio) == 2:
            audio, tuple_rate = audio
            sampling_rate = int(tuple_rate)
        if sampling_rate is None:
            raise ValueError("sampling_rate is required for array audio")
        try:
            rate = int(sampling_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("sampling_rate must be a positive integer") from exc
        if rate <= 0:
            raise ValueError("sampling_rate must be a positive integer")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 2:
            # Accept samples-by-channels or channels-by-samples.
            if samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
                samples = samples.mean(axis=0)
            else:
                samples = samples.mean(axis=1)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("audio must be a non-empty mono or multi-channel waveform")
        if not np.isfinite(samples).all():
            raise ValueError("audio contains non-finite samples")
        if rate != WhisperASR.SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("Audio resampling requires scipy from the asr-whisper extra") from exc
            from math import gcd
            divisor = gcd(rate, WhisperASR.SAMPLE_RATE)
            samples = resample_poly(
                samples, WhisperASR.SAMPLE_RATE // divisor, rate // divisor
            ).astype(np.float32, copy=False)
        return np.ascontiguousarray(samples), WhisperASR.SAMPLE_RATE

    def transcribe(self, audio: Any, sampling_rate: int | None = None) -> str:
        samples, rate = self._audio_input(audio, sampling_rate)
        if self._model is None or self._processor is None or self._torch is None:
            self.load()
        inputs = self._processor(
            samples, sampling_rate=rate, return_tensors="pt"
        )
        features = inputs.input_features.to("cpu")
        with self._torch.inference_mode():
            generated = self._model.generate(
                input_features=features, max_new_tokens=self.max_new_tokens
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return str(text).strip()


def create_whisper_asr(config: dict, config_path: str | Path = "config.json") -> WhisperASR:
    from noorix.paths import resolve_config_path
    model_value = config.get("model_path")
    if not model_value:
        raise ValueError("Whisper ASR requires model_path")
    return WhisperASR(
        resolve_config_path(model_value, config_path),
        language=str(config.get("language", "urdu")),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
    )

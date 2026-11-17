"""Concrete NOORIX agents backed by existing local adapters."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any, Mapping
from noorix.agents.base import AgentResult, AgentState
from noorix.capabilities import capability_status
from noorix.currency import CurrencyEstimator
from noorix.pkr_currency import PytorchPKRCurrencyEstimator
from noorix.objects import create_object_detector
from noorix.ocr import PrintedEnglishOCR
from noorix.speech import SpeechOutput
from noorix.whisper_asr import create_whisper_asr
from noorix.paths import resolve_config_path

class OCRAgent:
    name = "ocr"
    description = "Extract printed page text using configured Tesseract languages."
    def __init__(self, adapter: PrintedEnglishOCR | None = None):
        self.adapter = adapter or PrintedEnglishOCR()
    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "A camera frame is required.")
        text = self.adapter.read(payload)
        return AgentResult(self.name, AgentState.COMPLETED,
                           text or "No clear printed text detected.", data=text)

class ObjectDetectionAgent:
    name = "object_detection"
    description = "Detect supported objects and report image-relative positions."
    def __init__(self, config: Mapping[str, Any], adapter: Any | None = None, config_path: str | Path = "config.json"):
        self.config, self.adapter = dict(config), adapter
        self.config_path = Path(config_path)
    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "A camera frame is required.")
        if self.adapter is None:
            if not self.config.get("model_path"):
                return AgentResult(self.name, AgentState.NEEDS_SETUP,
                                   "Object detection model is not configured.")
            try:
                self.adapter = create_object_detector(self.config, self.config_path)
            except (OSError, ValueError, TypeError) as exc:
                return AgentResult(self.name, AgentState.NEEDS_SETUP,
                                   f"Object detection setup error: {exc}")
        detections = self.adapter.detect(payload)
        if not detections:
            return AgentResult(self.name, AgentState.COMPLETED, "No supported objects detected.",
                data=[], warnings=("A missing detection does not mean the area is clear.",))
        lines = [f"{x['label']} — image {x['position']} ({x['confidence']:.0%})" for x in detections]
        return AgentResult(self.name, AgentState.COMPLETED, "\n".join(lines), data=detections,
            warnings=("Positions describe the image only, not distance or safe navigation.",))

class CurrencyAgent:
    name = "currency"
    description = "Estimate Pakistani banknote denomination and side using a configured model."
    def __init__(self, config: Mapping[str, Any], adapter: Any | None = None, config_path: str | Path = "config.json"):
        self.config, self.adapter = dict(config), adapter
        self.config_path = Path(config_path)
    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "A camera frame is required.")
        if self.adapter is None:
            backend = self.config.get("backend", "onnx")
            model_path = self.config.get("model_path")
            if not model_path:
                return AgentResult(self.name, AgentState.NEEDS_SETUP, "Currency model path is not configured.")
            resolved_model = resolve_config_path(model_path, self.config_path)
            if backend == "pytorch_mobilenet_v2":
                labels_path = self.config.get("labels_path")
                if not labels_path:
                    return AgentResult(self.name, AgentState.NEEDS_SETUP, "PyTorch currency labels_path is not configured.")
                self.adapter = PytorchPKRCurrencyEstimator(
                    resolved_model,
                    resolve_config_path(labels_path, self.config_path),
                    input_size=int(self.config.get("input_size", 224)),
                )
            elif backend == "onnx":
                labels = self.config.get("labels")
                if not labels:
                    return AgentResult(self.name, AgentState.NEEDS_SETUP, "ONNX currency denomination labels are not configured.")
                self.adapter = CurrencyEstimator(resolved_model, labels, input_size=int(self.config.get("input_size", 224)))
            else:
                return AgentResult(self.name, AgentState.NEEDS_SETUP, f"Unsupported currency backend: {backend}")
        prediction = self.adapter.predict(payload)
        confidence = float(prediction["confidence"])
        min_confidence = float(self.config.get("min_confidence", 0.60))
        if not 0.0 <= min_confidence <= 1.0:
            return AgentResult(
                self.name, AgentState.NEEDS_SETUP,
                "Currency min_confidence must be between 0.0 and 1.0.",
            )
        if not 0.0 <= confidence <= 1.0:
            return AgentResult(
                self.name, AgentState.ERROR,
                "Currency model returned an invalid confidence score.",
            )
        if confidence < min_confidence:
            return AgentResult(
                self.name,
                AgentState.COMPLETED,
                f"Low confidence ({confidence:.0%}; minimum {min_confidence:.0%}). "
                "No reliable denomination detected. Reposition the note and try again.",
                data={"recognized": False, "confidence": confidence,
                      "minimum_confidence": min_confidence},
                warnings=(
                    "The model's top class is not reliable enough to report as a denomination.",
                    "This estimate does not authenticate banknotes.",
                ),
            )
        denomination = prediction["denomination"]
        side = prediction.get("side")
        side_text = f", {side} side" if side else ""
        return AgentResult(
            self.name, AgentState.COMPLETED,
            f"Possible denomination: {denomination} PKR{side_text} (score {confidence:.0%}).",
            data={**prediction, "recognized": True, "minimum_confidence": min_confidence},
            warnings=("This is an estimate, not banknote authentication. Check the note independently.",),
        )

class SpeechAgent:
    name = "speech"
    description = "Speak supplied text using an installed local system voice."
    def __init__(self, adapter: SpeechOutput | None = None):
        self.adapter = adapter or SpeechOutput()
    def run(self, payload: Any = None) -> AgentResult:
        if not isinstance(payload, str) or not payload.strip():
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "Non-empty text is required.")
        self.adapter.speak(payload)
        return AgentResult(self.name, AgentState.COMPLETED, "Speech playback finished.")

class WhisperASRAgent:
    name = "speech_to_text"
    description = "Transcribe local Urdu or English audio with a downloaded Whisper model."
    def __init__(self, config: Mapping[str, Any], adapter: Any | None = None, config_path: str | Path = "config.json"):
        self.config, self.adapter = dict(config), adapter
        self.config_path = Path(config_path)
    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "An audio file or waveform is required.")
        if self.adapter is None:
            if not self.config.get("model_path"):
                return AgentResult(self.name, AgentState.NEEDS_SETUP, "Whisper ASR model_path is not configured.")
            try:
                self.adapter = create_whisper_asr(self.config, self.config_path)
            except (OSError, ValueError, TypeError) as exc:
                return AgentResult(self.name, AgentState.NEEDS_SETUP, f"Whisper ASR setup error: {exc}")
        if isinstance(payload, Mapping) and "audio" not in payload:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "Audio mapping must contain an 'audio' field.")
        transcript = self.adapter.transcribe(payload)
        if not transcript:
            return AgentResult(self.name, AgentState.COMPLETED, "No clear speech was transcribed.", data="")
        return AgentResult(self.name, AgentState.COMPLETED, transcript, data={"transcript": transcript},
            warnings=("Speech recognition can make mistakes; verify important names, numbers and instructions.",))

class SafetyAgent:
    name = "safety"
    description = "Attach conservative warnings; never infer that an area is safe."
    def run(self, payload: Any = None) -> AgentResult:
        results = payload if isinstance(payload, (list, tuple)) else []
        warnings = [
            "NOORIX is not a replacement for a white cane, guide dog or trained mobility support.",
            "Image-relative positions are not distance, obstacle avoidance or navigation guidance.",
            "Currency predictions are estimates only and do not authenticate banknotes.",
        ]
        failed = [item.agent for item in results if not item.successful]
        if failed:
            warnings.append("Some specialist tasks did not complete: " + ", ".join(failed))
        return AgentResult(self.name, AgentState.COMPLETED,
            "Safety review attached to the current session.", data={"reviewed": len(results)},
            warnings=tuple(warnings))

class SystemHealthAgent:
    name = "system_health"
    description = "Report installed adapters and missing runtime/model requirements."
    def __init__(self, config_path: str | Path = "config.json"):
        self.config_path = Path(config_path)
    def run(self, payload: Any = None) -> AgentResult:
        status = capability_status()
        config_exists = self.config_path.is_file()
        status["model_configuration"] = "present" if config_exists else "config.json_missing"
        status["tesseract_executable"] = "installed" if shutil.which("tesseract") else "not_found"
        status["model_files"] = self._model_file_status() if config_exists else "not_checked_without_config"
        return AgentResult(self.name, AgentState.COMPLETED,
                           "Local system readiness collected.", data=status)
    def _model_file_status(self) -> dict[str, str]:
        import json
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"config": f"invalid: {exc}"}
        output = {}
        for key in ("object_detection", "currency", "asr"):
            section = config.get(key, {})
            if not isinstance(section, dict):
                output[key] = "invalid_configuration"
                continue
            model_path = section.get("model_path")
            if not model_path:
                output[key] = "not_configured"
                continue
            model = resolve_config_path(model_path, self.config_path)
            if key == "asr":
                if not model.is_dir():
                    output[key] = "model_path_not_directory"
                    continue
                processor_value = section.get("processor_path", model_path)
                processor = resolve_config_path(processor_value, self.config_path)
                if not processor.is_dir():
                    output[key] = "processor_path_missing"
                    continue
                has_weights = any((model / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
                has_model_config = (model / "config.json").is_file()
                has_tokenizer = (processor / "tokenizer.json").is_file()
                has_processor_config = any((processor / name).is_file() for name in ("preprocessor_config.json", "processor_config.json"))
                if not has_weights or not has_model_config:
                    output[key] = "model_files_incomplete"
                elif not has_tokenizer or not has_processor_config:
                    output[key] = "processor_files_incomplete"
                else:
                    output[key] = "available"
                continue
            if not model.is_file():
                output[key] = "model_file_missing"
                continue
            if key == "currency" and section.get("backend") == "pytorch_mobilenet_v2":
                labels_path = section.get("labels_path")
                if not labels_path:
                    output[key] = "labels_path_not_configured"
                    continue
                labels = resolve_config_path(labels_path, self.config_path)
                output[key] = "model_available; labels_available" if labels.is_file() else "model_available; labels_file_missing"
            else:
                output[key] = "available"
        return output

"""Concrete NOORIX agents backed by existing local adapters."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any, Mapping
from noorix.agents.base import AgentResult, AgentState
from noorix.capabilities import capability_status
from noorix.currency import CurrencyEstimator
from noorix.objects import create_object_detector
from noorix.ocr import PrintedEnglishOCR
from noorix.speech import SpeechOutput
from noorix.paths import load_json_document, require_regular_file, resolve_config_path
from noorix.privacy import safe_error_summary

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
                                   safe_error_summary(exc, "Object detection setup failed"))
        detections = self.adapter.detect(payload)
        if not detections:
            return AgentResult(self.name, AgentState.COMPLETED, "No supported objects detected.",
                data=[], warnings=("A missing detection does not mean the area is clear.",))
        lines = [f"{x['label']} — image {x['position']} ({x['confidence']:.0%})" for x in detections]
        return AgentResult(self.name, AgentState.COMPLETED, "\n".join(lines), data=detections,
            warnings=("Positions describe the image only, not distance or safe navigation.",))

class CurrencyAgent:
    name = "currency"
    description = "Estimate Pakistani banknote denomination using a configured model."
    def __init__(self, config: Mapping[str, Any], adapter: CurrencyEstimator | None = None, config_path: str | Path = "config.json"):
        self.config, self.adapter = dict(config), adapter
        self.config_path = Path(config_path)
    def run(self, payload: Any = None) -> AgentResult:
        if payload is None:
            return AgentResult(self.name, AgentState.NEEDS_SETUP, "A camera frame is required.")
        if self.adapter is None:
            model_path, labels = self.config.get("model_path"), self.config.get("labels")
            if not model_path or not labels:
                return AgentResult(self.name, AgentState.NEEDS_SETUP,
                    "Currency model and verified denomination labels are not configured.")
            self.adapter = CurrencyEstimator(resolve_config_path(model_path, self.config_path), labels,
                input_size=int(self.config.get("input_size", 224)))
        prediction = self.adapter.predict(payload)
        confidence = float(prediction["confidence"])
        return AgentResult(self.name, AgentState.COMPLETED,
            f"Possible denomination: {prediction['denomination']} (score {confidence:.0%}).",
            data=prediction,
            warnings=("This is an estimate, not banknote authentication. Check the note independently.",))

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
        model_files = self._model_file_status()
        if model_files.get("config") == "missing":
            status["model_configuration"] = "config.json_missing"
            status["model_files"] = "not_checked_without_config"
        else:
            status["model_configuration"] = "unsafe_or_invalid" if "config" in model_files else "present"
            status["model_files"] = model_files
        status["tesseract_executable"] = "installed" if shutil.which("tesseract") else "not_found"
        return AgentResult(self.name, AgentState.COMPLETED,
                           "Local system readiness collected.", data=status)

    def _model_file_status(self) -> dict[str, str]:
        try:
            config = load_json_document(self.config_path)
            if not isinstance(config, dict):
                raise ValueError("configuration root must be an object")
        except FileNotFoundError:
            return {"config": "missing"}
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            return {"config": safe_error_summary(exc, "Configuration unreadable")}
        output = {}
        for key in ("object_detection", "currency"):
            section = config.get(key, {})
            model_path = section.get("model_path") if isinstance(section, dict) else None
            if not model_path:
                output[key] = "not_configured"
            else:
                try:
                    require_regular_file(resolve_config_path(model_path, self.config_path), label=f"{key} model")
                    output[key] = "available"
                except FileNotFoundError:
                    output[key] = "file_missing"
                except (OSError, TypeError, ValueError):
                    output[key] = "unsafe_or_invalid_path"
        return output

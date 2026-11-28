"""Concrete NOORIX agents backed by existing local adapters."""
from __future__ import annotations
import math
import numbers
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
        try:
            text = self.adapter.read(payload)
        except Exception as exc:
            return AgentResult(
                self.name, AgentState.ERROR,
                safe_error_summary(exc, "OCR task failed"),
                warnings=("Text recognition did not complete; do not rely on partial output.",),
            )
        if text is not None and not isinstance(text, str):
            return AgentResult(
                self.name, AgentState.ERROR,
                "OCR task failed (TypeError).",
                warnings=("Text recognition returned an invalid result; do not rely on it.",),
            )
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
        try:
            detections = self.adapter.detect(payload)
            if not isinstance(detections, (list, tuple)):
                raise TypeError("object detector returned an invalid collection")
            normalized_detections = []
            for detection in detections:
                if not isinstance(detection, Mapping):
                    raise TypeError("object detector returned an invalid detection")
                label = detection.get("label")
                position = detection.get("position")
                confidence_value = detection.get("confidence")
                if (
                    not isinstance(label, str) or not label.strip()
                    or not isinstance(position, str) or not position.strip()
                    or isinstance(confidence_value, bool)
                    or not isinstance(confidence_value, numbers.Real)
                ):
                    raise TypeError("object detector returned invalid detection fields")
                confidence_value = float(confidence_value)
                if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                    raise ValueError("object detector returned an invalid confidence")
                # Snapshot adapter-owned mappings while still inside the error boundary.
                normalized = dict(detection)
                normalized.update(
                    label=label.strip(),
                    position=position.strip(),
                    confidence=confidence_value,
                )
                normalized_detections.append(normalized)
            detections = normalized_detections
        except Exception as exc:
            return AgentResult(
                self.name, AgentState.ERROR,
                safe_error_summary(exc, "Object detection failed"),
                warnings=("Object detection did not complete; do not assume the scene is clear.",),
            )
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
            input_size = self.config.get("input_size", 224)
            if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
                return AgentResult(
                    self.name,
                    AgentState.NEEDS_SETUP,
                    "Currency input_size must be a positive integer.",
                )
            try:
                resolved_model_path = resolve_config_path(model_path, self.config_path)
                self.adapter = CurrencyEstimator(
                    resolved_model_path, labels, input_size=input_size
                )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                return AgentResult(
                    self.name,
                    AgentState.NEEDS_SETUP,
                    safe_error_summary(exc, "Currency model setup failed"),
                )
        try:
            prediction = self.adapter.predict(payload)
            if not isinstance(prediction, Mapping):
                raise TypeError("currency adapter returned an invalid prediction")
            denomination = prediction["denomination"]
            raw_confidence = prediction["confidence"]
            if not isinstance(denomination, str) or not denomination.strip():
                raise ValueError("currency adapter returned an invalid denomination")
            # Do not coerce booleans or numeric strings into plausible scores.
            if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, numbers.Real):
                raise TypeError("currency confidence must be a real number")
            confidence = float(raw_confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("currency confidence is outside the expected range")
            # Return a stable snapshot rather than an adapter-owned mutable mapping.
            prediction = dict(prediction)
            prediction["denomination"] = denomination.strip()
            prediction["confidence"] = confidence
        except Exception as exc:
            return AgentResult(
                self.name, AgentState.ERROR,
                safe_error_summary(exc, "Currency prediction failed"),
                warnings=("Currency estimation did not complete; do not rely on a denomination.",),
            )
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
        try:
            self.adapter.speak(payload)
        except Exception as exc:
            return AgentResult(
                self.name, AgentState.ERROR,
                safe_error_summary(exc, "Speech playback failed"),
                warnings=("No successful speech playback was confirmed.",),
            )
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
        status: dict[str, Any] = {}
        failures: list[str] = []

        try:
            capabilities = capability_status()
            if not isinstance(capabilities, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in capabilities.items()
            ):
                raise TypeError("capability inventory returned an invalid result")
            status.update(capabilities)
        except Exception as exc:
            status["capability_inventory"] = safe_error_summary(
                exc, "Capability inventory unavailable"
            )
            failures.append("capability_inventory")

        try:
            model_files = self._model_file_status()
            if not isinstance(model_files, dict):
                raise TypeError("model status returned an invalid result")
            if model_files.get("config") == "missing":
                status["model_configuration"] = "config.json_missing"
                status["model_files"] = "not_checked_without_config"
            else:
                status["model_configuration"] = (
                    "unsafe_or_invalid" if "config" in model_files else "present"
                )
                status["model_files"] = model_files
                if "config" in model_files:
                    failures.append("model_configuration")
                for model_name, model_state in model_files.items():
                    if model_state in ("unsafe_or_invalid_path", "invalid_configuration_section"):
                        failures.append(model_name + "_model_configuration")
        except Exception as exc:
            status["model_configuration"] = safe_error_summary(
                exc, "Model readiness check failed"
            )
            status["model_files"] = "not_checked_after_error"
            failures.append("model_configuration")

        try:
            status["tesseract_executable"] = (
                "installed" if shutil.which("tesseract") else "not_found"
            )
        except Exception as exc:
            status["tesseract_executable"] = safe_error_summary(
                exc, "Tesseract lookup failed"
            )
            failures.append("tesseract_lookup")

        if failures:
            return AgentResult(
                self.name,
                AgentState.ERROR,
                "Local system readiness is incomplete.",
                data=status,
                warnings=("Some readiness checks failed: " + ", ".join(failures),),
            )
        return AgentResult(
            self.name,
            AgentState.COMPLETED,
            "Local system readiness collected.",
            data=status,
        )

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
            if key in config and not isinstance(config[key], dict):
                output[key] = "invalid_configuration_section"
                continue
            section = config.get(key, {})
            model_path = section.get("model_path")
            if model_path is None or model_path == "":
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

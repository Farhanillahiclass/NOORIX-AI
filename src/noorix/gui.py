"""Tk desktop interface for webcam preview and explicitly requested analysis."""
from __future__ import annotations

import base64
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


class NoorixApp:
    def __init__(self, root: tk.Tk, camera_index: int, config_path: Path):
        import cv2
        from noorix.camera import WebcamCapture

        self.cv2 = cv2
        self.WebcamCapture = WebcamCapture
        self.root = root
        self.camera_index = camera_index
        self.camera = None
        self.current_frame = None
        self.photo = None
        self.after_id = None
        self.detector = None
        self.currency = None
        self.config = self._read_config(config_path)

        root.title("NOORIX AI — See Beyond. Understand More.")
        root.geometry("980x700")
        root.minsize(720, 540)
        root.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="NOORIX AI", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(header, text="See Beyond. Understand More.  |  Powered by MRS").pack(anchor="w")

        controls = ttk.Frame(root, padding=(12, 4))
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Start camera", command=self.start_camera)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop camera", command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Read English text", command=self.read_text).pack(side="left", padx=4)
        ttk.Button(controls, text="Analyze objects / currency", command=self.analyze).pack(side="left", padx=4)
        ttk.Button(controls, text="Speak result", command=self.speak_result).pack(side="left", padx=4)

        self.preview = ttk.Label(root, text="Camera is off. Select Start camera.", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=12, pady=8)
        self.status = tk.StringVar(value="Ready. No image is saved automatically.")
        ttk.Label(root, textvariable=self.status, padding=10, wraplength=900).pack(fill="x")
        ttk.Label(
            root,
            text="Image-relative positions are not distance or safe-navigation guidance. Do not replace a white cane or guide dog.",
            padding=(12, 0, 12, 12), wraplength=900
        ).pack(fill="x")

    @staticmethod
    def _read_config(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc

    def start_camera(self):
        try:
            self.camera = self.WebcamCapture(self.camera_index)
            self.camera.start()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status.set("Camera is active. Frames are processed locally and not saved.")
            self._update_frame()
        except Exception as exc:
            self.camera = None
            messagebox.showerror("Camera error", str(exc))
            self.status.set(f"Camera unavailable: {exc}")

    def _update_frame(self):
        if self.camera is None:
            return
        try:
            frame = self.camera.read()
            self.current_frame = frame.copy()
            height, width = frame.shape[:2]
            scale = min(900 / width, 480 / height, 1.0)
            if scale < 1:
                frame = self.cv2.resize(frame, (int(width * scale), int(height * scale)))
            ok, encoded = self.cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Could not render camera frame")
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
            self.preview.configure(image=self.photo, text="")
            self.after_id = self.root.after(30, self._update_frame)
        except Exception as exc:
            self.status.set(f"Camera stopped: {exc}")
            self.stop_camera()

    def stop_camera(self):
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.preview.configure(image="", text="Camera is off. Select Start camera.")
        self.photo = None
        self.current_frame = None
        self.status.set("Camera stopped safely.")

    def read_text(self):
        if self.current_frame is None:
            self.status.set("Start the camera first.")
            return
        try:
            from noorix.ocr import PrintedEnglishOCR
            text = PrintedEnglishOCR().read(self.current_frame)
            self.status.set(f"English OCR: {text or 'No clear printed English text found.'}")
        except Exception as exc:
            self.status.set(f"OCR unavailable: {exc}")

    def analyze(self):
        if self.current_frame is None:
            self.status.set("Start the camera first.")
            return
        results = []
        try:
            model = self.config.get("object_detection", {})
            if model.get("model_path") and model.get("labels"):
                from noorix.objects import ObjectDetector
                if self.detector is None:
                    self.detector = ObjectDetector(
                        model["model_path"], model["labels"],
                        confidence_threshold=float(model.get("confidence_threshold", 0.5)),
                    )
                detections = self.detector.detect(self.current_frame)
                results.extend(
                    f"{item['label']} at image-{item['position']} ({item['confidence']:.0%})"
                    for item in detections
                )
            currency = self.config.get("currency", {})
            if currency.get("model_path") and currency.get("labels"):
                from noorix.currency import CurrencyEstimator
                if self.currency is None:
                    self.currency = CurrencyEstimator(
                        currency["model_path"], currency["labels"],
                        input_size=int(currency.get("input_size", 224)),
                    )
                estimate = self.currency.predict(self.current_frame)
                results.append(
                    f"Possible note: {estimate['denomination']} (model score {estimate['confidence']:.0%}; estimate only)"
                )
            self.status.set("; ".join(results) if results else
                "No models configured. Add verified ONNX model paths and exact labels to config.json.")
        except Exception as exc:
            self.status.set(f"Analysis unavailable: {exc}")

    def speak_result(self):
        text = self.status.get()
        if not text.strip():
            return
        def worker():
            try:
                from noorix.speech import SpeechOutput
                SpeechOutput().speak(text)
            except Exception as exc:
                error = str(exc)
                self.root.after(0, lambda: self.status.set(f"Speech unavailable: {error}"))
        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        self.stop_camera()
        self.root.destroy()


def launch(camera_index: int = 0, config_path: Path | None = None) -> int:
    root = tk.Tk()
    NoorixApp(root, camera_index, config_path or Path("config.json"))
    root.mainloop()
    return 0

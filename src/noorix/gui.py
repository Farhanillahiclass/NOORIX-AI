"""NOORIX AI desktop dashboard.

The interface is intentionally honest about prototype readiness: camera preview
works independently, while OCR and model-backed analysis require configured
local dependencies and validated model files.
"""
from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


BG = "#0b1220"
PANEL = "#111c2e"
PANEL_ALT = "#17243a"
BORDER = "#26364f"
TEXT = "#e8eef8"
MUTED = "#9aabc2"
ACCENT = "#59c3c3"
BLUE = "#6ea8fe"
WARNING = "#f0bd68"


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
        self._configure_window()
        self._build_dashboard()
        self._set_status("Ready", "Camera is off. Start the camera to begin.", ACCENT)

    @staticmethod
    def _read_config(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc

    def _configure_window(self):
        self.root.title("NOORIX AI | Accessibility Assistant")
        self.root.geometry("1180x780")
        self.root.minsize(900, 640)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 9))
        style.configure("Accent.TButton", background=ACCENT, foreground="#07131d")
        style.map("Accent.TButton", background=[("active", "#79d7d2")])
        style.configure("Secondary.TButton", background=PANEL_ALT, foreground=TEXT)
        style.map("Secondary.TButton", background=[("active", BORDER)])
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Sidebar.TFrame", background="#0e1929")

    def _build_dashboard(self):
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg="#0e1929", width=220, padx=20, pady=24)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="NOORIX", bg="#0e1929", fg=TEXT,
                 font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(sidebar, text="AI ACCESSIBILITY", bg="#0e1929", fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 28))
        self._nav_item(sidebar, "◉", "Overview", True)
        self._nav_item(sidebar, "▣", "Live camera", False)
        self._nav_item(sidebar, "Aa", "Text reading", False)
        self._nav_item(sidebar, "◎", "Object & currency", False)
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", pady=22)
        tk.Label(sidebar, text="PROJECT STATUS", bg="#0e1929", fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(sidebar, text="Research prototype\nLocal processing",
                 bg="#0e1929", fg=TEXT, justify="left",
                 font=("Segoe UI", 10), pady=10).pack(anchor="w")
        tk.Label(sidebar, text="POWERED BY MRS", bg="#0e1929", fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="bottom", anchor="w")

        main = tk.Frame(shell, bg=BG, padx=28, pady=24)
        main.pack(side="left", fill="both", expand=True)
        top = tk.Frame(main, bg=BG)
        top.pack(fill="x")
        titlebox = tk.Frame(top, bg=BG)
        titlebox.pack(side="left")
        tk.Label(titlebox, text="Good to see you", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(titlebox, text="Accessibility dashboard", bg=BG, fg=TEXT,
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(top, text="SEE BEYOND. UNDERSTAND MORE.", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(side="right", anchor="center")

        cards = tk.Frame(main, bg=BG)
        cards.pack(fill="x", pady=(22, 18))
        self.camera_card_value = self._metric_card(cards, "CAMERA", "OFF", "Waiting for camera")
        self.ocr_card_value = self._metric_card(cards, "ENGLISH OCR", "READY TO CHECK", "Tesseract required")
        self.model_card_value = self._metric_card(cards, "AI MODELS", "NOT CONFIGURED", "ONNX model files required")

        body = tk.Frame(main, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        preview_head = tk.Frame(left, bg=PANEL, padx=18, pady=15)
        preview_head.pack(fill="x")
        tk.Label(preview_head, text="LIVE CAMERA", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(preview_head, text="LOCAL PREVIEW", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(side="right")
        self.preview = tk.Label(left, text="Camera is off\n\nPress  Start camera  to begin",
                                bg="#070d17", fg=MUTED, font=("Segoe UI", 13),
                                justify="center")
        self.preview.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        right = tk.Frame(body, bg=BG, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._section_title(right, "QUICK ACTIONS")
        ttk.Button(right, text="▶  Start camera", style="Accent.TButton",
                   command=self.start_camera).pack(fill="x", pady=(4, 8))
        ttk.Button(right, text="■  Stop camera", style="Secondary.TButton",
                   command=self.stop_camera).pack(fill="x", pady=4)
        ttk.Button(right, text="Aa  Read English text", style="Secondary.TButton",
                   command=self.read_text).pack(fill="x", pady=4)
        ttk.Button(right, text="◎  Analyze scene", style="Secondary.TButton",
                   command=self.analyze).pack(fill="x", pady=4)
        ttk.Button(right, text="♫  Speak result", style="Secondary.TButton",
                   command=self.speak_result).pack(fill="x", pady=4)
        self._section_title(right, "LATEST RESULT", pady=(24, 8))
        self.result = tk.Text(right, height=8, wrap="word", bg=PANEL, fg=TEXT,
                              insertbackground=TEXT, relief="flat", padx=12, pady=12,
                              font=("Segoe UI", 9), highlightbackground=BORDER,
                              highlightthickness=1)
        self.result.pack(fill="x")
        self.result.insert("1.0", "No analysis yet. Start the camera and choose an action.")
        self.result.configure(state="disabled")

        footer = tk.Frame(main, bg=BG, pady=14)
        footer.pack(fill="x")
        self.status_dot = tk.Label(footer, text="●", bg=BG, fg=ACCENT)
        self.status_dot.pack(side="left")
        self.status_label = tk.Label(footer, text="Ready", bg=BG, fg=TEXT,
                                     font=("Segoe UI", 9, "bold"))
        self.status_label.pack(side="left", padx=(7, 10))
        self.status_detail = tk.Label(footer, text="", bg=BG, fg=MUTED,
                                      font=("Segoe UI", 9))
        self.status_detail.pack(side="left")
        tk.Label(main, text="Image-relative position is not distance or navigation guidance. "
                 "This prototype does not replace a white cane or guide dog.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8), wraplength=900,
                 justify="left").pack(anchor="w")

    def _nav_item(self, parent, icon, label, active):
        bg = PANEL_ALT if active else "#0e1929"
        fg = ACCENT if active else MUTED
        item = tk.Frame(parent, bg=bg, padx=10, pady=10)
        item.pack(fill="x", pady=3)
        tk.Label(item, text=icon, bg=bg, fg=fg, width=3,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(item, text=label, bg=bg, fg=TEXT if active else MUTED,
                 font=("Segoe UI", 10, "bold" if active else "normal")).pack(side="left")

    def _metric_card(self, parent, title, value, detail):
        card = tk.Frame(parent, bg=PANEL, padx=16, pady=13,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        value_label = tk.Label(card, text=value, bg=PANEL, fg=TEXT,
                               font=("Segoe UI", 12, "bold"))
        value_label.pack(anchor="w", pady=(7, 3))
        tk.Label(card, text=detail, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")
        return value_label

    def _section_title(self, parent, text, pady=(0, 10)):
        tk.Label(parent, text=text, bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=pady)

    def _set_status(self, title, detail, color=ACCENT):
        self.status_label.configure(text=title)
        self.status_detail.configure(text=detail)
        self.status_dot.configure(fg=color)

    def _set_result(self, text):
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)
        self.result.configure(state="disabled")

    def start_camera(self):
        if self.camera is not None:
            return
        try:
            self.camera = self.WebcamCapture(self.camera_index)
            self.camera.start()
            self.camera_card_value.configure(text="LIVE", fg=ACCENT)
            self._set_status("Camera active", "Frames remain local and are not saved automatically.")
            self._update_frame()
        except Exception as exc:
            self.camera = None
            self.camera_card_value.configure(text="UNAVAILABLE", fg=WARNING)
            self._set_status("Camera error", str(exc), WARNING)
            messagebox.showerror("NOORIX AI — Camera error", str(exc))

    def _update_frame(self):
        if self.camera is None:
            return
        try:
            from PIL import Image, ImageTk

            frame = self.camera.read()
            self.current_frame = frame.copy()
            height, width = frame.shape[:2]
            target_w = max(320, self.preview.winfo_width() - 12)
            target_h = max(240, self.preview.winfo_height() - 12)
            scale = min(target_w / width, target_h / height)
            resized = self.cv2.resize(frame, (int(width * scale), int(height * scale)))
            rgb = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=image, master=self.root)
            self.preview.configure(image=self.photo, text="")
            self.after_id = self.root.after(33, self._update_frame)
        except Exception as exc:
            self._set_status("Preview stopped", str(exc), WARNING)
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
        if hasattr(self, "camera_card_value"):
            self.camera_card_value.configure(text="OFF", fg=TEXT)
        if hasattr(self, "preview"):
            self.preview.configure(image="", text="Camera is off\n\nPress  Start camera  to begin")
        self.photo = None
        self.current_frame = None
        if hasattr(self, "status_label"):
            self._set_status("Camera stopped", "Camera released safely.")

    def read_text(self):
        if self.current_frame is None:
            self._set_result("Start the camera first.")
            self._set_status("Camera required", "Start the camera before reading text.", WARNING)
            return
        try:
            from noorix.ocr import PrintedEnglishOCR
            text = PrintedEnglishOCR().read(self.current_frame)
            output = text or "No clear printed English text found."
            self._set_result(output)
            self._set_status("Text reading complete", "English OCR result is shown at right.")
        except Exception as exc:
            self._set_result(f"OCR unavailable: {exc}")
            self._set_status("OCR unavailable", "Install and configure the Tesseract executable.", WARNING)

    def analyze(self):
        if self.current_frame is None:
            self._set_result("Start the camera first.")
            self._set_status("Camera required", "Start the camera before scene analysis.", WARNING)
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
                    f"{item['label']} — image {item['position']} ({item['confidence']:.0%})"
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
                    f"Possible banknote: {estimate['denomination']} "
                    f"(score {estimate['confidence']:.0%}; estimate only)"
                )
            output = "\n".join(results) if results else (
                "AI models are not configured yet. Add validated ONNX model paths "
                "and matching labels in config.json."
            )
            self._set_result(output)
            self._set_status("Analysis complete" if results else "Models not configured",
                             "Predictions are estimates and require evaluation.",
                             ACCENT if results else WARNING)
        except Exception as exc:
            self._set_result(f"Analysis unavailable: {exc}")
            self._set_status("Analysis error", "Check model files, labels and configuration.", WARNING)

    def speak_result(self):
        text = self.result.get("1.0", "end").strip()
        if not text:
            return
        def worker():
            try:
                from noorix.speech import SpeechOutput
                SpeechOutput().speak(text)
            except Exception as exc:
                error = str(exc)
                self.root.after(0, lambda: self._set_status(
                    "Speech unavailable", error, WARNING
                ))
        threading.Thread(target=worker, daemon=True).start()
        self._set_status("Speech requested", "Local voice availability depends on Windows voices.")

    def close(self):
        self.stop_camera()
        self.root.destroy()


def launch(camera_index: int = 0, config_path: Path | None = None) -> int:
    root = tk.Tk()
    NoorixApp(root, camera_index, config_path or Path("config.json"))
    root.mainloop()
    return 0

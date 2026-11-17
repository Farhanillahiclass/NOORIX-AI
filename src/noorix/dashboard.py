"""NOORIX-AI rebuilt desktop dashboard.

The interface is a fresh navigation-first workspace. It calls existing local
adapters and test runners; it does not replace model implementations.
"""
from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from noorix.paths import resolve_config_path

PALETTE = {
    "bg": "#07111f", "rail": "#091727", "surface": "#0d1c2e",
    "surface2": "#12253a", "surface3": "#172d45", "line": "#203951",
    "text": "#eaf3ff", "muted": "#91a6bf", "cyan": "#27d9ee",
    "blue": "#6aa8ff", "green": "#39d6a0", "amber": "#f4bd55",
    "red": "#ff7188", "purple": "#b5a0ff",
}


class NoorixDashboard:
    """Fresh Tkinter dashboard with live workspace and in-app testing."""

    NAV = [
        ("OVERVIEW", ["Command Center"]),
        ("VISION", ["Camera & Scene", "Object Detection", "English OCR", "Handwriting OCR", "Urdu OCR (Experimental)", "PKR Currency"]),
        ("VOICE", ["Voice Interaction", "Text to Speech"]),
        ("INTELLIGENCE", ["Model Management", "Testing Center"]),
        ("SYSTEM", ["Activity Logs", "Diagnostics", "Settings", "About Farhan"]),
    ]

    def __init__(self, root: tk.Tk, camera_index: int = 0, config_path: Path | None = None):
        self.root = root
        self.camera_index = camera_index
        self.config_path = Path(config_path or "config.json").expanduser().resolve()
        self.config = self._load_config()
        self.camera = None
        self.frame = None
        self.preview_image = None
        self.camera_after = None
        self.test_running = False
        self.history: list[str] = []
        self.detections: list[dict] = []
        self.section = "Command Center"
        self._configure_window()
        self._build_shell()
        self.show_section("Command Center")
        self.refresh_system()

    def _load_config(self):
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _configure_window(self):
        self.root.title("NOORIX-AI | Offline Accessibility Workspace")
        self.root.geometry("1500x920")
        self.root.minsize(1120, 720)
        self.root.configure(bg=PALETTE["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text, size=10, color="text", bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=kwargs.pop("bg", parent.cget("bg")),
                        fg=PALETTE.get(color, color), font=("Segoe UI", size, "bold" if bold else "normal"),
                        **kwargs)

    def button(self, parent, text, command, primary=False, **kwargs):
        bg = PALETTE["cyan"] if primary else PALETTE["surface3"]
        fg = "#06131e" if primary else PALETTE["text"]
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=PALETTE["blue"], activeforeground="#06131e",
                         relief="flat", bd=0, padx=14, pady=9, cursor="hand2",
                         font=("Segoe UI", 9, "bold"), **kwargs)

    def panel(self, parent, **kwargs):
        return tk.Frame(parent, bg=PALETTE["surface"], highlightbackground=PALETTE["line"],
                        highlightthickness=1, **kwargs)

    def _build_shell(self):
        shell = tk.Frame(self.root, bg=PALETTE["bg"])
        shell.pack(fill="both", expand=True)

        # Scrollable navigation rail: smaller displays can reach every section.
        sidebar = tk.Frame(shell, bg=PALETTE["rail"], width=238)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self.sidebar_canvas = tk.Canvas(sidebar, bg=PALETTE["rail"], highlightthickness=0, bd=0)
        sidebar_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_scroll.pack(side="right", fill="y")
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_inner = tk.Frame(self.sidebar_canvas, bg=PALETTE["rail"], padx=15, pady=18)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw")
        self.sidebar_inner.bind("<Configure>", lambda _e: self.sidebar_canvas.configure(
            scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfigure(
            self.sidebar_window, width=e.width))

        brand = tk.Frame(self.sidebar_inner, bg=PALETTE["rail"])
        brand.pack(fill="x", pady=(3, 22))
        mark = tk.Frame(brand, bg=PALETTE["cyan"], width=42, height=42)
        mark.pack(side="left")
        mark.pack_propagate(False)
        self.label(mark, "N", 22, "#06131e", True).pack(expand=True)
        btxt = tk.Frame(brand, bg=PALETTE["rail"])
        btxt.pack(side="left", padx=10)
        self.label(btxt, "NOORIX-AI", 15, "text", True).pack(anchor="w")
        self.label(btxt, "SEE BEYOND  /  OFFLINE FIRST", 7, "cyan", True).pack(anchor="w", pady=(3, 0))

        self.nav_buttons = {}
        for group, items in self.NAV:
            self.label(self.sidebar_inner, group, 8, "muted", True).pack(anchor="w", pady=(11, 5))
            for item in items:
                btn = tk.Button(self.sidebar_inner, text="   " + item, anchor="w",
                    command=lambda name=item: self.show_section(name),
                    bg=PALETTE["rail"], fg=PALETTE["muted"], activebackground=PALETTE["surface2"],
                    activeforeground=PALETTE["cyan"], relief="flat", bd=0, padx=9, pady=7,
                    font=("Segoe UI", 9), cursor="hand2")
                btn.pack(fill="x", pady=1)
                self.nav_buttons[item] = btn
        privacy = self.panel(self.sidebar_inner)
        privacy.pack(fill="x", pady=(18, 8))
        self.label(privacy, "  LOCAL BY DEFAULT", 8, "cyan", True).pack(anchor="w", pady=(10, 3))
        self.label(privacy, "  Camera and model processing stays on this device.", 8, "muted",
                   wraplength=190, justify="left").pack(anchor="w", padx=4, pady=(0, 10))
        self.label(self.sidebar_inner, "RESEARCH PROTOTYPE  ·  MRS", 7, "muted", True).pack(anchor="w", pady=(2, 10))

        content = tk.Frame(shell, bg=PALETTE["bg"], padx=18, pady=12)
        content.pack(side="left", fill="both", expand=True)
        top = tk.Frame(content, bg=PALETTE["surface"], padx=12, pady=9,
                       highlightbackground=PALETTE["line"], highlightthickness=1)
        top.pack(fill="x")
        self.label(top, "MRS   /   NOORIX-AI", 9, "text", True).pack(side="left")
        self.label(top, "LOCAL WORKSPACE", 8, "muted", True).pack(side="left", padx=14)
        self.system_badge = self.label(top, "● OFFLINE MODE", 8, "green", True)
        self.system_badge.pack(side="right", padx=8)
        self.button(top, "Refresh status", self.refresh_system).pack(side="right")

        # Keep the header and status footer fixed; scroll only the active page.
        page_area = tk.Frame(content, bg=PALETTE["bg"])
        page_area.pack(fill="both", expand=True, pady=(12, 0))
        self.page_canvas = tk.Canvas(page_area, bg=PALETTE["bg"], highlightthickness=0, bd=0)
        page_scroll = ttk.Scrollbar(page_area, orient="vertical", command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=page_scroll.set)
        page_scroll.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)
        self.page = tk.Frame(self.page_canvas, bg=PALETTE["bg"])
        self.page_window = self.page_canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", lambda _e: self.page_canvas.configure(
            scrollregion=self.page_canvas.bbox("all")))
        self.page_canvas.bind("<Configure>", lambda e: self.page_canvas.itemconfigure(
            self.page_window, width=e.width))

        footer = tk.Frame(content, bg=PALETTE["bg"])
        footer.pack(fill="x", pady=(7, 0))
        self.status = self.label(footer, "● READY", 8, "cyan", True)
        self.status.pack(side="left")
        self.status_detail = self.label(footer, "Waiting for an action", 8, "muted")
        self.status_detail.pack(side="left", padx=10)
        self.label(footer, "NOORIX-AI  ·  Predictions require independent verification", 7, "muted").pack(side="right")

        def scroll_canvas(event, canvas):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-event.delta / 120), "units")
        self.root.bind_all("<MouseWheel>", lambda event: scroll_canvas(event, self.page_canvas))
        self.root.bind_all("<Shift-MouseWheel>", lambda event: scroll_canvas(event, self.sidebar_canvas))

        def set_status(self, title, detail="", color="cyan"):
        self.status.configure(text="● " + title.upper(), fg=PALETTE[color])
        self.status_detail.configure(text=detail)

    def show_section(self, name):
        self.section = name
        for item, btn in self.nav_buttons.items():
            active = item == name
            btn.configure(bg=PALETTE["surface2"] if active else PALETTE["rail"],
                          fg=PALETTE["text"] if active else PALETTE["muted"],
                          font=("Segoe UI", 9, "bold" if active else "normal"))
        for child in self.page.winfo_children():
            child.destroy()
        handlers = {
            "Command Center": self._overview,
            "Camera & Scene": self._camera_page,
            "Object Detection": lambda: self._feature_page("OBJECT DETECTION", "Run the configured object detector against a real camera frame or imported image.", self.run_scene),
            "English OCR": lambda: self._ocr_page("eng", "ENGLISH OCR"),
            "Handwriting OCR": lambda: self._placeholder("HANDWRITING OCR", "Handwriting recognition is not currently implemented by the printed Tesseract OCR adapter. Importing a page here will not be reported as a handwriting pass."),
            "Urdu OCR (Experimental)": lambda: self._ocr_page("urd", "URDU OCR · EXPERIMENTAL"),
            "PKR Currency": lambda: self._feature_page("PKR CURRENCY RECOGNITION", "Estimate a configured PKR denomination and front/back side from a real image. This is not banknote authentication.", self.run_currency),
            "Voice Interaction": self._voice_page,
            "Text to Speech": self._tts_page,
            "Model Management": self._models_page,
            "Testing Center": self._testing_page,
            "Activity Logs": self._logs_page,
            "Diagnostics": self._diagnostics_page,
            "Settings": self._settings_page,
            "About Farhan": self._about_page,
        }
        handlers.get(name, self._overview)()

    def page_heading(self, eyebrow, title, subtitle):
        self.label(self.page, eyebrow, 8, "cyan", True).pack(anchor="w")
        self.label(self.page, title, 23, "text", True).pack(anchor="w", pady=(4, 3))
        self.label(self.page, subtitle, 9, "muted").pack(anchor="w", pady=(0, 15))

    def metric(self, parent, title, value, detail, color="cyan"):
        box = self.panel(parent)
        self.label(box, title, 8, "muted", True).pack(anchor="w", padx=13, pady=(11, 5))
        self.label(box, value, 15, color, True).pack(anchor="w", padx=13)
        self.label(box, detail, 8, "muted", wraplength=210, justify="left").pack(anchor="w", padx=13, pady=(4, 11))
        return box

    def _overview(self):
        self.page_heading("NOORIX / COMMAND", "Your accessibility workspace",
                          "A focused control room for vision, reading, voice and local model validation.")
        metrics = tk.Frame(self.page, bg=PALETTE["bg"])
        metrics.pack(fill="x", pady=(0, 14))
        for title, value, detail, color in [
            ("RUNTIME", "LOCAL", "No cloud dependency for local inference", "green"),
            ("CAMERA", "STANDBY", "Start a live visual session", "blue"),
            ("MODEL SUITE", "6 TESTS", "Package · PKR · YOLO · OCR · ASR · TTS", "cyan"),
            ("VALIDATION", "NOT CERTIFIED", "Smoke tests are not accuracy benchmarks", "amber"),
        ]:
            self.metric(metrics, title, value, detail, color).pack(side="left", fill="both", expand=True, padx=(0, 9))
        row = tk.Frame(self.page, bg=PALETTE["bg"])
        row.pack(fill="both", expand=True)
        vision = self.panel(row)
        vision.pack(side="left", fill="both", expand=True, padx=(0, 12))
        head = tk.Frame(vision, bg=PALETTE["surface"], padx=14, pady=12)
        head.pack(fill="x")
        self.label(head, "LIVE VISION CANVAS", 9, "text", True).pack(side="left")
        self.label(head, "CAMERA / OCR / DETECTION", 7, "cyan", True).pack(side="right")
        self.preview = tk.Label(vision, text="CAMERA PREVIEW\n\nStart the camera to view a live frame.\nObject boxes appear only after a detector returns results.",
                                bg="#081522", fg=PALETTE["muted"], font=("Segoe UI", 11), justify="center")
        self.preview.pack(fill="both", expand=True, padx=12, pady=12)
        controls = tk.Frame(vision, bg=PALETTE["surface"], padx=12, pady=10)
        controls.pack(fill="x")
        self.button(controls, "START CAMERA", self.start_camera, True).pack(side="left", padx=(0, 6))
        self.button(controls, "STOP", self.stop_camera).pack(side="left")
        self.button(controls, "IMPORT IMAGE", self.import_image).pack(side="right")
        tools = self.panel(row, width=285)
        tools.pack(side="right", fill="y")
        tools.pack_propagate(False)
        self.label(tools, "QUICK OPERATIONS", 9, "text", True).pack(anchor="w", padx=14, pady=(14, 8))
        for text, command in [
            ("Run scene analysis", self.run_scene),
            ("Read printed English", lambda: self.run_ocr("eng")),
            ("Estimate PKR note", self.run_currency),
            ("Open Testing Center", lambda: self.show_section("Testing Center")),
            ("Manage local models", lambda: self.show_section("Model Management")),
        ]:
            self.button(tools, text, command).pack(fill="x", padx=12, pady=4)
        self.label(tools, "LATEST OUTPUT", 8, "muted", True).pack(anchor="w", padx=14, pady=(16, 5))
        self.output = tk.Text(tools, height=10, wrap="word", bg="#081522", fg=PALETTE["text"],
                              insertbackground=PALETTE["text"], relief="flat", padx=10, pady=10,
                              font=("Segoe UI", 9))
        self.output.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.output.insert("1.0", "No result yet. Choose an operation to begin.")
        self.output.configure(state="disabled")

    def _set_output(self, text):
        if hasattr(self, "output") and self.output.winfo_exists():
            self.output.configure(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("1.0", str(text))
            self.output.configure(state="disabled")

    def _camera_page(self):
        self._feature_page("CAMERA CONTROLS", "Live camera workspace. Camera feed is processed locally; start or stop capture explicitly.", self.run_scene)
        self.preview = tk.Label(self.page, text="CAMERA PREVIEW", bg="#081522", fg=PALETTE["muted"], height=14)
        self.preview.pack(fill="both", expand=True, pady=8)
        self.button(self.page, "START CAMERA", self.start_camera, True).pack(anchor="w", pady=5)
        self.button(self.page, "STOP CAMERA", self.stop_camera).pack(anchor="w", pady=5)
        self.button(self.page, "IMPORT IMAGE FOR TESTING", self.import_image).pack(anchor="w", pady=5)

    def _feature_page(self, title, description, action):
        self.page_heading("NOORIX / VISION MODULE", title, description)
        row = tk.Frame(self.page, bg=PALETTE["bg"])
        row.pack(fill="x", pady=10)
        self.metric(row, "INPUT SOURCE", "CAMERA / FILE", "Use a real camera frame or import a local image").pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.metric(row, "PROCESSING", "ON DEVICE", "No image upload is performed by this desktop app", "green").pack(side="left", fill="x", expand=True)
        self.button(self.page, "IMPORT IMAGE", self.import_image, True).pack(anchor="w", pady=7)
        self.button(self.page, "RUN MODULE", action).pack(anchor="w", pady=7)
        self.result_box = tk.Text(self.page, height=16, wrap="word", bg=PALETTE["surface"], fg=PALETTE["text"],
                                  insertbackground=PALETTE["text"], relief="flat", padx=14, pady=12)
        self.result_box.pack(fill="both", expand=True, pady=(12, 0))
        self.result_box.insert("1.0", "Module ready. Import an image or start the camera, then run the module.")
        self.result_box.configure(state="disabled")

    def _write_result_box(self, text):
        if hasattr(self, "result_box") and self.result_box.winfo_exists():
            self.result_box.configure(state="normal")
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", text)
            self.result_box.configure(state="disabled")
        if hasattr(self, "_placeholder_voice") and self._placeholder_voice.winfo_exists():
            self._placeholder_voice.delete("1.0", "end")
            self._placeholder_voice.insert("1.0", text)
        self._set_output(text)
        self.history.append(text[:500])

    def _placeholder(self, title, description):
        self.page_heading("NOORIX / EXPERIMENTAL CAPABILITY", title, description)
        self.metric(self.page, "CAPABILITY STATUS", "NOT IMPLEMENTED", "No verified model adapter is connected", "amber").pack(anchor="w", pady=8)
        self.label(self.page, "This module is intentionally marked unavailable. It will not show a fake PASS or produce placeholder predictions.", 10, "muted", wraplength=800, justify="left").pack(anchor="w", pady=12)

    def _ocr_page(self, language, title):
        self._feature_page(title, "Printed text OCR only. Handwriting recognition is a separate capability and is not implied here.", lambda: self.run_ocr(language))

    def _voice_page(self):
        self.page_heading("NOORIX / VOICE", "Voice interaction", "Transcribe a local audio file using the configured Whisper model.")
        self.button(self.page, "IMPORT AUDIO FILE", self.transcribe_audio, True).pack(anchor="w", pady=8)
        self._placeholder_voice = tk.Text(self.page, height=18, bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", padx=12, pady=12)
        self._placeholder_voice.pack(fill="both", expand=True, pady=10)
        self._placeholder_voice.insert("1.0", "No transcription yet. Supported audio formats depend on installed audio codecs.")

    def _tts_page(self):
        self.page_heading("NOORIX / SPEECH OUTPUT", "Text to speech", "Speak text with an installed local Windows system voice.")
        self.tts_text = tk.Text(self.page, height=8, bg=PALETTE["surface"], fg=PALETTE["text"], insertbackground=PALETTE["text"], relief="flat", padx=12, pady=12)
        self.tts_text.pack(fill="x", pady=8)
        self.tts_text.insert("1.0", "Hello. This is NOORIX-AI.")
        self.button(self.page, "SPEAK TEXT", self.speak_text, True).pack(anchor="w", pady=6)
        self.button(self.page, "CHECK LOCAL VOICES", self.check_voices).pack(anchor="w", pady=6)

    def _models_page(self):
        self.page_heading("NOORIX / LOCAL ASSETS", "Model management", "Inspect configured model paths and import local model packages.")
        self.models_text = tk.Text(self.page, bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", padx=14, pady=14, font=("Consolas", 9))
        self.models_text.pack(fill="both", expand=True, pady=8)
        self._render_models()
        self.button(self.page, "REFRESH MODEL STATUS", self._render_models, True).pack(anchor="w", pady=6)
        self.button(self.page, "OPEN MODEL PACKAGE FOLDER", self.open_model_folder).pack(anchor="w", pady=6)

    def _render_models(self):
        if not hasattr(self, "models_text") or not self.models_text.winfo_exists():
            return
        lines = [f"CONFIG: {self.config_path}", ""]
        for key in ("object_detection", "currency", "asr"):
            settings = self.config.get(key, {})
            value = settings.get("model_path") if isinstance(settings, dict) else None
            path = resolve_config_path(value, self.config_path) if value else None
            state = "NOT CONFIGURED" if not value else ("FILES PRESENT · NOT VALIDATED" if path and path.exists() else "MISSING")
            lines += [f"{key.upper():18} {state}", f"  {path or 'No model path configured'}"]
            if key == "currency" and isinstance(settings, dict) and settings.get("labels_path"):
                labels = resolve_config_path(settings["labels_path"], self.config_path)
                lines.append(f"  labels: {'PRESENT' if labels.is_file() else 'MISSING'}")
            if key == "asr" and isinstance(settings, dict) and settings.get("processor_path"):
                processor = resolve_config_path(settings["processor_path"], self.config_path)
                lines.append(f"  processor: {'PRESENT' if processor.is_dir() else 'MISSING'}")
        lines += ["", "PKR labels: " + str(self.config.get("currency", {}).get("labels_path", "not configured")),
                  "ASR processor: " + str(self.config.get("asr", {}).get("processor_path", "same as model directory"))]
        self.models_text.configure(state="normal")
        self.models_text.delete("1.0", "end")
        self.models_text.insert("1.0", "\n".join(lines))
        self.models_text.configure(state="disabled")

    def _testing_page(self):
        self.page_heading("NOORIX / VALIDATION LAB", "Testing Center", "Run tests inside the desktop app. Smoke tests are not accuracy certification.")
        bar = tk.Frame(self.page, bg=PALETTE["bg"])
        bar.pack(fill="x", pady=(0, 10))
        self.button(bar, "RUN ALL SMOKE TESTS", lambda: self.run_tests(None), True).pack(side="left", padx=(0, 7))
        self.button(bar, "OPEN REPORTS", self.open_reports).pack(side="left")
        self.test_progress = ttk.Progressbar(self.page, mode="determinate", maximum=6)
        self.test_progress.pack(fill="x", pady=8)
        self.test_summary = self.label(self.page, "READY · 6 checks available", 9, "muted", True)
        self.test_summary.pack(anchor="w", pady=(0, 8))
        self.test_rows = tk.Frame(self.page, bg=PALETTE["bg"])
        self.test_rows.pack(fill="both", expand=True)
        self.test_keys = [("package","Package integrity"),("currency","PKR currency"),("objects","YOLO object detection"),("ocr","Printed OCR"),("asr","Whisper speech-to-text"),("tts","Local text-to-speech")]
        for key, title in self.test_keys:
            line = self.panel(self.test_rows)
            line.pack(fill="x", pady=3)
            self.label(line, title, 10, "text", True).pack(side="left", padx=12, pady=10)
            status = self.label(line, "NOT RUN", 8, "muted", True)
            status.pack(side="right", padx=12)
            self.button(line, "TEST", lambda k=key: self.run_tests(k)).pack(side="right", padx=6, pady=4)
            setattr(self, "test_" + key, status)
        note = self.panel(self.page)
        note.pack(fill="x", pady=(10, 0))
        self.label(note, "REAL INPUT VALIDATION", 8, "amber", True).pack(anchor="w", padx=12, pady=(9, 2))
        self.label(note, "Camera, handwriting, Urdu OCR, and labelled accuracy evaluation are separate workflows. They are not counted as PASS by these six smoke tests.", 8, "muted", wraplength=900, justify="left").pack(anchor="w", padx=12, pady=(0, 9))

    def run_tests(self, key):
        if self.test_running:
            return
        self.test_running = True
        self.test_progress.configure(value=0)
        self.test_summary.configure(text="RUNNING · Keep the dashboard open")
        for _, title in self.test_keys:
            pass
        def worker():
            try:
                from noorix.model_testing import run_all_tests, run_single_test
                def progress(item):
                    completed = len(getattr(progress, "items", [])) + 1
                    progress.items = getattr(progress, "items", []) + [item]
                    self.root.after(0, lambda n=completed, x=item: self._test_progress(n, x))
                report = run_all_tests(self.config_path, progress=progress) if key is None else run_single_test(key, self.config_path)
                self.root.after(0, lambda r=report: self._tests_complete(r))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._tests_failed(e))
        threading.Thread(target=worker, daemon=True).start()

    def _test_progress(self, completed, item):
        self.test_progress.configure(value=completed)
        self.test_summary.configure(text=f"RUNNING · {completed}/6 · {item.name} · {item.status}")
        mapping = {"Model package integrity":"package","PKR currency model":"currency","YOLO object detection":"objects","Printed OCR":"ocr","Whisper speech-to-text":"asr","Local text-to-speech":"tts"}
        key = mapping.get(item.name)
        if key:
            color = {"PASS":"green","FAIL":"red","SKIPPED":"amber","WARN":"amber"}.get(item.status,"muted")
            getattr(self, "test_" + key).configure(text=item.status, fg=PALETTE[color])

    def _tests_complete(self, report):
        self.test_running = False
        results = report.get("results", [])
        for item in results:
            name = item.get("name", "")
            mapping = {"Model package integrity":"package","PKR currency model":"currency","YOLO object detection":"objects","Printed OCR":"ocr","Whisper speech-to-text":"asr","Local text-to-speech":"tts"}
            key = mapping.get(name)
            if key:
                status = item.get("status", "UNKNOWN")
                color = {"PASS":"green","FAIL":"red","SKIPPED":"amber","WARN":"amber"}.get(status,"muted")
                getattr(self, "test_" + key).configure(text=status, fg=PALETTE[color])
        totals = report.get("totals", {})
        self.test_progress.configure(value=len(results))
        self.test_summary.configure(text=f"COMPLETE · PASS {totals.get('pass',0)}  FAIL {totals.get('fail',0)}  SKIPPED {totals.get('skipped',0)}  WARN {totals.get('warn',0)}")
        self.last_report = report
        if len(results) == 1:
            self.test_progress.configure(value=1)
        has_issues = bool(totals.get("fail", 0) or totals.get("skipped", 0) or totals.get("warn", 0))
        self.set_status("TESTS COMPLETE", "Production readiness remains NOT CERTIFIED",
                        "amber" if has_issues else "green")
        messagebox.showinfo("NOORIX-AI Testing Center", f"Tests complete.\nPASS: {totals.get('pass',0)}\nFAIL: {totals.get('fail',0)}\nSKIPPED: {totals.get('skipped',0)}\n\nProduction readiness: NOT CERTIFIED")

    def _tests_failed(self, exc):
        self.test_running = False
        self.test_summary.configure(text="TEST RUN FAILED · " + str(exc))
        self.set_status("TEST ERROR", str(exc), "red")

    def open_reports(self):
        folder = Path("reports/model-tests").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        self._open_path(folder)

    def _logs_page(self):
        self.page_heading("NOORIX / SESSION TRACE", "Activity logs", "Local in-memory actions from this app session.")
        box = tk.Text(self.page, bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", padx=12, pady=12)
        box.pack(fill="both", expand=True)
        box.insert("1.0", "\n\n".join(reversed(self.history)) if self.history else "No actions recorded yet.")
        box.configure(state="disabled")

    def _diagnostics_page(self):
        self.page_heading("NOORIX / SYSTEM HEALTH", "Diagnostics", "Runtime and configured model-file checks.")
        self.diag = tk.Text(self.page, bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", padx=12, pady=12)
        self.diag.pack(fill="both", expand=True)
        self.refresh_system()

    def _settings_page(self):
        self.page_heading("NOORIX / PREFERENCES", "Settings", "Current local configuration. Edit config.json to change model paths and runtime options.")
        self.label(self.page, f"Configuration file\n{self.config_path}", 10, "text", wraplength=900, justify="left").pack(anchor="w", pady=8)
        self.button(self.page, "RELOAD CONFIGURATION", self.reload_config, True).pack(anchor="w", pady=6)
        self.button(self.page, "OPEN CONFIG FILE", lambda: self._open_path(self.config_path)).pack(anchor="w", pady=6)

    def _about_page(self):
        self.page_heading("NOORIX / CREATOR", "About Farhan", "NOORIX-AI · Offline-first accessibility research prototype")
        self.label(self.page, "Muhammad Farhan", 20, "text", True).pack(anchor="w", pady=(10, 4))
        self.label(self.page, "AI Student · Researcher & Developer at MRS", 10, "cyan", True).pack(anchor="w")
        self.label(self.page, "Building practical AI systems with a focus on accessibility, local-first software and responsible model validation.", 10, "muted", wraplength=850, justify="left").pack(anchor="w", pady=12)
        for title, url in [("LinkedIn","https://linkedin.com/in/muhammadfarhanmrs"),("GitHub","https://github.com/Farhanillahiclass"),("Kaggle","https://kaggle.com/muhammadfarhanmrs"),("YouTube","https://youtube.com/@FarhanMRSDeveloper"),("MRS","https://mrsplatform.tech/")]:
            self.button(self.page, title + "  ↗", lambda u=url: __import__("webbrowser").open(u)).pack(anchor="w", pady=4)

    def refresh_system(self):
        import shutil
        details = [f"Python: {os.sys.version.split()[0]}", f"Config: {self.config_path}",
                   f"Tesseract: {'installed' if shutil.which('tesseract') else 'not found'}",
                   f"Camera: {'running' if self.camera else 'stopped'}",
                   "Handwriting OCR: NOT IMPLEMENTED — no verified handwriting model is configured"]
        if shutil.which("tesseract"):
            try:
                import pytesseract
                languages = sorted(pytesseract.get_languages(config=""))
                details.append("Tesseract languages: " + ", ".join(languages))
                details.append("Urdu OCR language data: " + ("available" if "urd" in languages else "missing"))
            except Exception as exc:
                details.append(f"Tesseract language check: {type(exc).__name__}: {exc}")
        for key in ("object_detection", "currency", "asr"):
            settings = self.config.get(key, {})
            value = settings.get("model_path") if isinstance(settings, dict) else None
            path = resolve_config_path(value, self.config_path) if value else None
            details.append(f"{key}: {'not configured' if not value else ('files present — not validated' if path and path.exists() else 'missing')}")
        if hasattr(self, "diag") and self.diag.winfo_exists():
            self.diag.configure(state="normal")
            self.diag.delete("1.0", "end")
            self.diag.insert("1.0", "\n".join(details))
            self.diag.configure(state="disabled")
        self.set_status("SYSTEM STATUS UPDATED", "Offline-first local diagnostics", "green")

    def reload_config(self):
        self.config = self._load_config()
        self.refresh_system()
        self._render_models()
        messagebox.showinfo("NOORIX-AI", "Configuration reloaded.")

    def _current_frame(self):
        if self.frame is None:
            messagebox.showwarning("NOORIX-AI", "Start the camera or import an image first.")
        return self.frame

    def import_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.webp"),("All files","*.*")])
        if not path:
            return
        try:
            import cv2
            self.cv2 = cv2
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Could not decode image")
            self.frame = image
            self._show_frame(image)
            self.set_status("IMAGE LOADED", Path(path).name, "green")
            self.history.append("Imported image: " + path)
        except Exception as exc:
            messagebox.showerror("Image import failed", str(exc))

    def _show_frame(self, frame):
        try:
            from PIL import Image, ImageTk
            display_frame = frame.copy()
            for detection in getattr(self, "detections", []):
                box = detection.get("box")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    self.cv2.rectangle(display_frame, (x1, y1), (x2, y2), (238, 217, 39), 2)
                    caption = str(detection.get("label", "object")) + " " + str(round(float(detection.get("confidence", 0)) * 100)) + "%"
                    self.cv2.putText(display_frame, caption, (x1, max(18, y1 - 7)),
                                     self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 217, 39), 2)
            rgb = self.cv2.cvtColor(display_frame, self.cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((950, 560))
            self.preview_image = ImageTk.PhotoImage(image)
            if hasattr(self, "preview") and self.preview.winfo_exists():
                self.preview.configure(image=self.preview_image, text="")
        except Exception:
            pass

    def start_camera(self):
        if self.camera is not None:
            return
        try:
            import cv2
            from noorix.camera import WebcamCapture
            self.cv2 = cv2
            self.camera = WebcamCapture(self.camera_index)
            self.camera.start()
            self.set_status("CAMERA LIVE", "Frames remain local to this device", "green")
            self._camera_tick()
        except Exception as exc:
            self.camera = None
            messagebox.showerror("Camera unavailable", str(exc))

    def _camera_tick(self):
        if self.camera is None:
            return
        try:
            self.frame = self.camera.read()
            self._show_frame(self.frame)
            self.camera_after = self.root.after(40, self._camera_tick)
        except Exception as exc:
            self.stop_camera()
            self.set_status("CAMERA ERROR", str(exc), "red")

    def stop_camera(self):
        if self.camera_after:
            try:
                self.root.after_cancel(self.camera_after)
            except tk.TclError:
                pass
            self.camera_after = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.set_status("CAMERA STOPPED", "Camera resource released", "cyan")

    def _run_agent(self, name, frame):
        from noorix.agents import AgentCoordinator, CurrencyAgent, ObjectDetectionAgent, OCRAgent, SafetyAgent, FrameQualityAgent
        agents = [FrameQualityAgent(), OCRAgent(), ObjectDetectionAgent(self.config.get("object_detection", {}), config_path=self.config_path),
                  CurrencyAgent(self.config.get("currency", {}), config_path=self.config_path), SafetyAgent()]
        coordinator = AgentCoordinator(agents)
        return coordinator.run(name, frame)

    def run_scene(self):
        frame = self._current_frame()
        if frame is None: return
        self._run_async("Scene analysis", lambda: [self._run_agent("object_detection", frame), self._run_agent("currency", frame)])

    def run_currency(self):
        frame = self._current_frame()
        if frame is None: return
        self._run_async("PKR currency", lambda: [self._run_agent("currency", frame)])

    def run_ocr(self, language="eng"):
        frame = self._current_frame()
        if frame is None: return
        def work():
            from noorix.ocr import PrintedEnglishOCR
            return PrintedEnglishOCR(language=language).read(frame)
        self._run_async("OCR", work)

    def _run_async(self, label, operation):
        self.set_status(label + " running", "Local inference in progress", "cyan")
        def worker():
            try:
                result = operation()
                detections = []
                failed = False
                if isinstance(result, list):
                    text = "\n\n".join(f"{x.agent.upper()} · {x.state.value.upper()}\n{x.summary}\n" + "\n".join(x.warnings) for x in result)
                    failed = any(not getattr(x, "successful", False) for x in result)
                    for item in result:
                        if item.agent == "object_detection" and isinstance(item.data, list):
                            detections = item.data
                else:
                    text = str(result) if result else "No text detected."
                self.root.after(0, lambda: self._operation_complete(label, text, failed=failed, detections=detections))
            except Exception as exc:
                self.root.after(0, lambda: self._operation_complete(label, f"{type(exc).__name__}: {exc}", True))
        threading.Thread(target=worker, daemon=True).start()

    def _operation_complete(self, label, text, failed=False, detections=None):
        if detections is not None:
            self.detections = detections
            if self.frame is not None:
                self._show_frame(self.frame)
        self._write_result_box(text)
        self.set_status(label + (" failed" if failed else " complete"), "Verify results independently", "red" if failed else "green")

    def transcribe_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio","*.wav *.flac *.ogg *.mp3 *.m4a"),("All files","*.*")])
        if not path: return
        def work():
            from noorix.agents import WhisperASRAgent
            agent = WhisperASRAgent(self.config.get("asr", {}), config_path=self.config_path)
            return agent.run(Path(path))
        self.set_status("TRANSCRIBING", Path(path).name, "cyan")
        def worker():
            try:
                result = work()
                text = result.summary + ("\n\n" + "\n".join(result.warnings) if result.warnings else "")
                self.root.after(0, lambda: self._voice_complete(text))
            except Exception as exc:
                self.root.after(0, lambda: self._voice_complete(str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _voice_complete(self, text):
        if hasattr(self, "_placeholder_voice") and self._placeholder_voice.winfo_exists():
            self._placeholder_voice.delete("1.0", "end")
            self._placeholder_voice.insert("1.0", text)
        self.history.append(text)
        self.set_status("VOICE TASK COMPLETE", "Check transcription before relying on it", "green")

    def speak_text(self):
        text = self.tts_text.get("1.0", "end").strip()
        if not text: return
        def worker():
            try:
                from noorix.speech import SpeechOutput
                SpeechOutput().speak(text)
                self.root.after(0, lambda: self.set_status("SPEECH COMPLETE", "Local system voice", "green"))
            except Exception as exc:
                self.root.after(0, lambda: self.set_status("SPEECH ERROR", str(exc), "red"))
        threading.Thread(target=worker, daemon=True).start()

    def check_voices(self):
        try:
            from noorix.speech import SpeechOutput
            voices = SpeechOutput.available_voices()
            messagebox.showinfo("Local voices", "\n".join(v.get("name","Unknown") for v in voices) or "No local voices found.")
        except Exception as exc:
            messagebox.showerror("Voice check failed", str(exc))

    def open_model_folder(self):
        root = Path(os.environ.get("NOORIX_MODELS_DIR", self.config_path.parent)).resolve()
        self._open_path(root)

    @staticmethod
    def _open_path(path):
        path = Path(path)
        if not path.exists():
            messagebox.showwarning("Path unavailable", f"Not found: {path}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Could not open path", str(exc))

    def close(self):
        self.stop_camera()
        self.root.destroy()


def launch(camera_index: int = 0, config_path: Path | None = None) -> int:
    root = tk.Tk()
    NoorixDashboard(root, camera_index, config_path)
    root.mainloop()
    return 0

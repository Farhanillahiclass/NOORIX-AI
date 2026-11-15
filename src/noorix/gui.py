"""NOORIX AI — polished desktop command center.

A lightweight Tkinter interface with a camera workspace, honest capability
states, accessible contrast, and working local actions. No web service required.
"""
from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog


C = {
    "bg": "#080f1d", "sidebar": "#0c1728", "panel": "#101e31",
    "panel2": "#14253b", "panel3": "#0b1626", "line": "#263b55",
    "text": "#edf5ff", "muted": "#91a6bf", "cyan": "#55e1d0",
    "blue": "#6daaff", "purple": "#a78bfa", "amber": "#f5c16c",
    "green": "#65d6a5", "red": "#ff8585",
}


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
        self.config_path = config_path
        self.agent_coordinator = None
        self.session_history = []
        self.active_section = "Overview"
        self.config = self._read_config(config_path)
        self._setup_window()
        self._build_ui()
        self._set_status("SYSTEM READY", "Start the camera to begin a local session.", C["cyan"])

    @staticmethod
    def _read_config(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc

    def _setup_window(self):
        self.root.title("NOORIX AI — See Beyond. Understand More.")
        self.root.geometry("1320x850")
        self.root.minsize(1040, 700)
        self.root.configure(bg=C["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _label(self, parent, text, size=10, color=None, bold=False, **kwargs):
        return tk.Label(
            parent, text=text, bg=kwargs.pop("bg", parent.cget("bg")),
            fg=color or C["text"], font=("Segoe UI", size, "bold" if bold else "normal"),
            **kwargs,
        )

    def _panel(self, parent, bg=None, padx=16, pady=14):
        return tk.Frame(parent, bg=bg or C["panel"], padx=padx, pady=pady,
                        highlightbackground=C["line"], highlightthickness=1)

    def _button(self, parent, text, command, primary=False, compact=False):
        bg = C["cyan"] if primary else C["panel2"]
        fg = "#06141c" if primary else C["text"]
        button = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=C["blue"] if primary else C["line"],
            activeforeground="#07131d" if primary else C["text"],
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 9 if compact else 10, "bold"),
            padx=12 if compact else 15, pady=8 if compact else 11,
            anchor="w",
        )
        return button

    def _build_ui(self):
        shell = tk.Frame(self.root, bg=C["bg"])
        shell.pack(fill="both", expand=True)

        side = tk.Frame(shell, bg=C["sidebar"], width=218, padx=18, pady=22)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = tk.Frame(side, bg=C["sidebar"])
        brand.pack(fill="x", pady=(2, 28))
        self._label(brand, "N", 22, C["cyan"], True, bg=C["panel2"], width=2, height=1).pack(side="left")
        name = tk.Frame(brand, bg=C["sidebar"])
        name.pack(side="left", padx=9)
        self._label(name, "NOORIX AI", 15, C["text"], True).pack(anchor="w")
        self._label(name, "SEE BEYOND", 7, C["cyan"], True).pack(anchor="w")

        self._label(side, "WORKSPACE", 8, C["muted"], True).pack(anchor="w", pady=(0, 8))
        self._nav(side, "◈", "Overview", True)
        self._nav(side, "◉", "Live vision", False)
        self._nav(side, "Aa", "Text reading", False)
        self._nav(side, "⌗", "Scene analysis", False)
        self._nav(side, "⚙", "System status", False)
        self._nav(side, "✦", "Agent console", False)
        self._nav(side, "◷", "Session history", False)
        self._nav(side, "⚙", "Settings", False)

        spacer = tk.Frame(side, bg=C["sidebar"])
        spacer.pack(fill="both", expand=True)
        privacy = self._panel(side, bg=C["panel3"], padx=12, pady=12)
        privacy.pack(fill="x", pady=(10, 14))
        note = self._label(
            privacy,
            "PRIVACY FIRST\\n\\nFrames are processed locally.\\nNo automatic image saving.",
            9, C["muted"], justify="left", wraplength=165,
        )
        note.pack(anchor="w")
        self._label(side, "POWERED BY MRS", 8, C["muted"], True).pack(anchor="w")

        main = tk.Frame(shell, bg=C["bg"], padx=24, pady=20)
        main.pack(side="left", fill="both", expand=True)

        top = tk.Frame(main, bg=C["bg"])
        top.pack(fill="x")
        heading = tk.Frame(top, bg=C["bg"])
        heading.pack(side="left")
        self._label(heading, "INTELLIGENT ACCESSIBILITY PLATFORM", 8, C["cyan"], True).pack(anchor="w")
        self._label(heading, "Your vision workspace", 23, C["text"], True).pack(anchor="w", pady=(4, 0))
        righttop = tk.Frame(top, bg=C["bg"])
        righttop.pack(side="right", anchor="center")
        self._button(righttop, "↻  Refresh readiness", self.refresh_overview, compact=True).pack(side="left", padx=(0, 8))
        self._label(righttop, "●  LOCAL MODE", 9, C["green"], True,
                    bg=C["panel"], padx=12, pady=8).pack(side="right")

        metrics = tk.Frame(main, bg=C["bg"])
        metrics.pack(fill="x", pady=(20, 16))
        self.camera_metric = self._metric(metrics, "CAMERA INPUT", "STANDBY", "Device 0 · Not connected", C["cyan"])
        self.ocr_metric = self._metric(metrics, "PRINTED ENGLISH OCR", "ADAPTER READY", "Tesseract executable needed", C["blue"])
        self.model_metric = self._metric(metrics, "AI INFERENCE", "AWAITING MODELS", "ONNX weights not configured", C["purple"])

        workspace = tk.Frame(main, bg=C["bg"])
        workspace.pack(fill="both", expand=True)
        left = tk.Frame(workspace, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        right = tk.Frame(workspace, bg=C["bg"], width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        preview_panel = self._panel(left, padx=0, pady=0)
        preview_panel.pack(fill="both", expand=True)
        phead = tk.Frame(preview_panel, bg=C["panel"], padx=18, pady=13)
        phead.pack(fill="x")
        self._label(phead, "LIVE VISION", 10, C["text"], True).pack(side="left")
        self.preview_state = self._label(phead, "●  CAMERA OFF", 8, C["muted"], True)
        self.preview_state.pack(side="right")
        self.preview = tk.Label(
            preview_panel, text="NO LIVE FEED\n\nConnect your camera to start seeing.",
            bg="#060c16", fg=C["muted"], font=("Segoe UI", 12),
            justify="center", compound="center",
        )
        self.preview.pack(fill="both", expand=True, padx=12, pady=12)
        pfoot = tk.Frame(preview_panel, bg=C["panel"], padx=14, pady=12)
        pfoot.pack(fill="x")
        self._button(pfoot, "▶  Start camera", self.start_camera, True).pack(side="left", padx=(0, 8))
        self._button(pfoot, "■  Stop", self.stop_camera, False, True).pack(side="left")
        self._label(pfoot, "IMAGE POSITION ONLY · NOT NAVIGATION", 7, C["muted"], True).pack(side="right")

        self._label(right, "AI CONTROL DECK", 8, C["muted"], True).pack(anchor="w", pady=(0, 8))
        actions = self._panel(right, padx=14, pady=14)
        actions.pack(fill="x")
        self._label(actions, "Quick actions", 14, C["text"], True).pack(anchor="w", pady=(0, 12))
        self._button(actions, "Aa   Read printed English", self.read_text, True).pack(fill="x", pady=4)
        self._button(actions, "⌗   Analyze objects & currency", self.analyze).pack(fill="x", pady=4)
        self._button(actions, "♫   Speak latest result", self.speak_result).pack(fill="x", pady=4)

        self._label(right, "INTELLIGENCE OUTPUT", 8, C["muted"], True).pack(anchor="w", pady=(18, 8))
        output_panel = self._panel(right, padx=13, pady=12)
        output_panel.pack(fill="both", expand=True)
        self._label(output_panel, "Latest insight", 12, C["text"], True).pack(anchor="w")
        self.result = tk.Text(
            output_panel, height=10, wrap="word", bg=C["panel"], fg=C["muted"],
            insertbackground=C["text"], relief="flat", bd=0, padx=0, pady=10,
            font=("Segoe UI", 9), highlightthickness=0,
        )
        self.result.pack(fill="both", expand=True)
        self.result.insert("1.0", "Your results will appear here.\n\nStart the camera, then choose an action.")
        self.result.configure(state="disabled")

        features = tk.Frame(main, bg=C["bg"])
        features.pack(fill="x", pady=(15, 12))
        self._feature(features, "01", "TEXT READER", "Printed English · OCR", C["blue"])
        self._feature(features, "02", "OBJECT AWARENESS", "Image-relative positions", C["cyan"])
        self._feature(features, "03", "CURRENCY ESTIMATE", "Pakistani banknote classes", C["purple"])

        footer = tk.Frame(main, bg=C["bg"])
        footer.pack(fill="x")
        self.status_dot = self._label(footer, "●", 9, C["cyan"], True)
        self.status_dot.pack(side="left")
        self.status_title = self._label(footer, "SYSTEM READY", 8, C["text"], True)
        self.status_title.pack(side="left", padx=(7, 10))
        self.status_detail = self._label(footer, "Waiting for user action", 8, C["muted"])
        self.status_detail.pack(side="left")
        self._label(main, "Research prototype · Estimates are not authentication · Never use as a replacement for a white cane or guide dog.",
                    8, C["muted"]).pack(anchor="w", pady=(8, 0))

    def _nav(self, parent, symbol, title, active):
        if not hasattr(self, "_nav_buttons"):
            self._nav_buttons = []
        actions = {
            "Overview": self.refresh_overview,
            "Live vision": self.start_camera,
            "Text reading": self.read_text,
            "Scene analysis": self.analyze,
            "System status": self.show_system_status,
            "Agent console": self.show_agents_console,
            "Session history": self.show_session_history,
            "Settings": self.show_settings,
        }
        button = tk.Button(
            parent, text=f"{symbol}   {title}", command=lambda t=title: self._navigate(t),
            bg=C["panel2"] if active else C["sidebar"],
            fg=C["text"] if active else C["muted"],
            activebackground=C["panel2"], activeforeground=C["cyan"],
            relief="flat", bd=0, cursor="hand2", anchor="w",
            font=("Segoe UI", 9, "bold" if active else "normal"),
            padx=12, pady=10,
        )
        button.pack(fill="x", pady=3)
        self._nav_buttons.append((title, button))

    def _navigate(self, title):
        self.active_section = title
        for name, button in getattr(self, "_nav_buttons", []):
            selected = name == title
            button.configure(
                bg=C["panel2"] if selected else C["sidebar"],
                fg=C["text"] if selected else C["muted"],
                font=("Segoe UI", 9, "bold" if selected else "normal"),
            )
        actions = {
            "Overview": self.refresh_overview,
            "Live vision": self.start_camera,
            "Text reading": self.read_text,
            "Scene analysis": self.analyze,
            "System status": self.show_system_status,
            "Agent console": self.show_agents_console,
            "Session history": self.show_session_history,
            "Settings": self.show_settings,
        }
        action = actions.get(title)
        if action:
            action()

    def _record_activity(self, action, summary):
        from datetime import datetime
        self.session_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": str(action),
            "summary": str(summary).replace("\\n", " ")[:240],
        })
        self.session_history = self.session_history[-50:]

    def refresh_overview(self):
        import shutil
        from noorix.paths import resolve_config_path

        if getattr(self, "camera", None) is not None:
            self.camera_metric.configure(text="LIVE", fg=C["green"])
        else:
            self.camera_metric.configure(text="STANDBY", fg=C["cyan"])

        if shutil.which("tesseract"):
            self.ocr_metric.configure(text="READY", fg=C["green"])
        else:
            self.ocr_metric.configure(text="NEEDS SETUP", fg=C["amber"])

        object_config = self.config.get("object_detection", {})
        model_value = object_config.get("model_path") if isinstance(object_config, dict) else None
        if model_value:
            model_path = resolve_config_path(model_value, self.config_path)
            self.model_metric.configure(
                text="MODEL AVAILABLE" if model_path.is_file() else "MODEL MISSING",
                fg=C["green"] if model_path.is_file() else C["amber"],
            )
        else:
            self.model_metric.configure(text="NOT CONFIGURED", fg=C["purple"])

        camera_state = "Camera live" if getattr(self, "camera", None) is not None else "Camera standby"
        self._set_status("OVERVIEW UPDATED", f"{camera_state} · {len(getattr(self, 'session_history', []))} session events", C["cyan"])

    def show_session_history(self):
        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — Session History")
        window.geometry("720x520")
        window.configure(bg=C["bg"])
        self._label(window, "SESSION ACTIVITY", 17, C["text"], True, bg=C["bg"]).pack(anchor="w", padx=20, pady=(18, 4))
        self._label(window, "In-memory event log only · cleared when the app closes", 9, C["muted"], bg=C["bg"]).pack(anchor="w", padx=20, pady=(0, 12))
        output = tk.Text(window, wrap="word", bg=C["panel"], fg=C["text"], relief="flat", bd=0, font=("Consolas", 9), padx=12, pady=12)
        output.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        entries = getattr(self, "session_history", [])
        if entries:
            for item in reversed(entries):
                output.insert("end", f"[{item['time']}] {item['action']}\\n{item['summary']}\\n\\n")
        else:
            output.insert("end", "No activity recorded in this session yet.")
        output.configure(state="disabled")
        self._button(window, "Clear session history", lambda: self._clear_session_history(output), compact=True).pack(anchor="e", padx=20, pady=(0, 16))

    def _clear_session_history(self, output):
        self.session_history.clear()
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("end", "Session history cleared.")
        output.configure(state="disabled")

    def show_settings(self):
        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — Local Settings")
        window.geometry("560x430")
        window.configure(bg=C["bg"])
        self._label(window, "LOCAL CONFIGURATION", 17, C["text"], True, bg=C["bg"]).pack(anchor="w", padx=20, pady=(18, 4))
        self._label(window, "Settings are stored in your local config file, not uploaded.", 9, C["muted"], bg=C["bg"]).pack(anchor="w", padx=20, pady=(0, 16))
        form = self._panel(window, padx=16, pady=14)
        form.pack(fill="x", padx=20)
        self._label(form, "Camera device index", 9, C["muted"], True).pack(anchor="w")
        camera_var = tk.StringVar(value=str(self.camera_index))
        camera_entry = tk.Entry(form, textvariable=camera_var, bg=C["panel3"], fg=C["text"], insertbackground=C["text"], relief="flat")
        camera_entry.pack(fill="x", pady=(5, 14), ipady=7)
        self._label(form, "Object confidence threshold (0.05–0.95)", 9, C["muted"], True).pack(anchor="w")
        object_config = self.config.setdefault("object_detection", {})
        threshold_var = tk.StringVar(value=str(object_config.get("confidence_threshold", 0.3)))
        threshold_entry = tk.Entry(form, textvariable=threshold_var, bg=C["panel3"], fg=C["text"], insertbackground=C["text"], relief="flat")
        threshold_entry.pack(fill="x", pady=(5, 12), ipady=7)

        def save_settings():
            try:
                camera_index = int(camera_var.get())
                threshold = float(threshold_var.get())
                if camera_index < 0 or not 0.05 <= threshold <= 0.95:
                    raise ValueError("Camera index must be non-negative and threshold must be between 0.05 and 0.95.")
                updated = dict(self.config)
                updated_object = dict(updated.get("object_detection", {}))
                updated_object["confidence_threshold"] = threshold
                updated["object_detection"] = updated_object
                self.config_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
                self.config = updated
                self.camera_index = camera_index
                self.agent_coordinator = None
                self.refresh_overview()
                messagebox.showinfo("NOORIX AI", "Local settings saved. Restart the camera if its device index changed.", parent=window)
                window.destroy()
            except (ValueError, OSError, TypeError) as exc:
                messagebox.showerror("Settings not saved", str(exc), parent=window)

        self._button(window, "Save local settings", save_settings, True).pack(anchor="e", padx=20, pady=16)

    def _get_agent_coordinator(self):
        if self.agent_coordinator is None:
            from noorix.agents import (
                AgentCoordinator, CurrencyAgent, ObjectDetectionAgent,
                OCRAgent, SafetyAgent, SpeechAgent, SystemHealthAgent,
                FrameQualityAgent, VoiceCommandAgent,
            )
            self.agent_coordinator = AgentCoordinator([
                OCRAgent(),
                ObjectDetectionAgent(self.config.get("object_detection", {}), config_path=self.config_path),
                CurrencyAgent(self.config.get("currency", {}), config_path=self.config_path),
                SpeechAgent(),
                SafetyAgent(),
                SystemHealthAgent(self.config_path),
                FrameQualityAgent(),
                VoiceCommandAgent(),
            ])
        return self.agent_coordinator

    def show_agents_console(self):
        coordinator = self._get_agent_coordinator()
        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — Specialist Agent Console")
        window.geometry("780x620")
        window.minsize(680, 520)
        window.configure(bg=C["bg"])

        header = tk.Frame(window, bg=C["bg"], padx=20, pady=18)
        header.pack(fill="x")
        self._label(header, "SPECIALIST AGENTS", 9, C["cyan"], True).pack(anchor="w")
        self._label(header, "Local intelligence control", 19, C["text"], True).pack(anchor="w", pady=(4, 0))
        self._label(header, "Each agent performs one bounded task. Missing models are shown honestly.", 9, C["muted"]).pack(anchor="w", pady=(5, 0))

        body = tk.Frame(window, bg=C["bg"], padx=20, pady=8)
        body.pack(fill="both", expand=True)
        list_panel = self._panel(body, padx=12, pady=10)
        list_panel.pack(side="left", fill="y", padx=(0, 12))
        output_panel = self._panel(body, padx=12, pady=10)
        output_panel.pack(side="left", fill="both", expand=True)
        self._label(list_panel, "REGISTERED AGENTS", 8, C["muted"], True).pack(anchor="w", pady=(0, 8))
        output = tk.Text(output_panel, wrap="word", bg=C["panel"], fg=C["text"],
                         relief="flat", bd=0, font=("Segoe UI", 9), padx=10, pady=10)
        output.pack(fill="both", expand=True)
        output.insert("1.0", "Choose an agent to run. Camera-based agents require an active camera.\n")
        output.configure(state="disabled")

        def show_result(result):
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("end", f"AGENT: {result.agent.upper()}\nSTATE: {result.state.value.upper()}\n\n{result.summary}\n")
            for warning in result.warnings:
                output.insert("end", "\n⚠ " + warning)
            if isinstance(result.data, dict):
                output.insert("end", "\n\nDETAILS\n")
                for key, value in result.data.items():
                    output.insert("end", f"{key}: {value}\n")
            output.configure(state="disabled")

        def run_named(name, payload=None):
            result = coordinator.run(name, payload)
            show_result(result)

        entries = [
            ("OCR Agent", "ocr", lambda: self.current_frame),
            ("Object Detection", "object_detection", lambda: self.current_frame),
            ("Currency Agent", "currency", lambda: self.current_frame),
            ("Frame Quality", "frame_quality", lambda: self.current_frame),
            ("Voice Command Parser", "voice_command", lambda: simpledialog.askstring("NOORIX command", "Enter a supported English / Urdu command (text only):", parent=window)),
            ("Speech Agent", "speech", lambda: self.result.get("1.0", "end").strip()),
            ("Safety Agent", "safety", lambda: []),
            ("System Health", "system_health", lambda: None),
        ]
        for title, name, payload in entries:
            row = tk.Frame(list_panel, bg=C["panel"], pady=3)
            row.pack(fill="x")
            self._label(row, title, 9, C["text"], True, bg=C["panel"]).pack(anchor="w", pady=(2, 4))
            button = self._button(row, "Run agent", lambda n=name, p=payload: run_named(n, p()))
            button.pack(fill="x")
            tk.Frame(list_panel, bg=C["line"], height=1).pack(fill="x", pady=7)

        def run_full_vision():
            if self.current_frame is None:
                show_result(coordinator.run("object_detection", None))
                return
            results = coordinator.run_vision(self.current_frame)
            output.configure(state="normal")
            output.delete("1.0", "end")
            for result in results:
                output.insert("end", f"{result.agent.upper()} · {result.state.value.upper()}\n{result.summary}\n")
                for warning in result.warnings:
                    output.insert("end", "⚠ " + warning + "\n")
                output.insert("end", "\n")
            output.configure(state="disabled")

        self._button(list_panel, "Run vision pipeline", run_full_vision, True).pack(fill="x", pady=(8, 0))
        self._label(window, "NO automatic navigation · No image upload · No model accuracy guarantee", 8, C["muted"]).pack(anchor="w", padx=22, pady=(0, 12))

    def show_system_status(self):
        from noorix.agents import SystemHealthAgent
        result = SystemHealthAgent(self.config_path).run()
        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — System Health")
        window.geometry("650x480")
        window.configure(bg=C["bg"])
        self._label(window, "SYSTEM HEALTH", 17, C["text"], True, bg=C["bg"]).pack(anchor="w", padx=20, pady=(18, 8))
        text = tk.Text(window, wrap="word", bg=C["panel"], fg=C["text"],
                       relief="flat", bd=0, font=("Consolas", 10), padx=14, pady=14)
        text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        for key, value in result.data.items():
            text.insert("end", f"{key}: {value}\n")
        text.configure(state="disabled")

    def _metric(self, parent, title, value, detail, accent):
        card = self._panel(parent, padx=14, pady=11)
        card.pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._label(card, title, 8, C["muted"], True).pack(anchor="w")
        val = self._label(card, value, 12, accent, True)
        val.pack(anchor="w", pady=(7, 3))
        self._label(card, detail, 8, C["muted"]).pack(anchor="w")
        return val

    def _feature(self, parent, number, title, detail, accent):
        card = self._panel(parent, padx=13, pady=10)
        card.pack(side="left", fill="x", expand=True, padx=(0, 9))
        self._label(card, number, 8, accent, True).pack(anchor="w")
        self._label(card, title, 9, C["text"], True).pack(anchor="w", pady=(4, 2))
        self._label(card, detail, 8, C["muted"]).pack(anchor="w")

    def _set_status(self, title, detail, color):
        self.status_title.configure(text=title)
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
            self.camera_metric.configure(text="LIVE", fg=C["green"])
            self.preview_state.configure(text="●  LIVE FEED", fg=C["green"])
            self._set_status("CAMERA ACTIVE", "Frames are processed locally and not saved.", C["green"])
            self._update_frame()
        except Exception as exc:
            self.camera = None
            self.camera_metric.configure(text="UNAVAILABLE", fg=C["amber"])
            self._set_status("CAMERA ERROR", str(exc), C["amber"])
            messagebox.showerror("NOORIX AI — Camera error", str(exc))

    def _update_frame(self):
        if self.camera is None:
            return
        try:
            from PIL import Image, ImageTk
            frame = self.camera.read()
            self.current_frame = frame.copy()
            width = max(360, self.preview.winfo_width() - 18)
            height = max(260, self.preview.winfo_height() - 18)
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self.preview.configure(image=self.photo, text="")
            self.after_id = self.root.after(33, self._update_frame)
        except Exception as exc:
            self._set_status("PREVIEW STOPPED", str(exc), C["amber"])
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
        if hasattr(self, "camera_metric"):
            self.camera_metric.configure(text="STANDBY", fg=C["cyan"])
            self.preview_state.configure(text="●  CAMERA OFF", fg=C["muted"])
            self.preview.configure(image="", text="NO LIVE FEED\n\nConnect your camera to start seeing.")
        self.photo = None
        self.current_frame = None
        if hasattr(self, "status_title"):
            self._set_status("SYSTEM READY", "Camera released safely.", C["cyan"])

    def read_text(self):
        if self.current_frame is None:
            self._set_result("Start the camera first.")
            self._set_status("CAMERA REQUIRED", "Start the camera before reading text.", C["amber"])
            return
        try:
            from noorix.ocr import PrintedEnglishOCR
            output = PrintedEnglishOCR().read(self.current_frame) or "No clear printed English text found."
            self._set_result(output)
            if hasattr(self, "_record_activity"):
                self._record_activity("Text reading", output)
            self.ocr_metric.configure(text="REQUEST COMPLETED", fg=C["green"])
            self._set_status("TEXT READING COMPLETE", "English OCR output is ready.", C["green"])
        except Exception as exc:
            self._set_result(f"OCR unavailable: {exc}")
            self._set_status("OCR UNAVAILABLE", "Tesseract executable is required.", C["amber"])

    def _resolve_model_path(self, value):
        from noorix.paths import resolve_config_path
        return resolve_config_path(value, self.config_path)

    def analyze(self):
        if self.current_frame is None:
            self._set_result("Start the camera first.")
            self._set_status(
                "CAMERA REQUIRED",
                "Start the camera before scene analysis.",
                C["amber"],
            )
            return

        try:
            coordinator = self._get_agent_coordinator()
            results = coordinator.run_vision(self.current_frame)

            sections = []
            for result in results:
                sections.append(
                    f"AGENT: {result.agent.upper()}\n"
                    f"STATE: {result.state.value.upper()}\n"
                    f"{result.summary}"
                )

                for warning in result.warnings:
                    sections.append(f"WARNING: {warning}")

                if result.data is not None:
                    sections.append(
                        "DETAILS:\n" +
                        json.dumps(
                            result.data,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    )

            output = "\n\n".join(sections)
            self._set_result(output or "No agent results were returned.")
            if hasattr(self, "_record_activity"):
                self._record_activity("Scene analysis", output or "No agent results were returned.")

            model_results = [
                result for result in results
                if result.agent in ("object_detection", "currency")
            ]
            completed = [
                result for result in model_results
                if result.successful
            ]
            errors = [
                result for result in results
                if result.state.value == "error"
            ]

            if errors:
                self.model_metric.configure(
                    text="SOME TASKS FAILED",
                    fg=C["amber"],
                )
                self._set_status(
                    "PARTIAL ANALYSIS",
                    "One or more agents could not complete their tasks.",
                    C["amber"],
                )
            elif completed:
                self.model_metric.configure(
                    text="INFERENCE COMPLETE",
                    fg=C["green"],
                )
                self._set_status(
                    "ANALYSIS COMPLETE",
                    "Agent results are estimates and require independent validation.",
                    C["green"],
                )
            else:
                self.model_metric.configure(
                    text="AWAITING MODELS",
                    fg=C["purple"],
                )
                self._set_status(
                    "MODELS NOT READY",
                    "Configure compatible models before requesting predictions.",
                    C["amber"],
                )

        except Exception as exc:
            self._set_result(f"Analysis unavailable: {type(exc).__name__}: {exc}")
            self._set_status(
                "ANALYSIS ERROR",
                "Check agent registration and configuration.",
                C["amber"],
            )

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
                try:
                    self.root.after(0, lambda: self._set_status("SPEECH UNAVAILABLE", error, C["amber"]))
                except tk.TclError:
                    pass
        threading.Thread(target=worker, daemon=True).start()
        self._set_status("SPEECH REQUESTED", "Uses an installed local system voice.", C["cyan"])

    def close(self):
        self.stop_camera()
        self.root.destroy()


def launch(camera_index: int = 0, config_path: Path | None = None) -> int:
    root = tk.Tk()
    NoorixApp(root, camera_index, config_path or Path("config.json"))
    root.mainloop()
    return 0

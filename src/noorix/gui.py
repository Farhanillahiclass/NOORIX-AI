"""NOORIX AI — polished desktop command center.

A lightweight Tkinter interface with a camera workspace, honest capability
states, accessible contrast, and working local actions. No web service required.
"""
from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk


C = {
    "bg": "#07111f", "sidebar": "#0b1728", "panel": "#101f33",
    "panel2": "#162a43", "panel3": "#0b1728", "line": "#21364f",
    "text": "#e6f0ff", "muted": "#91a4bd", "cyan": "#22d3ee",
    "blue": "#60a5fa", "purple": "#a78bfa", "amber": "#fbbf24",
    "green": "#34d399", "red": "#fb7185",
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
        self.auto_mode = False
        self.auto_after_id = None
        self.auto_busy = False
        self.last_spoken_signature = None
        self.active_section = "Overview"
        self.config = self._read_config(config_path)
        try:
            self.camera_index = int(self.config.get("camera_index", camera_index))
        except (TypeError, ValueError):
            self.camera_index = camera_index
        self._setup_window()
        self._build_ui()
        self._set_status("SYSTEM READY", "Start the camera to begin a local session.", C["cyan"])
        self.refresh_overview()

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
        self.root.geometry("1360x820")
        self.root.minsize(1080, 680)
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
        """Dark navy AI workspace with consistent contrast and hybrid navigation."""
        shell = tk.Frame(self.root, bg=C["bg"])
        shell.pack(fill="both", expand=True)

        # Persistent navigation rail.
        side = tk.Frame(shell, bg=C["sidebar"], width=210, padx=16, pady=18,
                        highlightbackground=C["line"], highlightthickness=1)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        brand = tk.Frame(side, bg=C["sidebar"])
        brand.pack(fill="x", pady=(2, 28))
        logo = tk.Frame(brand, bg="#172b4d", width=42, height=42)
        logo.pack(side="left")
        logo.pack_propagate(False)
        self._label(logo, "N", 23, "#ffffff", True, bg="#172b4d").pack(expand=True)
        brand_text = tk.Frame(brand, bg=C["sidebar"])
        brand_text.pack(side="left", padx=10)
        self._label(brand_text, "NOORIX AI", 15, C["text"], True).pack(anchor="w")
        self._label(brand_text, "SEE BEYOND", 7, C["cyan"], True).pack(anchor="w")

        self._label(side, "WORKSPACE", 8, C["muted"], True).pack(anchor="w", pady=(0, 9))
        self._nav(side, "⌂", "Overview", True)
        self._nav(side, "◉", "Live vision", False)
        self._nav(side, "Aa", "Text reading", False)
        self._nav(side, "⌗", "Scene analysis", False)
        self._label(side, "MANAGEMENT", 8, C["muted"], True).pack(anchor="w", pady=(20, 9))
        self._nav(side, "✦", "Agent console", False)
        self._nav(side, "✓", "Model testing", False)
        self._nav(side, "▤", "System status", False)
        self._nav(side, "◷", "Session history", False)
        self._nav(side, "⚙", "Settings", False)
        self._nav(side, "◎", "About Farhan", False)

        spacer = tk.Frame(side, bg=C["sidebar"])
        spacer.pack(fill="both", expand=True)
        privacy = self._panel(side, bg=C["panel3"], padx=11, pady=11)
        privacy.pack(fill="x", pady=(8, 12))
        self._label(privacy, "PRIVATE BY DESIGN", 8, C["cyan"], True).pack(anchor="w")
        self._label(privacy, "Frames are processed locally. Nothing is saved automatically.",
                    8, C["muted"], justify="left", wraplength=165).pack(anchor="w", pady=(5, 0))
        self._label(side, "POWERED BY MRS", 8, C["muted"], True).pack(anchor="w")

        # Main workspace and hybrid top command bar.
        main = tk.Frame(shell, bg=C["bg"], padx=22, pady=16)
        main.pack(side="left", fill="both", expand=True)
        topbar = tk.Frame(main, bg=C["panel"], padx=15, pady=10,
                          highlightbackground=C["line"], highlightthickness=1)
        topbar.pack(fill="x", pady=(0, 18))
        self._label(topbar, "MRS  /  NOORIX AI", 9, C["text"], True).pack(side="left")
        self._label(topbar, "LOCAL WORKSPACE", 8, C["muted"], True).pack(side="left", padx=(18, 0))
        top_actions = tk.Frame(topbar, bg=C["panel"])
        top_actions.pack(side="right")
        self._button(top_actions, "Refresh", self.refresh_overview, compact=True).pack(side="left", padx=(0, 7))
        self._label(top_actions, "● LOCAL MODE", 8, C["green"], True,
                    bg=C["panel2"], padx=10, pady=7).pack(side="left")

        title_row = tk.Frame(main, bg=C["bg"])
        title_row.pack(fill="x")
        heading = tk.Frame(title_row, bg=C["bg"])
        heading.pack(side="left")
        self._label(heading, "INTELLIGENCE OVERVIEW", 8, C["cyan"], True).pack(anchor="w")
        self._label(heading, "Your vision workspace", 24, C["text"], True).pack(anchor="w", pady=(4, 0))
        self._label(heading, "A clear view of your camera, recognition tools and local AI agents.",
                    9, C["muted"]).pack(anchor="w", pady=(4, 0))
        self.auto_button = self._button(title_row, "Enable Auto Scan", self.toggle_auto_mode, True, True)
        self.auto_button.pack(side="right", anchor="center")

        metrics = tk.Frame(main, bg=C["bg"])
        metrics.pack(fill="x", pady=(18, 16))
        self.camera_metric = self._metric(metrics, "CAMERA INPUT", "STANDBY",
                                          f"Device {self.camera_index} · Not connected", C["blue"])
        self.ocr_metric = self._metric(metrics, "PRINTED TEXT OCR", "CHECKING",
                                       "Tesseract availability", C["cyan"])
        self.model_metric = self._metric(metrics, "OBJECT MODEL", "CHECKING",
                                         "Model configuration", C["purple"])
        self.currency_metric = self._metric(metrics, "CURRENCY MODEL", "NOT CONFIGURED",
                                             "Verified model required", C["amber"])

        workspace = tk.Frame(main, bg=C["bg"])
        workspace.pack(fill="both", expand=True)
        left = tk.Frame(workspace, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        right = tk.Frame(workspace, bg=C["bg"], width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        preview_panel = self._panel(left, padx=0, pady=0)
        preview_panel.pack(fill="both", expand=True)
        phead = tk.Frame(preview_panel, bg=C["panel"], padx=16, pady=12)
        phead.pack(fill="x")
        self._label(phead, "LIVE VISION CANVAS", 9, C["text"], True).pack(side="left")
        self.preview_state = self._label(phead, "● CAMERA OFF", 8, C["muted"], True)
        self.preview_state.pack(side="right")
        self.preview = tk.Label(
            preview_panel,
            text="NO CAMERA FEED\n\nConnect your camera to begin live analysis.",
            bg="#0b1728", fg=C["muted"], font=("Segoe UI", 12),
            justify="center", compound="center",
        )
        self.preview.pack(fill="both", expand=True, padx=12, pady=12)
        pfoot = tk.Frame(preview_panel, bg=C["panel"], padx=12, pady=11)
        pfoot.pack(fill="x")
        self._button(pfoot, "Start camera", self.start_camera, True, True).pack(side="left", padx=(0, 7))
        self._button(pfoot, "Stop", self.stop_camera, False, True).pack(side="left")
        self._label(pfoot, "LOCAL FRAME ANALYSIS · NOT NAVIGATION", 7, C["muted"], True).pack(side="right")

        self._label(right, "TOOLS & RESULTS", 8, C["muted"], True).pack(anchor="w", pady=(0, 8))
        actions = self._panel(right, padx=13, pady=13)
        actions.pack(fill="x", pady=(0, 10))
        self._label(actions, "Quick actions", 13, C["text"], True).pack(anchor="w", pady=(0, 9))
        self._button(actions, "Read printed page", self.read_text, True, True).pack(fill="x", pady=3)
        self._button(actions, "Analyze objects & currency", self.analyze, False, True).pack(fill="x", pady=3)
        self._button(actions, "Speak latest result", self.speak_result, False, True).pack(fill="x", pady=3)

        output_panel = self._panel(right, padx=12, pady=11)
        output_panel.pack(fill="both", expand=True)
        self._label(output_panel, "LATEST INTELLIGENCE", 9, C["text"], True).pack(anchor="w")
        self.result = tk.Text(
            output_panel, height=12, wrap="word", bg=C["panel3"], fg=C["text"],
            insertbackground=C["text"], relief="flat", bd=0, padx=9, pady=9,
            font=("Segoe UI", 9), highlightthickness=1,
            highlightbackground=C["line"],
        )
        self.result.pack(fill="both", expand=True, pady=(8, 0))
        self.result.insert("1.0", "No analysis yet.\n\nStart the camera and choose a task, or enable Auto Scan.")
        self.result.configure(state="disabled")

        footer = tk.Frame(main, bg=C["bg"], pady=9)
        footer.pack(fill="x")
        self.status_dot = self._label(footer, "●", 9, C["cyan"], True)
        self.status_dot.pack(side="left")
        self.status_title = self._label(footer, "SYSTEM READY", 8, C["text"], True)
        self.status_title.pack(side="left", padx=(7, 9))
        self.status_detail = self._label(footer, "Waiting for user action", 8, C["muted"])
        self.status_detail.pack(side="left")
        self._label(main, "NOORIX AI · Powered by MRS  |  Research prototype · Not a mobility or navigation aid.",
                    7, C["muted"]).pack(anchor="w", pady=(2, 0))

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
            "Model testing": self.show_model_testing,
            "Session history": self.show_session_history,
            "Settings": self.show_settings,
            "About Farhan": self.show_about,
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
            "About Farhan": self.show_about,
        }
        action = actions.get(title)
        if action:
            action()

    def _record_activity(self, action, summary):
        from datetime import datetime
        self.session_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": str(action),
            "summary": str(summary).replace("\n", " ")[:240],
        })
        self.session_history = self.session_history[-50:]

    def refresh_overview(self):
        import shutil
        from noorix.paths import resolve_config_path

        camera_live = getattr(self, "camera", None) is not None
        self.camera_metric.configure(
            text="LIVE" if camera_live else "STANDBY",
            fg=C["green"] if camera_live else C["blue"],
        )
        camera_detail = f"Device {self.camera_index} · " + ("Connected" if camera_live else "Not connected")
        if camera_live:
            capture_info = self.camera.capture_info
            actual_width = capture_info.get("actual_width", 0)
            actual_height = capture_info.get("actual_height", 0)
            if actual_width and actual_height:
                camera_detail += f" · {actual_width}×{actual_height}"
        self.camera_input_detail.configure(text=camera_detail)

        tesseract = shutil.which("tesseract")
        ocr_config = self.config.get("ocr", {})
        language = ocr_config.get("language", "eng") if isinstance(ocr_config, dict) else "eng"
        if not tesseract:
            ocr_state, ocr_detail, ocr_color = "NEEDS SETUP", "Tesseract executable not found", C["amber"]
        else:
            try:
                import pytesseract
                installed_languages = set(pytesseract.get_languages(config=""))
                requested = set(language.split("+"))
                missing = sorted(requested - installed_languages)
                if missing:
                    ocr_state = "LANGUAGE DATA MISSING"
                    ocr_detail = "Install Tesseract data: " + ", ".join(missing)
                    ocr_color = C["amber"]
                else:
                    ocr_state, ocr_detail, ocr_color = "READY", f"OCR languages: {language}", C["green"]
            except Exception:
                ocr_state, ocr_detail, ocr_color = "CHECK REQUIRED", "Could not verify Tesseract language data", C["amber"]
        self.ocr_metric.configure(text=ocr_state, fg=ocr_color)
        self.printed_text_ocr_detail.configure(text=ocr_detail)

        object_config = self.config.get("object_detection", {})
        object_value = object_config.get("model_path") if isinstance(object_config, dict) else None
        object_path = resolve_config_path(object_value, self.config_path) if object_value else None
        object_ready = bool(object_path and object_path.is_file())
        self.model_metric.configure(
            text="MODEL FILE READY" if object_ready else ("MODEL FILE MISSING" if object_value else "NOT CONFIGURED"),
            fg=C["green"] if object_ready else (C["amber"] if object_value else C["purple"]),
        )
        self.object_model_detail.configure(
            text=str(object_path) if object_value else "Configure object_detection.model_path"
        )

        currency_config = self.config.get("currency", {})
        currency_value = currency_config.get("model_path") if isinstance(currency_config, dict) else None
        currency_labels = currency_config.get("labels") if isinstance(currency_config, dict) else None
        currency_path = resolve_config_path(currency_value, self.config_path) if currency_value else None
        currency_ready = bool(currency_path and currency_path.is_file() and currency_labels)
        self.currency_metric.configure(
            text="MODEL FILE READY" if currency_ready else ("MODEL FILE MISSING" if currency_value else "NOT CONFIGURED"),
            fg=C["green"] if currency_ready else (C["amber"] if currency_value else C["amber"]),
        )
        self.currency_model_detail.configure(
            text=str(currency_path) if currency_value else "Requires a compatible trained model and labels"
        )

        camera_state = "Camera live" if camera_live else "Camera standby"
        self._set_status(
            "READINESS REFRESHED",
            f"{camera_state} · OCR {ocr_state.lower()} · Objects {'ready' if object_ready else 'not ready'} · Currency {'ready' if currency_ready else 'not ready'}",
            C["cyan"],
        )

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
                output.insert("end", f"[{item['time']}] {item['action']}\n{item['summary']}\n\n")
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

    def show_about(self):
        """Show the creator profile and selected public project portfolio."""
        import webbrowser

        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — About the Creator")
        window.geometry("780x680")
        window.minsize(640, 540)
        window.configure(bg=C["bg"])

        header = self._panel(window, bg="#172b4d", padx=22, pady=20)
        header.pack(fill="x", padx=18, pady=(18, 12))
        self._label(header, "NOORIX AI  /  ABOUT THE CREATOR", 9, "#67e8f9", True,
                    bg="#172b4d").pack(anchor="w")
        self._label(header, "Muhammad Farhan", 24, "#ffffff", True,
                    bg="#172b4d").pack(anchor="w", pady=(8, 2))
        self._label(header, "AI/ML Student  ·  Junior Researcher at MRS  ·  Builder of practical AI systems",
                    10, "#dbeafe", bg="#172b4d").pack(anchor="w")

        body = tk.Frame(window, bg=C["bg"], padx=20)
        body.pack(fill="both", expand=True)
        bio = (
            "I am an AI/ML student and Junior Researcher at Muslim Revive Skills (MRS), "
            "interested in building useful, privacy-aware software and intelligent systems. "
            "My public project work explores Python development, AI/ML, computer vision, "
            "OCR, local-first assistants, backend engineering and workflow automation. "
            "I learn by designing, implementing, testing and documenting practical projects."
        )
        card = self._panel(body, padx=18, pady=16)
        card.pack(fill="x", pady=(0, 10))
        self._label(card, "ABOUT", 9, C["cyan"], True).pack(anchor="w")
        self._label(card, bio, 10, C["text"], justify="left",
                    wraplength=690).pack(anchor="w", pady=(8, 0))

        projects = [
            ("NOORIX-AI", "Offline-first camera accessibility research prototype: OCR, object detection, PKR denomination estimation and spoken feedback."),
            ("NEXORA", "Hybrid institutional cybersecurity intelligence and research platform with evidence-linked, human-reviewed workflows."),
            ("FlowForge-AI", "A modular n8n workflow library with 20 reusable business automation workflows and validation guides."),
            ("ILLI AI", "A local-first Windows assistant exploring voice interaction and desktop automation."),
            ("Phishing URL Detector", "A URL-text classifier with a documented held-out evaluation and automated tests."),
            ("GYRUS", "A voice-oriented assistant project with media controls, scheduling and a desktop dashboard."),
            ("AutoLoom AI", "A documented library of 100 AI-assisted automation workflows with setup guides and validation."),
            ("CIVID", "A source-verified civilian-impact research dataset with provenance and review requirements."),
            ("PyGuide AI", "A beginner-friendly Python learning chatbot with CLI and Streamlit interfaces."),
            ("Python Handbook", "A structured Python learning resource built from notebooks and practical examples."),
        ]
        project_card = self._panel(body, padx=18, pady=14)
        project_card.pack(fill="both", expand=True, pady=(0, 10))
        self._label(project_card, "SELECTED PROJECTS", 9, C["cyan"], True).pack(anchor="w")
        for name, description in projects:
            line = tk.Frame(project_card, bg=C["panel"])
            line.pack(fill="x", pady=(8, 0))
            self._label(line, name, 10, C["text"], True).pack(anchor="w")
            self._label(line, description, 9, C["muted"], justify="left",
                        wraplength=690).pack(anchor="w", pady=(2, 0))

        links = tk.Frame(body, bg=C["bg"])
        links.pack(fill="x", pady=(0, 14))
        for title, url in [
            ("GitHub", "https://github.com/Farhanillahiclass"),
            ("LinkedIn", "https://www.linkedin.com/in/muhammadfarhanmrs"),
            ("Kaggle", "https://www.kaggle.com/muhammadfarhanmrs"),
            ("Hugging Face", "https://huggingface.co/muhammedfarhan"),
            ("MRS Platform", "https://mrsplatform.tech/"),
        ]:
            self._button(links, title, lambda target=url: webbrowser.open(target),
                         compact=True).pack(side="left", padx=(0, 6))

        self._label(body, "Profile summary based on public portfolio links and repository documentation.",
                    8, C["muted"]).pack(anchor="w", pady=(0, 10))

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
        object_config = self.config.get("object_detection", {})
        if not isinstance(object_config, dict):
            object_config = {}
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
                updated["camera_index"] = camera_index
                self.config_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
                OCRAgent, SafetyAgent, SpeechAgent, SystemHealthAgent, WhisperASRAgent,
                FrameQualityAgent, VoiceCommandAgent,
            )
            from noorix.ocr import PrintedEnglishOCR
            ocr_config = self.config.get("ocr", {})
            ocr_language = ocr_config.get("language", "eng") if isinstance(ocr_config, dict) else "eng"
            self.agent_coordinator = AgentCoordinator([
                OCRAgent(adapter=PrintedEnglishOCR(language=ocr_language)),
                ObjectDetectionAgent(self.config.get("object_detection", {}), config_path=self.config_path),
                CurrencyAgent(self.config.get("currency", {}), config_path=self.config_path),
                SpeechAgent(),
                WhisperASRAgent(self.config.get('asr', {}), config_path=self.config_path),
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

        def choose_audio_file():
            selected = filedialog.askopenfilename(
                parent=window,
                title="Select audio for local Whisper transcription",
                filetypes=[("Audio files", "*.wav *.flac *.ogg *.mp3 *.m4a"), ("All files", "*.*")],
            )
            return selected or None

        entries = [
            ("OCR Agent", "ocr", lambda: self.current_frame),
            ("Object Detection", "object_detection", lambda: self.current_frame),
            ("Currency Agent", "currency", lambda: self.current_frame),
            ("Frame Quality", "frame_quality", lambda: self.current_frame),
            ("Voice Command Parser", "voice_command", lambda: simpledialog.askstring("NOORIX command", "Enter a supported English / Urdu command (text only):", parent=window)),
            ("Speech Agent", "speech", lambda: self.result.get("1.0", "end").strip()),
            ("Whisper Speech-to-Text", "speech_to_text", choose_audio_file),
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

    def show_model_testing(self):
        """Open the dashboard-driven local model testing centre."""
        import webbrowser

        window = tk.Toplevel(self.root)
        window.title("NOORIX AI — Model Testing Center")
        window.geometry("980x740")
        window.minsize(760, 600)
        window.configure(bg=C["bg"])

        header = tk.Frame(window, bg=C["bg"], padx=22, pady=18)
        header.pack(fill="x")
        self._label(header, "MODEL TESTING CENTER", 9, C["cyan"], True).pack(anchor="w")
        self._label(header, "Test from your dashboard", 21, C["text"], True).pack(anchor="w", pady=(4, 0))
        self._label(
            header,
            "Run every supported smoke test together or select one model below. Tests execute locally and sequentially.",
            9, C["muted"], wraplength=900, justify="left"
        ).pack(anchor="w", pady=(6, 0))

        controls = tk.Frame(window, bg=C["bg"], padx=22, pady=8)
        controls.pack(fill="x")
        run_all_button = self._button(controls, "RUN ALL TESTS", None, True)
        run_all_button.pack(side="left", padx=(0, 9))
        progress = ttk.Progressbar(controls, mode="indeterminate", length=190)
        progress.pack(side="left", padx=(4, 10))
        progress_label = self._label(controls, "Ready", 8, C["muted"], True)
        progress_label.pack(side="left")

        single_panel = self._panel(window, padx=12, pady=10)
        single_panel.pack(fill="x", padx=22, pady=(4, 10))
        self._label(single_panel, "INDIVIDUAL SMOKE TESTS", 8, C["cyan"], True).pack(anchor="w", pady=(0, 8))
        grid = tk.Frame(single_panel, bg=C["panel"])
        grid.pack(fill="x")
        test_options = [
            ("Package Integrity", "package"),
            ("PKR Currency", "currency"),
            ("YOLO Detection", "objects"),
            ("Printed OCR", "ocr"),
            ("Whisper ASR", "asr"),
            ("Text-to-Speech", "tts"),
        ]
        test_buttons = []
        for index, (label, key) in enumerate(test_options):
            button = self._button(
                grid, label, lambda selected=key: start(selected), compact=True
            )
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=4, pady=4)
            test_buttons.append(button)
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1)

        self._label(
            single_panel,
            "Handwritten OCR, experimental Urdu OCR and live camera validation are not included in these six smoke-test adapters yet. They must not be marked PASS until their dedicated real-input tests exist.",
            8, C["amber"], wraplength=900, justify="left"
        ).pack(anchor="w", pady=(8, 0))

        report_panel = self._panel(window, padx=12, pady=10)
        report_panel.pack(fill="x", padx=22, pady=(0, 10))
        self._label(report_panel, "LATEST REPORTS", 8, C["cyan"], True).pack(anchor="w")
        report_actions = tk.Frame(report_panel, bg=C["panel"])
        report_actions.pack(fill="x", pady=(7, 0))
        json_button = self._button(report_actions, "Open JSON Report", None, compact=True)
        json_button.pack(side="left", padx=(0, 7))
        markdown_button = self._button(report_actions, "Open Markdown Report", None, compact=True)
        markdown_button.pack(side="left")
        json_button.configure(state="disabled")
        markdown_button.configure(state="disabled")
        latest = {"json": None, "markdown": None}

        output_panel = self._panel(window, padx=12, pady=12)
        output_panel.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        output = tk.Text(
            output_panel, wrap="word", bg=C["panel3"], fg=C["text"],
            relief="flat", bd=0, font=("Consolas", 9), padx=12, pady=12
        )
        output.pack(fill="both", expand=True)
        output.insert("end", "READY — Choose RUN ALL TESTS or an individual test.\n")
        output.insert("end", "A PASS means basic local loading/execution only, not accuracy certification.\n")
        output.configure(state="disabled")
        state = {"running": False}

        def append_line(line):
            if not window.winfo_exists():
                return
            output.configure(state="normal")
            output.insert("end", line + "\n")
            output.see("end")
            output.configure(state="disabled")

        def open_report(kind):
            path = latest.get(kind)
            if path and Path(path).is_file():
                webbrowser.open(Path(path).resolve().as_uri())
            else:
                messagebox.showwarning("Report unavailable", "Run a test first to create this report.", parent=window)

        json_button.configure(command=lambda: open_report("json"))
        markdown_button.configure(command=lambda: open_report("markdown"))

        def finish(report):
            if not window.winfo_exists():
                return
            state["running"] = False
            run_all_button.configure(state="normal", text="RUN ALL TESTS")
            for button in test_buttons:
                button.configure(state="normal")
            progress.stop()
            progress_label.configure(text="Finished")
            totals = report.get("totals", {})
            append_line("")
            append_line("TOTALS: " + json.dumps(totals))
            append_line("JSON REPORT: " + report.get("report_json", "not saved"))
            append_line("MARKDOWN REPORT: " + report.get("report_markdown", "not saved"))
            append_line("PRODUCTION READINESS: NOT CERTIFIED")
            latest["json"] = report.get("report_json")
            latest["markdown"] = report.get("report_markdown")
            json_button.configure(state="normal" if latest["json"] else "disabled")
            markdown_button.configure(state="normal" if latest["markdown"] else "disabled")
            self._record_activity("Model testing", "PASS={} FAIL={} SKIPPED={} WARN={}".format(
                totals.get("pass", 0), totals.get("fail", 0),
                totals.get("skipped", 0), totals.get("warn", 0)
            ))
            self.root.after(0, self.refresh_overview)

        def start(test_key=None):
            if state["running"]:
                return
            state["running"] = True
            run_all_button.configure(state="disabled", text="TESTS RUNNING…")
            for button in test_buttons:
                button.configure(state="disabled")
            progress.start(12)
            progress_label.configure(text="Running locally…")
            if test_key:
                append_line(f"Starting individual test: {test_key}")
            else:
                append_line("Starting all configured local smoke tests…")

            def on_progress(item):
                line = "[{}] {} ({:.3f}s): {}".format(
                    item.status, item.name, item.duration_seconds, item.message
                )
                window.after(0, append_line, line)

            def worker():
                try:
                    from noorix.model_testing import run_all_tests, run_single_test
                    report_dir = self.config_path.parent / "reports" / "model-tests"
                    if test_key:
                        report = run_single_test(test_key, self.config_path, report_dir)
                        item = report["results"][0]
                        on_progress(type("ProgressItem", (), item)())
                    else:
                        report = run_all_tests(self.config_path, report_dir, progress=on_progress)
                except Exception as exc:
                    report = {
                        "totals": {"fail": 1}, "results": [],
                        "report_json": "", "report_markdown": ""
                    }
                    window.after(
                        0, append_line,
                        "RUNNER ERROR: {}: {}".format(type(exc).__name__, exc)
                    )
                window.after(0, finish, report)

            threading.Thread(target=worker, name="noorix-dashboard-tests", daemon=True).start()

        run_all_button.configure(command=lambda: start())
        self._label(
            window,
            "Note: the smoke suite currently covers package integrity, PKR, YOLO, printed OCR, Whisper ASR and installed TTS voices. Real camera, handwritten OCR, Urdu OCR and labelled accuracy tests remain separate validation work.",
            8, C["muted"], wraplength=920, justify="left"
        ).pack(anchor="w", padx=24, pady=(0, 14))

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
        detail_label = self._label(card, detail, 8, C["muted"])
        detail_label.pack(anchor="w")
        key = title.lower().replace(" ", "_")
        setattr(self, f"{key}_detail", detail_label)
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
            capture_info = self.camera.capture_info
            actual_width = capture_info.get("actual_width", 0)
            actual_height = capture_info.get("actual_height", 0)
            camera_detail = f"Device {self.camera_index} · Connected"
            if actual_width and actual_height:
                camera_detail += f" · {actual_width}×{actual_height}"
            self.camera_input_detail.configure(text=camera_detail)
            self.preview_state.configure(text="●  LIVE FEED", fg=C["green"])
            self._set_status("CAMERA ACTIVE", camera_detail + " · Frames stay local.", C["green"])
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
        self.disable_auto_mode()
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
            self.preview.configure(image="", text="CAMERA PREVIEW\n\nStart your camera to begin.")
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
            ocr_config = self.config.get("ocr", {})
            language = ocr_config.get("language", "eng") if isinstance(ocr_config, dict) else "eng"
            output = PrintedEnglishOCR(language=language).read(self.current_frame) or "No clear printed text found."
            from noorix.agents.frame_quality import FrameQualityAgent
            quality = FrameQualityAgent().run(self.current_frame)
            quality_notes = list(quality.warnings)
            if quality_notes:
                output += "\n\nIMAGE QUALITY CHECK\n" + "\n".join("• " + note for note in quality_notes)
            else:
                output += "\n\nIMAGE QUALITY CHECK\nFrame quality checks passed."
            self._set_result(output)
            if hasattr(self, "_record_activity"):
                self._record_activity("Text reading", output)
            self.ocr_metric.configure(text="REQUEST COMPLETED", fg=C["green"])
            detail = f"OCR output ready ({language})"
            if quality_notes:
                detail += " · Improve focus, lighting or contrast"
            self._set_status("TEXT READING COMPLETE", detail, C["green"] if not quality_notes else C["amber"])
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


    def toggle_auto_mode(self):
        if self.auto_mode:
            self.disable_auto_mode()
            self._set_status("AUTO SCAN PAUSED", "Automatic analysis has stopped.", C["cyan"])
            return
        if self.camera is None or self.current_frame is None:
            self._set_status("CAMERA REQUIRED", "Start the camera before enabling Auto Scan.", C["amber"])
            return
        self.auto_mode = True
        self.auto_button.configure(text="Pause Auto Scan", bg="#fee2e2", fg="#991b1b")
        self._set_status("AUTO SCAN ACTIVE", "Scanning text and configured recognition models every 5 seconds.", C["green"])
        self._auto_scan_tick()

    def disable_auto_mode(self):
        self.auto_mode = False
        if getattr(self, "auto_after_id", None) is not None:
            try:
                self.root.after_cancel(self.auto_after_id)
            except tk.TclError:
                pass
            self.auto_after_id = None
        if hasattr(self, "auto_button"):
            self.auto_button.configure(text="Enable Auto Scan", bg=C["panel2"], fg=C["text"])

    def _auto_scan_tick(self):
        if not self.auto_mode or self.camera is None:
            return
        if self.auto_busy:
            self.auto_after_id = self.root.after(1000, self._auto_scan_tick)
            return
        if self.current_frame is None:
            self.auto_after_id = self.root.after(1000, self._auto_scan_tick)
            return
        frame = self.current_frame.copy()
        self.auto_busy = True

        def worker():
            try:
                coordinator = self._get_agent_coordinator()
                results = [
                    coordinator.run("frame_quality", frame),
                    coordinator.run("ocr", frame),
                    coordinator.run("object_detection", frame),
                    coordinator.run("currency", frame),
                ]
                self.root.after(0, lambda r=results: self._auto_scan_complete(r))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.root.after(0, lambda e=error: self._auto_scan_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _auto_scan_complete(self, results):
        self.auto_busy = False
        lines = ["AUTO SCAN RESULTS"]
        spoken_parts = ["NOORIX کیمرہ تجزیہ"]
        for result in results:
            lines.append(f"{result.agent.upper()} · {result.state.value.upper()}")
            lines.append(result.summary)
            if result.warnings:
                lines.extend(f"Warning: {warning}" for warning in result.warnings)
            if result.agent == "object_detection" and isinstance(result.data, list):
                for item in result.data[:8]:
                    label = str(item.get("label", "object"))
                    position = str(item.get("position", "unknown"))
                    lines.append(f"  • {label} — {position}")
                    spoken_parts.append(self._urdu_object_phrase(label, position))
            elif result.agent == "currency" and isinstance(result.data, dict):
                denomination = str(result.data.get("denomination", ""))
                confidence = float(result.data.get("confidence", 0.0))
                lines.append(f"  • Possible note: {denomination} ({confidence:.0%})")
                spoken_parts.append(f"ممکنہ کرنسی نوٹ {denomination} ہے۔ اعتماد کا اسکور {confidence:.0%} ہے۔ براہ کرم خود بھی تصدیق کریں۔")
            elif result.agent == "ocr" and result.data:
                lines.append("  • Extracted page text:")
                lines.append(str(result.data)[:1800])
                spoken_parts.append("صفحے پر یہ عبارت پڑھی گئی ہے۔ " + str(result.data)[:700])
            elif result.agent == "ocr" and not result.successful:
                spoken_parts.append("متن پڑھنے کا نظام دستیاب نہیں۔")
        output = "\n".join(lines)
        self._set_result(output)
        self._record_activity("Auto scan", output)
        signature = output[:1200]
        if signature != self.last_spoken_signature:
            self.last_spoken_signature = signature
            self._speak_urdu(" ".join(spoken_parts))
        self._set_status("AUTO SCAN COMPLETE", "Results are estimates; missing models are reported as not configured.", C["green"])
        if self.auto_mode and self.camera is not None:
            self.auto_after_id = self.root.after(5000, self._auto_scan_tick)

    def _auto_scan_failed(self, error):
        self.auto_busy = False
        self._set_result("Automatic analysis failed: " + error)
        self._set_status("AUTO SCAN ERROR", error, C["amber"])
        if self.auto_mode and self.camera is not None:
            self.auto_after_id = self.root.after(5000, self._auto_scan_tick)

    @staticmethod
    def _urdu_object_phrase(label, position):
        labels = {
            "person": "شخص", "chair": "کرسی", "table": "میز", "book": "کتاب",
            "bottle": "بوتل", "cup": "کپ", "cell phone": "موبائل فون",
            "laptop": "لیپ ٹاپ", "door": "دروازہ", "car": "گاڑی",
            "backpack": "بیگ", "handbag": "ہینڈ بیگ",
        }
        positions = {"left": "بائیں طرف", "centre": "درمیان میں", "center": "درمیان میں", "right": "دائیں طرف"}
        return f"{labels.get(label.lower(), label)} تصویر میں {positions.get(position.lower(), position)} نظر آ رہا ہے۔"

    def _speak_urdu(self, text):
        def worker():
            try:
                from noorix.speech import SpeechOutput
                voices = SpeechOutput.available_voices()
                urdu_voice = None
                for voice in voices:
                    description = (voice.get("name", "") + " " + voice.get("id", "")).lower()
                    language_data = str(voice.get("languages", "")).lower()
                    if "urdu" in description or "ur-pk" in description or "ur_" in description or "urdu" in language_data:
                        urdu_voice = voice.get("id")
                        break
                if not urdu_voice:
                    raise RuntimeError("No Urdu-compatible Windows speech voice was detected. Install an Urdu voice in Windows Speech settings.")
                SpeechOutput(voice_id=urdu_voice).speak(text)
            except Exception as exc:
                error = str(exc)
                try:
                    self.root.after(0, lambda e=error: self._set_status("URDU VOICE UNAVAILABLE", e, C["amber"]))
                except tk.TclError:
                    pass
        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        self.disable_auto_mode()
        self.stop_camera()
        self.root.destroy()


def launch(camera_index: int = 0, config_path: Path | None = None) -> int:
    root = tk.Tk()
    NoorixApp(root, camera_index, config_path or Path("config.json"))
    root.mainloop()
    return 0

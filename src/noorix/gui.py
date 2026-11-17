"""Compatibility API for older NOORIX integrations; launches the rebuilt UI."""
import json

from noorix.dashboard import launch

C = {
    "bg": "#07111f", "sidebar": "#091727", "panel": "#0d1c2e",
    "panel2": "#172d45", "line": "#203951", "text": "#eaf3ff",
    "muted": "#91a6bf", "cyan": "#27d9ee", "blue": "#6aa8ff",
    "purple": "#b5a0ff", "amber": "#f4bd55", "green": "#39d6a0",
    "red": "#ff7188",
}


class NoorixApp:
    """Legacy method facade retained for integrations and existing tests."""

    @staticmethod
    def analyze(self):
        if getattr(self, "current_frame", None) is None:
            self._set_result("Start the camera first.")
            self._set_status("CAMERA REQUIRED", "Start the camera before scene analysis.", C["amber"])
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
                    sections.append("DETAILS:\n" + json.dumps(result.data, ensure_ascii=False, indent=2, default=str))
            output = "\n\n".join(sections)
            self._set_result(output or "No agent results were returned.")
            if hasattr(self, "_record_activity"):
                self._record_activity("Scene analysis", output or "No agent results were returned.")
            model_results = [r for r in results if r.agent in ("object_detection", "currency")]
            completed = [r for r in model_results if r.successful]
            errors = [r for r in results if r.state.value == "error"]
            if errors:
                self.model_metric.configure(text="SOME TASKS FAILED", fg=C["amber"])
                self._set_status("PARTIAL ANALYSIS", "One or more agents could not complete their tasks.", C["amber"])
            elif completed:
                self.model_metric.configure(text="INFERENCE COMPLETE", fg=C["green"])
                self._set_status("ANALYSIS COMPLETE", "Agent results are estimates and require independent validation.", C["green"])
            else:
                self.model_metric.configure(text="AWAITING MODELS", fg=C["purple"])
                self._set_status("MODELS NOT READY", "Configure compatible models before requesting predictions.", C["amber"])
        except Exception as exc:
            self._set_result(f"Analysis unavailable: {type(exc).__name__}: {exc}")
            self._set_status("ANALYSIS ERROR", "Check agent registration and configuration.", C["amber"])


__all__ = ["launch", "NoorixApp", "C"]

from types import SimpleNamespace

from noorix.agents.base import AgentResult, AgentState
from noorix.gui import C, NoorixApp


class FakeMetric:
    def __init__(self):
        self.values = {}

    def configure(self, **kwargs):
        self.values.update(kwargs)


def make_app(frame=object(), coordinator=None):
    app = SimpleNamespace()
    app.current_frame = frame
    app.result_text = ""
    app.status = None
    app.metric = FakeMetric()
    app.model_metric = app.metric

    app._set_result = lambda value: setattr(app, "result_text", value)
    app._set_status = lambda *values: setattr(app, "status", values)
    app._get_agent_coordinator = lambda: coordinator

    return app


class FakeCoordinator:
    def __init__(self, results):
        self.results = results
        self.received_frame = None

    def run_vision(self, frame):
        self.received_frame = frame
        return self.results


def test_analyze_requires_camera_frame():
    app = make_app(frame=None)

    NoorixApp.analyze(app)

    assert app.result_text == "Start the camera first."
    assert app.status[0] == "CAMERA REQUIRED"


def test_analyze_displays_agent_results_and_completes():
    frame = object()
    results = [
        AgentResult(
            agent="object_detection",
            state=AgentState.COMPLETED,
            summary="Objects detected.",
            data=[{"label": "chair", "confidence": 0.91}],
            warnings=("Image position is not distance.",),
        ),
        AgentResult(
            agent="currency",
            state=AgentState.COMPLETED,
            summary="Currency estimate completed.",
        ),
    ]
    coordinator = FakeCoordinator(results)
    app = make_app(frame=frame, coordinator=coordinator)

    NoorixApp.analyze(app)

    assert coordinator.received_frame is frame
    assert "AGENT: OBJECT_DETECTION" in app.result_text
    assert "Objects detected." in app.result_text
    assert "WARNING: Image position is not distance." in app.result_text
    assert '"label": "chair"' in app.result_text
    assert app.metric.values["text"] == "INFERENCE COMPLETE"
    assert app.status[0] == "ANALYSIS COMPLETE"


def test_analyze_reports_partial_failure():
    results = [
        AgentResult(
            agent="object_detection",
            state=AgentState.ERROR,
            summary="Detector failed.",
        ),
        AgentResult(
            agent="currency",
            state=AgentState.COMPLETED,
            summary="Currency estimate completed.",
        ),
    ]
    app = make_app(coordinator=FakeCoordinator(results))

    NoorixApp.analyze(app)

    assert "Detector failed." in app.result_text
    assert app.metric.values["text"] == "SOME TASKS FAILED"
    assert app.status[0] == "PARTIAL ANALYSIS"


def test_analyze_handles_unexpected_exception():
    class BrokenCoordinator:
        def run_vision(self, frame):
            raise RuntimeError("test failure")

    app = make_app(coordinator=BrokenCoordinator())

    NoorixApp.analyze(app)

    assert "Analysis unavailable: RuntimeError: test failure" in app.result_text
    assert app.status[0] == "ANALYSIS ERROR"

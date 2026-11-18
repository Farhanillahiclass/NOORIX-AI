from noorix.agents import AgentCoordinator, AgentResult, AgentState


class StubAgent:
    def __init__(self, name, summary=None, state=AgentState.COMPLETED):
        self.name = name
        self.summary = summary or name
        self.state = state
        self.payloads = []

    def run(self, payload=None):
        self.payloads.append(payload)
        return AgentResult(self.name, self.state, self.summary)


def test_vision_pipeline_runs_ocr_and_all_registered_vision_agents_in_order():
    agents = [
        StubAgent("frame_quality"),
        StubAgent("ocr", "Printed text extracted"),
        StubAgent("object_detection"),
        StubAgent("currency"),
        StubAgent("safety"),
    ]
    coordinator = AgentCoordinator(agents)
    frame = object()

    results = coordinator.run_vision(frame)

    assert [result.agent for result in results] == [
        "frame_quality", "ocr", "object_detection", "currency", "safety"
    ]
    assert all(agent.payloads[0] is frame for agent in agents[:4])
    assert agents[4].payloads[0] == results[:-1]


def test_vision_pipeline_skips_optional_agents_that_are_not_registered():
    agents = [StubAgent("object_detection"), StubAgent("safety")]
    results = AgentCoordinator(agents).run_vision(object())

    assert [result.agent for result in results] == ["object_detection", "safety"]
    assert agents[1].payloads[0] == results[:-1]


def test_vision_pipeline_keeps_ocr_setup_failure_visible():
    agents = [
        StubAgent("ocr", "Tesseract is not configured", AgentState.NEEDS_SETUP),
        StubAgent("object_detection"),
        StubAgent("safety"),
    ]
    results = AgentCoordinator(agents).run_vision(object())

    assert results[0].agent == "ocr"
    assert results[0].state == AgentState.NEEDS_SETUP
    assert results[-1].agent == "safety"
    assert agents[-1].payloads[0] == results[:-1]


def test_vision_pipeline_returns_empty_list_without_registered_agents():
    assert AgentCoordinator().run_vision(object()) == []

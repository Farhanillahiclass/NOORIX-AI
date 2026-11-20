"""Small deterministic coordinator for local specialist agents."""
from __future__ import annotations

from typing import Any, Iterable

from noorix.agents.base import Agent, AgentResult, AgentState


class AgentCoordinator:
    """Runs only explicitly requested agents; no hidden autonomous actions."""

    def __init__(self, agents: Iterable[Agent] = ()):
        self._agents: dict[str, Agent] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: Agent) -> None:
        name = getattr(agent, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every agent must have a non-empty name")
        if name in self._agents:
            raise ValueError(f"Agent already registered: {name}")
        if not callable(getattr(agent, "run", None)):
            raise TypeError(f"Agent {name!r} must implement run(payload)")
        self._agents[name] = agent

    def names(self) -> tuple[str, ...]:
        return tuple(self._agents)

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"Unknown NOORIX agent: {name}") from exc

    def run(self, name: str, payload: Any = None) -> AgentResult:
        agent = self.get(name)
        try:
            result = agent.run(payload)
            if not isinstance(result, AgentResult):
                raise TypeError(f"{name} returned an invalid result type")
            if result.agent != name:
                raise ValueError(f"{name} returned a result for {result.agent}")
            return result
        except Exception as exc:
            return AgentResult(
                name,
                AgentState.ERROR,
                f"Agent execution failed ({type(exc).__name__}). Details are omitted to protect local content and paths.",
                warnings=("The task did not complete; do not rely on a prediction.",),
            )

    def run_vision(self, frame: Any) -> list[AgentResult]:
        """Run the available frame-quality, OCR, object and currency agents.

        Optional agents are included only when registered. OCR is run before
        visual object/currency inference so a caller receives one consolidated
        set of observations. The safety agent reviews every result produced.
        """
        results: list[AgentResult] = []
        for name in ("frame_quality", "ocr", "object_detection", "currency"):
            if name in self._agents:
                results.append(self.run(name, frame))
        if "safety" in self._agents:
            results.append(self.run("safety", list(results)))
        return results

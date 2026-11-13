"""Common contracts shared by local NOORIX specialist agents."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

class AgentState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_SETUP = "needs_setup"
    ERROR = "error"

@dataclass(frozen=True)
class AgentResult:
    agent: str
    state: AgentState
    summary: str
    data: Any = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def successful(self) -> bool:
        return self.state == AgentState.COMPLETED

class Agent(Protocol):
    name: str
    description: str
    def run(self, payload: Any = None) -> AgentResult:
        """Execute one bounded task and return a structured result."""

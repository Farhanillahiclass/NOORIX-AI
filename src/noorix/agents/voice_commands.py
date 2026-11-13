"""Deterministic command parser; it does not record audio or execute actions."""
from __future__ import annotations

import re
from typing import Any

from noorix.agents.base import AgentResult, AgentState


class VoiceCommandAgent:
    """Map already-transcribed text to a small, explicit command allowlist.

    Speech-to-text is deliberately a separate capability. This agent accepts
    text only and returns an intent; the caller must confirm and dispatch it.
    """

    name = "voice_command"
    description = "Parse a small allowlist of English, Roman Urdu and Urdu commands."

    _COMMANDS = {
        "start_camera": (
            "start camera", "open camera", "camera start", "camera kholo",
            "camera shuru karo", "camera chalao", "کیمرہ شروع کرو", "کیمرہ کھولو",
        ),
        "stop_camera": (
            "stop camera", "close camera", "camera band karo", "camera roko",
            "کیمرہ بند کرو", "کیمرہ روک دو",
        ),
        "read_text": (
            "read text", "read this", "text parho", "matn parho",
            "متن پڑھو", "یہ پڑھو",
        ),
        "analyze_scene": (
            "analyze scene", "detect objects", "scene analyze karo",
            "manzar ka tajziya karo", "منظر کا تجزیہ کرو", "اشیاء پہچانو",
        ),
        "repeat_result": (
            "repeat", "repeat result", "dobara sunao", "phir sunao",
            "دوبارہ سناؤ", "پھر سناؤ",
        ),
        "stop_speech": (
            "stop speaking", "silence", "awaaz band karo", "bolna band karo",
            "آواز بند کرو", "خاموش ہو جاؤ",
        ),
    }

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.casefold().strip()
        return re.sub(r"[\s،,.!?؟؛:]+", " ", text).strip()

    def run(self, payload: Any = None) -> AgentResult:
        if not isinstance(payload, str) or not payload.strip():
            return AgentResult(
                self.name, AgentState.NEEDS_SETUP,
                "Provide transcribed command text. Microphone speech recognition is not included yet.",
            )
        command = self._normalize(payload)
        for intent, phrases in self._COMMANDS.items():
            if command in {self._normalize(phrase) for phrase in phrases}:
                return AgentResult(
                    self.name, AgentState.COMPLETED,
                    f"Recognized command intent: {intent}.",
                    data={"intent": intent, "requires_user_confirmation": True},
                    warnings=("This parser does not execute commands or control the camera by itself.",),
                )
        return AgentResult(
            self.name, AgentState.NEEDS_SETUP,
            "Command not recognized. Try a supported short command.",
            data={"supported_intents": list(self._COMMANDS)},
        )

"""Offline speech output adapter using the system's installed voices.

Requires the optional speech extra. Voice availability and language support
depend on the operating system; this module never downloads audio.
"""
from __future__ import annotations


class SpeechOutput:
    """Speak text locally through pyttsx3."""

    def __init__(self, rate: int = 165, volume: float = 1.0, voice_id: str | None = None):
        if rate <= 0:
            raise ValueError("rate must be greater than zero")
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id

    def speak(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError("Offline speech requires: pip install -e '.[speech]'") from exc
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()

    @staticmethod
    def available_voices() -> list[dict[str, str]]:
        """Return locally installed voices; this does not guarantee Urdu support."""
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError("Offline speech requires: pip install -e '.[speech]'") from exc
        engine = pyttsx3.init()
        try:
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": [
                        item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)
                        for item in getattr(voice, "languages", [])
                    ],
                }
                for voice in engine.getProperty("voices")
            ]
        finally:
            engine.stop()

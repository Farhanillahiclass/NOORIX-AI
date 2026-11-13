"""Speech output interface placeholder."""


class SpeechOutput:
    """A future adapter for offline speech engines and prepared audio clips."""

    def speak(self, text: str) -> None:
        """Speak text once a supported speech backend has been configured."""
        raise NotImplementedError("Speech backend is not integrated yet")

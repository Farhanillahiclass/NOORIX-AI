import pytest

from noorix.agents import AgentState, FrameQualityAgent, VoiceCommandAgent


def test_frame_quality_requires_frame():
    result = FrameQualityAgent().run()
    assert result.state == AgentState.NEEDS_SETUP


def test_frame_quality_flags_dark_frame():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = FrameQualityAgent().run(frame)
    assert result.successful
    assert result.data["brightness_mean"] == 0.0
    assert any("dark" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("phrase", "intent"),
    [
        ("start camera", "start_camera"),
        ("camera shuru karo", "start_camera"),
        ("کیمرہ بند کرو", "stop_camera"),
        ("متن پڑھو", "read_text"),
        ("منظر کا تجزیہ کرو", "analyze_scene"),
        ("دوبارہ سناؤ", "repeat_result"),
        ("آواز بند کرو", "stop_speech"),
    ],
)
def test_voice_command_agent_recognizes_allowlisted_phrases(phrase, intent):
    result = VoiceCommandAgent().run(phrase)
    assert result.successful
    assert result.data["intent"] == intent
    assert result.data["requires_user_confirmation"] is True


def test_voice_command_agent_does_not_execute_unknown_text():
    result = VoiceCommandAgent().run("go forward quickly")
    assert result.state == AgentState.NEEDS_SETUP
    assert "supported_intents" in result.data


def test_voice_command_agent_explains_transcription_requirement():
    result = VoiceCommandAgent().run()
    assert result.state == AgentState.NEEDS_SETUP
    assert "transcribed" in result.summary

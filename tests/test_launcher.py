"""Checks the one-command local desktop launcher contract."""
from pathlib import Path


def test_git_bash_launcher_starts_dashboard_and_resolves_model_package():
    launcher = Path("NOORIX.sh")
    assert launcher.is_file()
    source = launcher.read_text(encoding="utf-8")
    assert 'NOORIX_MODELS_DIR="/d/NOORIX_MODELS/EXTRACTED"' in source
    assert 'python -m noorix --gui --config "$CONFIG"' in source
    assert 'python -m pip install -e ".[all,pytorch-currency,asr-whisper]"' in source

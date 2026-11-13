"""Minimal application entry point; recognition backends are not wired yet."""

from noorix.capabilities import capability_status


def main() -> int:
    print("NOORIX AI — See Beyond. Understand More.")
    print("Research prototype scaffold; live recognition is not enabled.")
    for name, status in capability_status().items():
        print(f"- {name}: {status}")
    return 0

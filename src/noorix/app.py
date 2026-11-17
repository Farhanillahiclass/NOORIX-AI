"""NOORIX AI command-line application and safe camera preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from noorix.capabilities import capability_status


def camera_preview(device_index: int = 0) -> int:
    """Show webcam frames. Q, Escape, closing the window, or Ctrl+C exits."""
    try:
        import cv2
        from noorix.camera import WebcamCapture
    except ImportError as exc:
        print(f"Camera dependency unavailable: {exc}")
        return 1

    camera = WebcamCapture(device_index=device_index)
    window_name = "NOORIX AI - Camera Preview (Q / Esc to close)"
    print("Starting NOORIX AI camera preview. Focus the preview window and press Q or Esc to close.")
    try:
        camera.start()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.read()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(15)
            if key != -1 and (key & 0xFF) in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        print("Camera stopped safely.")
        return 0
    except KeyboardInterrupt:
        print("\nStop requested. Releasing camera safely.")
        return 0
    except (RuntimeError, cv2.error) as exc:
        print(f"Camera error: {exc}")
        return 1
    finally:
        camera.stop()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


def check_environment(config_path: str | Path = "config.json") -> int:
    print("NOORIX AI — See Beyond. Understand More.")
    print(f"Config: {Path(config_path)}")
    for name, status in capability_status(config_path).items():
        print(f"- {name}: {status}")
    print("- NOTE: model files present does not mean inference has been validated.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NOORIX AI accessibility research prototype")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--camera-preview", action="store_true", help="Open a live webcam preview")
    modes.add_argument("--gui", action="store_true", help="Open the desktop interface")
    modes.add_argument("--check", action="store_true", help="Show installed backend readiness")
    modes.add_argument("--transcribe", type=Path, metavar="AUDIO_FILE", help="Transcribe a local audio file with configured Whisper ASR")
    modes.add_argument("--test-all", action="store_true", help="Run all configured local model smoke tests sequentially")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Optional local model configuration")
    args = parser.parse_args(argv)

    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.camera_preview:
        return camera_preview(args.camera)
    if args.test_all:
        from noorix.model_testing import run_all_tests
        try:
            report = run_all_tests(args.config)
            for item in report["results"]:
                print("[{}] {} ({}s): {}".format(item["status"], item["name"], item["duration_seconds"], item["message"]))
            print("TOTALS: " + json.dumps(report["totals"]))
            print("JSON REPORT: " + report["report_json"])
            print("MARKDOWN REPORT: " + report["report_markdown"])
            print("PRODUCTION READINESS: NOT CERTIFIED — smoke tests are not accuracy or safety validation.")
            return 1 if report["totals"]["fail"] else 0
        except Exception as exc:
            print("Model test runner failed: {}: {}".format(type(exc).__name__, exc))
            return 1
    if args.transcribe:
        try:
            from noorix.agents import WhisperASRAgent
            config = json.loads(args.config.read_text(encoding="utf-8"))
            agent = WhisperASRAgent(config.get("asr", {}), config_path=args.config)
            result = agent.run(args.transcribe)
            print(result.summary)
            for warning in result.warnings:
                print("WARNING:", warning)
            return 0 if result.successful else 1
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read local configuration: {exc}")
            return 1
    if args.gui:
        try:
            from noorix.gui import launch
            return launch(camera_index=args.camera, config_path=args.config)
        except (ImportError, RuntimeError) as exc:
            print(f"Could not start desktop interface: {exc}")
            return 1
    return check_environment(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

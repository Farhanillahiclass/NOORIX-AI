"""NOORIX AI command-line application and safe camera preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from noorix.capabilities import capability_status
from noorix.privacy import safe_error_summary


def camera_preview(device_index: int = 0) -> int:
    """Show webcam frames. Q, Escape, closing the window, or Ctrl+C exits."""
    try:
        import cv2
        from noorix.camera import WebcamCapture
    except ImportError as exc:
        print(safe_error_summary(exc, "Camera dependency unavailable"))
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
    except Exception as exc:
        print(safe_error_summary(exc, "Camera preview failed"))
        return 1
    finally:
        camera.stop()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


def check_environment() -> int:
    print("NOORIX AI — See Beyond. Understand More.")
    for name, status in capability_status().items():
        print(f"- {name}: {status}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NOORIX AI accessibility research prototype")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--camera-preview", action="store_true", help="Open a live webcam preview")
    modes.add_argument("--gui", action="store_true", help="Open the desktop interface")
    modes.add_argument("--check", action="store_true", help="Show installed backend readiness")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Optional local model configuration")
    args = parser.parse_args(argv)

    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.camera_preview:
        return camera_preview(args.camera)
    if args.gui:
        try:
            from noorix.gui import launch
            return launch(camera_index=args.camera, config_path=args.config)
        except Exception as exc:
            print(safe_error_summary(exc, "Could not start desktop interface"))
            return 1
    return check_environment()


if __name__ == "__main__":
    raise SystemExit(main())

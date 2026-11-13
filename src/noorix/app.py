"""NOORIX AI application entry point."""

import argparse

from noorix.capabilities import capability_status


def camera_preview() -> int:
    """Display live webcam preview with safe exit handling."""
    import cv2
    from noorix.camera import WebcamCapture

    print("Starting NOORIX AI camera...")
    print("Press Q or ESC in the camera window to close.")

    try:
        with WebcamCapture() as camera:
            while True:
                frame = camera.read()
                cv2.imshow("NOORIX AI - Camera Preview", frame)

                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):
                    print("Exit key detected.")
                    break

                # Allow the window close button to stop the preview.
                try:
                    if cv2.getWindowProperty(
                        "NOORIX AI - Camera Preview",
                        cv2.WND_PROP_VISIBLE,
                    ) < 1:
                        break
                except cv2.error:
                    break

    except KeyboardInterrupt:
        print("\nCamera interrupted by user.")
    except (RuntimeError, ImportError) as exc:
        print(f"Camera error: {exc}")
        return 1
    finally:
        cv2.destroyAllWindows()

    print("Camera stopped safely.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NOORIX AI")
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Open live webcam preview",
    )
    args = parser.parse_args(argv)

    print("NOORIX AI — See Beyond. Understand More.")

    if args.camera_preview:
        return camera_preview()

    print("Research prototype; live recognition is not enabled.")

    for name, status in capability_status().items():
        print(f"- {name}: {status}")

    print("\nUse --camera-preview to test the webcam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

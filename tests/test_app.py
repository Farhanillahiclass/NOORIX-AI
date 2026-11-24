from noorix.app import main


def test_check_mode_reports_readiness(capsys):
    assert main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "NOORIX AI" in output
    assert "pakistani_currency_estimation" in output


def test_negative_camera_index_is_rejected():
    try:
        main(["--camera-preview", "--camera", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject a negative camera index")


def test_modes_are_mutually_exclusive():
    try:
        main(["--gui", "--check"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject conflicting modes")



def test_camera_preview_error_does_not_print_driver_message(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    from noorix import camera

    class FakeCV2Error(Exception):
        pass

    class BrokenCamera:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise FakeCV2Error("private driver path \x1b[31msecret\x1b[0m")

        def stop(self):
            pass

    fake_cv2 = SimpleNamespace(
        error=FakeCV2Error,
        destroyAllWindows=lambda: None,
        waitKey=lambda _delay: -1,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(camera, "WebcamCapture", BrokenCamera)

    from noorix.app import camera_preview

    assert camera_preview() == 1
    output = capsys.readouterr().out
    assert "Camera preview failed (OSError)." in output
    assert "private driver path" not in output
    assert chr(27) not in output

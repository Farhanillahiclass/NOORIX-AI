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

import pytest

from noorix.camera import WebcamCapture


class FakeCapture:
    def __init__(self, opened=True, frame="frame"):
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return (self.frame is not None, self.frame)

    def release(self):
        self.released = True


def test_camera_requires_start_before_read():
    camera = WebcamCapture(capture_factory=lambda _: FakeCapture())
    with pytest.raises(RuntimeError, match="not started"):
        camera.read()


def test_camera_reads_and_releases():
    fake = FakeCapture(frame="sample")
    camera = WebcamCapture(capture_factory=lambda _: fake)
    camera.start()
    assert camera.read() == "sample"
    camera.stop()
    assert fake.released


def test_camera_releases_device_when_open_fails():
    fake = FakeCapture(opened=False)
    camera = WebcamCapture(capture_factory=lambda _: fake)
    with pytest.raises(RuntimeError, match="Could not open"):
        camera.start()
    assert fake.released


def test_context_manager_releases_camera():
    fake = FakeCapture()
    camera = WebcamCapture(capture_factory=lambda _: fake)
    with camera as active:
        assert active.read() == "frame"
    assert fake.released


def test_failed_frame_read_is_reported():
    fake = FakeCapture(frame=None)
    camera = WebcamCapture(capture_factory=lambda _: fake)
    camera.start()
    with pytest.raises(RuntimeError, match="Could not read"):
        camera.read()
    camera.stop()



def test_camera_releases_handle_when_open_status_raises():
    class BrokenCapture(FakeCapture):
        def isOpened(self):
            raise OSError("driver diagnostic with private detail")

    fake = BrokenCapture()
    camera = WebcamCapture(capture_factory=lambda _: fake)
    with pytest.raises(RuntimeError, match="Could not initialize"):
        camera.start()
    assert fake.released
    assert not camera.is_running
    assert camera.capture_info["release_error"] is None


def test_camera_stop_detaches_handle_even_if_driver_release_raises():
    class BrokenReleaseCapture(FakeCapture):
        def release(self):
            raise OSError("driver diagnostic with private detail")

    fake = BrokenReleaseCapture()
    camera = WebcamCapture(capture_factory=lambda _: fake)
    camera.start()
    camera.stop()
    assert not camera.is_running
    assert camera.capture_info["release_error"] == "OSError"
    camera.stop()  # repeated stop remains safe

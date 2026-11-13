import pytest

from noorix.position import image_relative_position


def test_left_third():
    assert image_relative_position(10, 300) == "left"


def test_centre_third_includes_boundaries():
    assert image_relative_position(100, 300) == "centre"
    assert image_relative_position(200, 300) == "centre"


def test_right_third():
    assert image_relative_position(250, 300) == "right"


def test_invalid_frame_width():
    with pytest.raises(ValueError):
        image_relative_position(10, 0)


def test_coordinate_outside_frame():
    with pytest.raises(ValueError):
        image_relative_position(301, 300)

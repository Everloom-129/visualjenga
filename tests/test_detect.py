"""
Tests for src/detect.py

Unit tests cover the pure-Python parsing logic (_parse_points).
The GPU test checks end-to-end detection on a tiny image using the real model.
"""

import pytest
from src.detect import _parse_points, DetectedObject


# ---------------------------------------------------------------------------
# _parse_points — no GPU required
# ---------------------------------------------------------------------------

class TestParsePoints:

    def test_single_point(self):
        text = '<point x="42.5" y="30.0">cat</point>'
        objs = _parse_points(text)
        assert len(objs) == 1
        obj = objs[0]
        assert obj.label == "cat"
        assert pytest.approx(obj.x_frac, abs=1e-4) == 0.425
        assert pytest.approx(obj.y_frac, abs=1e-4) == 0.300

    def test_multiple_points(self):
        text = (
            '<point x="10.0" y="20.0">dog</point> '
            '<point x="80.0" y="90.0">table</point>'
        )
        objs = _parse_points(text)
        assert len(objs) == 2
        assert objs[0].label == "dog"
        assert objs[1].label == "table"

    def test_no_points_returns_empty(self):
        assert _parse_points("I see a cat and a table.") == []

    def test_x_coordinate_clamped_to_unit(self):
        # x > 100 should clamp to 1.0
        text = '<point x="120.0" y="50.0">ghost</point>'
        objs = _parse_points(text)
        assert len(objs) == 1
        assert objs[0].x_frac == 1.0
        assert pytest.approx(objs[0].y_frac, abs=1e-4) == 0.5

    def test_empty_label_replaced(self):
        text = '<point x="50.0" y="50.0">  </point>'
        objs = _parse_points(text)
        assert objs[0].label == "object"

    def test_whitespace_around_label(self):
        text = '<point x="50.0" y="50.0">  laptop  </point>'
        objs = _parse_points(text)
        assert objs[0].label == "laptop"


# ---------------------------------------------------------------------------
# DetectedObject.pixel_coords
# ---------------------------------------------------------------------------

class TestDetectedObjectPixelCoords:

    def test_centre_of_image(self):
        obj = DetectedObject(label="test", x_frac=0.5, y_frac=0.5)
        x, y = obj.pixel_coords(100, 200)
        assert x == 50
        assert y == 100

    def test_top_left(self):
        obj = DetectedObject(label="test", x_frac=0.0, y_frac=0.0)
        assert obj.pixel_coords(640, 480) == (0, 0)

    def test_bottom_right(self):
        obj = DetectedObject(label="test", x_frac=1.0, y_frac=1.0)
        assert obj.pixel_coords(640, 480) == (640, 480)


# ---------------------------------------------------------------------------
# GPU test — requires real model
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_molmo_detector_returns_points(small_rgb_image):
    from src.detect import MolmoDetector
    detector = MolmoDetector()
    objs = detector.detect(small_rgb_image)
    # On a random noise image Molmo may return nothing, but it must not crash
    assert isinstance(objs, list)
    for o in objs:
        assert 0.0 <= o.x_frac <= 1.0
        assert 0.0 <= o.y_frac <= 1.0

"""
Tests for src/detect.py

Unit tests cover the pure-Python parsing logic (_parse_points).
The GPU test checks end-to-end detection on a tiny image using the real model.
"""

import pytest
from src.detect import _parse_points, _sanitise_label, _nms_points, DetectedObject


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
        assert objs[0].label == "object_1"

    def test_whitespace_around_label(self):
        text = '<point x="50.0" y="50.0">  laptop  </point>'
        objs = _parse_points(text)
        assert objs[0].label == "laptop"

    def test_points_tag_closed(self):
        text = (
            '<points x1="19.5" y1="21.5" x2="40.5" y2="82.0" '
            'alt="objects">objects</points>'
        )
        objs = _parse_points(text)
        assert len(objs) == 2
        assert pytest.approx(objs[0].x_frac, abs=1e-4) == 0.195
        assert pytest.approx(objs[0].y_frac, abs=1e-4) == 0.215
        assert objs[0].label == "objects"

    def test_points_tag_truncated_no_closing(self):
        """Molmo output truncated by max_new_tokens — no </points> tag.
        Points are spread far apart so NMS keeps all of them."""
        text = (
            '<points x1="10.0" y1="10.0" x2="50.0" y2="50.0" '
            'x3="80.0" y3="20.0" x4="20.0" y4="80.0"'
        )
        objs = _parse_points(text)
        assert len(objs) == 4
        assert pytest.approx(objs[0].x_frac, abs=1e-4) == 0.100
        assert pytest.approx(objs[0].y_frac, abs=1e-4) == 0.100
        assert objs[3].label == "object_4"

    def test_points_tag_truncated_with_closing_angle(self):
        """Truncated after > but before </points>."""
        text = (
            '<points x1="10.0" y1="20.0" x2="50.0" y2="60.0">'
            'some partial text here'
        )
        objs = _parse_points(text)
        assert len(objs) == 2

    def test_points_tag_truncated_mid_attribute(self):
        """Truncated in the middle of an attribute value."""
        text = (
            '<points x1="10.0" y1="20.0" x2="50.0" y2="60.0" '
            'x3="70.0" y3="8'
        )
        objs = _parse_points(text)
        assert len(objs) == 2  # x3/y3 pair is incomplete, only first 2 parsed

    def test_points_tag_with_alt_label(self):
        text = (
            '<points x1="19.5" y1="21.5" x2="40.5" y2="82.0" '
            'alt="foreground objects">foreground objects</points>'
        )
        objs = _parse_points(text)
        assert len(objs) == 2
        assert objs[0].label == "foreground objects"

    def test_points_tag_truncated_alt_used_as_label(self):
        """Truncated output with alt attr should use alt as label."""
        text = (
            '<points x1="19.5" y1="21.5" x2="40.5" y2="82.0" '
            'alt="distinct objects"'
        )
        objs = _parse_points(text)
        assert len(objs) == 2
        assert objs[0].label == "distinct objects"


# ---------------------------------------------------------------------------
# _sanitise_label — no GPU
# ---------------------------------------------------------------------------

class TestSanitiseLabel:

    def test_normal_label_returned(self):
        assert _sanitise_label("cat", fallback="x") == "cat"

    def test_empty_string_uses_fallback(self):
        assert _sanitise_label("", fallback="obj") == "obj"

    def test_whitespace_only_uses_fallback(self):
        assert _sanitise_label("   ", fallback="obj") == "obj"

    def test_label_over_max_len_uses_fallback(self):
        long_label = "a" * 61
        assert _sanitise_label(long_label, fallback="obj") == "obj"

    def test_echoed_prompt_text_uses_fallback(self):
        echoed = "distinct object in this image. Include all foreground objects individually."
        assert _sanitise_label(echoed, fallback="obj") == "obj"

    def test_label_exactly_at_limit_kept(self):
        label = "a" * 60
        assert _sanitise_label(label, fallback="obj") == label


# ---------------------------------------------------------------------------
# _nms_points — no GPU
# ---------------------------------------------------------------------------

class TestNmsPoints:

    def _pt(self, x, y, label="obj"):
        return DetectedObject(label=label, x_frac=x, y_frac=y)

    def test_far_apart_points_all_kept(self):
        pts = [self._pt(0.1, 0.1), self._pt(0.5, 0.5), self._pt(0.9, 0.9)]
        result = _nms_points(pts, min_dist=0.05)
        assert len(result) == 3

    def test_identical_points_collapsed_to_one(self):
        pts = [self._pt(0.5, 0.5)] * 10
        result = _nms_points(pts, min_dist=0.05)
        assert len(result) == 1

    def test_first_point_wins(self):
        pts = [self._pt(0.5, 0.5, "first"), self._pt(0.51, 0.51, "second")]
        result = _nms_points(pts, min_dist=0.05)
        assert len(result) == 1
        assert result[0].label == "first"

    def test_empty_input(self):
        assert _nms_points([], min_dist=0.05) == []

    def test_evenly_spaced_diagonal_collapsed(self):
        """40 points tracing a diagonal (the surfer bug) should collapse to a few."""
        import math
        pts = [self._pt(i / 40, i / 40) for i in range(40)]
        result = _nms_points(pts, min_dist=0.05)
        # Expect roughly ceil(sqrt(2) / 0.05) ≈ 28, but definitely far fewer than 40
        assert len(result) < 40

    def test_real_bug_reproduction(self):
        """Reproduce the exact surfer detect.json: 40 diagonal points → significantly fewer.

        The 40 points span ~88% of the image diagonally, so NMS at 0.05 reduces
        them to ~14 (one kept per 0.05-radius zone along the diagonal).
        SAM2's IoU dedup (segment.py) is the second stage that collapses these
        remaining points to 1 mask since they all point at the same object.
        """
        raw = [
            {"x_frac": 0.29 + i * 0.005, "y_frac": 0.886 - i * 0.022}
            for i in range(40)
        ]
        pts = [DetectedObject(label="surfer", x_frac=d["x_frac"], y_frac=d["y_frac"]) for d in raw]
        result = _nms_points(pts, min_dist=0.05)
        # NMS removes ~65% of points; SAM2 IoU dedup handles the rest
        assert len(result) < len(pts)
        assert len(result) <= 15


# ---------------------------------------------------------------------------
# End-to-end: echoed prompt label from <points> tag (the actual surfer bug)
# ---------------------------------------------------------------------------

class TestEchoedPromptLabel:

    def test_long_inner_text_becomes_object_N(self):
        """When Molmo echoes the prompt as inner_text, labels must be object_N."""
        echoed = "distinct object in this image. Include all foreground objects individually."
        text = (
            f'<points x1="10.0" y1="10.0" x2="80.0" y2="80.0">'
            f'{echoed}</points>'
        )
        objs = _parse_points(text)
        for obj in objs:
            assert len(obj.label) <= 60
            assert obj.label != echoed


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

"""
Tests for src/segment.py

Unit tests cover mask IoU and deduplication logic.
The GPU test checks end-to-end SAM2 segmentation.
"""

import numpy as np
import pytest
from src.segment import _mask_iou, _dedup_masks


# ---------------------------------------------------------------------------
# _mask_iou — no GPU required
# ---------------------------------------------------------------------------

class TestMaskIou:

    def test_identical_masks_iou_is_one(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        assert pytest.approx(_mask_iou(mask, mask), abs=1e-6) == 1.0

    def test_no_overlap_iou_is_zero(self):
        a = np.zeros((64, 64), dtype=bool)
        a[0:10, 0:10] = True
        b = np.zeros((64, 64), dtype=bool)
        b[50:60, 50:60] = True
        assert pytest.approx(_mask_iou(a, b), abs=1e-6) == 0.0

    def test_half_overlap(self):
        a = np.zeros((64, 64), dtype=bool)
        a[0:20, 0:20] = True  # 400 pixels
        b = np.zeros((64, 64), dtype=bool)
        b[0:20, 10:30] = True  # 400 pixels, 200 overlap
        iou = _mask_iou(a, b)
        # intersection=200, union=600 → 1/3
        assert pytest.approx(iou, abs=1e-4) == 200 / 600

    def test_empty_masks(self):
        empty = np.zeros((64, 64), dtype=bool)
        # Both empty: denominator = 0+1e-6; result near 0
        assert _mask_iou(empty, empty) < 0.001


# ---------------------------------------------------------------------------
# _dedup_masks — no GPU required
# ---------------------------------------------------------------------------

class TestDedupMasks:

    def test_no_duplicates_kept(self):
        a = np.zeros((64, 64), dtype=bool); a[0:10, 0:10] = True
        b = np.zeros((64, 64), dtype=bool); b[50:60, 50:60] = True
        result = _dedup_masks([a, b], threshold=0.85)
        assert len(result) == 2

    def test_identical_masks_deduped(self):
        a = np.zeros((64, 64), dtype=bool); a[10:30, 10:30] = True
        b = a.copy()
        result = _dedup_masks([a, b], threshold=0.85)
        assert len(result) == 1

    def test_larger_mask_kept_on_duplicate(self):
        small = np.zeros((64, 64), dtype=bool); small[10:20, 10:20] = True   # 100 px
        large = np.zeros((64, 64), dtype=bool); large[9:21, 9:21] = True     # 144 px, high IoU
        result = _dedup_masks([small, large], threshold=0.5)
        assert len(result) == 1
        assert result[0].sum() == large.sum()

    def test_empty_input(self):
        assert _dedup_masks([], threshold=0.85) == []

    def test_single_mask_returned_unchanged(self, square_mask):
        result = _dedup_masks([square_mask])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], square_mask)


# ---------------------------------------------------------------------------
# GPU test — requires real SAM2 model
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_sam2_segmenter_basic(small_rgb_image):
    from src.segment import SAM2Segmenter
    seg = SAM2Segmenter()
    W, H = small_rgb_image.size
    # Point at the centre
    masks = seg.segment(small_rgb_image, [(W // 2, H // 2)])
    assert len(masks) >= 1
    assert masks[0].shape == (H, W)
    assert masks[0].dtype == bool
    # At least one pixel should be True
    assert masks[0].any()

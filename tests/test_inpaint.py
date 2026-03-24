"""
Tests for src/inpaint.py

Unit tests cover mask dilation and mask-to-PIL conversion.
The GPU test checks SD1.5 inpainting with real model.
"""

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
from src.inpaint import _dilate_mask, _mask_to_pil


# ---------------------------------------------------------------------------
# _dilate_mask — no GPU required
# ---------------------------------------------------------------------------

class TestDilateMask:

    def test_dilation_grows_mask(self, square_mask):
        original_count = square_mask.sum()
        dilated = _dilate_mask(square_mask, pixels=5)
        assert dilated.sum() > original_count

    def test_zero_dilation_unchanged(self, square_mask):
        dilated = _dilate_mask(square_mask, pixels=0)
        np.testing.assert_array_equal(dilated, square_mask)

    def test_dilated_contains_original(self, square_mask):
        dilated = _dilate_mask(square_mask, pixels=3)
        # Every pixel that was True stays True
        assert np.all(dilated[square_mask])

    def test_empty_mask_stays_empty(self):
        empty = np.zeros((64, 64), dtype=bool)
        dilated = _dilate_mask(empty, pixels=5)
        assert dilated.sum() == 0

    def test_full_mask_stays_full(self):
        full = np.ones((64, 64), dtype=bool)
        dilated = _dilate_mask(full, pixels=5)
        assert dilated.sum() == full.sum()


# ---------------------------------------------------------------------------
# _mask_to_pil — no GPU required
# ---------------------------------------------------------------------------

class TestMaskToPil:

    def test_output_is_grayscale(self, square_mask):
        pil = _mask_to_pil(square_mask, size=(64, 64))
        assert pil.mode == "L"

    def test_size_matches_requested(self, square_mask):
        pil = _mask_to_pil(square_mask, size=(128, 128))
        assert pil.size == (128, 128)

    def test_true_pixels_are_white(self, square_mask):
        pil = _mask_to_pil(square_mask, size=(64, 64))
        arr = np.array(pil)
        # Where mask is True, PIL should be 255
        assert (arr[square_mask] == 255).all()

    def test_false_pixels_are_black(self, square_mask):
        pil = _mask_to_pil(square_mask, size=(64, 64))
        arr = np.array(pil)
        assert (arr[~square_mask] == 0).all()


# ---------------------------------------------------------------------------
# GPU test — requires SD1.5
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_sd_inpainter_returns_n_images(small_rgb_image, square_mask):
    from src.inpaint import SDInpainter
    inpainter = SDInpainter()
    results = inpainter.inpaint(small_rgb_image, square_mask, n=2)
    assert len(results) == 2
    for img in results:
        assert isinstance(img, Image.Image)
        assert img.size == small_rgb_image.size


@pytest.mark.gpu
@pytest.mark.slow
def test_sd_inpainter_remove_returns_single_image(small_rgb_image, square_mask):
    from src.inpaint import SDInpainter
    inpainter = SDInpainter()
    result = inpainter.remove(small_rgb_image, square_mask)
    assert isinstance(result, Image.Image)
    assert result.size == small_rgb_image.size

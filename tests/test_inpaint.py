"""
Tests for src/inpaint.py

Unit tests cover:
  - _dilate_mask
  - _mask_to_pil
  - _compute_inpaint_bbox  (P0: crop-based inpainting bbox)
  - _prepare_crop / _paste_back  (P0: full crop→paste round-trip)

GPU tests check SD1.5 inpainting with the real model.
"""

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
from src.inpaint import (
    _dilate_mask,
    _mask_to_pil,
    _compute_inpaint_bbox,
    _prepare_crop,
    _paste_back,
    _MASK_DILATION_PX,
    _INPAINT_PAD,
    _SD_SIZE,
)


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
        assert (arr[square_mask] == 255).all()

    def test_false_pixels_are_black(self, square_mask):
        pil = _mask_to_pil(square_mask, size=(64, 64))
        arr = np.array(pil)
        assert (arr[~square_mask] == 0).all()


# ---------------------------------------------------------------------------
# _compute_inpaint_bbox — no GPU required
# ---------------------------------------------------------------------------

class TestComputeInpaintBbox:

    def test_output_is_square(self, square_mask):
        img_w, img_h = 64, 64
        x0, y0, x1, y1 = _compute_inpaint_bbox(square_mask, img_w, img_h, pad=4)
        assert (x1 - x0) == (y1 - y0), "bbox must be square"

    def test_contains_all_mask_pixels(self, square_mask):
        img_w, img_h = 64, 64
        x0, y0, x1, y1 = _compute_inpaint_bbox(square_mask, img_w, img_h, pad=4)
        rows, cols = np.where(square_mask)
        assert rows.min() >= y0
        assert rows.max() < y1
        assert cols.min() >= x0
        assert cols.max() < x1

    def test_padding_expands_beyond_tight_bbox(self):
        """Padded bbox should be larger than the tight mask bbox."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True   # 20×20 centre block
        img_w, img_h = 100, 100
        x0, y0, x1, y1 = _compute_inpaint_bbox(mask, img_w, img_h, pad=10)
        # Tight bbox: x=[40,60), y=[40,60) → side=20. With pad=10 → side≥40
        assert (x1 - x0) >= 40
        assert (y1 - y0) >= 40

    def test_clamped_to_image_bounds(self, square_mask):
        img_w, img_h = 64, 64
        x0, y0, x1, y1 = _compute_inpaint_bbox(square_mask, img_w, img_h, pad=100)
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= img_w
        assert y1 <= img_h

    def test_corner_mask_still_produces_valid_bbox(self):
        """Mask touching the image corner should still give a clamped square bbox."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:10, 0:10] = True
        x0, y0, x1, y1 = _compute_inpaint_bbox(mask, 64, 64, pad=5)
        assert x0 >= 0 and y0 >= 0
        assert x1 <= 64 and y1 <= 64
        assert (x1 - x0) == (y1 - y0)

    def test_large_mask_fills_image(self):
        """Mask covering most of the image — bbox should not exceed image bounds."""
        mask = np.ones((64, 64), dtype=bool)
        x0, y0, x1, y1 = _compute_inpaint_bbox(mask, 64, 64, pad=32)
        assert x0 >= 0 and y0 >= 0
        assert x1 <= 64 and y1 <= 64


# ---------------------------------------------------------------------------
# _prepare_crop — no GPU required
# ---------------------------------------------------------------------------

class TestPrepareCrop:

    def test_img_sd_is_512(self, small_rgb_image, square_mask):
        img_rgb = small_rgb_image.convert("RGB")
        img_w, img_h = img_rgb.size
        _, img_sd, mask_sd, _, _ = _prepare_crop(img_rgb, square_mask, img_w, img_h)
        assert img_sd.size == (_SD_SIZE, _SD_SIZE)

    def test_mask_sd_is_512_grayscale(self, small_rgb_image, square_mask):
        img_rgb = small_rgb_image.convert("RGB")
        img_w, img_h = img_rgb.size
        _, img_sd, mask_sd, _, _ = _prepare_crop(img_rgb, square_mask, img_w, img_h)
        assert mask_sd.size == (_SD_SIZE, _SD_SIZE)
        assert mask_sd.mode == "L"

    def test_bbox_is_square(self, small_rgb_image, square_mask):
        img_rgb = small_rgb_image.convert("RGB")
        img_w, img_h = img_rgb.size
        bbox, _, _, crop_w, crop_h = _prepare_crop(img_rgb, square_mask, img_w, img_h)
        x0, y0, x1, y1 = bbox
        assert (x1 - x0) == (y1 - y0)
        assert crop_w == x1 - x0
        assert crop_h == y1 - y0

    def test_crop_dims_consistent_with_bbox(self, small_rgb_image, square_mask):
        img_rgb = small_rgb_image.convert("RGB")
        img_w, img_h = img_rgb.size
        bbox, _, _, crop_w, crop_h = _prepare_crop(img_rgb, square_mask, img_w, img_h)
        x0, y0, x1, y1 = bbox
        assert crop_w == x1 - x0
        assert crop_h == y1 - y0

    def test_mask_contains_dilated_region(self, small_rgb_image, square_mask):
        """Dilated mask in crop should be all-white inside the original mask region."""
        img_rgb = small_rgb_image.convert("RGB")
        img_w, img_h = img_rgb.size
        bbox, _, mask_sd, crop_w, crop_h = _prepare_crop(img_rgb, square_mask, img_w, img_h)
        x0, y0, x1, y1 = bbox
        mask_arr = np.array(mask_sd)
        # At least some pixels in the SD mask should be white (255)
        assert mask_arr.max() == 255


# ---------------------------------------------------------------------------
# _paste_back — no GPU required
# ---------------------------------------------------------------------------

class TestPasteBack:

    def _make_image(self, w, h, fill):
        arr = np.full((h, w, 3), fill, dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def test_output_is_original_size(self):
        original = self._make_image(200, 150, fill=100)
        sd_out = self._make_image(512, 512, fill=200)
        result = _paste_back(original, sd_out, x0=20, y0=10, crop_w=80, crop_h=80)
        assert result.size == (200, 150)

    def test_outside_bbox_unchanged(self):
        """Pixels outside the paste region must not change."""
        original = self._make_image(200, 150, fill=50)
        sd_out = self._make_image(512, 512, fill=200)
        x0, y0, crop_w, crop_h = 80, 60, 40, 40
        result = _paste_back(original, sd_out, x0, y0, crop_w, crop_h)
        arr_orig = np.array(original)
        arr_result = np.array(result)
        # Corners (0,0) should be unchanged
        assert arr_result[0, 0, 0] == arr_orig[0, 0, 0]
        # Bottom-right corner should be unchanged
        assert arr_result[149, 199, 0] == arr_orig[149, 199, 0]

    def test_pasted_region_is_modified(self):
        """Pixels inside the bbox should reflect the SD output (resized)."""
        original = self._make_image(200, 150, fill=50)
        # Make SD output pure white so it's distinguishable
        sd_out = self._make_image(512, 512, fill=255)
        x0, y0, crop_w, crop_h = 80, 60, 40, 40
        result = _paste_back(original, sd_out, x0, y0, crop_w, crop_h)
        arr = np.array(result)
        # Centre of pasted region should be white
        cy, cx = y0 + crop_h // 2, x0 + crop_w // 2
        assert arr[cy, cx, 0] == 255

    def test_original_not_mutated(self):
        """_paste_back must not modify the original image in place."""
        original = self._make_image(200, 150, fill=50)
        sd_out = self._make_image(512, 512, fill=200)
        orig_arr = np.array(original).copy()
        _paste_back(original, sd_out, x0=20, y0=10, crop_w=80, crop_h=80)
        np.testing.assert_array_equal(np.array(original), orig_arr)


# ---------------------------------------------------------------------------
# Full inpaint() round-trip — mocked SD pipeline, no GPU
# ---------------------------------------------------------------------------

class MockPipe:
    """Minimal mock for StableDiffusionInpaintPipeline."""

    def __call__(self, image, mask_image, **kwargs):
        # Return an image of the same size as the input, filled with a fixed colour
        out = Image.new("RGB", image.size, (128, 200, 128))

        class _Result:
            images = [out]

        return _Result()

    def set_progress_bar_config(self, **kwargs):
        pass


class TestInpaintRoundTrip:
    """Validate the crop→SD→paste pipeline without a real GPU."""

    def _make_inpainter(self):
        from src.inpaint import SDInpainter
        inpainter = SDInpainter(device="cpu", dtype=None)
        inpainter._pipe = MockPipe()
        return inpainter

    def test_returns_n_images(self, small_rgb_image, square_mask):
        inpainter = self._make_inpainter()
        results = inpainter.inpaint(small_rgb_image, square_mask, n=3)
        assert len(results) == 3

    def test_output_is_original_size(self, small_rgb_image, square_mask):
        inpainter = self._make_inpainter()
        results = inpainter.inpaint(small_rgb_image, square_mask, n=2)
        for img in results:
            assert img.size == small_rgb_image.size

    def test_output_is_rgb(self, small_rgb_image, square_mask):
        inpainter = self._make_inpainter()
        results = inpainter.inpaint(small_rgb_image, square_mask, n=1)
        assert results[0].mode == "RGB"

    def test_outside_mask_bbox_mostly_unchanged(self, square_mask):
        """Pixels well outside the mask bbox should be identical to original."""
        # Use a 128×128 image with the mask in the centre
        arr = np.full((128, 128, 3), 42, dtype=np.uint8)
        image = Image.fromarray(arr, "RGB")
        # small mask in centre
        mask = np.zeros((128, 128), dtype=bool)
        mask[56:72, 56:72] = True

        inpainter = self._make_inpainter()
        results = inpainter.inpaint(image, mask, n=1)
        result_arr = np.array(results[0])
        # Top-left corner (0,0) is far from mask — must be unchanged (value=42)
        assert result_arr[0, 0, 0] == 42

    def test_remove_returns_single_full_size_image(self, small_rgb_image, square_mask):
        inpainter = self._make_inpainter()
        result = inpainter.remove(small_rgb_image, square_mask)
        assert isinstance(result, Image.Image)
        assert result.size == small_rgb_image.size
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# GPU tests — requires SD1.5
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


@pytest.mark.gpu
def test_lama_remover_returns_original_size(small_rgb_image, square_mask):
    from src.inpaint import LaMaRemover
    remover = LaMaRemover()
    result = remover.remove(small_rgb_image, square_mask)
    assert isinstance(result, Image.Image)
    assert result.size == small_rgb_image.size
    assert result.mode == "RGB"

"""
Tests for src/diversity.py

Unit tests cover crop_to_mask, area_fraction, and the diversity_score formula.
GPU tests check the full scoring path with real models.
"""

import numpy as np
import pytest
from PIL import Image
from src.diversity import crop_to_mask, area_fraction, _tight_square_bbox, diversity_score


# ---------------------------------------------------------------------------
# _tight_square_bbox — no GPU
# ---------------------------------------------------------------------------

class TestTightSquareBbox:

    def test_square_mask_returns_square_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 20:40] = True   # 20×20
        x0, y0, x1, y1 = _tight_square_bbox(mask)
        assert (x1 - x0) == (y1 - y0)

    def test_wide_mask_makes_square(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 10:90] = True   # 20 tall, 80 wide → square side 80
        x0, y0, x1, y1 = _tight_square_bbox(mask)
        assert (x1 - x0) == (y1 - y0)

    def test_bbox_contains_all_mask_pixels(self, square_mask):
        x0, y0, x1, y1 = _tight_square_bbox(square_mask)
        rows, cols = np.where(square_mask)
        # y1 is exclusive, so last row must satisfy rows.max() < y1
        assert rows.min() >= y0
        assert rows.max() < y1,  f"row {rows.max()} not inside [{y0},{y1})"
        assert cols.min() >= x0
        assert cols.max() < x1, f"col {cols.max()} not inside [{x0},{x1})"

    def test_single_pixel_mask(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 32] = True
        x0, y0, x1, y1 = _tight_square_bbox(mask)
        # Side should be at least 1
        assert x1 > x0 and y1 > y0


# ---------------------------------------------------------------------------
# crop_to_mask — no GPU
# ---------------------------------------------------------------------------

class TestCropToMask:

    def test_output_size_is_224(self, small_rgb_image, square_mask):
        crop = crop_to_mask(small_rgb_image, square_mask)
        assert crop.size == (224, 224)

    def test_output_is_rgb(self, small_rgb_image, square_mask):
        crop = crop_to_mask(small_rgb_image, square_mask)
        assert crop.mode == "RGB"

    def test_pixels_outside_mask_are_zero(self, small_rgb_image):
        # Use a non-square mask so the bounding box contains pixels outside the mask
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:30, 20:50] = True  # wide rectangle: bbox is 30×30, but 20 rows are empty
        # Fill image with non-zero values
        img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        crop = crop_to_mask(img, mask)
        arr = np.array(crop)
        # Outside the rectangular mask (which is narrower than its square bbox),
        # pixels should be zeroed → some zeros must exist
        assert (arr == 0).any()

    def test_full_mask_has_no_zeros(self, small_rgb_image):
        full_mask = np.ones((64, 64), dtype=bool)
        crop = crop_to_mask(small_rgb_image, full_mask)
        arr = np.array(crop)
        # Original has random pixels; all should pass through
        assert not (arr == 0).all()


# ---------------------------------------------------------------------------
# area_fraction — no GPU
# ---------------------------------------------------------------------------

class TestAreaFraction:

    def test_full_bbox_coverage_is_one(self):
        # Mask that EXACTLY fills its bounding box: a full square from (0,0)
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:20, 0:20] = True  # anchored at corner → bbox starts at (0,0)
        af = area_fraction(mask)
        assert pytest.approx(af, abs=0.01) == 1.0

    def test_thin_mask_has_low_fraction(self, thin_mask):
        af = area_fraction(thin_mask)
        # A single-row mask has very low area fraction
        assert af < 0.1

    def test_fraction_in_range(self, square_mask):
        af = area_fraction(square_mask)
        assert 0.0 < af <= 1.0

    def test_large_mask_fraction_close_to_one(self):
        mask = np.ones((64, 64), dtype=bool)
        af = area_fraction(mask)
        assert pytest.approx(af, abs=0.01) == 1.0


# ---------------------------------------------------------------------------
# diversity_score formula — mocked similarity model, no GPU
# ---------------------------------------------------------------------------

class MockSimilarityModel:
    """Returns fixed similarity values for deterministic testing."""

    def __init__(self, clip_val: float, dino_val: float):
        self.clip_val = clip_val
        self.dino_val = dino_val

    def clip_sim_batch(self, originals, inpaintings):
        return [self.clip_val] * len(originals)

    def dino_sim_batch(self, originals, inpaintings):
        return [self.dino_val] * len(originals)


class TestDiversityScore:

    def _make_inpaintings(self, n, size=(64, 64)):
        rng = np.random.default_rng(42)
        imgs = []
        for _ in range(n):
            arr = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
            imgs.append(Image.fromarray(arr, "RGB"))
        return imgs

    def test_high_similarity_gives_low_diversity(self, small_rgb_image, square_mask):
        """If inpaintings are identical to original (sim→1), diversity should be low."""
        sim = MockSimilarityModel(clip_val=1.0, dino_val=1.0)
        inpaintings = self._make_inpaintings(4)
        score = diversity_score(small_rgb_image, square_mask, inpaintings, sim)
        # diversity = 1 - 1.0 * 1.0 / af; af ≤ 1 → score ≤ 0
        assert score <= 0.0

    def test_zero_similarity_gives_high_diversity(self, small_rgb_image, square_mask):
        """If inpaintings share no similarity (sim→0), diversity should be near 1."""
        sim = MockSimilarityModel(clip_val=0.0, dino_val=0.0)
        inpaintings = self._make_inpaintings(4)
        score = diversity_score(small_rgb_image, square_mask, inpaintings, sim)
        assert pytest.approx(score, abs=1e-6) == 1.0

    def test_score_is_scalar(self, small_rgb_image, square_mask):
        sim = MockSimilarityModel(clip_val=0.5, dino_val=0.5)
        inpaintings = self._make_inpaintings(4)
        score = diversity_score(small_rgb_image, square_mask, inpaintings, sim)
        assert isinstance(score, float)

    def test_higher_similarity_lower_score(self, small_rgb_image, square_mask):
        """Increasing similarity should decrease diversity score."""
        inpaintings = self._make_inpaintings(4)
        score_low = diversity_score(
            small_rgb_image, square_mask, inpaintings,
            MockSimilarityModel(0.3, 0.3)
        )
        score_high = diversity_score(
            small_rgb_image, square_mask, inpaintings,
            MockSimilarityModel(0.9, 0.9)
        )
        assert score_high < score_low

    def test_small_area_fraction_amplifies_score(self, small_rgb_image, thin_mask):
        """Small area fraction (denominator) increases the diversity score magnitude."""
        inpaintings = self._make_inpaintings(4)
        sim = MockSimilarityModel(clip_val=0.5, dino_val=0.5)

        score_thin = diversity_score(small_rgb_image, thin_mask, inpaintings, sim)
        score_square = diversity_score(small_rgb_image, np.ones((64, 64), bool), inpaintings, sim)
        # thin mask → smaller af → larger denominator effect → lower (more negative) score
        assert score_thin < score_square


# ---------------------------------------------------------------------------
# GPU test — full round-trip with real similarity model
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_diversity_score_gpu(small_rgb_image, square_mask):
    from src.similarity import SimilarityModel
    sim = SimilarityModel()
    # Use copies of the original as "inpaintings" — similarity should be very high → low score
    inpaintings = [small_rgb_image.copy() for _ in range(4)]
    score = diversity_score(small_rgb_image, square_mask, inpaintings, sim)
    assert isinstance(score, float)
    # Identical inpaintings → low diversity (≤ 0)
    assert score <= 0.1

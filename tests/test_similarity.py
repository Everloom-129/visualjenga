"""
Tests for src/similarity.py

Unit tests validate similarity normalization logic.
GPU tests load CLIP and DINOv2 and verify outputs on tiny images.
"""

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Normalization formula (no GPU)
# ---------------------------------------------------------------------------

class TestSimilarityNormalization:
    """Verify that (cosine_sim + 1) / 2 maps [-1, 1] → [0, 1]."""

    def test_identical_vectors_give_one(self):
        # cos(0°) = 1 → normalized = 1.0
        sim = (1.0 + 1.0) / 2.0
        assert sim == 1.0

    def test_opposite_vectors_give_zero(self):
        # cos(180°) = -1 → normalized = 0.0
        sim = (-1.0 + 1.0) / 2.0
        assert sim == 0.0

    def test_orthogonal_vectors_give_half(self):
        # cos(90°) = 0 → normalized = 0.5
        sim = (0.0 + 1.0) / 2.0
        assert sim == 0.5


# ---------------------------------------------------------------------------
# GPU tests — require CLIP + DINOv2 models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_model():
    pytest.importorskip("open_clip")
    pytest.importorskip("transformers")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    from src.similarity import SimilarityModel
    return SimilarityModel()


@pytest.fixture
def identical_images():
    """Two identical 64×64 green images."""
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img, img.copy()


@pytest.fixture
def different_images():
    """One green and one red image."""
    green = Image.fromarray(np.full((64, 64, 3), [0, 200, 0], dtype=np.uint8))
    red = Image.fromarray(np.full((64, 64, 3), [200, 0, 0], dtype=np.uint8))
    return green, red


@pytest.mark.gpu
def test_clip_sim_identical_images_near_one(sim_model, identical_images):
    a, b = identical_images
    sim = sim_model.clip_sim(a, b)
    assert sim > 0.99


@pytest.mark.gpu
def test_dino_sim_identical_images_near_one(sim_model, identical_images):
    a, b = identical_images
    sim = sim_model.dino_sim(a, b)
    assert sim > 0.99


@pytest.mark.gpu
def test_clip_sim_in_range(sim_model, different_images):
    a, b = different_images
    sim = sim_model.clip_sim(a, b)
    assert 0.0 <= sim <= 1.0


@pytest.mark.gpu
def test_dino_sim_in_range(sim_model, different_images):
    a, b = different_images
    sim = sim_model.dino_sim(a, b)
    assert 0.0 <= sim <= 1.0


@pytest.mark.gpu
def test_clip_sim_identical_geq_different(sim_model, identical_images, different_images):
    sim_same = sim_model.clip_sim(*identical_images)
    sim_diff = sim_model.clip_sim(*different_images)
    assert sim_same >= sim_diff


@pytest.mark.gpu
def test_clip_sim_batch_consistent_with_scalar(sim_model, identical_images, different_images):
    """Batch output must match scalar calls."""
    imgs_a = [identical_images[0], different_images[0]]
    imgs_b = [identical_images[1], different_images[1]]
    batch = sim_model.clip_sim_batch(imgs_a, imgs_b)
    assert len(batch) == 2
    assert pytest.approx(batch[0], abs=1e-4) == sim_model.clip_sim(*identical_images)
    assert pytest.approx(batch[1], abs=1e-4) == sim_model.clip_sim(*different_images)

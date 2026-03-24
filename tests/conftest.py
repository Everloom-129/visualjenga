"""
Shared fixtures and pytest configuration.

Marks:
  - @pytest.mark.gpu   : requires a CUDA GPU + loaded models
  - @pytest.mark.slow  : slow tests (multiple inpaintings, etc.)
"""

import numpy as np
import pytest
from PIL import Image


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU and loaded models")
    config.addinivalue_line("markers", "slow: slow test (multiple model forward passes)")


# ---------------------------------------------------------------------------
# Small synthetic fixtures — no GPU required
# ---------------------------------------------------------------------------

@pytest.fixture
def small_rgb_image():
    """64×64 random RGB image."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def square_mask():
    """64×64 bool mask with a 20×20 block in the centre."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[22:42, 22:42] = True
    return mask


@pytest.fixture
def thin_mask():
    """64×64 mask that is a single row — stress-tests area_fraction."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[32, 10:54] = True
    return mask


@pytest.fixture
def multi_object_masks():
    """Two non-overlapping 16×16 masks in a 64×64 image."""
    a = np.zeros((64, 64), dtype=bool)
    a[4:20, 4:20] = True
    b = np.zeros((64, 64), dtype=bool)
    b[44:60, 44:60] = True
    return a, b

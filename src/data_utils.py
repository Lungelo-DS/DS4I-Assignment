"""Load the handwriting dataset from the R .rds file into NumPy / pandas."""

from pathlib import Path

import numpy as np
import pandas as pd
import rdata

# Repo root is one level above this src/ folder, so paths work from any notebook.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "Assignment 2" / "handwriting.rds"


def load_handwriting(path=DATA_PATH, scale=True):
    """Return (images, metadata).

    images   : float32 array, shape (25000, 28, 28). Values in [0, 1] if scale=True,
               otherwise the raw 0 (background) to 255 (ink) integers.
    metadata : DataFrame with image_id, person_id, digit, session, replicate.
               Row i describes images[i].
    """
    data = rdata.read_rds(str(path))

    images = np.asarray(data["images"])[..., 0]  # drop the trailing channel dim
    if scale:
        images = images.astype("float32") / 255.0

    metadata = data["metadata"].reset_index(drop=True).astype(int)
    return images, metadata


def select(images, metadata, **filters):
    """Subset images and metadata, e.g. select(x, meta, person_id=3, digit=2)."""
    mask = np.ones(len(metadata), dtype=bool)
    for column, value in filters.items():
        mask &= metadata[column].to_numpy() == value
    return images[mask], metadata[mask].reset_index(drop=True)

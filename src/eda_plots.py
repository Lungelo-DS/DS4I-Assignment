"""Plotting helpers for the exploratory data analysis (Task 1)."""

import matplotlib.pyplot as plt
import numpy as np

from data_utils import select


def show_grid(images, metadata, row_var, col_var, row_values, col_values,
              fixed=None, replicate=1, title=None):
    """Grid of single images: one row per row_value, one column per col_value.

    `fixed` holds the variables kept constant, e.g. {"session": 1}.
    Example: rows = people, columns = digits, fixed session 1, replicate 1.
    """
    fixed = dict(fixed or {})
    fig, axes = plt.subplots(len(row_values), len(col_values),
                             figsize=(1.1 * len(col_values), 1.1 * len(row_values)),
                             squeeze=False)
    for r, row_value in enumerate(row_values):
        for c, col_value in enumerate(col_values):
            filters = {**fixed, row_var: row_value, col_var: col_value,
                       "replicate": replicate}
            imgs, _ = select(images, metadata, **filters)
            ax = axes[r, c]
            ax.imshow(imgs[0], cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{col_var} {col_value}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{row_var} {row_value}", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_mean_images(images, metadata, row_var, col_var, row_values, col_values,
                     fixed=None, title=None):
    """Same layout as show_grid, but each cell is the average of all matching images.

    Averaging over the 50 replicates shows the 'typical' shape and how spread out it is.
    """
    fixed = dict(fixed or {})
    fig, axes = plt.subplots(len(row_values), len(col_values),
                             figsize=(1.1 * len(col_values), 1.1 * len(row_values)),
                             squeeze=False)
    for r, row_value in enumerate(row_values):
        for c, col_value in enumerate(col_values):
            imgs, _ = select(images, metadata,
                             **{**fixed, row_var: row_value, col_var: col_value})
            ax = axes[r, c]
            ax.imshow(imgs.mean(axis=0), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{col_var} {col_value}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{row_var} {row_value}", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def image_features(images, threshold=0.5):
    """Simple, interpretable per-image measurements of handwriting style.

    Returns a dict of 1-D arrays (one value per image):
      ink     : total ink (sum of pixel values) - stroke thickness / size
      width   : columns containing ink above `threshold`
      height  : rows containing ink above `threshold`
      x_centre, y_centre : ink-weighted centre of the digit
      slant   : ink-weighted covariance of x and y divided by variance of y.
                Positive = leans right (top of digit further right).
    """
    n = len(images)
    rows = np.arange(28)[None, :, None]   # y coordinate (0 = top)
    cols = np.arange(28)[None, None, :]   # x coordinate (0 = left)

    ink = images.sum(axis=(1, 2))
    weight = images / ink[:, None, None]
    x_centre = (weight * cols).sum(axis=(1, 2))
    y_centre = (weight * rows).sum(axis=(1, 2))

    dx = cols - x_centre[:, None, None]
    dy = rows - y_centre[:, None, None]
    cov_xy = (weight * dx * dy).sum(axis=(1, 2))
    var_y = (weight * dy ** 2).sum(axis=(1, 2))
    # Image y increases downwards, so flip the sign to make "leans right" positive.
    slant = -cov_xy / var_y

    on = images > threshold
    width = on.any(axis=1).sum(axis=1)
    height = on.any(axis=2).sum(axis=1)

    return {"ink": ink, "width": width, "height": height,
            "x_centre": x_centre, "y_centre": y_centre, "slant": slant}

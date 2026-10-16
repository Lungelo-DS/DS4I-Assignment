"""Exploratory data analysis computations and figures (Task 1).

All functions take the scaled image array (n, 28, 28) and the metadata DataFrame
returned by `data_utils.load_handwriting()`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from eda_plots import image_features


def find_row(meta, person, digit, session, replicate):
    """Row number of one specific image."""
    m = ((meta.person_id == person) & (meta.digit == digit)
         & (meta.session == session) & (meta.replicate == replicate))
    return int(np.where(m)[0][0])


def image_panel(images, rows, row_labels=None, col_labels=None, cell=0.5, fig=None):
    """Grid of images; `rows` is a list of lists of row numbers into `images`."""
    n_r, n_c = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_r, n_c, figsize=(cell * n_c, cell * n_r + 0.3), squeeze=False)
    for r in range(n_r):
        for c in range(n_c):
            ax = axes[r, c]
            ax.imshow(images[rows[r][c]], cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if r == 0 and col_labels is not None and col_labels[c]:
                ax.set_title(col_labels[c], fontsize=6)
            if c == 0 and row_labels is not None:
                ax.set_ylabel(row_labels[r], fontsize=6)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


def structure_checks(images, meta):
    """Table of basic dataset checks (size, balance, missing values, blank pixels)."""
    cells = meta.groupby(["person_id", "digit", "session"]).size()
    return pd.DataFrame({
        "Check": ["Images", "Image size", "Pixel range (raw)", "Missing values",
                  "Writers / digits / sessions", "Images per writer x digit x session",
                  "Blank pixels (%)"],
        "Value": [f"{len(images):,}", "28 x 28", "0 to 255", int(meta.isna().sum().sum()),
                  f"{meta.person_id.nunique()} / {meta.digit.nunique()} / {meta.session.nunique()}",
                  ", ".join(map(str, cells.unique())), f"{100 * (images == 0).mean():.1f}"],
    })


def eta_squared_table(images, meta):
    """Share of total pixel variance explained by each grouping (between-group SS / total SS)."""
    X = images.reshape(len(images), -1).astype("float64")
    grand = X.mean(axis=0)
    total = ((X - grand) ** 2).sum()
    groupings = {"Digit": ["digit"], "Writer": ["person_id"], "Session": ["session"],
                 "Writer x digit": ["person_id", "digit"],
                 "Writer x digit x session": ["person_id", "digit", "session"]}
    rows = []
    for name, cols in groupings.items():
        g = meta.groupby(cols).ngroup().to_numpy()
        n = np.bincount(g)
        means = np.zeros((n.size, X.shape[1]))
        np.add.at(means, g, X)
        means /= n[:, None]
        rows.append({"Grouping": name, "Groups": int(n.size),
                     "Share of pixel variance": (n * ((means - grand) ** 2).sum(1)).sum() / total})
    return pd.DataFrame(rows)


def style_features(images, meta):
    """Metadata plus interpretable style measurements (ink, width, height, slant, centre)."""
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(image_features(images))], axis=1)


def pair_distances(images, meta, n=5000, seed=1):
    """Euclidean pixel distance for three kinds of same-digit pairs.

    Returns a long DataFrame with columns 'pair type' and 'distance'.
    """
    X = images.reshape(len(images), -1)
    P, D, S = meta.person_id.values, meta.digit.values, meta.session.values
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(len(X))
        same_digit = D == D[i]
        masks = {
            "same writer, same session": same_digit & (P == P[i]) & (S == S[i]) & (idx != i),
            "same writer, different session": same_digit & (P == P[i]) & (S != S[i]),
            "different writer": same_digit & (P != P[i]),
        }
        for name, m in masks.items():
            j = rng.choice(idx[m])
            out.append((name, float(np.linalg.norm(X[i] - X[j]))))
    return pd.DataFrame(out, columns=["pair type", "distance"])


def pca_2d(images):
    """First two principal components of the flattened images and their variance share."""
    X = images.reshape(len(images), -1)
    pca = PCA(n_components=2).fit(X)
    return pca.transform(X), float(pca.explained_variance_ratio_.sum())

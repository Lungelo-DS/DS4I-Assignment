"""Helpers for the Quarto report: plot style, markdown tables, results loading, git history."""

import json
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.grid": True,
        "grid.alpha": 0.3, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def show_table(df, index=False, floatfmt=".3f"):
    """Print a DataFrame as a markdown table (use with `#| output: asis`).

    Whole-number columns are printed without decimals, other numeric columns with
    `floatfmt`; numeric columns are right-aligned and text columns left-aligned.
    """
    out = df.copy()
    if index:
        if out.index.name is None:
            out.index.name = ""
        out = out.reset_index()
    align = []
    for c in out.columns:
        col = out[c]
        numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
        align.append("right" if numeric else "left")
        if numeric:
            vals = col.dropna()
            if len(vals) and np.allclose(vals, np.round(vals)):
                out[c] = col.map(lambda v: "" if pd.isna(v) else f"{int(round(v))}")
            else:
                out[c] = col.map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    print(out.to_markdown(index=False, disable_numparse=True, colalign=align))


def load_csv(name):
    return pd.read_csv(RESULTS / name)


def load_json(name):
    return json.load(open(RESULTS / name))


def load_npz(name):
    return np.load(RESULTS / name)


def git_history():
    """Commits per author and merged pull requests, read from the local git repository.

    Returns (commits_by_author DataFrame, number of merged PRs), or (None, None) if git
    is not available (e.g. the bundle was copied without its .git folder).
    """
    try:
        log = subprocess.run(["git", "-C", str(REPO_ROOT), "log", "--format=%an|%s"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None, None
    rows = [line.split("|", 1) for line in log.splitlines() if "|" in line]
    df = pd.DataFrame(rows, columns=["author", "message"])
    merged = int(df.message.str.startswith("Merge pull request").sum())
    by_author = (df.assign(merge=df.message.str.startswith("Merge pull request"))
                   .groupby("author")
                   .agg(commits=("message", "size"), **{"PR merges": ("merge", "sum")})
                   .sort_values("commits", ascending=False))
    return by_author, merged

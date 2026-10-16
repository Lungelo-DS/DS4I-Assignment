# DS4I-Assignment

**STA5073Z – Data Science for Industry 2026, Assignment 2: Image classification with neural networks**
University of Cape Town, Department of Statistical Sciences · Due 16 October 2026

## Overview

This project uses neural networks to identify **who wrote a handwritten digit**, rather than which digit it is. We build:

- a **CNN** that predicts the writer ID from a single image, and
- a **Siamese network** that decides whether two images were written by the same person.

We also study how the choice of train/validation/test split affects the results, which digits carry the most information about the writer, and how changes in a writer's style between recording sessions affect performance.

## Data

`Assignment 2/handwriting.rds` contains 25,000 simulated (not real) grayscale images of the digits 0–4, each 28 × 28 pixels.

| | |
|---|---|
| Writers | 20 (`person_id` 1–20) |
| Digits | 0, 1, 2, 3, 4 |
| Sessions per writer | 5 (`session` 1–5) |
| Replicates per writer × digit × session | 50 (`replicate` 1–50) |
| Total | 20 × 5 × 5 × 50 = 25,000 images |

Pixel values run from 0 (background) to 255 (ink) and are scaled to [0, 1] before modelling. Each writer has a consistent style that shifts somewhat between sessions, with smaller variation between images within a session.

## Tasks

1. **Exploratory data analysis**: examples across people, digits and sessions; sources of variation; limitations of the simulated data.
2. **CNN writer identification**: predict writer ID; overall and per-digit performance.
3. **Siamese network**: same-writer matching using contrastive or triplet loss.
4. **Train/validation/test strategy**: what kind of generalisation is measured; leakage risks.
5. **Hyperparameter optimisation**: limited search for the CNN or Siamese model; sensitivity of results.
6. **Comparison**: CNN vs Siamese; which digits carry more writer information; effect of sessions.
7. **Collaboration** with Git and GitHub (branches, pull requests).
8. **Report** (max 5,000 words) with plagiarism declaration and AI use statement.
9. **GitHub Pages** website hosting the report.

## Repository structure

```
DS4I-Assignment/
├── Assignment 2/                 # brief (PDF) and data (handwriting.rds)
├── src/
│   ├── data_utils.py             # load the .rds file into NumPy / pandas
│   ├── eda_plots.py              # plotting and style measurements for the EDA
│   └── siamese.py                # Siamese network: triplets, pairs, training, metrics
├── notebooks/                    # run in this order
│   ├── 01_eda.ipynb                          # Task 1
│   ├── 03_cnn_writer_id.ipynb                # Task 2 (CNN), Task 4 (random vs session split)
│   ├── 04_siamese_writer_matching.ipynb      # Task 3 (Siamese), Task 4 (writer-disjoint folds)
│   ├── 05_hyperparameter_sensitivity.ipynb   # Task 5
│   └── 06_cnn_vs_siamese.ipynb               # Task 6 (needs 04 and 05 to have been run)
├── results/                      # CSVs and model weights written by notebooks 04 and 05
├── requirements.txt
└── README.md
```

## Getting started

Requires Python 3.10 or newer, in a version TensorFlow supports. Notebooks 03–06 train neural networks on the CPU and take roughly 5–30 minutes each.

```bash
git clone https://github.com/Lungelo-DS/DS4I-Assignment.git
cd DS4I-Assignment
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Open the folder in VS Code, select the `.venv` interpreter, and run the notebooks in `notebooks/`.

The data is read in Python with the `rdata` package:

```python
from data_utils import load_handwriting
images, meta = load_handwriting()   # images: (25000, 28, 28) in [0, 1]
```

## Collaboration workflow

- `main` holds reviewed work only. Nobody commits to it directly.
- Each piece of work happens on its own branch (e.g. `eda-overview`, `eda-sessions`).
- Changes reach `main` through a pull request reviewed by the other team member.
- Pull `main` before starting new work: `git pull origin main`.

## Team

| Member | GitHub | Responsibilities |
|---|---|---|
| _Name_ | [@Lungelo-DS](https://github.com/Lungelo-DS) | Repo setup, EDA overview (people × digits) |
| _Name_ | _@second-account_ | EDA sessions and data limitations |

## Licence

MIT. See [LICENSE](LICENSE).

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
├── report.qmd                    # the report (Quarto, Python chunks) -> PDF and HTML
├── index.qmd, _quarto.yml        # GitHub Pages landing page; output goes to docs/
├── references.bib
├── run_experiments.py            # trains every model and saves results/ (~70 min on a CPU)
├── Assignment 2/                 # brief (PDF) and data (handwriting.rds)
├── src/                          # modules imported by the report and notebooks
│   ├── data_utils.py             # load the .rds file into NumPy / pandas
│   ├── eda_plots.py              # style measurements (ink, width, slant, ...)
│   ├── eda_stats.py              # EDA tables and figures (Task 1)
│   ├── cnn.py                    # CNN writer identification (Task 2)
│   ├── siamese.py                # triplet-loss Siamese network (Tasks 3, 5, 6)
│   ├── contrastive.py            # contrastive-loss Siamese, L. Nkejane's design (Task 3)
│   ├── experiments.py            # all training runs, cached in results/
│   └── report_utils.py           # tables, plot style, git history for the report
├── notebooks/                    # exploratory notebooks (same analyses, step by step)
├── results/                      # saved results and model weights
├── docs/                         # rendered website (GitHub Pages)
├── requirements.txt
└── README.md
```

## Reproducing the report

Requires Python 3.10 or newer, in a version TensorFlow supports.

```bash
git clone https://github.com/Lungelo-DS/DS4I-Assignment.git
cd DS4I-Assignment
python -m venv .venv
.venv\Scripts\activate           # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_experiments.py         # skips steps whose results already exist; --force to redo all
quarto render                     # writes docs/report.pdf, docs/report.html and docs/index.html
```

Rendering needs Quarto (1.4 or newer) and a LaTeX installation for the PDF (`quarto install tinytex`).

## Collaboration workflow

- `main` holds reviewed work only. Nobody commits to it directly.
- Each piece of work happens on its own branch (e.g. `eda-overview`, `eda-sessions`).
- Changes reach `main` through a pull request reviewed by the other team member.
- Pull `main` before starting new work: `git pull origin main`.

## Team

| Member | GitHub | Responsibilities |
|---|---|---|
| Lungelo Phewa | [@Lungelo-DS](https://github.com/Lungelo-DS), [@LungeloPhewa](https://github.com/LungeloPhewa) | [to complete] |
| Lehlonono Nkejane | [to complete] | [to complete] |
| Ishmael Ngobeni | [@IshmaelNgobeni](https://github.com/IshmaelNgobeni) | [to complete] |

## Licence

MIT. See [LICENSE](LICENSE).

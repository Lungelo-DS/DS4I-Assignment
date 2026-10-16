"""Train every model used in the report and save the results to results/.

Usage (from the repository root):
    python run_experiments.py           # skip steps whose results already exist
    python run_experiments.py --force   # recompute everything (about 70 minutes on a CPU)

Then render the report:
    quarto render report.qmd
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import experiments  # noqa: E402

if __name__ == "__main__":
    experiments.run_all(force="--force" in sys.argv)

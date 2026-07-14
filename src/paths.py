"""Filesystem layout. Everything is resolved relative to the repository root so
the code runs unchanged regardless of where the repository is checked out.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
ONLINE_Q_DIR = DATA_DIR / "online_q"
RESULTS_DIR = DATA_DIR / "results"
REALISM_DIR = DATA_DIR / "realism"

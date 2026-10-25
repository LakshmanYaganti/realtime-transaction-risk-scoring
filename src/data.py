"""Dataset loading for the credit-card fraud dataset.

The ~150 MB CSV is never committed (size + Kaggle terms). Resolution order:
explicit path -> ``$CREDITCARD_DATA_PATH`` -> ``data/creditcard.csv`` ->
``/tmp/creditcard.csv`` -> download from the public TensorFlow mirror into
``data/creditcard.csv``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_path(path: str | None = None) -> Path:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("CREDITCARD_DATA_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / "data" / "creditcard.csv")
    candidates.append(Path("/tmp/creditcard.csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = REPO_ROOT / "data" / "creditcard.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {MIRROR_URL} -> {target}")
    urllib.request.urlretrieve(MIRROR_URL, target)
    return target


def load_data(path: str | None = None) -> pd.DataFrame:
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df.sort_values("Time").reset_index(drop=True)


def time_ordered_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 split — the honest split for a streaming system.

    The model is always trained on the past and evaluated on the future,
    which is how it would be deployed. (A shuffled split, used in the
    fraud-detection-ml project, leaks future fraud patterns into training.)
    """
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test

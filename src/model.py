"""Risk model: offline training on the real fraud dataset, online scoring.

The model scores the transaction itself (PCA features + amount + time of
day). Velocity features are deliberately *not* model inputs: the offline
training data has no entity history, so training on replay-time velocity
features would mean training on synthetic-entity artifacts. Instead the
enrichment features are attached to the scored event for investigators and
drive rule-based escalation in ``src.routing`` — the same score-plus-rules
split used by production fraud stacks.

Preprocessing mirrors the fraud-detection-ml project: ``Time`` and ``Amount``
are robust-scaled (median/IQR) while the PCA components pass through.
``hour_of_day`` is derived from ``Time`` because fraud rates are strongly
time-of-day dependent in this dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.events import FEATURE_COLUMNS

RAW_COLUMNS = FEATURE_COLUMNS + ["Amount", "Time"]
MODEL_COLUMNS = FEATURE_COLUMNS + ["Amount", "hour_of_day"]


def build_features(df: pd.DataFrame, scaler: RobustScaler | None = None) -> tuple[pd.DataFrame, RobustScaler]:
    """Build the model feature frame; fit the scaler when none is given."""
    work = df.copy()
    work["hour_of_day"] = (work["Time"] % 86_400) / 3600.0
    if scaler is None:
        scaler = RobustScaler()
        work[["Time", "Amount"]] = scaler.fit_transform(work[["Time", "Amount"]])
    else:
        work[["Time", "Amount"]] = scaler.transform(work[["Time", "Amount"]])
    return work[MODEL_COLUMNS], scaler


class RiskModel:
    """A fitted estimator + its fitted preprocessing, saved as one bundle."""

    def __init__(self, estimator, scaler: RobustScaler, name: str = "model"):
        self.estimator = estimator
        self.scaler = scaler
        self.name = name

    def score_features(self, feature_dict: dict[str, float]) -> float:
        """Fraud probability for one transaction's raw feature dict.

        Fast path for single-event (streaming) scoring: identical math to
        ``build_features`` — robust-scale Time/Amount with the fitted
        scaler, derive hour_of_day — but on a raw NumPy vector. Building a
        one-row DataFrame per event costs ~10ms in pandas overhead, which
        would dominate the streaming latency budget; this path is ~50x
        cheaper. Equivalence with the DataFrame path is covered by a test.
        """
        import warnings

        raw_time = float(feature_dict["Time"])
        raw_amount = float(feature_dict["Amount"])
        scaled = self.scaler.transform(np.array([[raw_time, raw_amount]], dtype=float))[0]
        hour = (raw_time % 86_400) / 3600.0
        vector = np.array(
            [[feature_dict[c] for c in FEATURE_COLUMNS] + [scaled[1], hour]], dtype=float
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.estimator.predict_proba(vector)[0]
        classes = list(self.estimator.classes_)
        return float(proba[classes.index(1)] if 1 in classes else proba[-1])

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Fraud probabilities for a whole frame of raw transactions."""
        X, _ = build_features(df, self.scaler)
        proba = self.estimator.predict_proba(X[MODEL_COLUMNS])
        classes = list(self.estimator.classes_)
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"estimator": self.estimator, "scaler": self.scaler, "name": self.name}, path)

    @classmethod
    def load(cls, path: str) -> "RiskModel":
        import joblib

        bundle = joblib.load(path)
        return cls(bundle["estimator"], bundle["scaler"], bundle.get("name", "model"))

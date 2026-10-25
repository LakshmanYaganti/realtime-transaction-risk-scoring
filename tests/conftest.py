import sys
from pathlib import Path

# Make the repository root importable (`src`, `streaming`) when running pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from src.events import FEATURE_COLUMNS, TransactionEvent
from src.model import MODEL_COLUMNS, RiskModel


def make_event(
    event_id: str = "e1",
    timestamp: float = 0.0,
    card_id: str = "card_00001",
    merchant_id: str = "merch_00001",
    amount: float = 50.0,
    feature_value: float = 0.0,
    label: int | None = None,
) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        timestamp=timestamp,
        card_id=card_id,
        merchant_id=merchant_id,
        amount=amount,
        features={c: feature_value for c in FEATURE_COLUMNS},
        label=label,
    )


@pytest.fixture()
def tiny_model() -> RiskModel:
    """A real (if tiny) model trained on synthetic separable data.

    Fraud-like rows get a strongly negative V14 (the single most fraud-
    indicative component in the real dataset); legitimate rows sit at 0.
    """
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for _ in range(200):
        rows.append({c: float(rng.normal(0, 0.5)) for c in FEATURE_COLUMNS} | {"Amount": 25.0, "Time": 3600.0})
        labels.append(0)
    for _ in range(40):
        row = {c: float(rng.normal(0, 0.5)) for c in FEATURE_COLUMNS}
        row["V14"] = float(rng.normal(-8, 0.5))
        row |= {"Amount": 25.0, "Time": 3600.0}
        rows.append(row)
        labels.append(1)
    frame = pd.DataFrame(rows)
    scaler = RobustScaler()
    work = frame.copy()
    work["hour_of_day"] = (work["Time"] % 86_400) / 3600.0
    work[["Time", "Amount"]] = scaler.fit_transform(work[["Time", "Amount"]])
    est = LogisticRegression(max_iter=1000)
    est.fit(work[MODEL_COLUMNS], np.array(labels))
    return RiskModel(est, scaler, "tiny-test-model")

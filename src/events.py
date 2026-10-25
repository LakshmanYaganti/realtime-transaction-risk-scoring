"""Transaction event types and synthetic entity assignment.

The credit-card fraud dataset this project replays is PCA-anonymized: it has
no card or merchant identifiers. Velocity features need entities, so the
replay assigns *synthetic* card/merchant IDs with a seeded generator
(Zipf-distributed, like real payment traffic where a few cards and merchants
account for a disproportionate share of volume). The assignment is purely a
harness for the enrichment machinery — it is deterministic for a given seed
and is never presented as real cardholder data. In a live deployment the
events arriving on the ``transactions.raw`` topic already carry real
``card_id`` / ``merchant_id`` values and this module's generator is unused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]


@dataclass
class TransactionEvent:
    """One transaction as published to the raw transactions topic."""

    event_id: str
    timestamp: float  # seconds; dataset "Time" (seconds since first txn)
    card_id: str
    merchant_id: str
    amount: float
    features: dict[str, float]  # V1..V28 PCA components
    label: int | None = None  # ground truth when replaying labeled data; never scored on

    def model_features(self) -> dict[str, float]:
        """Feature dict consumed by the risk model (see src.model)."""
        feats = dict(self.features)
        feats["Amount"] = float(self.amount)
        feats["Time"] = float(self.timestamp)
        return feats


@dataclass
class ScoredEvent:
    """A transaction after enrichment, scoring, and routing."""

    event: TransactionEvent
    risk_score: float
    band: str  # "high" | "medium" | "low"
    queue: str  # investigation queue / output topic the event is routed to
    enriched: dict[str, float]  # velocity/context features from enrichment
    routing_reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0  # end-to-end processing latency for this event


def assign_entities(
    df: pd.DataFrame,
    seed: int = 42,
    n_cards: int = 5_000,
    n_merchants: int = 500,
    zipf_exponent: float = 0.6,
) -> pd.DataFrame:
    """Assign deterministic synthetic card/merchant IDs to dataset rows.

    Returns a copy of ``df`` with ``card_id`` and ``merchant_id`` columns.
    IDs are drawn i.i.d. from a Zipf-weighted pool, so repeat activity (and
    therefore non-trivial velocity features) actually occurs during replay.
    """
    rng = np.random.default_rng(seed)

    def _zipf_ids(n: int, pool: int, prefix: str) -> np.ndarray:
        ranks = np.arange(1, pool + 1)
        weights = 1.0 / np.power(ranks, zipf_exponent)
        weights = weights / weights.sum()
        idx = rng.choice(pool, size=n, p=weights)
        return np.array([f"{prefix}_{i:05d}" for i in idx])

    out = df.copy()
    out["card_id"] = _zipf_ids(len(out), n_cards, "card")
    out["merchant_id"] = _zipf_ids(len(out), n_merchants, "merch")
    return out


def row_to_event(row: pd.Series, event_id: str) -> TransactionEvent:
    """Build a TransactionEvent from a dataset row with entity columns."""
    return TransactionEvent(
        event_id=event_id,
        timestamp=float(row["Time"]),
        card_id=str(row["card_id"]),
        merchant_id=str(row["merchant_id"]),
        amount=float(row["Amount"]),
        features={c: float(row[c]) for c in FEATURE_COLUMNS},
        label=int(row["Class"]) if "Class" in row and not pd.isna(row["Class"]) else None,
    )

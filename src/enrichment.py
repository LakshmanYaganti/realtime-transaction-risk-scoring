"""Velocity / context enrichment.

Given a transaction event and the state store, compute the features an
investigator (and the routing rules) need beyond the raw transaction:

- per-card counts and amount sums over 10-minute / 1-hour / 24-hour windows
- per-merchant counts over 1-hour / 24-hour windows
- time since the card's previous transaction
- amount deviation vs. the card's own history (ratio to mean and z-score)

All windows look strictly *backwards* from the event's timestamp: the current
event is not in its own history. The pipeline records the event in the store
only after enrichment, which is what keeps that invariant true.
"""

from __future__ import annotations

import numpy as np

from src.events import TransactionEvent
from src.state import StateStore

WINDOW_10M = 600
WINDOW_1H = 3600
WINDOW_24H = 24 * 3600


def _window_stats(history: list[tuple[float, float]], as_of: float, window: float) -> tuple[int, float]:
    """(count, amount_sum) of history entries within ``window`` seconds before ``as_of``."""
    lo = as_of - window
    count = 0
    total = 0.0
    for ts, amount in history:
        if lo <= ts < as_of:
            count += 1
            total += amount
    return count, total


def compute_velocity_features(store: StateStore, event: TransactionEvent) -> dict[str, float]:
    """Compute enrichment features for ``event`` from prior activity only."""
    ts = event.timestamp
    card_hist = store.card_history(event.card_id, ts)
    merch_hist = store.merchant_history(event.merchant_id, ts)

    card_10m_count, _ = _window_stats(card_hist, ts, WINDOW_10M)
    card_1h_count, card_1h_sum = _window_stats(card_hist, ts, WINDOW_1H)
    card_24h_count, card_24h_sum = _window_stats(card_hist, ts, WINDOW_24H)
    merch_1h_count, _ = _window_stats(merch_hist, ts, WINDOW_1H)
    merch_24h_count, _ = _window_stats(merch_hist, ts, WINDOW_24H)

    if card_hist:
        last_ts = max(t for t, _ in card_hist if t < ts) if any(t < ts for t, _ in card_hist) else None
        time_since_last = float(ts - last_ts) if last_ts is not None else -1.0
    else:
        time_since_last = -1.0

    prior_amounts = np.array([a for t, a in card_hist if t < ts], dtype=float)
    if prior_amounts.size > 0:
        mean_amt = float(prior_amounts.mean())
        amount_ratio = float(event.amount / mean_amt) if mean_amt > 0 else 1.0
        if prior_amounts.size >= 2 and float(prior_amounts.std()) > 0:
            amount_zscore = float((event.amount - mean_amt) / float(prior_amounts.std()))
        else:
            amount_zscore = 0.0
    else:
        amount_ratio = 1.0
        amount_zscore = 0.0

    return {
        "card_txn_count_10m": float(card_10m_count),
        "card_txn_count_1h": float(card_1h_count),
        "card_txn_count_24h": float(card_24h_count),
        "card_amount_sum_1h": float(card_1h_sum),
        "card_amount_sum_24h": float(card_24h_sum),
        "merchant_txn_count_1h": float(merch_1h_count),
        "merchant_txn_count_24h": float(merch_24h_count),
        "time_since_last_card_txn_s": float(time_since_last),
        "is_first_card_txn": 1.0 if not prior_amounts.size else 0.0,
        "amount_vs_card_avg_ratio": float(amount_ratio),
        "amount_vs_card_zscore": float(amount_zscore),
    }

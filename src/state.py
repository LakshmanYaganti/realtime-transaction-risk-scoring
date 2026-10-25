"""State store for velocity features.

Velocity features are computed from recent per-card and per-merchant
history, so the consumer needs a state store. The interface is a Protocol so
the in-memory implementation used for local runs and tests can be swapped
for a Redis-backed implementation in production (sliding-window sorted sets
per entity, with TTLs matching the longest window) without touching the
enrichment logic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol, runtime_checkable

MAX_WINDOW_S = 24 * 3600  # longest velocity window: 24h


@runtime_checkable
class StateStore(Protocol):
    """Minimal history interface the enrichment step depends on."""

    def card_history(self, card_id: str, as_of: float) -> list[tuple[float, float]]:
        """(timestamp, amount) pairs for a card, oldest first, pruned to 24h."""
        ...

    def merchant_history(self, merchant_id: str, as_of: float) -> list[tuple[float, float]]:
        """(timestamp, amount) pairs for a merchant, oldest first, pruned to 24h."""
        ...

    def add(self, card_id: str, merchant_id: str, timestamp: float, amount: float) -> None:
        """Record a transaction in both the card and merchant histories."""
        ...


class InMemoryStateStore:
    """Process-local StateStore backed by per-entity deques.

    Entries older than the 24h window (relative to the newest timestamp seen
    for that entity) are pruned lazily on read and on write, so memory stays
    proportional to 24h of traffic per active entity. Not shared across
    processes — that is exactly the gap the Redis implementation fills in
    production.
    """

    def __init__(self, max_window_s: float = MAX_WINDOW_S):
        self.max_window_s = float(max_window_s)
        self._cards: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._merchants: dict[str, deque[tuple[float, float]]] = defaultdict(deque)

    def _prune(self, history: deque[tuple[float, float]], as_of: float) -> None:
        cutoff = as_of - self.max_window_s
        while history and history[0][0] < cutoff:
            history.popleft()

    def card_history(self, card_id: str, as_of: float) -> list[tuple[float, float]]:
        history = self._cards.get(card_id)
        if not history:
            return []
        self._prune(history, as_of)
        return list(history)

    def merchant_history(self, merchant_id: str, as_of: float) -> list[tuple[float, float]]:
        history = self._merchants.get(merchant_id)
        if not history:
            return []
        self._prune(history, as_of)
        return list(history)

    def add(self, card_id: str, merchant_id: str, timestamp: float, amount: float) -> None:
        card = self._cards[card_id]
        card.append((timestamp, amount))
        self._prune(card, timestamp)
        merchant = self._merchants[merchant_id]
        merchant.append((timestamp, amount))
        self._prune(merchant, timestamp)

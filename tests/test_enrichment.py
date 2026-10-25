"""Enrichment math and velocity-window behavior."""

from src.enrichment import compute_velocity_features
from src.state import InMemoryStateStore
from tests.conftest import make_event


def _store_with_card_history(store, card_id, entries):
    for ts, amount in entries:
        store.add(card_id, "merch_other", ts, amount)
    return store


def test_first_transaction_has_empty_history_features():
    store = InMemoryStateStore()
    feats = compute_velocity_features(store, make_event(timestamp=1000.0))
    assert feats["card_txn_count_10m"] == 0
    assert feats["card_txn_count_1h"] == 0
    assert feats["card_txn_count_24h"] == 0
    assert feats["time_since_last_card_txn_s"] == -1.0
    assert feats["is_first_card_txn"] == 1.0
    assert feats["amount_vs_card_avg_ratio"] == 1.0


def test_velocity_counts_respect_windows():
    store = InMemoryStateStore()
    # Prior activity for this card: 100s ago, 30min ago, 3h ago, 30h ago (pruned).
    _store_with_card_history(
        store,
        "card_00001",
        [(9_900.0, 10.0), (8_200.0, 10.0), (-800.0, 10.0), (-98_000.0, 10.0)],
    )
    feats = compute_velocity_features(store, make_event(timestamp=10_000.0))
    assert feats["card_txn_count_10m"] == 1  # only the 100s-ago txn
    assert feats["card_txn_count_1h"] == 2  # 100s and 30min ago
    assert feats["card_txn_count_24h"] == 3  # everything except the 30h-ago txn
    assert feats["card_amount_sum_1h"] == 20.0
    assert feats["time_since_last_card_txn_s"] == 100.0
    assert feats["is_first_card_txn"] == 0.0


def test_amount_deviation_against_card_history():
    store = InMemoryStateStore()
    _store_with_card_history(store, "card_00001", [(100.0, 10.0), (200.0, 10.0), (300.0, 10.0)])
    feats = compute_velocity_features(store, make_event(timestamp=400.0, amount=100.0))
    assert feats["amount_vs_card_avg_ratio"] == 10.0  # 100 vs. mean of 10
    # All prior amounts identical -> std is 0 -> z-score defined as 0.
    assert feats["amount_vs_card_zscore"] == 0.0


def test_amount_zscore_with_varied_history():
    store = InMemoryStateStore()
    _store_with_card_history(store, "card_00001", [(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)])
    feats = compute_velocity_features(store, make_event(timestamp=400.0, amount=50.0))
    # mean 20, population std ~8.165 -> (50-20)/8.165
    assert abs(feats["amount_vs_card_zscore"] - 3.674) < 0.01
    assert feats["amount_vs_card_avg_ratio"] == 2.5


def test_merchant_counts_are_independent_of_card_counts():
    store = InMemoryStateStore()
    # Three other cards at the same merchant in the last hour.
    for i, card in enumerate(["card_a", "card_b", "card_c"]):
        store.add(card, "merch_00001", 900.0 + i, 15.0)
    feats = compute_velocity_features(store, make_event(timestamp=1000.0))
    assert feats["merchant_txn_count_1h"] == 3
    assert feats["card_txn_count_1h"] == 0


def test_state_store_prunes_beyond_24h():
    store = InMemoryStateStore()
    store.add("card_x", "merch_x", 0.0, 10.0)
    store.add("card_x", "merch_x", 100_000.0, 10.0)  # >24h later prunes the first
    history = store.card_history("card_x", as_of=100_000.0)
    assert history == [(100_000.0, 10.0)]


def test_event_does_not_count_toward_its_own_velocity():
    store = InMemoryStateStore()
    event = make_event(timestamp=500.0)
    feats_before = compute_velocity_features(store, event)
    store.add(event.card_id, event.merchant_id, event.timestamp, event.amount)
    feats_after = compute_velocity_features(store, make_event(event_id="e2", timestamp=500.0))
    assert feats_before["card_txn_count_1h"] == 0
    # A *different* event at the same instant also sees count 0 (strictly-before window).
    assert feats_after["card_txn_count_1h"] == 0

"""Routing bands, queue mapping, and velocity escalation rules."""

from src.routing import QUEUE_AUTO, QUEUE_HIGH, QUEUE_MEDIUM, RoutingConfig, band_for_score, route

CFG = RoutingConfig()


def _enriched(**overrides):
    base = {
        "card_txn_count_10m": 0.0,
        "card_txn_count_1h": 0.0,
        "card_txn_count_24h": 10.0,
        "amount_vs_card_avg_ratio": 1.0,
        "_amount": 20.0,
    }
    base.update(overrides)
    return base


def test_score_band_boundaries():
    assert band_for_score(0.0, CFG) == "low"
    assert band_for_score(0.29, CFG) == "low"
    assert band_for_score(0.30, CFG) == "medium"
    assert band_for_score(0.79, CFG) == "medium"
    assert band_for_score(0.80, CFG) == "high"
    assert band_for_score(1.0, CFG) == "high"


def test_band_to_queue_mapping():
    assert route(0.95, _enriched(), CFG).queue == QUEUE_HIGH
    assert route(0.50, _enriched(), CFG).queue == QUEUE_MEDIUM
    assert route(0.05, _enriched(), CFG).queue == QUEUE_AUTO


def test_velocity_burst_escalates_low_to_medium():
    decision = route(0.05, _enriched(card_txn_count_10m=4.0), CFG)
    assert decision.band == "medium"
    assert decision.queue == QUEUE_MEDIUM
    assert any("velocity:card_txn_count_10m" in r for r in decision.reasons)
    assert any("escalated_to:medium" in r for r in decision.reasons)


def test_sustained_hourly_volume_escalates():
    decision = route(0.05, _enriched(card_txn_count_1h=10.0), CFG)
    assert decision.band == "medium"
    assert any("velocity:card_txn_count_1h" in r for r in decision.reasons)


def test_velocity_burst_escalates_medium_to_high():
    decision = route(0.50, _enriched(card_txn_count_10m=6.0), CFG)
    assert decision.band == "high"
    assert decision.queue == QUEUE_HIGH


def test_high_band_is_not_escalated_further():
    decision = route(0.95, _enriched(card_txn_count_10m=20.0), CFG)
    assert decision.band == "high"
    assert not any(r.startswith("escalated_to") for r in decision.reasons)


def test_amount_ratio_rule_needs_a_large_amount():
    # Ratio 10x but a $5 transaction: no escalation (trivial amount guard).
    small = route(0.05, _enriched(amount_vs_card_avg_ratio=10.0, _amount=5.0), CFG)
    assert small.band == "low"
    # Same ratio at $500: escalates.
    large = route(0.05, _enriched(amount_vs_card_avg_ratio=10.0, _amount=500.0), CFG)
    assert large.band == "medium"


def test_amount_ratio_rule_needs_established_history():
    # Ratio 10x at $500, but only 2 prior transactions: the "average" is
    # noise, so the rule must not fire.
    decision = route(
        0.05,
        _enriched(amount_vs_card_avg_ratio=10.0, _amount=500.0, card_txn_count_24h=2.0),
        CFG,
    )
    assert decision.band == "low"
    assert decision.queue == QUEUE_AUTO


def test_below_velocity_threshold_no_escalation():
    decision = route(0.05, _enriched(card_txn_count_10m=3.0, card_txn_count_1h=9.0), CFG)
    assert decision.band == "low"
    assert decision.queue == QUEUE_AUTO

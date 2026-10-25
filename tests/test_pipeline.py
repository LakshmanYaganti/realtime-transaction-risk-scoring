"""Model scoring sanity, latency tracking, end-to-end pipeline, entities."""

import time

import numpy as np
import pandas as pd
import pytest

from src.events import FEATURE_COLUMNS, assign_entities
from src.latency import LatencyTracker
from src.pipeline import StreamingPipeline
from tests.conftest import make_event


def test_model_score_is_a_probability(tiny_model):
    event = make_event()
    score = tiny_model.score_features(event.model_features())
    assert 0.0 <= score <= 1.0


def test_fast_scoring_path_matches_frame_path(tiny_model):
    # score_features (NumPy fast path, used in streaming) must agree with
    # score_frame (DataFrame path, used in training/evaluation).
    event = make_event(amount=123.45, timestamp=73_000.0)
    event.features["V14"] = -3.5
    frame = pd.DataFrame([event.model_features()])
    via_frame = float(tiny_model.score_frame(frame)[0])
    via_fast = tiny_model.score_features(event.model_features())
    assert via_fast == pytest.approx(via_frame, abs=1e-9)


def test_model_scores_fraud_like_event_higher(tiny_model):
    legit = make_event(event_id="legit")
    fraud_like = make_event(event_id="fraud")
    fraud_like.features["V14"] = -8.0  # the pattern the tiny model was trained on
    assert tiny_model.score_features(fraud_like.model_features()) > tiny_model.score_features(
        legit.model_features()
    )


def test_latency_tracker_percentiles_and_sla():
    tracker = LatencyTracker(sla_ms=300_000)
    for ms in range(1, 101):
        tracker.add(float(ms))
    summary = tracker.summary()
    assert summary["count"] == 100
    assert summary["mean_ms"] == pytest.approx(50.5)
    assert summary["p50_ms"] == pytest.approx(50.5, abs=1.0)
    assert summary["p95_ms"] == pytest.approx(95.05, abs=1.0)
    assert summary["p99_ms"] == pytest.approx(99.01, abs=1.0)
    assert summary["max_ms"] == 100.0
    assert summary["pct_within_sla"] == 100.0


def test_latency_tracker_empty_and_breach():
    assert LatencyTracker().summary()["count"] == 0
    tracker = LatencyTracker(sla_ms=100)
    tracker.add(50.0)
    tracker.add(150.0)  # breaches the 100ms SLA
    assert tracker.summary()["pct_within_sla"] == 50.0


def test_end_to_end_replay_on_tiny_fixture(tiny_model):
    pipeline = StreamingPipeline(tiny_model)
    events = [
        make_event(event_id="t1", timestamp=0.0, amount=20.0),
        make_event(event_id="t2", timestamp=60.0, amount=20.0),
        make_event(event_id="t3", timestamp=120.0, amount=20.0),
    ]
    scored = [pipeline.process(e, published_at=time.perf_counter()) for e in events]

    assert all(s.queue in ("review-high", "review-medium", "auto-approve") for s in scored)
    assert all(0.0 <= s.risk_score <= 1.0 for s in scored)
    assert all(s.latency_ms >= 0.0 for s in scored)
    assert len(pipeline.latency) == 3
    # Velocity history builds up: the third event sees the two before it.
    assert scored[0].enriched["card_txn_count_1h"] == 0
    assert scored[2].enriched["card_txn_count_1h"] == 2
    assert scored[2].enriched["time_since_last_card_txn_s"] == 60.0


def test_pipeline_escalates_burst_traffic(tiny_model):
    pipeline = StreamingPipeline(tiny_model)
    last = None
    for i in range(6):
        event = make_event(event_id=f"b{i}", timestamp=float(i * 60), amount=20.0)
        last = pipeline.process(event)
    # 6th event within the hour: 5 prior card txns -> velocity escalation fired,
    # so even this low-scoring event cannot be in the auto-approve queue.
    assert last is not None
    assert last.enriched["card_txn_count_1h"] == 5
    assert last.queue != "auto-approve"


def test_assign_entities_is_deterministic():
    df = pd.DataFrame(
        {
            "Time": [0.0, 60.0, 120.0],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 0, 1],
            **{c: [0.0, 0.0, 0.0] for c in FEATURE_COLUMNS},
        }
    )
    first = assign_entities(df, seed=7)
    second = assign_entities(df, seed=7)
    assert list(first["card_id"]) == list(second["card_id"])
    assert list(first["merchant_id"]) == list(second["merchant_id"])
    assert first["card_id"].str.startswith("card_").all()
    assert first["merchant_id"].str.startswith("merch_").all()


def test_serialization_round_trip():
    from streaming.consumer import deserialize_event, scored_to_payload
    from streaming.producer import serialize_event

    event = make_event(event_id="rt1", timestamp=42.0, amount=99.5, label=None)
    restored = deserialize_event(serialize_event(event))
    assert restored.event_id == event.event_id
    assert restored.card_id == event.card_id
    assert restored.amount == event.amount
    assert restored.features == event.features
    assert restored.label is None  # labels never travel on the topic

    class _Scored:
        pass

    scored = _Scored()
    scored.event = event
    scored.risk_score = 0.42
    scored.band = "medium"
    scored.queue = "review-medium"
    scored.routing_reasons = ["score_band:medium"]
    scored.enriched = {"card_txn_count_1h": 1.0}
    scored.latency_ms = 1.23
    payload = scored_to_payload(scored)
    assert b"review-medium" in payload

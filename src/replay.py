"""Replay benchmark: stream held-out transactions through the pipeline.

Replays transactions from the held-out test period, in chronological order,
through the exact same ``StreamingPipeline`` the Kafka consumer uses —
enrichment, scoring, routing — and measures real per-event processing
latency. Results are merged into ``reports/metrics.json``.

This is a *replay*, not a live stream: it measures the pipeline's processing
latency on this machine and does not include Kafka broker transit time. The
README states this plainly, next to the numbers.

Usage:
    python -m src.replay [--data PATH] [--limit 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.data import load_data, time_ordered_split
from src.events import assign_entities, row_to_event
from src.latency import SLA_MS
from src.model import RiskModel
from src.pipeline import StreamingPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "risk_model.joblib"))
    parser.add_argument("--metrics", default=str(REPO_ROOT / "reports" / "metrics.json"))
    args = parser.parse_args()

    model = RiskModel.load(args.model)
    df = load_data(args.data)
    _, _, test = time_ordered_split(df)
    replay_df = (
        assign_entities(test, seed=args.seed, n_cards=5_000, n_merchants=500, zipf_exponent=0.6)
        .head(args.limit)
        .reset_index(drop=True)
    )
    print(f"replaying {len(replay_df)} test-period transactions with model '{model.name}'")

    pipeline = StreamingPipeline(model)
    queues: Counter[str] = Counter()
    bands: Counter[str] = Counter()
    escalations = 0
    fraud_total = 0
    fraud_in_review = 0  # fraud routed to review-high or review-medium
    fraud_high = 0
    wall_start = time.perf_counter()

    for i, (_, row) in enumerate(replay_df.iterrows()):
        event = row_to_event(row, event_id=f"replay-{i:06d}")
        published_at = time.perf_counter()  # event "publication" in the replay harness
        scored = pipeline.process(event, published_at=published_at)
        queues[scored.queue] += 1
        bands[scored.band] += 1
        if any(r.startswith("escalated_to") for r in scored.routing_reasons):
            escalations += 1
        if event.label == 1:
            fraud_total += 1
            if scored.queue in ("review-high", "review-medium"):
                fraud_in_review += 1
            if scored.queue == "review-high":
                fraud_high += 1

    wall_s = time.perf_counter() - wall_start
    latency = pipeline.latency.summary()

    replay_metrics = {
        "events_replayed": int(len(replay_df)),
        "entity_assignment": (
            "synthetic seeded Zipf IDs (n_cards=5000, n_merchants=500, exponent=0.6); "
            "the source dataset is anonymized and carries no real card/merchant identifiers"
        ),
        "latency": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in latency.items()},
        "throughput_events_per_s": round(len(replay_df) / wall_s, 2) if wall_s > 0 else 0.0,
        "sla_ms": SLA_MS,
        "routing_distribution": dict(queues),
        "band_distribution": dict(bands),
        "velocity_escalations": int(escalations),
        "fraud_in_replay": int(fraud_total),
        "fraud_routed_to_review": int(fraud_in_review),
        "fraud_routed_to_review_high": int(fraud_high),
        "fraud_review_capture_rate": (
            round(fraud_in_review / fraud_total, 4) if fraud_total else None
        ),
    }

    print(json.dumps(replay_metrics, indent=2))

    metrics_path = Path(args.metrics)
    metrics: dict = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    metrics["replay"] = replay_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

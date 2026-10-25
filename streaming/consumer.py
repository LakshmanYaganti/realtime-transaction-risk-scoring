"""Kafka consumer: enrich, score, and route transactions in real time.

Consumes ``transactions.raw``, runs each event through
``src.pipeline.StreamingPipeline`` (the same core the tests and the replay
benchmark exercise), and publishes the scored event to the queue topic its
routing decision selected: ``review-high``, ``review-medium``, or
``auto-approve``. Prints rolling latency stats so the 5-minute delivery SLA
can be watched live.

Usage:
    docker compose up -d
    python -m src.train                 # once, to build models/risk_model.joblib
    python -m streaming.consumer

Requires a running broker and a trained model bundle; importing this module
requires neither.
"""

from __future__ import annotations

import argparse
import json
import time

from src.events import TransactionEvent
from src.pipeline import StreamingPipeline

QUEUE_TOPICS = ("review-high", "review-medium", "auto-approve")


def deserialize_event(raw: bytes) -> TransactionEvent:
    payload = json.loads(raw.decode("utf-8"))
    return TransactionEvent(
        event_id=str(payload["event_id"]),
        timestamp=float(payload["timestamp"]),
        card_id=str(payload["card_id"]),
        merchant_id=str(payload["merchant_id"]),
        amount=float(payload["amount"]),
        features={k: float(v) for k, v in payload["features"].items()},
    )


def scored_to_payload(scored) -> bytes:
    payload = {
        "event_id": scored.event.event_id,
        "timestamp": scored.event.timestamp,
        "card_id": scored.event.card_id,
        "merchant_id": scored.event.merchant_id,
        "amount": scored.event.amount,
        "risk_score": scored.risk_score,
        "band": scored.band,
        "queue": scored.queue,
        "routing_reasons": scored.routing_reasons,
        "enriched": scored.enriched,
        "latency_ms": scored.latency_ms,
    }
    return json.dumps(payload).encode("utf-8")


def main() -> None:
    from kafka import KafkaConsumer, KafkaProducer  # live-broker only

    from src.model import RiskModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transactions.raw")
    parser.add_argument("--group-id", default="risk-scoring")
    parser.add_argument("--model", default="models/risk_model.joblib")
    parser.add_argument("--report-every", type=int, default=500)
    args = parser.parse_args()

    model = RiskModel.load(args.model)
    pipeline = StreamingPipeline(model)
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    print(f"consuming {args.topic} with model '{model.name}' ... Ctrl+C to stop")
    try:
        for message in consumer:
            published_at = time.perf_counter()
            event = deserialize_event(message.value)
            scored = pipeline.process(event, published_at=published_at)
            producer.send(scored.queue, key=event.card_id.encode("utf-8"), value=scored_to_payload(scored))
            if len(pipeline.latency) % args.report_every == 0:
                s = pipeline.latency.summary()
                print(
                    f"processed={s['count']} mean={s['mean_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
                    f"p99={s['p99_ms']:.2f}ms within_sla={s['pct_within_sla']:.2f}%"
                )
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        producer.flush()
        producer.close()
        consumer.close()


if __name__ == "__main__":
    main()

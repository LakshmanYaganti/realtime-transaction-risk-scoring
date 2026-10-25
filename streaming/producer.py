"""Kafka producer: publish transactions to the ``transactions.raw`` topic.

In production this role is played by the payment platform itself; here the
producer replays the dataset (or any CSV with the dataset's schema) onto the
topic so the consumer can be run end-to-end locally with docker-compose.

Usage:
    docker compose up -d          # start Kafka
    python -m streaming.producer --rate 100 --limit 5000

Requires a running broker; importing this module does not.
"""

from __future__ import annotations

import argparse
import json
import time

from src.events import FEATURE_COLUMNS, TransactionEvent, assign_entities, row_to_event


def serialize_event(event: TransactionEvent) -> bytes:
    payload = {
        "event_id": event.event_id,
        "timestamp": event.timestamp,
        "card_id": event.card_id,
        "merchant_id": event.merchant_id,
        "amount": event.amount,
        "features": event.features,
    }
    return json.dumps(payload).encode("utf-8")


def main() -> None:
    from kafka import KafkaProducer  # imported here: only needed with a live broker

    from src.data import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transactions.raw")
    parser.add_argument("--data", default=None)
    parser.add_argument("--limit", type=int, default=5_000)
    parser.add_argument("--rate", type=float, default=100.0, help="events per second (0 = as fast as possible)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = assign_entities(load_data(args.data), seed=args.seed).head(args.limit)
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    for i, (_, row) in enumerate(df.iterrows()):
        event = row_to_event(row, event_id=f"txn-{i:06d}")
        producer.send(args.topic, key=event.card_id.encode("utf-8"), value=serialize_event(event))
        sent += 1
        if delay:
            time.sleep(delay)
    producer.flush()
    producer.close()
    print(f"published {sent} events to {args.topic}")


if __name__ == "__main__":
    main()

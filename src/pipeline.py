"""The streaming pipeline: enrich -> score -> route, latency-tracked.

This is the consumer's core. It knows nothing about Kafka: the live consumer
(``streaming/consumer.py``) deserializes topic messages into
``TransactionEvent`` objects and hands them here; the replay benchmark
(``src/replay.py``) does exactly the same thing from a CSV. Both paths share
this code, which is what makes the benchmark meaningful and the logic
testable without a broker.

Latency definition: ``published_at`` is a ``time.perf_counter()`` reading
taken by the caller the moment the event is handed to the pipeline (in the
live consumer: the moment the message is taken off the topic). The recorded
latency is the elapsed time until the routing decision is made — enrichment
+ scoring + routing. Broker transit time is not included in local replay;
see the README's "Honesty notes" section.
"""

from __future__ import annotations

import time

from src.enrichment import compute_velocity_features
from src.events import ScoredEvent, TransactionEvent
from src.latency import LatencyTracker
from src.model import RiskModel
from src.routing import RoutingConfig, route
from src.state import InMemoryStateStore, StateStore


class StreamingPipeline:
    def __init__(
        self,
        model: RiskModel,
        store: StateStore | None = None,
        routing_config: RoutingConfig | None = None,
        latency_tracker: LatencyTracker | None = None,
    ):
        self.model = model
        self.store: StateStore = store if store is not None else InMemoryStateStore()
        self.routing_config = routing_config or RoutingConfig()
        self.latency = latency_tracker or LatencyTracker()

    def process(self, event: TransactionEvent, published_at: float | None = None) -> ScoredEvent:
        start = published_at if published_at is not None else time.perf_counter()

        enriched = compute_velocity_features(self.store, event)
        score = self.model.score_features(event.model_features())

        # The amount-ratio escalation rule needs the raw amount; pass it
        # through the enrichment dict under a private key, then strip it
        # from what gets attached to the scored event.
        enriched_for_routing = dict(enriched)
        enriched_for_routing["_amount"] = float(event.amount)
        decision = route(score, enriched_for_routing, self.routing_config)

        # Record history only after enrichment so an event never counts
        # toward its own velocity features.
        self.store.add(event.card_id, event.merchant_id, event.timestamp, event.amount)

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.latency.add(latency_ms)

        return ScoredEvent(
            event=event,
            risk_score=score,
            band=decision.band,
            queue=decision.queue,
            enriched=enriched,
            routing_reasons=decision.reasons,
            latency_ms=latency_ms,
        )

"""Risk-band routing: scored event -> investigation queue.

Bands are set from the model score, then velocity rules can *escalate* (never
de-escalate) an event when the enrichment features show burst behavior or an
amount far outside the card's own history. Every escalation records a
human-readable reason so an investigator can see why an event landed in
their queue.

Queues map 1:1 to Kafka topics in the live deployment:

- ``review-high``    — analyst review before/with priority
- ``review-medium``  — standard analyst review queue
- ``auto-approve``   — no review; event flows to downstream processing
"""

from __future__ import annotations

from dataclasses import dataclass

QUEUE_HIGH = "review-high"
QUEUE_MEDIUM = "review-medium"
QUEUE_AUTO = "auto-approve"

_BAND_ORDER = ["low", "medium", "high"]
_BAND_QUEUE = {"high": QUEUE_HIGH, "medium": QUEUE_MEDIUM, "low": QUEUE_AUTO}


@dataclass(frozen=True)
class RoutingConfig:
    high_threshold: float = 0.80  # score >= -> high
    medium_threshold: float = 0.30  # score >= -> medium
    # Burst rules, calibrated like production velocity rules: a handful of
    # transactions in a few minutes, or sustained high volume over an hour.
    # (A low single threshold over a full hour escalates ordinary repeat
    # customers — in an early replay calibration it pushed ~59% of all
    # traffic into review, which no investigations team can work.)
    velocity_card_count_10m: float = 4  # prior card txns within 10 min
    velocity_card_count_1h: float = 10  # prior card txns within 1 hour
    amount_ratio_threshold: float = 5.0  # amount vs. card's historical mean
    amount_ratio_min_amount: float = 100.0  # ...but only for non-trivial amounts
    amount_ratio_min_history: float = 5  # ...and only once the card has an
    # established baseline: a ratio against 1-2 prior transactions is noise,
    # and firing on it floods the review queue with ordinary customers.


@dataclass
class RoutingDecision:
    band: str
    queue: str
    reasons: list[str]


def band_for_score(score: float, config: RoutingConfig = RoutingConfig()) -> str:
    if score >= config.high_threshold:
        return "high"
    if score >= config.medium_threshold:
        return "medium"
    return "low"


def route(score: float, enriched: dict[str, float], config: RoutingConfig = RoutingConfig()) -> RoutingDecision:
    """Route a scored event, applying velocity escalation rules."""
    band = band_for_score(score, config)
    reasons = [f"score_band:{band}"]

    escalate = False
    if enriched.get("card_txn_count_10m", 0.0) >= config.velocity_card_count_10m:
        escalate = True
        reasons.append(
            f"velocity:card_txn_count_10m>={int(config.velocity_card_count_10m)}"
        )
    if enriched.get("card_txn_count_1h", 0.0) >= config.velocity_card_count_1h:
        escalate = True
        reasons.append(
            f"velocity:card_txn_count_1h>={int(config.velocity_card_count_1h)}"
        )
    ratio = enriched.get("amount_vs_card_avg_ratio", 1.0)
    # amount is not in the enriched dict; the ratio rule only fires alongside
    # a genuinely large ratio, a non-trivial amount (passed in by the
    # pipeline under "_amount"), and enough card history to trust the mean.
    if (
        ratio >= config.amount_ratio_threshold
        and enriched.get("_amount", 0.0) >= config.amount_ratio_min_amount
        and enriched.get("card_txn_count_24h", 0.0) >= config.amount_ratio_min_history
    ):
        escalate = True
        reasons.append(f"velocity:amount_vs_card_avg_ratio>={config.amount_ratio_threshold:g}")

    if escalate and band != "high":
        band = _BAND_ORDER[_BAND_ORDER.index(band) + 1]
        reasons.append(f"escalated_to:{band}")

    return RoutingDecision(band=band, queue=_BAND_QUEUE[band], reasons=reasons)

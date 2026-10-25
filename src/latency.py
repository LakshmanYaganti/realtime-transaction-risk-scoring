"""End-to-end latency tracking.

The production SLA this system was built against: a transaction's risk
indicators must reach the downstream investigation queues **within 5 minutes
of event publication**. This tracker records the per-event processing latency
measured by the pipeline and summarizes it against that SLA.
"""

from __future__ import annotations

import numpy as np

SLA_MS = 5 * 60 * 1000  # 5 minutes, the delivery SLA


class LatencyTracker:
    def __init__(self, sla_ms: float = SLA_MS):
        self.sla_ms = float(sla_ms)
        self._samples: list[float] = []

    def add(self, latency_ms: float) -> None:
        self._samples.append(float(latency_ms))

    def __len__(self) -> int:
        return len(self._samples)

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
                "pct_within_sla": 0.0,
                "sla_ms": self.sla_ms,
            }
        arr = np.asarray(self._samples, dtype=float)
        return {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
            "pct_within_sla": float((arr <= self.sla_ms).mean() * 100.0),
            "sla_ms": self.sla_ms,
        }

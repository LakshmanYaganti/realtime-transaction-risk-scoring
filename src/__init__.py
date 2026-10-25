"""Real-time transaction risk scoring: enrichment, scoring, routing.

The core package is deliberately Kafka-free: every module here can be
imported and tested without a running broker. The Kafka producer/consumer
entry points live in ``streaming/`` and are thin adapters over
``src.pipeline.StreamingPipeline``.
"""

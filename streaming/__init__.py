"""Kafka adapters for the streaming pipeline.

These modules are the only place kafka-python is used. The serialization
helpers are importable without a broker (and without kafka-python); the
``KafkaProducer`` / ``KafkaConsumer`` objects are constructed inside the
``main()`` entry points only.
"""

"""Message bus: Kafka when available, an in-memory stand-in otherwise (Stage 8)."""
import json
import logging
import os
import threading
from collections import deque

log = logging.getLogger("router.bus")


def bootstrap_servers() -> str:
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


class LocalBus:
    """Keeps recent messages in memory. Used in tests and when Kafka is not running."""
    name = "none"

    def __init__(self, keep: int = 1000):
        self.published = deque(maxlen=keep)

    def publish(self, topic: str, key: str, value: dict) -> None:
        self.published.append((topic, key, value))

    def flush(self, timeout: float = 5.0) -> None:
        pass


class KafkaBus:
    name = "kafka"

    def __init__(self, servers: str = None):
        from confluent_kafka import Producer

        self._p = Producer({"bootstrap.servers": servers or bootstrap_servers(), "log_level": 0,
                            "linger.ms": 5, "socket.timeout.ms": 5000, "message.timeout.ms": 10000})

    def publish(self, topic: str, key: str, value: dict) -> None:
        try:
            self._p.produce(topic, key=str(key), value=json.dumps(value, default=str).encode())
            self._p.poll(0)
        except BufferError:
            log.warning("kafka producer queue full, dropping message for %s", topic)
        except Exception as exc:  # the payment path must never fail because the bus is down
            log.warning("kafka publish failed: %s", exc)

    def flush(self, timeout: float = 5.0) -> None:
        self._p.flush(timeout)


class ConsumerThread(threading.Thread):
    """Reads JSON messages from one topic and hands each to `handler(dict)`."""

    def __init__(self, topic: str, group: str, handler, servers: str = None, offset_reset: str = "latest"):
        super().__init__(daemon=True, name=f"consumer-{topic}")
        self.topic, self.group, self.handler = topic, group, handler
        self.servers, self.offset_reset = servers or bootstrap_servers(), offset_reset
        self._halt = threading.Event()
        self.ready = threading.Event()

    def run(self) -> None:
        from confluent_kafka import Consumer

        c = Consumer({"bootstrap.servers": self.servers, "group.id": self.group, "log_level": 0,
                      "auto.offset.reset": self.offset_reset, "enable.auto.commit": True})
        c.subscribe([self.topic], on_assign=lambda *_: self.ready.set())
        try:
            while not self._halt.is_set():
                msg = c.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    log.warning("consumer error on %s: %s", self.topic, msg.error())
                    continue
                try:
                    self.handler(json.loads(msg.value()))
                except Exception:
                    log.exception("handler failed for a message on %s", self.topic)
        finally:
            c.close()

    def stop(self) -> None:
        self._halt.set()


def make_bus(mode: str = None):
    """Kafka when reachable (or demanded with EVENT_BUS=kafka), otherwise a local stand-in."""
    mode = mode or os.getenv("EVENT_BUS", "auto")
    if mode != "none":
        try:
            from confluent_kafka.admin import AdminClient

            AdminClient({"bootstrap.servers": bootstrap_servers(), "log_level": 0}).list_topics(timeout=3)
            return KafkaBus()
        except Exception:
            if mode == "kafka":
                raise
    return LocalBus()

"""Check that Postgres, Redis, Kafka and Grafana are reachable.

Run from the project root, after `docker compose up -d`:

    python scripts/check_services.py
"""
import os
import sys
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def check_postgres() -> str:
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "payment_router"),
        user=os.getenv("POSTGRES_USER", "router"),
        password=os.getenv("POSTGRES_PASSWORD", "change_me"),
        connect_timeout=5,
    )
    with conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0].split(",")[0]
    conn.close()
    return version


def check_redis() -> str:
    import redis

    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        socket_connect_timeout=5,
    )
    client.set("pr:healthcheck", "ok", ex=30)
    value = client.get("pr:healthcheck").decode()
    return f"set/get ok ({value}), redis {client.info()['redis_version']}"


def check_kafka() -> str:
    from confluent_kafka.admin import AdminClient

    admin = AdminClient(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "log_level": 0,  # silence librdkafka connection-retry spam
            "socket.timeout.ms": 5000,
        }
    )
    md = admin.list_topics(timeout=6)
    return f"{len(md.brokers)} broker(s), {len(md.topics)} topic(s)"


def check_grafana() -> str:
    with urllib.request.urlopen("http://localhost:3000/api/health", timeout=5) as r:
        body = r.read().decode()
    return body.replace("\n", "").replace(" ", "")


CHECKS = [
    ("PostgreSQL", check_postgres),
    ("Redis", check_redis),
    ("Kafka", check_kafka),
    ("Grafana", check_grafana),
]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"[ OK ] {name:<11} {detail}")
        except Exception as exc:  # report every service, not just the first failure
            failures += 1
            print(f"[FAIL] {name:<11} {type(exc).__name__}: {exc}")
    print()
    print("All services reachable." if failures == 0 else f"{failures} service(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

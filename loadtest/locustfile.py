"""Load test for POST /score.

    python -m loadtest.make_payloads
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 60s --csv loadtest/results

Reports p50/p95/p99 in the console and loadtest/results_stats.csv.
Note: each request also writes entity state to Redis, so the same payload
replayed repeatedly bumps that entity's counters. Fine for latency, not
meaningful for score quality.
"""
import json
import os
import random
from pathlib import Path

from locust import HttpUser, constant_throughput, task

PAYLOADS = [json.loads(line) for line in (Path(__file__).parent / "payloads.jsonl").read_text().splitlines()]


class Scorer(HttpUser):
    # Fixed rate: each simulated user sends RPS_PER_USER requests per second
    # (default 1), so `-u 20` is ~20 req/s. Latency is then reported at a stated
    # request rate, instead of at whatever saturates the server.
    wait_time = constant_throughput(float(os.getenv("RPS_PER_USER", "1")))

    @task
    def score(self):
        with self.client.post("/score", json=random.choice(PAYLOADS), catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"{r.status_code}: {r.text[:200]}")

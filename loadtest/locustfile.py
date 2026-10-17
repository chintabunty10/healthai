"""Load test: 100 concurrent users, mix of repeated (cacheable) and varied symptom queries.

Run headless:
  locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest/results
"""
import random

from locust import HttpUser, between, events, task

COMMON = ["fever, cough", "itching, skin rash", "headache, vomiting", "chest pain, breathlessness"]
POOL = ["fever", "cough", "headache", "fatigue", "vomiting", "chills", "skin rash", "joint pain",
        "abdominal pain", "dizziness", "nausea", "muscle pain", "itching", "sweating"]


class SymptomUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(7)
    def repeated_query(self):
        self.client.post("/api/predict", json={"symptoms": random.choice(COMMON)})

    @task(3)
    def varied_query(self):
        self.client.post("/api/predict", json={"symptoms": ", ".join(random.sample(POOL, 3))})

    @task(1)
    def health(self):
        self.client.get("/api/health")


# Pass/fail gate so the run is a real check, not just a report (override with env vars if needed).
import os

MAX_FAIL_RATIO = float(os.getenv("LOAD_MAX_FAIL_RATIO", "0.01"))
MAX_P95_MS = float(os.getenv("LOAD_MAX_P95_MS", "500"))


@events.quitting.add_listener
def _gate(environment, **kwargs):
    total = environment.stats.total
    p95 = total.get_response_time_percentile(0.95) or 0
    if total.fail_ratio > MAX_FAIL_RATIO:
        print(f"LOAD TEST FAILED: failure ratio {total.fail_ratio:.2%} > {MAX_FAIL_RATIO:.2%}")
        environment.process_exit_code = 1
    elif p95 > MAX_P95_MS:
        print(f"LOAD TEST FAILED: p95 {p95:.0f} ms > {MAX_P95_MS:.0f} ms")
        environment.process_exit_code = 1
    else:
        print(f"LOAD TEST PASSED: p95 {p95:.0f} ms, failures {total.fail_ratio:.2%}, rps {total.total_rps:.1f}")

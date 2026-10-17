"""Load test: 100 concurrent users, mix of repeated (cacheable) and varied symptom queries.

Run headless:
  locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest/results
"""
import random

from locust import HttpUser, between, task

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

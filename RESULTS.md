# Load test result

Tool: Locust 2.46.7, `loadtest/locustfile.py` (70% repeated cacheable queries, 30% varied queries, plus `/api/health`).
Command: `locust -f loadtest\locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest\results`
Date: 2026-10-20. Machine: local Windows PC; app and Locust on the same machine.
Outcome printed by the gate: `LOAD TEST PASSED: p95 190 ms, failures 0.00%, rps 140.6`

| Type | Name | # reqs | # fails | Avg (ms) | Min | Max | Med | req/s |
|---|---|---|---|---|---|---|---|---|
| GET | /api/health | 720 | 0 | 67 | 2 | 2295 | 22 | 12.4 |
| POST | /api/predict | 7,448 | 0 | 81 | 2 | 2675 | 30 | 128.2 |
| | Aggregated | 8,168 | 0 (0.00%) | 80 | 2 | 2675 | 29 | 140.6 |

Response time percentiles (ms, approximated by Locust)

| Name | 50% | 66% | 75% | 80% | 90% | 95% | 98% | 99% | 99.9% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| GET /api/health | 22 | 33 | 46 | 58 | 91 | 140 | 240 | 2200 | 2300 | 2300 |
| POST /api/predict | 30 | 53 | 75 | 89 | 130 | 190 | 320 | 2200 | 2600 | 2700 |
| Aggregated | 29 | 51 | 73 | 87 | 130 | 190 | 320 | 2200 | 2600 | 2700 |

Notes: the slowest requests occurred during the initial ramp-up (interpretation). `/api/explain` and image upload were not tested.

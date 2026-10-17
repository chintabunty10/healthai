import asyncio
import time

import pytest

import threading

from app.resilience import Busy, CircuitBreaker, Limiter, SingleFlight, ThreadSingleFlight


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_breaker_opens_after_threshold_then_probes_once_after_cooldown():
    clk = Clock()
    b = CircuitBreaker(failure_threshold=3, cooldown=10, clock=clk)
    assert b.state == "closed" and b.allow()
    b.record_failure(); b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clk.t = 11
    assert b.state == "half_open"
    assert b.allow() is True            # the single trial call
    assert b.allow() is False           # everyone else still blocked while the trial is in flight
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_breaker_failed_trial_reopens_with_fresh_cooldown():
    clk = Clock()
    b = CircuitBreaker(failure_threshold=1, cooldown=10, clock=clk)
    b.record_failure()
    clk.t = 10
    assert b.allow()
    b.record_failure()                   # trial failed
    assert b.state == "open"
    clk.t = 15
    assert b.state == "open"             # cool-down restarted at t=10
    clk.t = 20
    assert b.state == "half_open"


def test_success_resets_failure_count():
    b = CircuitBreaker(failure_threshold=3)
    b.record_failure(); b.record_failure(); b.record_success(); b.record_failure(); b.record_failure()
    assert b.state == "closed"


def test_singleflight_runs_identical_work_once():
    calls = []

    async def work():
        calls.append(1)
        await asyncio.sleep(0.05)
        return "result"

    async def main():
        sf = SingleFlight()
        res = await asyncio.gather(*[sf.do("k", work) for _ in range(10)])
        return res, sf.in_flight

    res, in_flight = asyncio.run(main())
    assert res == ["result"] * 10 and len(calls) == 1 and in_flight == 0


def test_singleflight_propagates_errors_to_all_waiters_and_recovers():
    async def boom():
        await asyncio.sleep(0.01)
        raise RuntimeError("upstream down")

    async def main():
        sf = SingleFlight()
        out = await asyncio.gather(*[sf.do("k", boom) for _ in range(3)], return_exceptions=True)
        again = await sf.do("k", lambda: asyncio.sleep(0, result="ok"))
        return out, again

    out, again = asyncio.run(main())
    assert all(isinstance(e, RuntimeError) for e in out) and again == "ok"


def test_singleflight_does_not_cancel_work_when_one_caller_is_cancelled():
    done = []

    async def work():
        await asyncio.sleep(0.1)
        done.append(1)
        return 1

    async def main():
        sf = SingleFlight()
        t1 = asyncio.create_task(sf.do("k", work))
        t2 = asyncio.create_task(sf.do("k", work))
        await asyncio.sleep(0.01)
        t1.cancel()
        return await t2

    assert asyncio.run(main()) == 1 and done == [1]


def test_limiter_bounds_concurrency_and_sheds_load():
    def slow(x):
        time.sleep(0.3)
        return x

    async def main():
        lim = Limiter(max_concurrent=2, queue_timeout=0.05)
        results = await asyncio.gather(*[lim.run(slow, i) for i in range(5)], return_exceptions=True)
        return results

    results = asyncio.run(main())
    ok = [r for r in results if not isinstance(r, Exception)]
    busy = [r for r in results if isinstance(r, Busy)]
    assert len(ok) == 2 and len(busy) == 3


def test_limiter_releases_slot_after_error():
    def fail():
        raise ValueError("x")

    async def main():
        lim = Limiter(1, 0.5)
        with pytest.raises(ValueError):
            await lim.run(fail)
        return await lim.run(lambda: "fine")

    assert asyncio.run(main()) == "fine"


def test_thread_singleflight_computes_once_for_a_burst_of_identical_calls():
    sf, calls, results = ThreadSingleFlight(), [], []

    def work():
        calls.append(1)
        time.sleep(0.1)
        return "v"

    ts = [threading.Thread(target=lambda: results.append(sf.do("k", work))) for _ in range(20)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results == ["v"] * 20 and len(calls) == 1


def test_thread_singleflight_shares_errors_and_allows_retry_afterwards():
    sf = ThreadSingleFlight()
    errors = []

    def boom():
        time.sleep(0.05)
        raise RuntimeError("x")

    def call():
        try:
            sf.do("k", boom)
        except RuntimeError as e:
            errors.append(e)

    ts = [threading.Thread(target=call) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(errors) == 4
    assert sf.do("k", lambda: "recovered") == "recovered"

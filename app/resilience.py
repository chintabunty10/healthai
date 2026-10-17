"""Small reliability primitives used to survive load and dependency failures.

CircuitBreaker  - stop calling a failing dependency for a cool-down, then probe once.
SingleFlight    - N identical concurrent requests share ONE upstream computation.
Limiter         - bound concurrent calls to a slow/expensive provider; shed load instead of queueing forever.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Awaitable, Callable


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold, self.cooldown, self._clock = failure_threshold, cooldown, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial = False
        self._lock = threading.Lock()

    def _state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.cooldown else "open"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def allow(self) -> bool:
        """closed: yes. open: no. half-open: exactly one trial call until its result is recorded."""
        with self._lock:
            st = self._state()
            if st == "closed":
                return True
            if st == "half_open" and not self._trial:
                self._trial = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._trial = 0, None, False

    def record_failure(self) -> None:
        with self._lock:
            self._trial = False
            self._failures += 1
            if self._opened_at is not None or self._failures >= self.failure_threshold:
                self._opened_at = self._clock()  # (re)open and restart the cool-down


class SingleFlight:
    """Coalesce concurrent identical work. The work runs as its own task, so one client
    disconnecting never cancels it for everyone else."""

    def __init__(self):
        self._tasks: dict[str, asyncio.Future] = {}

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def do(self, key: str, fn: Callable[[], Awaitable]):
        task = self._tasks.get(key)
        if task is None:
            task = asyncio.ensure_future(fn())
            self._tasks[key] = task
            task.add_done_callback(lambda t, k=key: self._finish(k, t))
        return await asyncio.shield(task)

    def _finish(self, key: str, task: asyncio.Future) -> None:
        self._tasks.pop(key, None)
        if not task.cancelled():
            task.exception()  # mark retrieved; awaiting requests still receive it


class Busy(Exception):
    """Raised when a Limiter could not admit the call within its queue timeout."""


class Limiter:
    def __init__(self, max_concurrent: int, queue_timeout: float = 2.0):
        self.max_concurrent, self.queue_timeout = max_concurrent, queue_timeout
        self._sem = asyncio.Semaphore(max_concurrent)
        self.active = 0

    async def run(self, fn: Callable, *args):
        """Run blocking `fn` in a worker thread, at most `max_concurrent` at once."""
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout)
        except asyncio.TimeoutError:
            raise Busy(f"{self.max_concurrent} calls already in flight")
        self.active += 1
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            self.active -= 1
            self._sem.release()

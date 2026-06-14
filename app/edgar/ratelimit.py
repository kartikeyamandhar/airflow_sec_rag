"""A simple thread-safe rate limiter.

EDGAR allows 10 requests/second per IP and temporarily blocks offenders. We keep
a minimum interval between requests, well under that ceiling. The clock and sleep
functions are injectable so the spacing logic can be tested without real time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Enforce a minimum interval between :meth:`acquire` calls."""

    def __init__(
        self,
        max_per_second: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError(f"max_per_second must be positive, got {max_per_second}")
        self._min_interval = 1.0 / max_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        """Block until the next request is allowed under the rate cap."""
        with self._lock:
            now = self._monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                self._sleep(wait)
                now = self._next_allowed
            self._next_allowed = now + self._min_interval

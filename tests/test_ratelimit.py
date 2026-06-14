"""Rate limiter spacing, tested with an injected fake clock (no real sleeping)."""

import pytest

from app.edgar.ratelimit import RateLimiter


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_requests() -> None:
    clock = _Clock()
    limiter = RateLimiter(10.0, monotonic=clock.monotonic, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    # First call is free; the next two each wait one min-interval (0.1s at 10/s).
    assert len(clock.sleeps) == 2
    assert all(abs(s - 0.1) < 1e-9 for s in clock.sleeps)


def test_rate_limiter_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0.0)

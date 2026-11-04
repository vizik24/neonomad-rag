"""In-memory rate limits for /ask: per client IP (per minute and per day), plus a daily cap on all questions.

The per-IP limits stop one person or script hammering the endpoint. The daily cap is what bounds the
Anthropic bill: a script rotating through many IPs gets past the per-IP limits but not the cap.

Each ECS task keeps its own counts, so N tasks allow N x these limits. That's the trade-off for needing no
shared store (Redis): size the daily cap with the task count in mind.
"""
import threading
import time
from collections import deque

SWEEP_EVERY_S = 3600  # drop the timestamps of IPs that haven't come back, so memory doesn't creep up


class SlidingWindow:
    """At most `limit` events per `window_s` seconds for each key."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window_s = limit, window_s
        self.events: dict[str, deque] = {}

    def _prune(self, key: str, now: float) -> deque | None:
        q = self.events.get(key)
        while q and q[0] <= now - self.window_s:
            q.popleft()
        if q is not None and not q:
            del self.events[key]
            return None
        return q

    def wait(self, key: str, now: float) -> float:
        """0 if another event is allowed now, else the seconds until one is."""
        q = self._prune(key, now)
        if q is None or len(q) < self.limit:
            return 0.0
        return q[0] + self.window_s - now

    def record(self, key: str, now: float) -> None:
        self.events.setdefault(key, deque()).append(now)

    def sweep(self, now: float) -> None:
        for key in list(self.events):
            self._prune(key, now)


class RateLimiter:
    def __init__(self, per_ip_minute: int, per_ip_day: int, daily_total: int):
        self.windows = [
            ("ip", SlidingWindow(per_ip_minute, 60)),
            ("ip", SlidingWindow(per_ip_day, 86400)),
            ("all", SlidingWindow(daily_total, 86400)),
        ]
        self.lock = threading.Lock()  # /ask runs in FastAPI's thread pool
        self.last_sweep = time.monotonic()

    def check(self, ip: str) -> tuple[float, str | None]:
        """Count a question from `ip` if every limit allows it: (0, None).

        Otherwise count nothing and return (seconds to wait, "ip" or "all"). Only allowed questions are
        recorded, so memory is bounded by the daily cap, however many requests are rejected.
        """
        now = time.monotonic()
        with self.lock:
            if now - self.last_sweep > SWEEP_EVERY_S:
                for _, window in self.windows:
                    window.sweep(now)
                self.last_sweep = now

            checks = [(window, ip if scope == "ip" else "*", scope) for scope, window in self.windows]
            blocked = [(window.wait(key, now), scope) for window, key, scope in checks]
            wait, scope = max(blocked)
            if wait > 0:
                return wait, scope
            for window, key, _ in checks:
                window.record(key, now)
            return 0.0, None

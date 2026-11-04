"""Rate limits for /ask, stored in Postgres: per client IP (per minute and per day), plus a daily cap on all
questions.

The per-IP limits stop one person or script hammering the endpoint. The daily cap is what bounds the
Anthropic bill: a script rotating through many IPs gets past the per-IP limits but not the cap.

In Postgres, not in memory, because on Lambda every concurrent request can be a separate, short-lived
instance: in-memory counts would reset on each cold start and never be shared. One small table, shared by
every instance, costs nothing extra on the database the app already uses.
"""
import hashlib

import psycopg

LOCK_ID = 7_291_401  # any constant: serialises checks so two instances can't both take the last slot

SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_limit_events (
    ip_hash text        NOT NULL,   -- sha256 of the IP: enough to count by, without storing the address
    at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rate_limit_events_at_idx ON rate_limit_events (at);
-- On Supabase, tables in public are reachable through its REST API with the site's public key. RLS with no
-- policies blocks that; the app connects as the table owner, which RLS doesn't apply to.
ALTER TABLE rate_limit_events ENABLE ROW LEVEL SECURITY;
"""

# One pass over the last day's events gives every count, and the oldest event in each window (which is when
# that window frees up a slot). Only allowed questions are recorded, so this is at most the daily cap's rows.
COUNTS_SQL = """
SELECT
    count(*) FILTER (WHERE ip_hash = %(ip)s AND at > now() - interval '1 minute'),
    extract(epoch FROM min(at) FILTER (WHERE ip_hash = %(ip)s AND at > now() - interval '1 minute')
                       + interval '1 minute' - now()),
    count(*) FILTER (WHERE ip_hash = %(ip)s),
    extract(epoch FROM min(at) FILTER (WHERE ip_hash = %(ip)s) + interval '1 day' - now()),
    count(*),
    extract(epoch FROM min(at) + interval '1 day' - now())
FROM rate_limit_events
WHERE at > now() - interval '1 day'
"""


def hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()


class RateLimiter:
    def __init__(self, per_ip_minute: int, per_ip_day: int, daily_total: int):
        self.limits = [("ip", per_ip_minute), ("ip", per_ip_day), ("all", daily_total)]

    def create_table(self, conn: psycopg.Connection) -> None:
        conn.execute(SCHEMA)

    def check(self, conn: psycopg.Connection, ip: str) -> tuple[float, str | None]:
        """Count a question from `ip` if every limit allows it: (0, None).

        Otherwise count nothing and return (seconds to wait, "ip" or "all").
        """
        ip_hash = hash_ip(ip)
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            row = conn.execute(COUNTS_SQL, {"ip": ip_hash}).fetchone()
            counts_and_waits = [(row[0], row[1]), (row[2], row[3]), (row[4], row[5])]
            blocked = [(float(wait or 0), scope)
                       for (scope, limit), (count, wait) in zip(self.limits, counts_and_waits) if count >= limit]
            if blocked:
                wait, scope = max(blocked)
                return max(wait, 1.0), scope
            conn.execute("INSERT INTO rate_limit_events (ip_hash) VALUES (%s)", (ip_hash,))
            conn.execute("DELETE FROM rate_limit_events WHERE at < now() - interval '1 day'")
        return 0.0, None

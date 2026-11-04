"""HTTP API for the support assistant: the production entry point, serving query.py's pipeline. Deployed on
AWS Lambda (Function URL) with the knowledge base and rate limits in Supabase Postgres.

    .venv/bin/uvicorn app:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Do you deliver to France?"}'

Runs query.py only. The self-check agent (agent.py) is deliberately not wired in: it costs 2-5x more per
question, and the production image doesn't contain it (see README, "Deployment").
"""
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager

import anthropic
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import query
from ingest import DATABASE_URL, EMBEDDING_MODEL
from ratelimit import RateLimiter

MAX_QUESTION_CHARS = 500  # a support question fits easily; caps the cost of any one request
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
UNAVAILABLE = f"The assistant is unavailable right now. Please email {query.SUPPORT_EMAIL}."

# Shared by every instance (stored in Postgres, see ratelimit.py). A question costs ~1-3p, so the daily cap
# bounds Anthropic spend at ~$3-9/day.
limiter = RateLimiter(
    per_ip_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "5")),
    per_ip_day=int(os.getenv("RATE_LIMIT_PER_DAY", "30")),
    daily_total=int(os.getenv("DAILY_QUESTION_LIMIT", "300")),
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("support")

resources: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loaded once per container, not per request: the embedding model takes a few seconds to load.
    resources["encoder"] = SentenceTransformer(EMBEDDING_MODEL)
    resources["client"] = anthropic.Anthropic(timeout=60, max_retries=2)
    resources["pool"] = ConnectionPool(
        DATABASE_URL, min_size=1, max_size=4, open=True,
        configure=register_vector,
        check=ConnectionPool.check_connection,  # drop connections the database closed (e.g. a restart)
        # Supabase's pooler in transaction mode can't keep prepared statements between transactions
        kwargs={"prepare_threshold": None},
    )
    try:
        with resources["pool"].connection(timeout=10) as conn:
            limiter.create_table(conn)
    except (psycopg.Error, PoolTimeout):
        log.exception("couldn't create the rate limit table; /ask will return 503 until the database is up")
    yield
    resources["pool"].close()


# No /docs or /openapi.json: the API is public, and nothing but the website needs its schema
app = FastAPI(title="Neo Nomad support assistant", lifespan=lifespan, docs_url=None, redoc_url=None,
              openapi_url=None)
if ALLOWED_ORIGINS:  # only needed when the website calls the API from the browser
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST"],
                       allow_headers=["content-type"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    id: str
    title: str
    heading: str
    url: str


class AskResponse(BaseModel):
    answerable: bool
    answer: str
    sources: list[Source]


def client_ip(request: Request) -> str:
    """The customer's address.

    - On Lambda, the Web Adapter passes the invocation's request context as a header, and its sourceIp is
      the address AWS saw, which the client can't fake.
    - Behind a load balancer (ECS), the ALB appends the real address to X-Forwarded-For: the LAST entry is
      the one it added (earlier entries are whatever the client sent, so they can be faked).
    - Otherwise (local dev), the socket's address.
    """
    context = request.headers.get("x-amzn-request-context")
    if context:
        try:
            return json.loads(context)["http"]["sourceIp"]
        except (ValueError, KeyError, TypeError):
            pass
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    try:
        with resources["pool"].connection() as conn:
            wait, scope = limiter.check(conn, client_ip(request))
    except (psycopg.Error, PoolTimeout):
        log.exception("rate limit check failed")
        raise HTTPException(503, UNAVAILABLE)  # fail closed: no check, no Claude call
    if not wait:
        return
    log.warning("rate limited scope=%s", scope)  # no IP: it's personal data, and the count is what matters
    minutes = math.ceil(wait / 60)
    detail = (f"You've asked a lot of questions in a short time. Please try again in {minutes} minute(s), "
              f"or email {query.SUPPORT_EMAIL}." if scope == "ip" else
              f"The assistant is very busy right now. Please try again later, or email {query.SUPPORT_EMAIL}.")
    raise HTTPException(429, detail, headers={"Retry-After": str(math.ceil(wait))})


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(rate_limit)])
def ask(request: AskRequest) -> AskResponse:
    """Sync on purpose: FastAPI runs it in a worker thread, and every step (model, DB, API) is blocking."""
    question = request.question.strip()
    if not question:
        raise HTTPException(422, "question is empty")

    started = time.perf_counter()
    client, calls = resources["client"], []
    try:
        queries = query.decompose(question, client, calls=calls)
        with resources["pool"].connection() as conn:
            chunks = query.retrieve_multi(queries, resources["encoder"], conn)
        result = query.generate(question, chunks, client, calls=calls)
    except (anthropic.APIError, anthropic.CredentialsError):
        log.exception("anthropic call failed")
        raise HTTPException(502, UNAVAILABLE)
    except (psycopg.OperationalError, PoolTimeout):
        log.exception("database unavailable")
        raise HTTPException(503, UNAVAILABLE)

    # Metadata only: customer questions can contain personal details, so they aren't logged.
    log.info(
        "answered answerable=%s queries=%d chunks=%d input_tokens=%d output_tokens=%d invalid_citations=%d "
        "latency_s=%.2f",
        result["answerable"], len(queries), len(chunks),
        sum(c["usage"]["input_tokens"] for c in calls), sum(c["usage"]["output_tokens"] for c in calls),
        len(result["invalid_citations"]), time.perf_counter() - started,
    )
    return AskResponse(answerable=result["answerable"], answer=result["answer"], sources=result["sources"])


@app.get("/ready")
def ready() -> dict:
    """The process is up and the model is loaded (lifespan has run). For the Lambda Web Adapter's readiness check."""
    return {"status": "ready"}


@app.get("/health")
def health() -> dict:
    """For the ECS/load balancer health check: the database is reachable and has been ingested."""
    try:
        with resources["pool"].connection(timeout=3) as conn:
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except (psycopg.Error, PoolTimeout):
        raise HTTPException(503, "database unavailable")
    if not chunks:
        raise HTTPException(503, "knowledge base is empty: run ingest.py")
    return {"status": "ok", "chunks": chunks}

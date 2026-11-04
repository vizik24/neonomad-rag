# Neo Nomad Support Assistant (RAG)

A customer support assistant for [Neo Nomad](https://neonomad.co.uk), my van conversion business. It answers questions about the products, installation, delivery, returns and DVLA motorhome rules from the company's own documents, with citations, and says so when it doesn't know.

**Results** (30-question eval, 3 runs each): **98.9% pass**, up from 91.1% at baseline. Multi-topic questions went from 60% to 100%, with **100% refusal accuracy** on unanswerable and prompt-injection questions. Deployed on AWS Lambda for close to zero running cost.

## How it works

```
INGESTION (offline)
  website HTML + PDFs ──► Markdown ──► heading-aware chunks ──► bge-small embeddings ──► Postgres + pgvector

QUERY
  question ──► Claude splits it into 1–3 search queries
           ──► hybrid search per query (vector + keyword, reciprocal rank fusion)
           ──► round-robin merge
           ──► Claude answers from those chunks only (structured output: answerable, answer, citations)
           ──► citations checked against what was retrieved
```

| Part | Choices |
|---|---|
| **Ingestion** | Markdown with front-matter metadata; chunks split on headings, each prefixed with its title and heading path; local `bge-small-en-v1.5` embeddings |
| **Retrieval** | Postgres + pgvector (HNSW) alongside full-text search (GIN), fused with weighted RRF; LLM query decomposition for questions spanning several documents |
| **Generation** | Claude with Pydantic structured output; grounding and prompt-injection rules in the system prompt; invalid citations dropped in code |
| **Agent** (optional) | LangGraph self-check loop: verify each claim against the sources, retry with feedback, hand off to a human if it still fails. Off in production for cost |
| **Evaluation** | 30 cases across 5 types × 3 runs; code checks plus an LLM judge (a different model from the app), validated with oracle, null and planted-error checks; paired comparison between versions |
| **Production** | FastAPI on AWS Lambda (container image, Function URL), Supabase Postgres, rate limiting in Postgres, no customer questions in logs |

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
docker compose up -d                 # Postgres 17 + pgvector on localhost:5434
.venv/bin/python ingest.py           # load -> chunk -> embed -> store

cp .env.example .env                 # add ANTHROPIC_API_KEY
.venv/bin/python query.py "Can I fit an e-bike in the BikerBed?"
.venv/bin/python query.py --retrieve-only --no-decompose "What size Allen key do I need?"   # no LLM calls
```

Other entry points:

```bash
.venv/bin/uvicorn app:app --port 8000                       # the API (POST /ask, GET /health)
.venv/bin/python eval/run_eval.py --reps 3                  # the eval
.venv/bin/pip install -r requirements-agent.txt             # the self-check agent
SELF_CHECK_ENABLED=true .venv/bin/python agent.py "Will a BikerBed let me register my van as a motorhome?"
```

## Evaluation

| Metric | Baseline | v1 (adopted) |
|---|---|---|
| Pass | 91.1% | **98.9%** |
| Multi-topic questions | 60% | **100%** |
| Retrieval hit | 89.9% | **100%** |
| Refuses when it should | 100% | **100%** |
| Grounded | 98.9% | 98.9% |

v1 is a one-prompt change to the query decomposition step, found by tracing the baseline failures back to retrieval. Cases are in `eval/cases.md`, and the per-question report is in `.claude/hillclimb/support_qa/report.html`.

**Limitations:** the questions are drafted rather than taken from real customers, 30 questions can't resolve small differences (the pass rate's confidence interval is about ±9 points), and the judge hasn't been calibrated against human grades.

## Deployment

One Docker image runs on AWS Lambda through the Lambda Web Adapter. The knowledge base and rate-limit counters live in the Supabase project the website already uses. `deploy/deploy_lambda.sh` builds, pushes and updates the function.

## Project layout

```
docs/                 source HTML/PDFs, converters, and the generated Markdown knowledge base
ingest.py             load -> chunk -> embed -> store
query.py              decompose -> hybrid retrieve -> generate -> check citations
agent.py              LangGraph version with a grounding check and retry loop (off by default)
app.py, ratelimit.py  FastAPI service and Postgres-backed rate limits
eval/                 test cases and the eval runner
deploy/               Lambda deploy script, ECS task definition template
Dockerfile            production image
```

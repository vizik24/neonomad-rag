# Production image: the FastAPI service (app.py) over query.py, plus ingest.py for one-off ingestion.
# Runs unchanged on AWS Lambda (via the Lambda Web Adapter below) or as a plain container (ECS, docker run).
# The self-check agent (agent.py, langgraph) is deliberately left out: see README, "Deployment".
FROM python:3.11-slim

# Lambda Web Adapter: a Lambda extension that turns invocations into HTTP requests to uvicorn, so the same
# FastAPI app serves Lambda. Outside Lambda it does nothing.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
# The adapter waits for this path before sending traffic. /ready, not /health: /health returns 503 until the
# database is ingested, which would stop the function ever starting.
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/ready

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build (GBs that Fargate can't use)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image: containers start without downloading from Hugging Face
COPY ingest.py .
RUN python -c "from ingest import EMBEDDING_MODEL; from sentence_transformers import SentenceTransformer; SentenceTransformer(EMBEDDING_MODEL)"
ENV HF_HUB_OFFLINE=1

COPY query.py app.py ratelimit.py ./
COPY docs/md docs/md

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# One worker: each loads the embedding model (~0.5 GB with torch); scale with more ECS tasks instead
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

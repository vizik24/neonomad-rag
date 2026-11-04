# Production image: the FastAPI service (app.py) over query.py, plus ingest.py for one-off ingestion tasks.
# The self-check agent (agent.py, langgraph) is deliberately left out: see README, "Deployment".
FROM python:3.11-slim

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

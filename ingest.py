"""Ingestion pipeline: load -> chunk -> embed -> store.

Reads the Markdown knowledge base in docs/md, splits it into heading-aware chunks,
embeds each chunk with a local sentence-transformers model and writes the result to
Postgres + pgvector. Run it again whenever the documents change: it rebuilds the table.

    docker compose up -d
    .venv/bin/python ingest.py
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs" / "md"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs locally on CPU
EMBEDDING_DIM = 384

MAX_WORDS = 250  # sections longer than this are split on paragraph boundaries
MIN_WORDS = 20   # sections shorter than this are merged into the next one (keeps one FAQ answer per chunk)


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    heading: str          # heading path, e.g. "BikerBed Installation > 7. Install the Drawer"
    text: str             # what is embedded and shown to the LLM
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 1. Load

def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split a '---' front matter block from the body. Returns (metadata, body)."""
    if not raw.startswith("---\n"):
        return {}, raw
    header, body = raw[4:].split("\n---\n", 1)
    metadata = {}
    for line in header.splitlines():
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Read every .md file into a dict of {id, text, metadata}."""
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        metadata, text = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append({"id": path.stem, "text": text, "metadata": metadata})
    return documents


# ---------------------------------------------------------------- 2. Chunk

HEADING = re.compile(r"^(#{1,6}) +(.+)$")


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Split Markdown into (heading path, body) sections at every heading."""
    sections, path, lines = [], [], []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append((list(path), body))
        lines.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1]       # drop deeper headings from the path
            path.extend([""] * (level - 1 - len(path)))  # pad if levels were skipped
            path.append(match.group(2).strip())
        else:
            lines.append(line)
    flush()
    return [([h for h in p if h], body) for p, body in sections]


def split_long(body: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split an oversized section on blank lines, overlapping one paragraph between pieces."""
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    pieces, current = [], []
    for para in paragraphs:
        if current and len(" ".join(current + [para]).split()) > max_words:
            pieces.append("\n\n".join(current))
            current = current[-1:]  # carry the last paragraph over for context
        current.append(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(doc: dict) -> list[Chunk]:
    """Turn one document into chunks, each prefixed with its title and heading path."""
    title = doc["metadata"].get("title", doc["id"])

    # Merge tiny sections into the next one so a heading + one line isn't its own chunk.
    # The merged section keeps the first heading path; later headings stay inline in the text.
    merged: list[tuple[list[str], str]] = []
    for path, body in split_sections(doc["text"]):
        if merged and len(merged[-1][1].split()) < MIN_WORDS:
            prev_path, prev_body = merged.pop()
            inline_heading = f"{path[-1]}\n\n" if path else ""
            path, body = prev_path, f"{prev_body}\n\n{inline_heading}{body}"
        merged.append((path, body))

    chunks = []
    for path, body in merged:
        heading = " > ".join(path) if path else title
        for piece in split_long(body):
            # Context header: the embedding and the LLM both see where this text came from
            text = f"{title}\n{heading}\n\n{piece}" if heading != title else f"{title}\n\n{piece}"
            chunks.append(Chunk(doc["id"], len(chunks), heading, text, doc["metadata"]))
    return chunks


# ---------------------------------------------------------------- 3. Embed

def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> list:
    """Embed chunk texts. Vectors are normalised, so cosine distance = 1 - dot product."""
    return model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=True)


# ---------------------------------------------------------------- 4. Store

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id          text PRIMARY KEY,               -- "<doc_id>:<chunk_index>", stable for citations
    doc_id      text NOT NULL,
    chunk_index int  NOT NULL,
    heading     text NOT NULL,
    title       text,
    source_url  text,
    doc_type    text,
    content     text NOT NULL,
    embedding   vector({EMBEDDING_DIM}) NOT NULL,
    -- Keyword search alongside vector search (hybrid retrieval), kept in sync automatically
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_doc_type_idx ON chunks (doc_type);

-- On Supabase, public tables are reachable through its REST API; RLS with no policies blocks that.
-- The app connects as the table owner, which RLS doesn't apply to.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
"""


def store_chunks(chunks: list[Chunk], embeddings, database_url: str = DATABASE_URL) -> None:
    """Rebuild the chunks table and insert every chunk with its embedding."""
    with psycopg.connect(database_url, prepare_threshold=None) as conn:  # works through Supabase's pooler
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (id, doc_id, chunk_index, heading, title, source_url, doc_type, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        f"{c.doc_id}:{c.chunk_index}", c.doc_id, c.chunk_index, c.heading,
                        c.metadata.get("title"), c.metadata.get("source_url"), c.metadata.get("doc_type"),
                        c.text, emb,
                    )
                    for c, emb in zip(chunks, embeddings)
                ],
            )


# ---------------------------------------------------------------- Pipeline

def main() -> None:
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
    sizes = [len(c.text.split()) for c in chunks]
    print(f"Created {len(chunks)} chunks (words: min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) // len(sizes)})")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)

    store_chunks(chunks, embeddings)
    print(f"Stored {len(chunks)} chunks in Postgres")


if __name__ == "__main__":
    main()

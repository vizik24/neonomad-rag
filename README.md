# Neo Nomad Support Assistant (RAG)

A retrieval-augmented customer support assistant for [Neo Nomad](https://neonomad.co.uk), my van conversion business. It answers customer questions about the BikerBed, kitchen and mattress (fit, installation, delivery, returns, warranty, DVLA motorhome rules) from the company's own documents, with citations.

This README covers the **ingestion pipeline** (how documents become searchable chunks in a vector database) and the **query pipeline** (how a question becomes a cited answer). Evaluation is documented as it's built.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
docker compose up -d              # Postgres 17 + pgvector on localhost:5434
.venv/bin/python ingest.py        # load -> chunk -> embed -> store

cp .env.example .env              # add ANTHROPIC_API_KEY (needed for answers, not for --retrieve-only)
.venv/bin/python query.py "Can I fit an e-bike in the BikerBed?"
.venv/bin/python query.py --retrieve-only --no-decompose "What size Allen key do I need?"   # free: no LLM calls
```

Optional, and off by default: the self-check agent (see [Self-check agent](#self-check-agent-agentpy)).

```bash
.venv/bin/pip install -r requirements-agent.txt
SELF_CHECK_ENABLED=true .venv/bin/python agent.py "Will fitting a BikerBed let me register my van as a motorhome?"
```

To regenerate the Markdown from source (only needed when the website or PDFs change):

```bash
.venv/bin/pip install -r requirements-docs.txt   # conversion libraries, kept out of the production image
cd docs
../.venv/bin/python convert_html.py
../.venv/bin/python convert_pdf.py
```

## Pipeline overview

```
docs/html/*.html ─┐  convert_html.py
                  ├──────────────────►  docs/md/*.md  ──► ingest.py ──► Postgres (pgvector)
docs/pdf/*.pdf  ──┘  convert_pdf.py     (13 docs)         load            chunks table
                                                           chunk           (88 rows)
                                                           embed
                                                           store
```

Ingestion runs offline, whenever documents change. The query pipeline (separate) embeds a question with the same model and searches this table.

## Corpus

| Source | Documents | Type |
|---|---|---|
| Website pages (saved HTML) | BikerBed, Kitchen, Mattress, FAQ, shipping, returns & warranty, T&Cs, privacy, about, contact, homepage | product, faq, policy, about, overview |
| BikerBed installation PDF | 11-step install guide | guide |
| DVLA guidance PDF (Open Government Licence) | Converting a vehicle into a motor caravan | regulation |

About 4,500 words in total. Small enough to inspect every chunk by hand, which is deliberate: at this size I can tell whether a retrieval failure is a chunking problem or an embedding problem.

## Design decisions

### 1. Convert everything to Markdown first, as a separate step

Raw HTML is mostly noise for retrieval: navigation, Tailwind classes, video tags, cross-sell cards and a trust footer repeated on every product page. Embedding that noise dilutes each chunk's meaning, and duplicated blocks compete with each other in search.

The converters (`docs/convert_html.py`, `docs/convert_pdf.py`) strip that noise and output Markdown because:
- **Headings survive.** They are the natural boundaries for chunking (see below).
- **It's human-readable**, so the knowledge base can be reviewed and corrected by hand before ingestion.
- **Conversion is decoupled from ingestion.** Fixing a PDF extraction glitch doesn't mean touching the embedding code, and vice versa.

Specific cleanup choices:
- FAQ questions are promoted to `##` headings so each Q&A becomes its own chunk.
- The FAQ block duplicated on the contact page is dropped (each `##` section is kept only in its canonical page).
- PDF print artefacts (browser headers/footers, page numbers, licence boilerplate) are removed, and URLs split across lines are rejoined so citations work.
- Known extraction errors are fixed with an explicit `FIXUPS` list rather than clever regexes. It's honest about being a manual correction and easy to audit.

### 2. Front matter carries metadata through to every chunk

Each Markdown file starts with:

```yaml
title: BikerBed
source_url: https://neonomad.co.uk/product?item=bikerbed
doc_type: product        # product | faq | policy | guide | regulation | about | overview | contact
```

Documents are loaded as `{id, text, metadata}` records, not concatenated into one string, so every chunk knows where it came from. This enables:
- **Citations:** answers can link to the exact page.
- **Filtering:** e.g. restrict a returns question to `doc_type = 'policy'`.
- **Trust signals:** the LLM can treat `regulation` content (DVLA) differently from marketing copy.

### 3. Chunk by heading structure, not fixed character windows

`chunk_document()` splits at every Markdown heading and tracks the heading path (e.g. `BikerBed Installation > Installation Steps > 7. Install the Drawer`). Fixed-size windows would cut install steps and FAQ answers in half. Splitting by heading keeps each chunk to one topic, which is what both the embedding and the LLM need.

Two size guards handle the edges:

| Rule | Value | Why |
|---|---|---|
| Merge sections under `MIN_WORDS` into the next one | 20 words | Avoids chunks that are just a heading and one line ("Version 8", "£1750.00"). |
| Split sections over `MAX_WORDS` on paragraph boundaries | 250 words | Only the longer DVLA sections hit this. Never splits mid-paragraph, and carries one paragraph of overlap so context isn't lost at the boundary. |

**Tuning note.** I first set `MIN_WORDS = 40`. That merged short FAQ answers together. "Can I fit an e-bike?" then retrieved a chunk headed *"Will my bikes get scratched?"* that happened to contain the e-bike answer further down. The embedding was diluted (similarity 0.675) and the chunk's label was misleading for citations. Dropping to 20 gave every FAQ its own chunk, and the same query now hits the exact answer at 0.756. Result: 88 chunks, mean 53 words.

### 4. Every chunk is prefixed with its title and heading path

```
BikerBed
BikerBed > Can I still use it if I have an e-bike instead of a regular mountain bike?

Yes. The bike rack is designed to hold either two standard bikes or one e-bike...
```

A bare answer like *"Yes. The bike rack is designed to hold..."* doesn't say which product it's about. Adding the context header puts the product name and the question into the embedding, which is what makes small chunks safe. It's a cheap version of "contextual retrieval".

### 5. Local embedding model: `BAAI/bge-small-en-v1.5`

- **Runs locally on CPU** (384 dimensions, about 130 MB). No API key, no per-call cost, and customer questions aren't sent to a third party just to be embedded.
- **Strong for its size** on retrieval benchmarks (MTEB). The corpus is small and in English, so a larger model isn't justified yet.
- **Vectors are normalised**, so cosine similarity equals the dot product and scores are comparable across queries.
- **Query side:** bge models expect queries to be prefixed with `"Represent this sentence for searching relevant passages: "`. Documents are embedded without a prefix. The query pipeline must do the same, and must use the same model. Changing the model means re-running ingestion.

The model is one constant (`EMBEDDING_MODEL`), so it's easy to compare against a hosted model (e.g. Voyage, OpenAI) in evaluation.

### 6. Postgres + pgvector, not a dedicated vector database

- **One database for vectors, text and metadata.** Filtering by `doc_type` is a plain `WHERE` clause, and everything lives in tools I already use in production (PostgreSQL/PostGIS).
- **Hybrid search is ready.** A generated `tsvector` column with a GIN index sits alongside the vector column, so keyword search (good for exact terms like "V5C", "4 mm Allen key", "T6") can be combined with semantic search later without re-ingesting.
- **HNSW index** (`vector_cosine_ops`) for approximate nearest-neighbour search. It's overkill at 88 rows, but it's the setup that scales.
- Runs in Docker (`pgvector/pgvector:pg17`) on port 5434, to avoid clashing with other local Postgres instances.

Schema:

| Column | Purpose |
|---|---|
| `id` | `"<doc_id>:<chunk_index>"`, a stable, readable ID used in citations |
| `doc_id`, `chunk_index`, `heading` | Where the chunk sits in its document |
| `title`, `source_url`, `doc_type` | Copied from front matter for citations and filtering |
| `content` | The text that was embedded, and that the LLM will read |
| `embedding` | `vector(384)` |
| `tsv` | Generated full-text index for keyword or hybrid search |

### 7. Full rebuild on every run

`ingest.py` drops and recreates the `chunks` table each time. With 13 documents and 88 chunks that takes seconds, and it guarantees the table exactly matches `docs/md` (no stale chunks from edited or deleted pages). Incremental upserts keyed on a content hash would be the next step if the corpus grew to thousands of documents.

## Sanity check

Top-ranked chunk for a few hand-written questions after ingestion:

| Question | Top chunk | Similarity |
|---|---|---|
| Can I fit an e-bike? | BikerBed > Can I still use it if I have an e-bike…? | 0.756 |
| What tools do I need to install it? | BikerBed Installation > Tools Required | 0.672 |
| How long do I have to return something? | Returns Policy > Eligibility for Returns | 0.831 |
| Will it make my van a motorhome on the V5? | DVLA > Evidence needed to support your application | 0.693 |
| Is a stove included with the kitchen? | Kitchen > What exactly comes with the kitchen? | 0.814 |
| Do you deliver to France? | Terms & Conditions > 3. Delivery & Shipping | 0.640 |

This is a spot check, not an evaluation. A labelled question set with retrieval hit rate is the next step.

## Query pipeline (`query.py`)

```
question ──► Claude: decompose into 1–3 search queries (one per topic)
                 │
                 ▼  for each query:
             embed (bge + query prefix) ──► vector top 20 ─┐
             Postgres full-text ──────────► keyword top 20 ┴─► weighted RRF ─► top 6 chunks
                 │
                 ▼
             round-robin merge across queries (dedupe)
                 │
                 ▼
             Claude (structured output: answerable, answer, citations)
                 │
                 ▼
             citation check: drop ids that weren't retrieved ─► answer + source links
```

### 8. Hybrid retrieval, fused by rank

Vector search handles paraphrase ("Can I fit an e-bike?" vs. "two standard bikes or one e-bike"). Keyword search handles exact tokens that embeddings blur, like "V5C", "T6" and "4 mm Allen key". Each retriever returns its top 20, and the lists are merged with **reciprocal rank fusion** (`score = Σ weight / (60 + rank)`). RRF uses ranks, not scores, so cosine similarity (0–1) and `ts_rank_cd` (unbounded) never need normalising against each other.

**Two tuning steps, both found by testing:**
1. `websearch_to_tsquery` ANDs every word, so "What size Allen key do I need?" matched nothing (no chunk contains "size"). I switched to OR-ing the terms.
2. OR-ing made keyword search noisy. Postgres ranking has no IDF, so "need" matched DVLA's "Evidence *needed*" as strongly as "Allen" matched the install guide. I gave keyword results **half weight** in the fusion. Vector search leads, and keyword search breaks ties and promotes exact-term matches. The right chunk now ranks first for every test question.

A proper BM25 implementation (e.g. `pg_search` / ParadeDB) would weight rare terms correctly. It's worth trying if the evaluation shows keyword-dependent misses.

### 9. Structured output instead of free text

Claude returns a Pydantic-validated object via `client.messages.parse(output_format=Answer)`:

```python
class Answer(BaseModel):
    answerable: bool        # do the sources actually contain the answer?
    answer: str
    citations: list[str]    # chunk ids, e.g. "bikerbed:11"
```

- **`answerable` makes refusals measurable.** The evaluation can count how often the model correctly says "I don't know" on out-of-scope questions (e.g. Renault Trafic fit), rather than parsing prose.
- **Citations are chunk ids, then checked in code.** Any cited id that wasn't in the retrieved set is a hallucinated citation. It's dropped from the answer and reported separately (`invalid_citations`), so it's both corrected and countable.

### 10. Grounding rules in the system prompt

- Answer **only** from the provided sources. If they don't contain the answer, say so and point to support@neonomad.co.uk. A confident wrong answer about van fit costs a customer £1,750.
- **Sources are wrapped in `<source>` tags and declared to be data, not instructions.** That's the basic prompt-injection defence: a document saying "ignore previous instructions" shouldn't change behaviour.
- **DVLA questions:** explain the guidance, never promise the outcome. The DVLA decides.
- Each source tag carries its `doc_type`, so the model can tell regulation from marketing copy.

### 11. Model and settings

- **`claude-opus-5-5` at `effort: "low"`** for both decomposition and answering. Grounded Q&A over a handful of short chunks doesn't need deep reasoning, and low effort keeps latency and cost down for a chat-style workload. If the evaluation shows quality holding, a cheaper model (Sonnet or Haiku) is the next lever to test. That's a measured decision, not a default.
- **Server-side refusal fallback** (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). If a safety classifier declines a request, the API retries it on a suitable model instead of returning nothing. If the whole chain still refuses, the code returns a polite hand-off to the support email.
- The retrieval step is exposed separately (`--retrieve-only --no-decompose`) so retrieval can be debugged and evaluated without paying for LLM calls.

### 12. Query decomposition for multi-topic questions

**The problem.** The first smoke test failed on *"Will fitting a BikerBed let me register my van as a motorhome?"* This is a **multi-hop** question: the answer needs BikerBed facts (how it's fixed in the van) *and* DVLA rules (what counts as a motor caravan). But "BikerBed" dominates the question's embedding, so all top 8 chunks were product pages and no DVLA chunk was retrieved:

```
bikerbed:12              sim 0.765  BikerBed > Does it work with all types of vans...?
bikerbed:7               sim 0.714  BikerBed > How much space does the BikerBed take up...?
BikerBedInstructions:7   sim 0.709  BikerBed Installation > 9. Secure the BikerBed to the Van
...                                 (8 of 8 chunks about the product, 0 from the DVLA guidance)
```

The model handled the gap safely: `answerable: false`, no promised outcome. It was still a wrong answer, because the knowledge base *does* contain the answer. Tuning chunk size, `k` or the fusion weights can't fix this, because the issue is that **one query vector can only point at one topic**.

**The fix has three parts:**

1. **Decompose (`decompose()`).** Before retrieval, Claude rewrites the question into 1–3 search queries, one per topic, each phrased the way the relevant document talks. The prompt describes what the knowledge base contains so the queries match it. The output is a validated `SearchPlan` (structured output, same pattern as the answer). Simple questions come back as a single query, so they're unaffected. The motorhome question becomes:
   ```
   - BikerBed features and installation in van (bed, fixed fittings)
   - DVLA requirements to register a van as a motor caravan (motorhome) conversion
   ```

2. **Retrieve per query, merge round-robin (`retrieve_multi()`).** Each sub-query gets its own hybrid search, then the result lists are interleaved: rank 1 from each query, then rank 2 from each, and so on, with duplicates removed. I chose round-robin over pooling everything into one RRF ranking deliberately. Pooled ranking would let the dominant topic crowd out the minority one again. Round-robin guarantees every sub-question's best chunks a place in the context.

3. **Allow combining sources in the answer prompt.** The original rules ("answer only from the sources") were correct but made the model reluctant to *combine* facts. The answer here isn't stated anywhere directly. It follows from two sources together: DVLA says equipment must be "rigidly fixed", and the install guide says the BikerBed is held by a ratchet strap. The prompt now says to combine sources, draw the conclusion that follows, cite each one, and say which parts depend on the customer's van.

**A second bug found by re-running.** After the fix, the answer was right on the first run but missed the key "rigidly fixed" rule on the second. Checking retrieval showed why: for the DVLA sub-query, the "internal features" chunk (the one with "rigidly fixed") ranks **6th**. Two limits were cutting it off: `TOP_K = 5` per query, and an overall cap of 8 chunks after merging, which also dropped the 5th DVLA chunk. Chunks average about 53 words, so the cap was saving almost nothing. I raised `TOP_K` to 6 and removed the overall cap (3 sub-queries × 6 chunks is still under 1,000 words of context). After that, **3 of 3 runs** retrieved the internal-features chunk and reached the same conclusion.

**Lesson: run the same question more than once.** Decomposition is an LLM step, so the sub-queries vary between runs (two or three queries, different wording). A pipeline that passes once can still be fragile. The evaluation needs repeated runs per question, not single passes.

**Cost.** Decomposition adds one small LLM call per question (low effort, short output) and up to 3× the retrieval queries, which are milliseconds on local Postgres. `--no-decompose` turns it off for free retrieval debugging.

**The answer now** (abridged, from `smoke_test.log`):

> Not on its own, I'm afraid. The DVLA only considers changing a van's body type to 'motor caravan' if it has a set of internal features (seats and a table, sleeping accommodation, cooking facilities and storage, **all rigidly fixed** to the living area…) and external features such as two or more windows…, a separate door…, and a high-top or pop-top roof. The BikerBed gives you the bed, but it's designed to be installed or removed in minutes with a **ratchet tie-down system**… That makes it unlikely to count as 'permanently' fitted… The DVLA makes the final decision and may inspect the van.

It cites both DVLA sections and the install guide. It answers the actual question, explains *why*, and doesn't promise an outcome.

**Why this is in plain Python, not LangGraph.** This is a fixed pipeline: decompose → retrieve → answer, always in that order. Nothing loops or branches, so a graph framework would add nothing. LangGraph becomes worth it with a **self-check loop**, which needs a conditional edge back to an earlier node. That's built separately in `agent.py` (see [Self-check agent](#self-check-agent-agentpy)).

### End-to-end smoke test

`scripts/smoke_test.sh` runs six questions through the full pipeline (output saved to `smoke_test.log`):

| Question | Before decomposition | After |
|---|---|---|
| Can I fit an e-bike in the BikerBed? | ✅ answerable | ✅ answerable ("two standard bikes or one e-bike") |
| Does the BikerBed fit a Renault Trafic? | ✅ correctly **not** answerable | ✅ correctly **not** answerable, sends the customer to support with their van details |
| What size Allen key do I need? | ✅ 4 mm | ✅ 4 mm, finger tight, no impact driver |
| Is a stove included with the kitchen? | ✅ No | ✅ No, lists what's included |
| Will fitting a BikerBed let me register my van as a motorhome? | ❌ retrieval miss, `answerable: false` | ✅ **"Not on its own"**: rigid-fixing rule plus strap mounting, DVLA decides (3/3 runs) |
| Do you deliver to France? | ✅ UK only | ✅ UK only, international by arrangement |

No hallucinated citation ids in any run. Simple questions decompose to a single query, so their retrieval is unchanged.

**Still open: over-citation.** Some answers cite loosely related chunks alongside the right one (e.g. "Shipping Costs" on the France answer). The evaluation should measure citation precision as well as recall.

## Self-check agent (`agent.py`)

`query.py` checks that cited ids were actually retrieved, but nothing checks that the answer's *claims* are in those sources. That's only measured afterwards, by the eval's judge. `agent.py` adds the check live, before the customer sees the answer:

```
decompose -> retrieve -> generate -> check --(all claims supported)--> answer
                            ^          |
                            +--retry---+   draft again, told which claims were unsupported
                                       |
                                       +--(still unsupported after 2 drafts)--> hand off to support
```

`.venv/bin/python agent.py --graph` prints the same graph as Mermaid, generated by LangGraph from the code.

**Disabled in production.** The check makes each question 2–5× more expensive (see Cost below), so production runs `query.py` and the agent is opt-in:

- **Flag:** `agent.py` refuses to answer, and `build_graph()` raises, unless `SELF_CHECK_ENABLED=true`. Any future entry point that imports it can't turn it on by accident.
- **Dependency:** `langgraph` lives in `requirements-agent.txt`, not `requirements.txt`. A production image built from `requirements.txt` doesn't contain it at all.

Turning it on later is one environment variable and one extra install, with no code change.

### 13. A LangGraph loop with a grounding check

- **Reuses the pipeline.** Each node calls `query.py`'s own functions. The graph only adds the `check` node, the retry edge and the handoff. `generate()` gained one optional argument, `feedback`, so the retry can say which claims to leave out. Without it, `query.py` behaves exactly as before.
- **The check.** A separate Claude call sees the sources, the question and the draft. It lists every factual claim and marks each one supported or not (structured output, like the eval judge). It uses the same rules the judge was tuned to: inferences that follow from a source count as supported, widening a policy doesn't, and pointing to support is always fine. Its examples are deliberately *not* taken from the eval cases, so the loop isn't tuned to the test.
- **Why a graph here.** This is the first part of the system where the next step depends on a result: done, retry or hand off. That's a conditional edge, and LangGraph makes it explicit and inspectable instead of a `while` loop buried in a function.
- **Two drafts, then hand off.** One retry fixes most problems (below). If the second draft still fails, the agent replaces it with a handoff to `support@neonomad.co.uk`. A wrong answer about fit or the DVLA costs the customer money, so "please ask the team" is the safer failure.
- **What gets skipped.** Answers that cite nothing ("I don't have that, please email support") make no claims, so they skip the check. Refusals that *do* cite sources still get checked. The first version skipped every refusal, and a test showed a Trafic refusal stating facts (VW Transporter, 1950mm) that went unverified.

**What it caught.** On the motorhome question, draft 1 said *"You'd likely need something like our kitchen unit as well."* No source says that, and it's exactly the kind of helpful-sounding overreach the eval found on p06. The check flagged it, and draft 2 left it out and passed.

**Cost.** Opus 5.5 pricing; tokens from the runs above.

| Question | LLM calls | Cost | Plain pipeline |
|---|---|---|---|
| Do your prices include VAT? (passed first time) | 3 | ~$0.025 | ~$0.014 |
| Discount code / Renault Trafic (refusals with citations) | 3 | ~$0.02–0.04 | ~$0.01 |
| Motorhome (one retry) | 5 | ~$0.15 | ~$0.03 |

The check roughly doubles the cost of a normal question, and a retry costs about 5× more, mostly in the check's output on long multi-source answers.

**Not yet evaluated.** The eval measures `query.py`, and that's what the 98.9% refers to. Pointing it at `agent.py` means changing the eval runner, which is locked during a hillclimb, so that's a separate, approved change. What the eval would show: whether the loop fixes slips like p06 without making correct answers worse, and whether it hands off answers that were actually fine. The five test runs here show it works; they don't show it's better.

## Evaluation (`eval/`)

The smoke test is six questions run once. The eval is the version you can make decisions with: 30 questions, each run 3 times, graded automatically, with error bars.

```bash
.venv/bin/python eval/run_eval.py --reps 3          # full run -> .claude/hillclimb/support_qa/baseline/
.venv/bin/python eval/run_eval.py --summary         # re-print the summary
.venv/bin/python eval/run_eval.py --check oracle    # grader sanity check (oracle | null | wrong)
```

### Baseline result

**30 questions × 3 runs = 90 answers. $2.31, 3.4 minutes, 0 errors.**

| Metric | Score (95% CI) | What it means |
|---|---|---|
| **Pass** (headline) | **91.1% ± 8.8** | Every check below passed |
| Answers when it should | 94.2% | Answerable questions not wrongly refused |
| Refuses when it should | **100%** | Unanswerable, injection and off-topic questions all refused, in all 21 runs |
| Key facts present | 95.1% ± 5.2 | |
| No must-not violations | **100%** | Never promised a DVLA outcome, invented a discount, or said it fits a Trafic |
| Grounded | 98.9% ± 2.2 | One unsupported material claim in 90 answers |
| Citations valid | **100%** | No made-up chunk ids |
| Retrieval hit | 89.9% ± 10.4 | Expected sources retrieved (answerable questions only) |

By question type: policy 100%, unanswerable 100%, adversarial 100%, direct 93.9%, **multi-topic 60%**. Median latency 4.6 s, $0.026 per answer including the judge.

**The safety behaviour is solid. The weak spot is retrieval on multi-topic questions.**

### What the failures show: brand names in sub-queries

7 of the 8 failing runs (across d06, m01, m02 and m05) trace back to **retrieval**, not generation. In most of them the model then did the safe thing and said it didn't have the information. The 8th (m01, run 3) retrieved the right chunks but left out the strap-mounting point. Reading the decomposed queries showed the pattern: the decomposition step tends to put the **brand name** into sub-queries ("**Neo Nomad** mattress piece sizes", "**BikerBed** warranty coverage"). Every page mentions the brand, so those words pull the embedding toward generic "about us" and product chunks and away from the specific one. I confirmed it with retrieval-only runs (free, no LLM):

| Sub-query | Top result |
|---|---|
| "**Neo Nomad** mattress piece sizes dimensions" | ❌ FAQ: *Are Neo Nomad beds made in-house?* |
| "mattress piece sizes dimensions" | ✅ **Mattress** (the right chunk) |
| "**Neo Nomad** kitchen features cooker hob sink installation" | ❌ FAQ: *made in-house?* |
| "kitchen features cooker hob sink installation" | ✅ **Kitchen > What exactly comes with the kitchen?** |
| "**BikerBed** warranty coverage for damage caused during self-installation" | ❌ Install step 11 |
| "warranty coverage for damage caused during installation" | ✅ **2-Year Warranty > What's Not Covered** |

This is the same mechanism as the original motorhome bug (one dominant term steers the whole query vector), now one level down, inside the decomposition step. It's also **intermittent**: d06 passed in 1 of 3 runs, depending on whether the decomposer happened to add "Neo Nomad". A single-run test would have called it fixed or broken by chance. That's why the eval runs everything 3 times.

### v1: brand-free sub-queries (adopted)

**Change** (`DECOMPOSE_PROMPT` in `query.py`, recorded in `.claude/hillclimb/support_qa/v1/change.md` and `change.patch`): two new rules. Leave out "Neo Nomad", and for policy topics name the policy ("warranty cover for damage during installation"), not the product ("BikerBed warranty").

**Cheap check first.** Before paying for a full run, I ran retrieval only on the failing questions. The new sub-queries ("warranty cover for damage caused during self-installation", "kitchen features cooking hob installation") put the right chunk first in each case.

**Result: 30 questions × 3 runs, paired against the baseline** (same questions, v1 minus baseline):

| Metric | Baseline | v1 | Paired difference (95% CI) |
|---|---|---|---|
| **Pass** | 91.1% | **98.9%** | +7.8 ± 9.2 points |
| Retrieval hit | 89.9% | **100%** | +10.1 ± 10.4 points |
| Answerable flag correct | 95.6% | **100%** | +4.4 ± 5.2 points |
| Key facts | 95.1% | **100%** | +4.9 ± 5.2 points |
| Grounded | 98.9% | 98.9% | ±0 |
| Refuses when it should | 100% | **100%** | no regression (21/21 runs) |

By type: **multi-topic 60% → 100%**, direct 93.9% → 100%, unanswerable and adversarial unchanged at 100%, policy 100% → 94.4% (see below). Cost ($2.31 → $2.36) and median latency (4.6 s → 4.7 s) were effectively unchanged.

- **Fixed (4 questions):** m05 kitchen + DVLA (0/3 → 3/3 runs), d06 mattress sizes (1/3 → 3/3), m01 motorhome (1/3 → 3/3), m02 install-damage warranty (2/3 → 3/3). All four were brand-name retrieval misses in the baseline.
- **New failure (1 question, 1 run):** p06 VAT. The answer added "the price you see on the site is the price you pay", which overstates the policy (shipping is extra, and pricing errors can be corrected). That's a generation overreach. Retrieval for p06 was correct in every run, so it isn't caused by this change. The judge was right to fail it, and it's the next thing to look at.
- **No regressions where a prompt change was most likely to cause one:** all unanswerable, injection and off-topic questions still refuse in every run.

**Reading it honestly.** Every targeted failure was fixed, nothing it could plausibly have broken got worse, and retrieval reached 100%. Strictly, the paired confidence interval on the headline pass rate (+7.8 ± 9.2) still just includes zero. With 30 questions, a change affecting 5 of them sits at the edge of what this eval can resolve. The per-question picture (4 fixed with a confirmed mechanism, 1 unrelated slip) is the stronger evidence. **v1 is adopted.** It's live in `query.py`.

**The credit outage, as a test of the runner.** The API account ran out of credit partway through the first v1 run. The runner logged every failed attempt to `errors.jsonl` as `api_error` (`credit balance is too low`), scored none of them as wrong answers, and on a later re-run resumed exactly the 27 missing attempts while keeping the 63 completed ones. Those error rows stay in the log, but each has been superseded by a successful result. In the end, every one of the 90 v1 results comes from a clean run.

### How the eval is built

**Questions (`eval/cases.jsonl`, reviewable as `eval/cases.md`).** 30 questions in five types: 11 direct, 6 policy, 5 multi-topic, 5 unanswerable, 3 adversarial (prompt injection, false premise, off-topic). They were drafted from the knowledge base and approved before running. Each case has:
- the expected `answerable` flag,
- **sources** that must be retrieved, matched by *document + a text snippet* rather than chunk id, so the cases survive re-chunking. A source can list alternatives when the same fact appears on several pages (prices are on the homepage too),
- **key facts** the answer must state, and **must-nots** that fail it outright.

**Grading: free code checks plus an AI judge.**

| Check | How | Why separate |
|---|---|---|
| `answerable_ok` | code | Refusal accuracy, reported both ways (answers when it should / refuses when it should). A system that always refuses would ace the unanswerable cases, and this catches that |
| `retrieval_hit` | code | Answerable questions only. **Not** part of pass: a correct answer from a different chunk shouldn't fail, but a low score says *where* to look |
| `citations_ok` | code | Any cited chunk id that wasn't retrieved |
| `facts`, `no_violation`, `grounded` | Claude Sonnet 5.5 judge, structured output | Text correctness needs judgement. The judge is a **different model** from the app (Opus 5.5) to avoid self-preference |

**Runner (`eval/run_eval.py`).**
- Calls the app's real `decompose → retrieve_multi → generate` functions. It doesn't re-implement them.
- Records the actual serving model and usage for every call, and fails loudly if the served model isn't the one requested.
- Writes each result as it completes, and resumes from where it stopped.
- Uses 4 concurrent workers, a 240 s hard limit per question, and the SDK's jittered retries.
- API errors, timeouts and grader crashes go to `errors.jsonl` and are **never scored as wrong answers**.
- **Harness gate:** a SHA of the runner and the questions must match an approval the owner records with `--approve-harness`. Questions or grading can't be changed silently between runs. Only `query.py`, the thing being improved, can change freely.

### Validating the grader (the part that took longest)

The first pilot (5 questions) scored 2/5, and **all 3 failures were grader errors**, not bad answers. It took three rounds to fix:

1. The judge flagged `support@neonomad.co.uk` as "unsupported". It didn't know the app is instructed to give that address. → Told the judge.
2. It flagged sound inferences ("no alterations" + "removed in 30 minutes" → "not permanently fixed"). → Added an explicit example of an allowed inference, and raised judge effort.
3. Its list of "unsupported claims" included claims it then called supported. **Any free-text entry counted as a failure.** → Changed the format: each claim gets `supported` and `material` flags, and only *unsupported + material* claims fail. Re-grading the same 10 answers turned all 3 false failures into passes with no other changes.

Two of the pilot's "retrieval misses" were **my case errors**: the right fact came from a page I hadn't listed (prices on the homepage). I then checked every case programmatically for facts appearing in more than one document and added alternatives.

**Pre-flight checks before the paid run:**

| Check | Expected | Result |
|---|---|---|
| Oracle: perfect answers built from the key facts | ~100% | 100% (30/30) |
| Null: empty answers | ~0% | 0% |
| Wrong: a confident answer to a different question | ~0% | 0% |
| Planted errors: lifetime warranty, 30-day returns, promised DVLA reclassification, "fits a Trafic", shipping refunded for transit damage | all fail | 5/5 fail |
| Induced API error (invalid key) | logged, not scored | `errors.jsonl`, `api_error` |

**A judgement call recorded in the cases.** p05 ("will you refund the shipping?") answered that shipping is refunded "if the kitchen arrived faulty **or damaged**". The policy only covers "a fault on our part", and transit damage isn't necessarily that. I decided this should fail, and wrote it into the case as a must-not rather than relying on the judge to generalise. "Arrived faulty" still passes.

### Limitations

- **Drafted, not real, questions.** All 30 were written from the docs, so they skew toward what the docs cover. Real customer emails would be better inputs.
- **Resolution.** With 30 questions × 3 runs, the pass-rate confidence interval is about ±9 points. That's enough to catch a regression or a big multi-topic improvement, but not a 2–3 point change.
- **Judge not yet calibrated against human labels.** The checks above test known-good and known-bad answers. A proper calibration would compare the judge with independent human grades on ~30 answers.
- **Cost of the judge.** 45% of the $0.026 per answer ($1.03 of $2.31) is the Sonnet judge.

## Deployment (AWS ECS)

`app.py` puts `query.py`'s pipeline behind a small FastAPI service, and the `Dockerfile` packages it for ECS on Fargate.

```
neonomad.co.uk ──HTTPS──► load balancer ──► ECS Fargate task (app.py) ──► RDS Postgres + pgvector
                                                 │
                                                 ├──► Anthropic API
                                                 └──► CloudWatch Logs     (secrets from Secrets Manager)
```

**API**

```bash
curl -s https://<host>/ask -H 'content-type: application/json' -d '{"question": "Do you deliver to France?"}'
# {"answerable": true, "answer": "...", "sources": [{"id": "faq:1", "title": "...", "heading": "...", "url": "..."}]}
curl -s https://<host>/health     # {"status": "ok", "chunks": 88}, or 503 if the database is down or empty
```

Run it locally with `.venv/bin/uvicorn app:app --port 8000`.

### 14. Production service and image

- **Plain pipeline only.** The service calls `query.py`. `agent.py` isn't copied into the image and `langgraph` isn't installed, so the 2–5× self-check cost can't reach production by accident. The conversion libraries (pymupdf etc.) are left out the same way, in `requirements-docs.txt`.
- **One image, two jobs.** The image holds `ingest.py` and `docs/md` too, so filling the production database is a one-off ECS task with the command overridden (below). There's no separate ingestion setup.
- **Model baked in.** The Dockerfile downloads bge-small at build time and sets `HF_HUB_OFFLINE=1`. Containers start without reaching Hugging Face, and every task runs exactly the same model as ingestion.
- **CPU-only torch.** The default torch wheel bundles CUDA, several GB that Fargate can't use. Even so, the image is 2.2 GB, ~700 MB of it torch. Moving the query embedding to ONNX would remove torch entirely; that's the next size win if cold starts matter.
- **ARM64 (Graviton).** It's about 20% cheaper on Fargate than x86, and a Mac builds it natively. Building on x86 CI needs `docker buildx build --platform linux/arm64`.
- **One worker per task.** Each worker loads its own copy of the model, so the service scales by adding ECS tasks, not workers. The endpoint is a sync function: FastAPI runs it in a thread, which suits the blocking model, database and API calls.
- **Connection pool** (`psycopg-pool`, 1–4 connections) checks each connection before use, so an RDS restart or failover doesn't leave dead connections behind.
- **Health check means "can answer".** `/health` checks the database is reachable *and* the chunks table isn't empty. A task pointed at an un-ingested database fails its health check instead of telling every customer "I don't know".
- **Logs hold metadata, not questions.** Each answer logs answerable, query and chunk counts, tokens, invalid citations and latency. Customer questions can contain names, addresses or order details, so the text isn't logged.
- **Cost and abuse limits in code.** Questions are capped at 500 characters. CORS allows only the origins in `ALLOWED_ORIGINS`. Anthropic and database failures return a 502/503 that points to the support email, not a stack trace.
- **Rate limiting in the app (`ratelimit.py`), free, no extra infrastructure.** Three sliding-window limits, checked before any Claude call, so a rejected request costs nothing:

  | Limit | Default | Env var | Stops |
  |---|---|---|---|
  | Per IP, per minute | 5 | `RATE_LIMIT_PER_MINUTE` | One script hammering the endpoint |
  | Per IP, per day | 30 | `RATE_LIMIT_PER_DAY` | One person or script grinding slowly all day |
  | All questions, per day | 300 | `DAILY_QUESTION_LIMIT` | A script rotating through many IPs: **this is what caps the bill**, at roughly $3–9/day per task |

  Over a limit → `429` with `Retry-After` and a message pointing to support. The client IP is the **last** `X-Forwarded-For` entry, the one the load balancer appends. Earlier entries come from the client and can be faked, so tasks must accept traffic only from the load balancer's security group. Limitations: counts are kept **per task** in memory (2 tasks = 2× the limits) and reset when a task restarts. Sharing them across tasks would need Redis, and AWS WAF is the stronger option if traffic grows. Rejected IPs aren't logged, only the count.

### Deploying

Placeholders are in `<ANGLE_BRACKETS>`. The task definition template is `deploy/ecs-task-definition.json`.

1. **Database:** create an RDS for PostgreSQL 17 instance (pgvector is supported; `ingest.py` runs `CREATE EXTENSION vector` itself). Put it in the same VPC, with a security group that allows port 5432 from the ECS tasks' security group only.
2. **Secrets:** in Secrets Manager, store `ANTHROPIC_API_KEY` and `DATABASE_URL` (`postgresql://<user>:<password>@<rds-endpoint>:5432/<db>?sslmode=require`). The task execution role needs `secretsmanager:GetSecretValue` on both, plus `logs:CreateLogGroup` for the log group.
3. **Image:**
   ```bash
   aws ecr create-repository --repository-name neonomad-support
   aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
   docker build -t <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/neonomad-support:<TAG> .
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/neonomad-support:<TAG>
   ```
4. **Task definition:** fill in the placeholders, then `aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json`.
5. **Ingest:** run the same task once with the command overridden (and again whenever `docs/md` changes, after rebuilding the image):
   ```bash
   aws ecs run-task --cluster <CLUSTER> --launch-type FARGATE --task-definition neonomad-support \
     --network-configuration 'awsvpcConfiguration={subnets=[<SUBNET>],securityGroups=[<TASK_SG>]}' \
     --overrides '{"containerOverrides":[{"name":"support-api","command":["python","ingest.py"]}]}'
   ```
6. **Service:** create an ECS service behind an Application Load Balancer with HTTPS, target port 8000 and health check path `/health`.

**Before it goes on the live site** (not built yet):
- **A spend limit in the Anthropic console.** In-app rate limiting caps normal abuse, but a monthly cap is the backstop for everything else (e.g. more tasks than planned).
- **The chat widget** on neonomad.co.uk that calls `/ask`.
- **AI disclosure** in that widget: tell customers they're talking to an automated assistant.

**Tested locally:** the rate limiter (each limit, IP handling and memory, with a fake clock), a real `429` through the API with no Claude call behind it, the API (health, validation, one answer) and the built image (health, an answer, a 503 with a wrong database password, the Docker health check reporting healthy, langgraph absent, running as a non-root user, and `ingest.py` run from the image). Nothing has been deployed to AWS.

## Known gaps in the knowledge base

- **Van compatibility list.** The website's "Will it fit?" search is a JavaScript widget, so the vehicle list isn't in the saved HTML. "Does it fit a Renault Trafic?" can't be answered yet, so the assistant should say so rather than guess.
- **Diagrams.** Install step 2 says "orientation matches the diagram provided", but the diagram isn't text.
- **Table product.** It's linked from the site, but the page hasn't been saved.

## Project layout

```
docs/
  html/              saved website pages (source)
  pdf/               install guide + DVLA guidance (source)
  md/                cleaned Markdown knowledge base (generated, reviewable)
  convert_html.py
  convert_pdf.py
ingest.py            load -> chunk -> embed -> store
query.py             decompose -> hybrid retrieve per query -> generate -> check citations
agent.py             the same pipeline as a LangGraph graph, plus a grounding check and retry loop (off by default)
scripts/smoke_test.sh  six end-to-end test questions -> smoke_test.log
eval/
  cases.jsonl        30 approved test questions (cases.md: readable version)
  run_eval.py        runner + grader (code checks + Sonnet 5.5 judge)
.claude/hillclimb/support_qa/
  baseline/          results.jsonl, traces/, errors.jsonl for the baseline run
  v1/                same for v1, plus change.md, change.patch and query.before.py
  report.html        per-case report (each row links to its transcript)
  _checks/           pilot runs and grader sanity checks (not part of results)
app.py               FastAPI service over query.py: POST /ask, GET /health (the production entry point)
ratelimit.py         in-memory rate limits for /ask: per IP per minute/day, plus a daily cap on all questions
Dockerfile           production image: app.py + query.py + ingest.py + docs/md, embedding model baked in
deploy/
  ecs-task-definition.json  Fargate task definition template (placeholders in <ANGLE_BRACKETS>)
docker-compose.yml   Postgres 17 + pgvector (local development)
requirements.txt     production dependencies
requirements-agent.txt  adds langgraph for agent.py (not installed in production)
requirements-docs.txt   conversion libraries for docs/convert_*.py (not installed in production)
```

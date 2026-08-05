# CLAUDE.md

Project spec for Vidhi, a retrieval augmented generation assistant. Read this before
making changes. Where it says MUST or NEVER, there is no discretion.

---

## 1. What this project is

Vidhi answers questions over a single large PDF, "Bharat's GST Smart Guide, 3rd
edition" (roughly 1500 pages), using retrieval augmented generation. Every answer is
grounded in retrieved excerpts from the manual and cites page numbers rather than
relying on the model's own knowledge. Each question is answered independently: there
is no conversation memory or chat history.

The project also includes a full evaluation harness that compares four increasingly
sophisticated pipeline configurations, and a DSPy-based prompt optimizer (MIPROv2),
so the value of hybrid retrieval and prompt optimization can each be measured
separately rather than assumed.

Live at https://vidhi.readiq.app.

## 2. Hard rules

MUST:
- Python 3.11 or later. Type hints on every function signature.
- Raw SQL via `asyncpg`. A thin repository module wraps queries.
- All external calls (LLM, embeddings, reranker, database) go through a retry wrapper.
- Every module is independently runnable and testable from the command line.
- Every stage of the query path writes timing into a trace object.
- Absolute imports. No relative imports beyond one level.
- Read config from environment variables via a single `app/config.py`. No hardcoded keys,
  model names, thresholds, or magic numbers scattered across files.

NEVER:
- NEVER use LangChain, LlamaIndex, or Haystack anywhere in `app/` or `ingest/`.
  (`langchain-core` arrives as a transitive dependency of `ragas` in the eval extra only.
  That is acceptable. Do not import from it directly.)
- NEVER use SQLAlchemy or any ORM. Raw SQL only.
- NEVER install `torch`, `transformers`, or `sentence-transformers` into the serving
  image. Cold start on Cloud Run is a latency concern.
- NEVER use a bare `except:` or `except Exception: pass`. Every handler logs and
  either degrades gracefully or re raises a typed exception.
- NEVER write an em dash or a double hyphen in any prose, comment, docstring, README,
  or generated document. Use a comma or restructure the sentence. Command line flags
  in shell scripts and workflow files are exempt from this rule.
- NEVER invent metric numbers, benchmark results, or page citations in the README.
  Leave a clearly marked `TBD` for anything not yet measured.
- NEVER add authentication, user accounts, or multi tenancy. Out of scope.
- NEVER build a chat interface with message history. Each query is independent.

## 3. Models and services

Read these from environment variables. Defaults shown are the intended values.

| Env var | Default | Notes |
|---|---|---|
| `EMBED_MODEL` | `gemini-embedding-2` | Multimodal capable, use text only here |
| `EMBED_DIM` | `768` | Truncated from 3072 default via `output_dimensionality` |
| `TASK_MODEL` | `gemini-3.1-flash-lite` | Serving model |
| `PROMPT_MODEL` | same as `TASK_MODEL` | Used by MIPROv2 to propose instructions |
| `JUDGE_MODEL` | same as `TASK_MODEL` | Used by RAGAS and the groundedness check |
| `GEMINI_API_KEY_TASK` | none | Project A key, serving and optimizing |
| `GEMINI_API_KEY_JUDGE` | none | Project B key, evaluation and judging |
| `DATABASE_URL` | none | Supabase Postgres connection string, pooler port |
| `RETRIEVAL_FLOOR` | `0.35` | Abstention threshold on top rerank score |
| `COMPILED_PROGRAM_PATH` | `artifacts/compiled_program.json` | |

Two separate API keys are deliberate: they belong to two different Google Cloud
projects, so evaluation traffic (which makes far more requests than serving) never
competes with live serving traffic. The task key MUST NEVER be used for judging, and
vice versa. Log a warning at startup if either is missing.

### Embedding rules, do not get these wrong

1. Pass `task_type="RETRIEVAL_DOCUMENT"` when embedding chunks at ingestion time.
2. Pass `task_type="RETRIEVAL_QUERY"` when embedding a user question at query time.
3. `gemini-embedding-2` was verified empirically (`output_dimensionality=768`) to
   already return unit norm vectors. There is no L2 normalization step. If a future
   model change is observed to return non-unit vectors, verify first rather than
   assuming, per section 15, before adding one.
4. `gemini-embedding-2` does NOT batch multiple documents per `embed_content` call,
   confirmed empirically, including against the SDK's own documented usage example
   (`contents=["text one", "text two"]` returns exactly one embedding, not two). Every
   embedding call in `app/llm.py` (`embed_text`) and `ingest/embed.py` embeds exactly
   one document per call. Concurrency, not batching, is the only way to speed this up:
   `ingest/embed.py` fans out a small number of calls at once with `asyncio.gather`
   (`--concurrency`, default 5).
5. Retry with exponential backoff and jitter on HTTP 429.
6. The embed script is resumable: it queries for rows where `embedding is null` and
   processes only those, so an interrupted run can be restarted safely. Read and write
   for this loop happen on the same held asyncpg connection, not a fresh
   `pool.acquire()` per call. Supabase's pooler was observed to not reliably make a
   write on one pooled connection visible to an immediately following read on a
   different pooled connection, which caused already-embedded rows to be reselected.
   `ingest/embed.py` supports `--skip-load` to skip re-upserting `chunks.jsonl` into
   the database on every run, since chunk text and metadata do not change between
   embedding runs, only which rows are missing an embedding.

### Reranker

`app/rerank.py` implements a single `async def rerank(query, candidates, top_k)`: one
listwise call to `TASK_MODEL`. It sends the query plus numbered candidate snippets,
asks for a JSON array of indices ordered best first with a relevance score from 0 to
1 each, and parses the result defensively.

If reranking fails or times out after 4 seconds, fall back to the fusion order, set
`degraded=True` on the trace, and continue. NEVER fail the request because reranking
failed.

## 4. Repository layout

```
.
├── CLAUDE.md
├── README.md
├── .env.example
├── requirements.txt
├── requirements-eval.txt
├── Dockerfile
├── .dockerignore
├── .github/workflows/deploy.yml
├── sql/
│   └── schema.sql
├── data/
│   ├── trainset.csv
│   └── manual.pdf            (gitignored, user supplies)
├── ingest/
│   ├── extract.py
│   ├── chunk.py
│   ├── embed.py
│   └── run_ingest.py
├── app/
│   ├── config.py
│   ├── errors.py
│   ├── db.py
│   ├── llm.py
│   ├── retrieve.py
│   ├── rerank.py
│   ├── programs.py
│   ├── guardrails.py
│   ├── trace.py
│   ├── pipeline.py
│   ├── api.py
│   └── static/
│       ├── index.html
│       └── metrics.html
├── eval/
│   ├── dataset.py
│   ├── metrics_retrieval.py
│   ├── metrics_ragas.py
│   ├── metrics_latency.py
│   ├── run_eval.py
│   ├── optimize.py
│   └── results/              (gitignored except .gitkeep)
├── artifacts/
│   └── .gitkeep
├── scripts/
│   ├── answer_cli.py
│   └── check_dspy_stream.py
└── tests/
    ├── test_chunk.py
    ├── test_fusion.py
    ├── test_guardrails.py
    └── test_metrics_retrieval.py
```

## 5. Database schema

Written to `sql/schema.sql` and applied against Supabase. Do not attempt to run
migrations from application code.

Both tables live under a `vidhi` schema rather than `public`, by deliberate choice.
Every query in `app/db.py` and `ingest/embed.py` qualifies the table name explicitly,
for example `select ... from vidhi.chunks`, rather than relying on `search_path`.
This is required, not stylistic: Supabase's pooler was observed to not reliably carry
a `set search_path` across statements on a pooled connection, which silently broke
lookups when the schema was addressed unqualified.

```sql
create extension if not exists vector;

create schema if not exists vidhi;

create table if not exists vidhi.chunks (
  id           bigserial primary key,
  chunk_uid    text unique not null,
  content      text not null,
  heading_path text,
  chapter      text,
  page_start   int not null,
  page_end     int not null,
  token_count  int,
  embedding    public.vector(768),
  tsv tsvector generated always as (to_tsvector('english', content)) stored
);

create index if not exists chunks_embedding_idx
  on vidhi.chunks using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

create index if not exists chunks_tsv_idx on vidhi.chunks using gin (tsv);

create table if not exists vidhi.traces (
  request_id   uuid primary key,
  query        text not null,
  route        text,
  sub_queries  jsonb,
  stages       jsonb,
  latency_ms   jsonb,
  tokens       jsonb,
  answer       text,
  citations    jsonb,
  grounded     boolean,
  abstained    boolean,
  degraded     boolean default false,
  error        text,
  created_at   timestamptz default now()
);

create index if not exists traces_created_idx on vidhi.traces (created_at desc);
```

`chunk_uid` is a deterministic hash of chapter plus page_start plus the first 200
characters of content, so reruns of ingestion upsert instead of duplicating.
Implemented in `ingest/chunk.py` as a sha256 hex digest, truncated to 32 characters.

**Ingestion status:** 2373 chunks ingested and fully embedded (gemini-embedding-2,
768 dimensions). 7 of 1321 pages suspected scanned (0.5 percent, no OCR needed).

## 6. Ingestion contract

`ingest/extract.py`
- Uses `pymupdf`. Writes `data/pages.jsonl`, one object per page: `page`, `text`, `char_count`.
- Logs any page with fewer than 100 characters of text as a suspected scanned page and
  prints a summary count at the end. If more than 5 percent of pages are suspect, print
  a loud warning telling the user OCR may be needed. Do not implement OCR unless asked.

`ingest/chunk.py`
- Heading detection was tuned to this specific manual's actual structure, confirmed by
  reading real page samples rather than assumed from a generic regex list. In this
  document: `Chapter N` is a hard chapter boundary only when a standalone `Synopsis`
  line follows within a few lines, which distinguishes a real chapter start from the
  table of contents (which lists `Chapter N` with no nearby `Synopsis`) and from the
  running header and footer (which abbreviate to `Chap. N`, never matching `Chapter N`).
  `Appendix N` is a second, equally weighted hard boundary for the reprinted-circulars
  section that follows the numbered chapters. `N.N` and `N.N.N` numbering is real but
  numbers sub-points under a synopsis topic, not a resetting per-chapter heading
  counter, so it is used only as a soft `heading_path` breadcrumb, never a hard
  boundary. `Section N` and `Rule N` occur only as inline citations, not headings, and
  short all-caps lines are dominated by front matter and table noise in this document.
  Neither is used for structure. The source book mislabels two different chapters both
  `Chapter 54`, so chapter boundaries are tracked by detection order, not by parsed
  chapter number.
- Packs text into chunks of about 700 tokens with 120 tokens of overlap. Use `tiktoken`
  `cl100k_base` purely as a token counter, it does not need to match the Gemini tokenizer.
- A chunk MUST NEVER span two chapters or two appendix entries.
- Carries `heading_path` as a breadcrumb string, `chapter`, `page_start`, `page_end`.
- Strips the running header and footer boilerplate (`Chap. N`, `Appx. N`, the book
  title line, bare page number lines) from chunk content before packing.
- Writes `data/chunks.jsonl` and prints a distribution summary: chunk count, token count
  percentiles, chunks per chapter.

`ingest/embed.py`
- Reads `chunks.jsonl`, upserts rows without embeddings (skippable with `--skip-load`
  once chunks are already loaded), then embeds one chunk per API call (see embedding
  rule 4), a small number at a time concurrently via `asyncio.gather` (`--concurrency`,
  default 5).
- Resumable: rows where `embedding is null` are selected and processed; an interrupted
  run can simply be restarted. The read and write for this check happen on one held
  connection for the whole run, see embedding rule 6.

`ingest/run_ingest.py`
- Orchestrates the three stages, each skippable by a flag, prints per stage timing.

## 7. Query path contract

`app/pipeline.py` exposes one function:

```python
async def answer_question(
    question: str,
    config: PipelineConfig,
) -> AsyncIterator[PipelineEvent]:
```

`PipelineConfig` selects the ablation variant so the same code path serves both the API
and the evaluation harness. This is important: evaluation MUST exercise production code,
not a parallel reimplementation.

```python
@dataclass
class PipelineConfig:
    name: str                  # "A_vanilla" | "B_hybrid" | "C_dspy" | "D_optimized"
    use_hybrid: bool           # False means dense only
    use_rerank: bool
    use_dspy: bool
    use_compiled: bool
    dense_k: int = 30
    sparse_k: int = 30
    fusion_k: int = 12
    final_k: int = 5
```

`PipelineEvent` is a small tagged union: `RouteEvent`, `RetrievalEvent`, `TokenEvent`,
`DoneEvent`, `ErrorEvent`. The API turns these into server sent events. The evaluation
harness consumes the same iterator and collects the final state.

Stages in order:

1. **Route.** DSPy `RouteQuery` when `use_dspy`, otherwise a hardcoded `lookup` route.
   On `out_of_scope`, skip retrieval entirely and emit the abstention message.
2. **Embed query** with `RETRIEVAL_QUERY` task type.
3. **Retrieve.** Dense and sparse searches run concurrently with `asyncio.gather`.
   Dense: `order by embedding <=> $1 limit $2`. Sparse: `ts_rank_cd` over `tsv`
   with `websearch_to_tsquery`. When `use_hybrid` is false, run dense only.
4. **Fuse.** Reciprocal rank fusion, `score = sum(1 / (60 + rank))` across lists.
   Pure function in `app/retrieve.py`, unit tested with a fixed example.
5. **Rerank** when enabled, keep `final_k`.
6. **Guardrail, pre generation.** If the top rerank score is below `RETRIEVAL_FLOOR`,
   abstain without calling the generation model at all. This both prevents hallucination
   and saves an unnecessary call.
7. **Generate.** Streams tokens. Cited pages MUST come only from the retrieved set.
8. **Guardrail, post generation.** Validate citations against retrieved pages, dropping
   any that do not appear and marking the answer low confidence. Then run the
   groundedness check. This runs AFTER the stream completes and is emitted as a separate
   event so it never delays the first token.
9. **Persist trace.** Fire and forget insert. A trace write failure MUST NEVER fail the
   request, log it and move on.

## 8. DSPy layer

`app/programs.py`. Pin the DSPy version in `requirements.txt`.

Three signatures:

- `RouteQuery`: input `question`. Outputs `route` as a literal of
  `lookup | procedure | multi_hop | out_of_scope`, and `sub_questions` as a list of
  strings, empty unless the route is `multi_hop`.
- `AnswerFromManual`: inputs `context` and `question`. Outputs `answer` as prose and
  `citations` as a list of integer page numbers. The docstring states that the answer
  must come only from the context and that the model must say it cannot find the answer
  rather than guessing.
- `CheckGrounded`: inputs `context` and `answer`. Outputs `unsupported_claims` as a list
  of strings and `grounded` as a boolean.

`RAGProgram(dspy.Module)` composes the router and `dspy.ChainOfThought(AnswerFromManual)`.
The retriever is injected as a callable, not constructed inside the module, so the
evaluation harness can swap retrieval configurations.

`CheckGrounded` is a separate program, deliberately outside `RAGProgram`, so it stays off
the streaming path and out of the optimizer's search space.

Enable the DSPy LM cache so optimizer retries and repeated evaluation runs do not
recall the model for identical inputs.

## 9. Optimizer

`eval/optimize.py`.

Use `MIPROv2` with `auto="light"`. Justification, which also goes in the README:
MIPROv2 proposes new instructions and bootstraps few shot demonstrations for every
predictor, then searches over the combinations with Bayesian optimization. Both halves
matter here because the grounding and abstention rules live in the instruction text, not
in the demonstrations. `BootstrapFewShotWithRandomSearch` searches only over
demonstration sets and cannot rewrite an instruction, so its ceiling is lower on a task
whose main failure mode is answering from parametric knowledge instead of abstaining.

The metric is deliberately part deterministic:

```
score = 0.6 * correctness + 0.4 * citation_validity
```

- `correctness`: an LLM judge on `JUDGE_MODEL` returning 0, 0.5, or 1 against `gold_answer`.
- `citation_validity`: fraction of predicted pages that fall inside the gold page set
  widened by plus or minus 2 pages. Purely arithmetic, no LLM call.

The deterministic 40 percent keeps the Bayesian search from chasing judge noise, which is
the usual reason a MIPROv2 run appears to accomplish nothing.

MIPROv2 compiles a bare `ChainOfThought(AnswerFromManual)` in isolation, not the full
`RAGProgram`: `RAGProgram.forward` only calls the answer predictor, so the router gets
no real training signal and would only waste search budget if included. The result is
spliced into a fresh `RAGProgram` before saving.

Save the result to `COMPILED_PROGRAM_PATH`. Also write `eval/results/optimize_log.json`
with per trial scores and the winning instruction text, because the README needs to show
what the optimizer actually changed.

## 10. Evaluation

`data/trainset.csv` is the single source of truth. Columns:

`id, question, gold_answer, gold_pages, category, answerable, split`

- `gold_pages`: semicolon separated integers, empty when `answerable` is false.
- `category`: `lookup | rate | procedure | multi_hop | out_of_scope`.
- `answerable`: `true` or `false`.
- `split`: `train` or `dev`. `train` feeds the optimizer, `dev` produces reported metrics.

`eval/dataset.py` loads and validates the CSV, failing loudly on malformed rows, and
converts `train` rows into DSPy examples.

### Metric groups

**Group 1, deterministic, zero LLM calls.** Run on the full dev split, always.
`eval/metrics_retrieval.py`:
- Recall at 5 and at 10, computed as whether any gold page appears in the retrieved
  page spans.
- MRR at 10.
- Citation validity rate.
- Abstention accuracy on `answerable=false` rows, and false abstention rate on
  `answerable=true` rows.
- Answer exact page overlap.

**Group 2, RAGAS.** `eval/metrics_ragas.py`. Faithfulness, answer relevancy,
context precision, context recall. Runs on `JUDGE_MODEL` with the judge API key.
Defaults to a subset of dev questions, size controlled by a flag.
Context precision and recall depend only on retrieval, so cache and reuse them across
configurations that share a retrieval setup. Configs C and D share retrieval, compute once.

**Group 3, production and latency.** `eval/metrics_latency.py`. This group measures:
- Time to first token, p50 and p95.
- Total wall clock, p50 and p95.
- Per stage latency breakdown, mean and p95: query embedding, dense search, sparse
  search, fusion, rerank, generation first token, generation total, groundedness check.
- Cold start latency, measured separately by hitting the deployed service after it has
  scaled to zero. Report warm and cold as distinct numbers, never blended.
- Tokens per second on the output stream.
- Input and output tokens per query, mean.
- Estimated cost per thousand queries at the published paid tier rate.

`eval/run_eval.py` runs one or more named configurations, writes
`eval/results/{config}_{timestamp}.json` with full per question records, and prints a
markdown comparison table ready to paste into the README. Fix all random seeds. Record
the model IDs, the git commit hash, and the timestamp in every result file.

The four configurations:

| Name | Retrieval | Prompting |
|---|---|---|
| `A_vanilla` | Dense only, top 5 | Plain string prompt, no DSPy |
| `B_hybrid` | Hybrid, fusion, rerank | Same plain prompt |
| `C_dspy` | Hybrid, fusion, rerank | DSPy, not compiled |
| `D_optimized` | Hybrid, fusion, rerank | DSPy, MIPROv2 compiled |

This isolates retrieval gains from prompt optimization gains. A single before and after
number would not, which is why a shallow evaluation misses this distinction.

## 11. Error handling

`app/errors.py` defines: `ConfigError`, `RetrievalError`, `RerankError`, `LLMError`,
`EmbeddingError`, `DatabaseError`, `GuardrailAbstain`.

Degradation table. Implement every row.

| Failure | Behaviour |
|---|---|
| Rerank error or timeout | Use fusion order, set `degraded=True`, continue |
| Query embedding fails | Sparse only retrieval, set `degraded=True`, note it in the response |
| Router fails | Default to the `lookup` route |
| Sparse search fails | Dense only, set `degraded=True` |
| Dense and sparse both fail | Raise `RetrievalError`, return HTTP 503 with a clear message |
| Zero chunks retrieved | Abstain, do not send an empty context to the model |
| LLM 429 | Two retries with exponential backoff and jitter, then a user visible quota message |
| LLM 5xx or timeout | Two retries, then HTTP 502 with a clear message |
| Groundedness check fails | Answer still ships, verification badge reads "not verified" |
| Trace insert fails | Log and continue, never fail the request |
| Malformed rerank or citation JSON | Log the raw text, fall back, never crash on a parse error |

The API layer has exactly one exception boundary that maps typed exceptions to status
codes and always returns a structured JSON error body carrying `request_id`.

## 12. API and UI

One service. FastAPI serves both the JSON and SSE API and a single static HTML page.
There is no separate frontend service, no npm, no build step, and no JavaScript
framework. The UI is one hand written file.

### 12.1 Application structure

`app/api.py` constructs the FastAPI app and owns the only exception boundary in the
project. Use a lifespan context manager, not the deprecated startup and shutdown events.

Lifespan startup, in this order:

1. Validate configuration. Raise `ConfigError` and refuse to start if `DATABASE_URL` or
   `GEMINI_API_KEY_TASK` is missing. Log a warning, but start, if
   `GEMINI_API_KEY_JUDGE` is missing, because judging is only needed by evaluation.
2. Create the asyncpg pool with `min_size=1`, `max_size=5`. Cloud Run scales
   horizontally so a large pool per instance is wasted, and the Supabase free tier caps
   total connections. Use the Supabase connection pooler port, not the direct database
   port, because scale to zero means connections churn.
3. Issue one trivial query to warm the pool so the first user request does not pay for
   connection setup.
4. Load the compiled DSPy program from `COMPILED_PROGRAM_PATH` if the file exists. Cache
   both the compiled and uncompiled programs in module state. Never load a program per
   request.
5. Cache the chunk count from a single `select count(*) from chunks`.

Lifespan shutdown closes the pool.

Do all of the above lazily where possible and avoid heavy work at import time, because
every millisecond here is cold start latency.

CORS is not needed. The page and the API share an origin. Do not add CORS middleware.

### 12.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Streaming answer, server sent events |
| `GET` | `/healthz` | Liveness and configuration echo |
| `GET` | `/trace/{request_id}` | Full stored trace as JSON |
| `GET` | `/program` | Active DSPy instruction text and demonstration count |
| `GET` | `/configs` | The four ablation config names and their settings |
| `GET` | `/metrics` | Latest eval, latency, and optimizer result files |
| `GET` | `/` | `app/static/index.html` |

`GET /healthz` returns the git commit hash, the resolved model IDs, `EMBED_DIM`, the
cached chunk count, and whether the compiled artifact was found. This doubles as the
post deploy smoke test target, so it MUST fail with a non 200 status when the chunk
count is zero or the database is unreachable.

`GET /program` exists so the UI can show the instruction that MIPROv2 actually produced.
Return the instruction text for each predictor, the number of demonstrations attached,
and whether the program in use is compiled or not.

`POST /ask` request body:

```json
{"question": "string", "config": "D_optimized"}
```

Validate with a Pydantic model. Reject an empty question, or one longer than 1000
characters, with HTTP 422 before touching any external service. Reject an unknown config
name with HTTP 422 and list the valid names in the error body.

### 12.3 The SSE contract

This contract is the interface between `app/pipeline.py` and `app/static/index.html`.
Write it once, in one place, and do not let the two drift.

Response headers on `/ask`:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

Each frame is `event: NAME` then `data: <single line of JSON>` then a blank line. JSON
MUST be serialized without embedded newlines, because a newline inside `data:` breaks
the frame.

| Event | Payload |
|---|---|
| `route` | `{"route": str, "sub_queries": [str]}` |
| `retrieval` | `{"candidates": [...], "degraded": bool}` |
| `token` | `{"text": str}` |
| `done` | `{"request_id": str, "answer": str, "citations": [int], "grounded": bool, "unsupported_claims": [str], "abstained": bool, "degraded": bool, "latency_ms": {...}, "tokens": {...}}` |
| `error` | `{"request_id": str, "code": str, "message": str}` |

Each candidate object carries `chunk_uid`, `page_start`, `page_end`, `heading_path`,
`dense_rank`, `sparse_rank`, `fusion_score`, and `rerank_score`. Use `null`, never a
sentinel number, for a rank the candidate did not receive. Cap the candidates list at
the top 12 so the payload stays small.

The `route` and `retrieval` events give the page something truthful to display during the
seconds before the first token, which is what makes the perceived latency acceptable.

Emit a heartbeat comment line, a line beginning with a colon, every 10 seconds during any
silent period. Reranking can take a few seconds and an idle connection through a proxy is
the classic cause of a stream that appears to hang.

`error` is always the final event when it occurs, and the response still ends cleanly.
Never abandon the connection mid stream, because the client cannot distinguish that from
a network failure.

### 12.4 Ordering and the trace write

Inside the `/ask` handler and before the `done` event is emitted:

1. Stream all `token` events.
2. Run citation validation.
3. Run the groundedness check.
4. Write the trace row.
5. Emit `done`.

Steps 2 through 4 happen after the last token, so they do not affect time to first token,
but they happen inside the request, so they are guaranteed to execute. A trace write
failure MUST be logged and swallowed. It MUST NEVER prevent the `done` event.

Handle client disconnection. If the client goes away mid stream, cancel the generation
rather than continuing needlessly. Catch `asyncio.CancelledError`, log it, attempt the
trace write with `abandoned=True` recorded in the `error` column, and re raise.

### 12.5 `app/static/index.html`

One file. Inline `<style>` and inline `<script>`. No external requests of any kind, no
CDN, no fonts, no images. Vanilla JavaScript only.

**Streaming client.** Do not use `EventSource`. It cannot send a POST body, and it
automatically reconnects when a stream ends, which on this project would silently re run
the query. Use `fetch` with a POST body, read `response.body.getReader()`, decode with
`TextDecoder`, and parse SSE frames manually:

- Keep a string buffer. Append each decoded chunk.
- Split the buffer on a double newline. Process every complete frame, retain the trailing
  partial frame in the buffer. Getting this wrong produces intermittent corruption that
  only appears under slow networks, so write it carefully.
- Skip frames whose first line begins with a colon, those are heartbeats.
- Wrap `JSON.parse` in try and catch. Log a malformed frame to the console and continue,
  never let one bad frame kill the stream.

**Layout, top to bottom.**

1. A header with the project name and a one line disclaimer stating that answers are
   informational, generated from a single reference manual, and not professional tax
   advice.
2. A short note stating that each query is handled independently and there is no
   conversation memory.
3. A config selector, populated from `GET /configs`, defaulting to `D_optimized`. Label
   it clearly as an ablation switch, so the difference between configurations is visible
   live instead of only in a README table.
4. A textarea for the question and a submit button. Enter submits, shift and enter
   inserts a newline.
5. A status line driven by which events have arrived: routing, then retrieving, then
   reranking, then generating, then verifying. Show elapsed milliseconds beside it.
6. The answer region. Append token text to a single text node. Set
   `white-space: pre-wrap` and do not parse markdown, a markdown library is not worth
   the dependency here.
7. A citations row rendering each cited page as a small pill.
8. A grounding badge with three states: verified, unsupported claims found, and not
   verified. Show the unsupported claims when there are any. Do not hide a failed check.
9. A degraded notice, shown only when the `done` event reports `degraded` as true, naming
   which stage fell back.
10. An `Inspect` element, collapsed by default, containing:
    - Route and any sub queries.
    - A candidates table with columns for page range, heading path, dense rank, sparse
      rank, fusion score, and rerank score. Mark the rows that survived into the final
      context.
    - The per stage latency breakdown as a simple bar or a table.
    - Input and output token counts.
    - The active instruction text from `GET /program`, fetched once on page load.
    - The `request_id`, so it can be pasted into `/trace/{request_id}`.

**Behaviour.** Disable the submit button while a stream is in flight. Reset all regions on
a new submission, since there is no history to preserve. Show the `error` payload inline
with its `request_id` rather than in an alert dialog. Set `aria-live="polite"` on the
answer region and `role="status"` on the status line. Do not use browser storage of any
kind.

Serve the file with `StaticFiles` mounted at a path such as `/static`, and add an explicit
`GET /` route returning the HTML with a no cache header so a stale cached page never
mismatches a redeployed API.

---

## 13. Deployment

One Cloud Run service named `rag`, in region `asia-south1`, deployed from one Dockerfile
by one GitHub Actions workflow. The Supabase project is in the geographically matching
region. Colocating the application and the database is the single largest latency
lever available, larger than any code optimization in this project.

### 13.1 Dockerfile

Single stage, `python:3.11-slim` base. Layer ordering matters for build cache:

1. Install only the runtime OS packages needed, if any. Do not install a compiler
   toolchain. Every dependency in `requirements.txt` MUST have a manylinux wheel.
2. Copy `requirements.txt` alone, then install, so dependency layers cache across code
   changes.
3. Copy `app/`, `artifacts/`, and `sql/`.
4. Create and switch to a non root user.
5. `CMD` runs uvicorn, binding to `0.0.0.0` and the `PORT` environment variable that
   Cloud Run injects. Never hardcode 8080.

Use uvicorn directly with a single worker. Cloud Run scales by adding instances, so
multiple workers inside one container just multiply memory for no throughput gain.

MUST NOT be installed: `torch`, `transformers`, `sentence-transformers`, `ragas`,
`langchain-core`, `streamlit`, `pymupdf`. Ingestion and evaluation dependencies belong
in `requirements-eval.txt` and never enter the serving image. Target an image under
400MB. A large image is measurable cold start latency.

Deviation, deliberate: `tiktoken` DOES enter the serving image, as an unavoidable
transitive dependency of `dspy` via `litellm`. It is a small pure tokenizer library
with no model weights, not one of the heavy libraries this rule actually exists to
exclude.

`.dockerignore` MUST exclude `data/`, `eval/`, `ingest/`, `tests/`, `scripts/`, `.git`,
`.github`, `__pycache__`, `*.pyc`, `.env`, `.venv`. The PDF is hundreds of megabytes and
MUST NEVER enter the build context.

### 13.2 Service configuration

| Setting | Value | Reason |
|---|---|---|
| Region | `asia-south1` | Matches Supabase, minimizes round trip |
| Memory | 512Mi | Sufficient without torch |
| CPU | 1 | |
| Startup CPU boost | enabled | Directly reduces cold start, no ongoing cost |
| Concurrency | 20 | Requests are IO bound |
| Min instances | 0 | Accepts a cold start in exchange for zero idle cost |
| Max instances | 3 | Caps runaway cost |
| Request timeout | 120s | Well above the worst realistic request |
| Ingress | all | |
| Authentication | allow unauthenticated | Public demo |

Min instances of zero is a deliberate choice: it is the sole cause of the cold start
figure in the latency table, and setting it to one eliminates that figure at a small
monthly cost. Reporting cold start honestly and explaining its cause reads far better
than blending it into the warm numbers.

### 13.3 Configuration and secrets

Secrets live in Google Secret Manager and are mounted as environment variables by Cloud
Run, so application code reads them from the environment exactly as it does locally and
`app/config.py` needs no special case.

Store as secrets: `GEMINI_API_KEY_TASK`, `GEMINI_API_KEY_JUDGE`, `DATABASE_URL`.
Grant the Cloud Run runtime service account the Secret Manager secret accessor role on
each one. Secret values MUST NEVER appear in the workflow file, in the repository, or
in deploy logs.

Non secret environment variables (`EMBED_MODEL`, `EMBED_DIM`, `TASK_MODEL`,
`JUDGE_MODEL`, `RETRIEVAL_FLOOR`, `COMPILED_PROGRAM_PATH`, `GIT_SHA`) and all secrets
are currently managed by hand on the Cloud Run service in the GCP console, not by the
workflow file's `--set-env-vars`/`--set-secrets` flags. This is a deliberate deviation
from having the workflow file be the single source of truth: `gcloud run deploy`
without those flags leaves whatever is already configured on the service untouched, so
console-set values persist across deploys as long as the deploy step never adds those
flags back.

### 13.4 The compiled program artifact

`artifacts/compiled_program.json` is committed to the repository and baked into the image.
It is a build input, not runtime state.

The service starts correctly without it and falls back to the uncompiled program, with
`/healthz` reporting `compiled: false`. After a MIPROv2 run produces the artifact,
commit it and redeploy. That redeploy is the moment configuration `D_optimized`
becomes live, and it should be a distinct commit with a clear message so the before and
after state is legible in the git history.

Do not fetch the artifact from object storage at startup. That adds a network call to
cold start and a failure mode, to solve a problem this project does not have.

### 13.5 GitHub Actions workflow

`.github/workflows/deploy.yml`. Triggers on push to `main` when `app/`, `artifacts/`,
`requirements.txt`, `Dockerfile`, or the workflow itself changes, and on manual dispatch.

Steps:

1. Checkout.
2. Authenticate to Google Cloud via Workload Identity Federation.
3. Configure Docker authentication for Artifact Registry.
4. Build, tagging the image with both the short commit SHA and `latest`. Pass the commit
   SHA in as a build argument so `/healthz` can report it. Never deploy an image
   referenced only by `latest`, because you lose the ability to identify what is running.
5. Push.
6. Deploy to Cloud Run with the settings in 13.2.
7. Smoke test. Curl `/healthz` on the returned service URL, parse the JSON, and fail the
   job unless the chunk count is greater than zero, the commit SHA matches the one just
   built, and every model ID is non empty. A deploy that reports success while the
   service cannot reach the database is worse than a failed deploy.
8. Print the service URL in the job summary.

Concurrency: cancel any in progress run for the same ref, so a rapid second push does not
race the first deploy.

Cloud Run keeps every revision, so rollback is redirecting traffic to a previous
revision rather than rebuilding.

### 13.6 Cold start

Mitigations: startup CPU boost enabled, image under 400MB, no torch, no work at import
time, and the pool warmed during lifespan rather than on the first request.

Measure cold start explicitly. `eval/metrics_latency.py` has a mode that forces a cold
path: wait out the scale to zero interval, issue one request, record time to first byte
and time to first token, and repeat at least five times. Report those numbers as their
own row, clearly labelled, never averaged together with warm requests.

Document, but do not automate, the optional Cloud Scheduler warm ping: a job hitting
`/healthz` every 5 minutes keeps one instance warm at a small ongoing cost.

### 13.7 Local parity

`docker build` then `docker run` with an env file and `PORT` set MUST produce a working
service identical to the deployed one. Include the two commands in the README quick
start. If the container works locally and fails on Cloud Run, the difference is
configuration, and the smoke test in step 7 is what tells you that within a minute
instead of an hour.

## 14. README requirements

The README is the primary document a reader sees first. It MUST contain:

1. What the system does, in plain language, near the top.
2. A live link and a quick start (local and deployed).
3. Architecture diagram as a mermaid block.
4. The four configuration ablation table with real measured numbers.
5. The production and latency table, warm and cold reported separately, with an honest
   paragraph on which numbers are Cloud Run scale-to-zero artefacts and which are
   architectural.
6. Optimizer section: which optimizer, why that one, what the alternatives would have
   traded off, the composite metric definition, the before and after instruction text,
   and the per trial score curve.
7. Modern techniques implemented, one short paragraph each.
8. Design tradeoffs, explicitly including why the reranker is an LLM call rather than a
   local cross encoder, and what would change with a different reranker provider.
9. Known limitations and what would be built next with more time. Be specific and honest.
   A named weakness reads better than a silent one.

Placeholders for unmeasured numbers MUST be written as `TBD` and never as invented values.

## 15. When you are unsure

If an external library API is uncertain, write the smallest possible script that
exercises it, run it, and report the result. Do not write many files on top of an
unverified assumption. If a decision in this document conflicts with something you
discover while running code, stop and report the conflict rather than silently choosing.

## 16. Confirmed facts and deviations, read before making changes

- **Schema location.** Tables are `vidhi.chunks` and `vidhi.traces`, not `public.*`.
  Every query in `app/db.py` qualifies the schema explicitly. See section 5.
- **No `l2_normalize`.** Not implemented, not tested. `gemini-embedding-2` at 768
  dimensions was confirmed to already return unit vectors. See embedding rule 3 in
  section 3.
- **One embedding per API call.** `app/llm.py` has `embed_text` (single document) and
  `embed_texts` (loops `embed_text`), not a real batch call. `gemini-embedding-2` does
  not batch multiple documents into one `embed_content` call despite its own SDK
  documented example implying it does. See embedding rule 4 in section 3.
- **`tiktoken` ships in the serving image.** Accepted transitive dependency of `dspy`
  via `litellm`. See section 13.1.
- **Supabase pooler does not reliably carry session state or make a write on one
  pooled connection immediately visible to a read on another pooled connection.**
  Two consequences for any new code that reads then writes, or writes then
  immediately reads, in a loop: hold one connection with `pool.acquire()` for the
  whole loop rather than acquiring per statement, and never rely on `search_path`,
  qualify table names explicitly instead.
- **Chunking is tuned to this book's real structure**, not a generic regex list.
  Read the comment block at the top of `ingest/chunk.py` before assuming `Section N`
  or `Rule N` mean anything structural in this document, they do not, they are inline
  citations.
- **DSPy streaming works correctly** for realistic length answers. A trivially short
  answer can arrive as zero or one `StreamResponse` events before the final
  `Prediction`, this is a Gemini API nuance, not a bug, do not add a workaround for
  it. See `scripts/check_dspy_stream.py`.
- **MIPROv2 optimizes only the `answer` predictor** of `RAGProgram`, compiled in
  isolation as a bare `ChainOfThought(AnswerFromManual)`, then spliced into a fresh
  `RAGProgram` before saving. See section 9.
- **The single reranker backend is a listwise LLM call** on `TASK_MODEL`. There is no
  separate hosted reranker integration; see the tradeoffs section of the README for
  what a hosted cross encoder would change.

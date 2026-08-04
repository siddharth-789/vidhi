# Claude Code prompts for build day

Paste these one at a time. Do not paste the next one until the checkpoint passes.
`CLAUDE.md` is auto loaded, so these stay short on purpose.

---

## Before you start

1. `git init`, drop `CLAUDE.md` at the repository root, `data/trainset.csv` in place,
   and the PDF at `data/manual.pdf`.
2. Create two Google Cloud projects and get a Gemini API key for each.
3. Create the Supabase project in the region closest to `asia-south1`.
4. Confirm the exact model IDs in AI Studio and correct section 3 of `CLAUDE.md` if they
   differ from the defaults written there.

---

## Prompt 1, foundation and ingestion — DONE

Phase 1 is complete. 2373 chunks ingested and fully embedded, 7 of 1321 pages (0.5
percent) suspected scanned, no OCR needed. Tables live under a `vidhi` Postgres schema,
not `public`. See CLAUDE.md section 17 for the full list of deviations and confirmed
facts from this phase, Phase 2 work must be consistent with those, not with whatever an
earlier section of CLAUDE.md assumed before it was checked against the real API and the
real document.

<details>
<summary>Original prompt, for reference</summary>

> Read CLAUDE.md fully, then execute Phase 1 only. Stop at the Phase 1 checkpoint.
>
> Start with scripts/check_dspy_stream.py and actually run it before writing anything
> else, so we know the DSPy streaming API behaves as section 8 assumes. Report what you
> observe. Then write the foundation files and the full ingest package.
>
> Do not write anything from Phase 2 or Phase 3. Do not create app/pipeline.py,
> app/retrieve.py, or anything under eval/ yet.
>
> Before you write any file, list the files you plan to create in order and wait for me
> to confirm.

</details>

---

## Prompt 2, retrieval and generation — START HERE IN A NEW CHAT

> Read CLAUDE.md fully, in particular section 17, then execute Phase 2. Stop at the
> Phase 2 checkpoint.
>
> Ingestion is complete: 2373 chunks in vidhi.chunks, all embedded. Do not re-run
> ingestion or touch anything under ingest/.
>
> Before writing app/pipeline.py or app/programs.py, verify with a small standalone
> script that DSPy's ChainOfThought and the router signature behave as section 8
> describes against the real gemini-3.1-flash-lite model, the same way
> scripts/check_dspy_stream.py verified streaming in Phase 1. Report what you observe
> before writing the full module.
>
> Build retrieval, fusion, reranking, the DSPy programs, guardrails, tracing, and the
> pipeline, plus a command line entry point that takes a question and a config name and
> prints the streamed answer followed by the full trace as formatted JSON.
>
> Default RERANK_PROVIDER to llm so this runs on a Gemini key alone.
>
> Unit test the fusion function against a hand computed example and the guardrail
> thresholds against synthetic scores. Both must pass before you tell me Phase 2 is done.
>
> Before you write any file, list the files you plan to create in order and wait for me
> to confirm.

**Checkpoint.** Run the CLI against three questions: one straightforward lookup, one that
spans two sections, and one about something entirely outside GST, for instance a question
about Roman history. The third MUST abstain without calling the generation model. Verify
the abstention in the trace, not just in the output text.

Then look at the trace for the first question and confirm the same chunk appears in both
the dense and the sparse candidate lists. If the two lists never overlap, something is
wrong with one of the two searches and hybrid retrieval is silently doing nothing useful.

---

## Prompt 3, evaluation and serving

> Execute Phase 3 from CLAUDE.md.
>
> Write all of eval/ first, then the API, then the UI, then the Dockerfiles and
> the GitHub Actions workflow, then the README skeleton with TBD placeholders everywhere
> a measured number belongs.
>
> Evaluation is the graded deliverable, so give eval/ the most care. The latency and
> production metrics in group 3 matter as much as the RAGAS scores, they are how we
> justify free tier serving numbers honestly.
>
> run_eval.py must accept a list of config names, must never blend warm and cold latency,
> and must print a markdown table I can paste straight into the README.
>
> optimize.py must estimate and print its request count and refuse to run without an
> explicit confirmation flag.

**Checkpoint.** Run `run_eval.py` for config A only, on five dev questions, to prove the
harness works before spending quota. Then A and B on the full dev split. Two columns of
the table filled is the end of build day.

---

## Status as of 2026-08-04, read this before resuming

`trainset.csv` is filled to 100 rows (45 train, 55 dev). Phase 3 (`eval/`, `app/api.py`,
`app/static/index.html`, Dockerfile, workflow, README skeleton) is built. Since then, real
runs against live quota surfaced several bugs, all fixed:

- `app/pipeline.py`, plain prompt citation regex missed comma separated page lists like
  `(p. 131, p. 134)`. Fixed.
- `app/static/index.html`, the `done` event handler only synced the answer text when
  `abstained` was true, so a DSPy config answer that streamed zero token events (a real
  Gemini/DSPy behavior, see CLAUDE.md section 17, not a bug) rendered as empty in the UI
  even though the trace had the full answer. Fixed: `done` always resyncs from
  `payload.answer`.
- `app/pipeline.py` and `app/api.py`, `PipelineConfig.use_compiled` was defined but never
  read, so `D_optimized` silently behaved identically to `C_dspy`. Fixed: `answer_question`
  now takes an optional `compiled_rag_program` and both `app/api.py` and
  `eval/run_eval.py` load and pass it when `use_compiled` is set.
- `eval/optimize.py`, the first MIPROv2 run compiled the full `RAGProgram` (both `route`
  and `answer` predictors). `RAGProgram.forward` only ever calls `self.answer`, so `route`
  got no real training signal, and the search spent its whole budget perturbing the
  predictor nobody scores. Fixed: compile a bare `ChainOfThought(AnswerFromManual)` in
  isolation, then splice the result into a fresh `RAGProgram` before saving, so the
  artifact's predictor keys still match what `app/pipeline.py` loads.
- `eval/metrics_ragas.py`, two bugs: `GoogleGenerativeAIEmbeddings` needs a `models/`
  prefixed name unlike `google.genai.Client` used everywhere else in this project, and
  `EvaluationResult` has no `__contains__`, so `metric_name in result` silently misbehaved.
  Both fixed. Also added `RunConfig(max_workers=2)` to `ragas.evaluate`, its default of 16
  concurrent judge calls thundering-herds the free tier's 15 requests per minute ceiling
  and the run barely progresses.
- `eval/run_eval.py` gained `--resume`, checkpointing each answered question to
  `eval/results/.checkpoint_{config}_{split}.jsonl` as it goes and skipping already
  checkpointed rows on restart. Added because key rotation and quota exhaustion forced
  three full restarts of `B_hybrid` before this existed.

Real measured results so far, `RERANK_PROVIDER=llm` (Cohere free tier could not sustain
the eval's call volume, degraded on a third of B_hybrid's first attempt):

| Config | n | recall@5 | recall@10 | MRR@10 | citation validity | abstention acc. | false abstention | page overlap |
|---|---|---|---|---|---|---|---|---|
| A_vanilla | 55 (full dev) | 0.740 | 0.740 | 0.514 | 0.680 | 0.000 | 0.000 | 0.300 |
| B_hybrid | 55 (full dev) | 0.800 | 0.800 | 0.543 | 0.750 | 1.000 | 0.040 | 0.323 |
| C_dspy | 20 (dev subset) | 0.800 | 0.800 | 0.639 | 0.889 | 0.000 | 0.100 | 0.361 |
| D_optimized | 20 (dev subset) | 0.800 | 0.800 | 0.591 | 1.000 | 0.000 | 0.100 | 0.472 |

C and D are on a 20 question subset, not the full 55, because DSPy's `ChainOfThought`
path makes 3 to 4 Gemini calls per question (route, rerank, generate, groundedness)
against a 15 requests per minute free tier ceiling, so a full 55 question run was
projected at 3 to 6 hours. Widen to the full dev split if a future session has more time
or a paid tier.

The MIPROv2 light compile ran and produced `artifacts/compiled_program.json`. Its honest
result: the search never beat the baseline instruction and zero demos (`eval/results/
optimize_log.json` has the full trial score curve), so `D_optimized`'s prompt is
currently identical to `C_dspy`'s, only the retrieval and generation path differ per the
table above. This is a real finding to report as-is in the README, not something to
paper over.

## Status as of 2026-08-05, everything below this line is newer than the block above

RAGAS, latency, and token tracking are done. Deployment is still not done, see the end
of this section. Bugs found and fixed today:

- `app/pipeline.py`, `trace.tokens` was defined in the dataclass and threaded through
  every event, but nothing ever wrote to it, so every `/ask` response reported
  `tokens: {}`. Fixed: `_generate_plain` reads `usage_metadata` off the last streamed
  Gemini chunk (confirmed empirically to carry cumulative totals), `_generate_dspy`
  reads `task_lm.history[-1]["usage"]` after the call. Known residual gap, not fixed:
  `dspy.LM` via litellm reports `completion_tokens` correctly but `prompt_tokens` as 0
  for calls made through `dspy.streamify`, affects configs C and D only, documented in
  the README section 9 rather than chased further.
- `eval/run_eval.py`, `context_texts` was populated from `RetrievalEvent.candidates[].
  heading_path`, not the actual retrieved chunk content, because the SSE contract in
  CLAUDE.md section 12.3 deliberately keeps `RetrievalEvent` slim with no content
  field. This silently fed RAGAS a heading string like "Chapter 3: Levy and Collection
  of Tax..." instead of the real excerpt, which is why an early RAGAS attempt scored
  `faithfulness: 0.05` and `context_recall: 0.0` on real answers, a data bug, not a
  real finding. Fixed: `app/pipeline.py` now records full chunk content into
  `trace.stages["retrieval"]["context_texts"]` (never sent over SSE, so the slim API
  contract is untouched), and `run_eval.py` reads it back via `app.db.get_trace` after
  each question, the same path `GET /trace/{request_id}` already exposes.
  Consequence: all four configs (A, B, C, D) were re-run at n=20 to get correct traces
  before RAGAS could be trusted, this is why the config table numbers changed slightly
  from the 2026-08-04 block above (which used the buggy A at n=55 and B at n=55).
- `eval/metrics_ragas.py`, fixing this required three iterations, documented in full in
  the README section 9's RAGAS note, because ragas's own concurrency control
  (`RunConfig.max_workers`) does not control the actual 429 recovery, that lives inside
  `langchain_google_genai`'s tenacity retry wrapper, which retries after a fixed short
  delay regardless of the server's suggested `retry_delay`. Every batched or
  concurrent shape (`max_workers` from 16 down to 1, `--ragas-n` from 20 down to 3)
  either stalled indefinitely or made only a few sub-evaluations of progress per 20
  minutes. It also does not compose with being called repeatedly from inside our own
  running event loop: `ragas.evaluate` calls `asyncio.run()` internally and the second
  call crashed with "pop from an empty deque" even with `nest_asyncio` applied. Working
  fix: one metric against one row per call, each dispatched via `asyncio.to_thread` (a
  fresh thread has no running loop), with an explicit `asyncio.sleep` between calls
  that sits entirely outside ragas's and langchain's retry logic. Slow, but the only
  shape observed to reliably finish. `python -m eval.backfill_ragas` now runs this way
  by default, no flag needed.
- `eval/metrics_latency.py`'s default question ("What is the GST rate for cotton
  textiles?") triggered abstention on some configs, which measures the pre generation
  guardrail path, not the intended full generation and groundedness path. Not a code
  bug, just a bad default for a latency probe; the actual runs used
  `--question "What is the time limit for claiming input tax credit?"`, a real
  in-scope lookup, instead.

RAGAS ran on 5 questions per config (`--ragas-n 5`), small on purpose given the pacing
constraint above; some individual metrics lost all retries on one row at that sample
size and are reported as `n/a` in the README rather than a fabricated number, see the
config table's footnote. Latency ran with 5 warm requests per config against a locally
running `uvicorn app.api:app` instance, not deployed. Real numbers for both are in
`README.md` sections 4 and 5 now, no more `TBD` there except the paid tier cost
estimate and the cold start row.

## Still left, deployment only

- Deployment to Cloud Run and the cold start measurement. Needs a GCP project, Artifact
  Registry repository, Workload Identity Federation or a service account key, and
  Supabase secrets configured in GitHub or by hand on the Cloud Run service (the
  workflow's env var and secret flags were deliberately removed per your instruction,
  those are now managed by hand in the console, see the comment left in
  `.github/workflows/deploy.yml`'s deploy step). None of this can be done from a coding
  session alone.
- Once deployed, `python -m eval.metrics_latency cold --url <service-url> --n 5
  --wait-seconds 900` measures cold start, kept separate from the warm numbers per
  CLAUDE.md section 13.6.
- Consider widening the config table to the full 55 question dev split, and RAGAS to
  a larger sample, once there is a full day's fresh quota and no deploy or other
  quota-consuming task competing for it.
- The estimated cost per thousand queries row in README section 5 needs the current
  AI Studio price for `gemini-3.1-flash-lite`, deliberately not hardcoded, see
  `eval/metrics_latency.py`'s `estimate_cost_per_thousand`.

## Quota discipline, still applies

The 500 requests per day per key ceiling and the 15 requests per minute ceiling are both
real and both bit this session, again, on both the task and judge keys. `.env` has
multiple `GEMINI_API_KEY_TASK` candidates, only one uncommented at a time; rotate
manually and re-run with `--resume` rather than restarting a config from question 1.
Judge key exhaustion can be worked around the same way, point `GEMINI_API_KEY_JUDGE` at
an unused task key.

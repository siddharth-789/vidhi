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

## Left for the next session

- RAGAS (group 2: faithfulness, answer relevancy, context precision, context recall) is
  wired and smoke tested (`python -m eval.metrics_ragas` passes against a synthetic
  sample) but not yet run against the real saved results. Two attempts today hit the
  judge key's daily quota then the per minute ceiling; the `RunConfig(max_workers=2)` fix
  above should let a modest `--ragas-n` (10 to 20) actually complete. Use
  `python -m eval.backfill_ragas --configs A_vanilla,B_hybrid,C_dspy,D_optimized
  --ragas-n 10`, it reads the already saved `eval/results/{config}_*.json` files and
  fills in `group2_ragas` without re-spending task key quota.
- Latency and production metrics (`eval/metrics_latency.py`, group 3) not yet run. Needs
  the API actually running locally or deployed first (`uvicorn app.api:app`), since it
  measures real `/ask` requests.
- Deployment to Cloud Run and the cold start measurement. Needs GCP project setup,
  Artifact Registry, Workload Identity Federation or a service account key, and Supabase
  secrets in GitHub, none of which can be done from the coding session alone.
- The README currently has the section 4 config table and section 6 optimizer section
  fillable from the numbers above; sections 5 (latency) and the deploy URL in section 3
  stay `TBD` until the two items above happen.
- Consider widening C_dspy and D_optimized to the full 55 question dev split once there
  is a full day's fresh quota and no other quota-consuming task competing for it.

## Quota discipline, still applies

The 500 requests per day per key ceiling and the 15 requests per minute ceiling are both
real and both bit this session. `.env` has multiple `GEMINI_API_KEY_TASK` candidates,
only one uncommented at a time; rotate manually and re-run with `--resume` rather than
restarting a config from question 1. Judge key exhaustion can be worked around the same
way, point `GEMINI_API_KEY_JUDGE` at an unused task key.

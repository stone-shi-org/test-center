# test-center plan

Single-user, internal web app to configure and run test suites as jobs, store results in a DB,
and render/compare them later. Built from scratch; `../tests` is reference only.

## Decisions
- Backend: Python, FastAPI, SQLAlchemy + Alembic, SQLite (Postgres-ready).
- Frontend: React + Vite (TypeScript).
- Deploy: Docker compose (backend serves built frontend; one container + one volume for data).
- Auth: none (internal, single user). Secrets come from env vars, never shown in the UI.
- Test data: `test-data/NOTSOFAR-1` (gitignored), mounted read-only into the container.
- v1 testers: live-stt (all endpoints), embedding, LLM. Rerank/others later via the same plugin API.
- Priority feature: **compare** runs side by side.
- live-stt gRPC: vendor `proto/livestt/v1/asr.proto` (no dependency on `live-stt` code), generate Python stubs in the build (`scripts/gen_proto.sh`), and use the built-in streaming client/harness in `backend/app/testers/live_stt/grpc_client.py`.

## Layout
```
backend/
  app/api/        REST: targets, testers, jobs, runs, results, compare
  app/runner/     DB-backed queue + worker, subprocess per tester run, timeouts, cancel
  app/db/         models + migrations
  app/testers/    plugins: live_stt/ (HTTP + gRPC + pb/), embedding/, llm/
proto/            vendored livestt/v1/asr.proto
scripts/          gen_proto.sh (build-time Python gRPC stub codegen)
frontend/         React/Vite app
test-data/        NOTSOFAR-1 etc. (not committed)
data/             sqlite db + artifacts (not committed)
docs/
docker-compose.yml, Dockerfile
```

## Tester plugin contract
Each tester provides:
- `config_schema` (JSON Schema; drives the UI form)
- `suites`: named sub-suites that can be selected independently
- `run(config, target, ctx) -> Result`; `ctx.log()` / `ctx.progress()` stream live output

`Result`:
- `status`: pass | fail | error
- `metrics`: flat `{name: number}` (e.g. `ttft_ms`, `p95_ms`, `wer`, `der`, `recall@5`, `tps`)
  with optional `unit` and `better: higher|lower` so compare can color regressions
- `cases`: per-case rows `{id, suite, status, metrics, detail}`
- `artifacts`: files saved under data/artifacts/<run_id>/

## Data model
- `target`: name, kind, base_url, auth env-var ref, extra (model name, etc.)
- `test_definition`: tester, suites, config JSON, target_id
- `job`: name, definition ids, schedule (manual | cron), enabled
- `run`: job_id, status, queued/started/finished, config snapshot, target snapshot, git/version info
- `result`: run_id, definition id, tester, status, metrics JSON, summary
- `case_result`: result_id, suite, case_id, status, metrics JSON, detail JSON
- `artifact`: result_id, name, path, mime

Runs snapshot config and target so old results stay comparable after edits.

## Testers
### live-stt
HTTP: `/v1/audio/transcriptions` (JSON + SSE stream), `/v1/audio/diarization`, TTS
(`/v1/audio/speech`), `/v1/models`, `/api/stats`, health, admin endpoints (read-only checks).
gRPC: streaming `Transcribe` - time to first partial, final latency, real-time factor,
behavior on long calls.
Accuracy on NOTSOFAR-1: WER (transcription), DER/speaker count (diarization), per-meeting cases.
Contract checks: status codes, response schema, error handling, concurrency limits.

### embedding
OpenAI-compatible embeddings: latency percentiles, sequential/batched/concurrent throughput,
consistency, vector norms, retrieval Recall@k / MRR on a built-in corpus.

### llm
OpenAI-compatible chat: TTFT, tokens/s, latency; capability cases (coding, tool-call JSON,
constraint following, reasoning, summarization); optional LLM judge via a second target.

## UI
Dashboard (latest status per tester) | Targets | Jobs (form generated from schema) |
Run detail (live log, then results) | History + metric trends |
**Compare** (pick 2+ runs: metric table with deltas and regression highlighting, per-case diff,
side-by-side artifacts).

## Phases
1. Skeleton: repo scaffolding, DB, plugin contract, runner, embedding tester, manual run, run/result pages, Docker.
2. LLM tester + live-stt HTTP endpoint tests.
3. Compare view (metric deltas, per-case diff) - pulled early because it is the priority.
4. Jobs with cron scheduling, history/trend charts.
5. live-stt gRPC + NOTSOFAR-1 accuracy suites.
6. Polish: notifications, artifact viewer, optional Bamboo hook.

## Open items
- ~~NOTSOFAR-1 layout and which subset/split to use (decide once copied in).~~ **Resolved (TES-32 / TES-33):** `test-data/NOTSOFAR-1` contains `eval_set/240629.1_eval_small_with_GT` (80 meetings, default `eval_small` split), `eval_set/240825.1_eval_full_with_GT` (129 meetings, `eval_full`), `dev_set/240825.1_dev1` (36 meetings, `dev`), and `train_set/240825.1_train` (72 meetings, `train`). Each `MTG_*` directory provides `gt_meeting_metadata.json`, `gt_transcription.json`, and single-channel far-field 16kHz mono 16-bit PCM WAVs (`sc_meetup_0/ch0.wav`, `sc_rockfall_0/ch0.wav`), loaded by `backend/app/testers/live_stt/dataset.py` and scored by `backend/app/testers/live_stt/accuracy.py` (`accuracy_wer`, `accuracy_streaming`, `accuracy_diarization` with SQLite result cache).
- ~~Where the live-stt proto/client comes from (vendor the .proto vs. pip-install the client).~~ **Resolved (TES-30):** Vendored `proto/livestt/v1/asr.proto` (no dependency on `live-stt` code), generate Python protobuf/gRPC stubs in the build (`scripts/gen_proto.sh` into `backend/app/testers/live_stt/pb/`), and use the built-in streaming client & harness (`backend/app/testers/live_stt/grpc_client.py`).



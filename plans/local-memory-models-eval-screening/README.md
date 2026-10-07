# LME η screening sweep (task 3720)

This directory holds the evidence behind η's screening verdict for PRD
`plans/local-memory-models-eval-prd.md`. The verdict sorts every `arms.yaml` LLM arm
under the survivor rule of `plans/local-memory-models-eval-preregistration.md` §8. This
file is provenance only: what ran, where, at which pins, and what happened to it. The
rule lives in the preregistration doc and in `fused_memory.arm_harness.screening`. The
findings are in `plans/local-memory-models-eval-screening-report.md`, which carries the
delivered-check marker; this file spells it only as `PRD[-]MARKER`.

`fused-memory/tests/arm_harness/test_lme_screening_artifacts.py` re-derives
`screening-verdict.json` from the committed evidence and requires it to be
byte-identical. It also ties the evidence to the committed `arms.yaml` slate, the
controls' code sha and corpus, and the preregistration sha. It carries no integration
marker, so the merge lane runs it.

## Layout

The layout's one definition is
`fused-memory/src/fused_memory/arm_harness/screening_evidence.py::ArmEvidencePaths`.
The driver writes through it and the loader reads through it.

| Path | What it is |
|---|---|
| `specs/<arm>.json` | The candidate `ArmSpec` each arm was screened under, from `slate.candidate_spec` |
| `arms/<arm>/commands.json` | Every stage's `CommandRecord`: argv, exit code, start and finish times, the last 4000 characters of output. Also the tap binding |
| `arms/<arm>/health.json` | α's `lms_healthcheck --arm <arm> --output` report, schema v6, taken after the run with the arm still up |
| `arms/<arm>/smoke-calls.jsonl` | The usage tap's record of the smoke's calls |
| `arms/<arm>/calls.jsonl` | The usage tap's record of every call the run made, one `CallRecord` per call |
| `runs/<arm>/<stamp>/` | The pinned harness run: `run.json`, `outcomes.jsonl`, `metrics/`, and `abort.json` when INV-4 stopped it. The run-local `journal/` is deliberately not committed |
| `release.json` | The sweep's one leading `lms_ctl stop-all` |
| `screening-verdict.json` | `harness.py screen` over this directory |

## Pins

- code_sha `055e9c15a0d756704017b0d77ffbe4185155402f`, the controls' (prereg §1). Every
  smoke, run and teardown came from a clean checkout at that sha. Each run's
  `code-sha-matches-checkout` and `preregistration-sha` pre-run checks passed and are
  recorded in its `run.json`.
- corpus_sha `850cf7c937c745b0d224a0ba975efc1c69080ad61d84c5369d06287cdd16f2e8`, the
  bytes of `fused-memory/scripts/local_memory_models_eval/corpus_manifest.json`. They
  are identical at the pin and on this branch. The driver refused to start unless they
  hashed to `preregistration-inputs.json`'s `corpus_sha`.
- preregistration_sha `865a017d79c6a210da157474e639fd15fe51f9e0`, which was
  `git log -1 --format=%H main -- plans/local-memory-models-eval-preregistration.md`
  at submit time.
- Run shape: the first 20 manifest episodes, `--concurrency 3`,
  `--index-configuration with-indices` (`screening_evidence.SCREENING_RUN_SHAPE`).
  The episode timeout was 120 s. The params are control A's (temperature 0.0,
  max_tokens 4096), read from
  `plans/local-memory-models-eval-controls/specs/incumbent-generic-a.json`.
- Tap port 8418. Every spec's `base_url` is `http://127.0.0.1:8418/v1`.
- Reasoning mode per arm, as `arms.yaml` declares and ζ ruled: qwen3.5-9b `on` with
  `--reasoning-parser qwen3`, phi-4-14b `off`, moe-stretch `off`. η did not vary it.
  The loader refuses any α health row whose `reasoning` differs from the manifest's.

## How it ran

One transient unit, submitted from this branch's worktree on 2026-10-07:

```bash
uv run --project fused-memory python fused-memory/scripts/local_memory_models_eval/screen_slate.py \
  --evidence-root /tmp/lme-eta-3720/evidence --pinned-checkout /tmp/lme-eta-3720/code \
  --preregistration-sha 865a017d79c6a210da157474e639fd15fe51f9e0
```

This ran `systemd-run --user --unit=lme-eta-screen --collect`. A bare
`--setenv=OPENAI_API_KEY` imported the key, so it never appeared on a command line.
The unit's stdout and stderr were appended to `/tmp/lme-eta-3720/lme-eta-screen.log`.
Inside the unit, `screen_slate.py --in-unit` ran:

1. one `lms_ctl stop-all`;
2. per arm, in manifest order: write the spec; `lms_ctl start`; `lms_ctl wait-ready`;
   the pinned `harness.py smoke`; the pinned `harness.py run`; `lms_healthcheck --arm`;
   then, in `finally`, `lms_ctl stop` and the pinned `harness.py teardown`.

The smoke and the run each went through their own usage-tap session.

Teardown runs once the run stage has been reached, because the run is the only stage
that writes the scratch graph. An arm that is never served is stopped, but not torn
down. The spec is written before `start` for every arm, so an unserved arm's evidence
validates just as a served arm's does.

The pinned checkout was `git clone --shared` of the main checkout into
`/tmp/lme-eta-3720/code`, at 055e9c15a0, with its own `uv sync --all-packages --frozen`
venv. The plan named `git worktree add --detach`, but the agent sandbox refused writes
to the main checkout's `.git/worktrees/`. A shared clone is the same clean tree at the
same sha and creates no ref in the main repository. Outputs went to
`/tmp/lme-eta-3720/evidence`, outside the clone, because `harness.py run` refuses a
dirty tree, untracked files included. They were copied here with
`rsync -a --exclude journal/`.

Wall clock, from the command records:

| Arm | First command → last (UTC) | Wall | wait-ready | run | Run stamp | Run exit |
|---|---|---|---|---|---|---|
| (release) | 19:59:08 → 19:59:09 | <1 s | | | | |
| qwen3.5-9b | 19:59:08 → 20:14:51 | 942 s | 571.9 s | 247.0 s | `20261007T200908Z` | 4 (INV-4 abort) |
| phi-4-14b | 20:14:51 → 20:24:38 | 587 s | 293.6 s | 263.2 s | `20261007T201954Z` | 0 |
| moe-stretch | 20:24:38 → 20:32:46 | 488 s | 163.5 s | 245.7 s | `20261007T202735Z` | 4 (INV-4 abort) |

The whole sweep took about 34 minutes. Every start, wait-ready, smoke, healthcheck,
stop and teardown exited 0. Both aborts were 5 consecutive `TimeoutError` episodes at
the 120 s budget. Each left a complete `run.json` and `abort.json`, so both are
results.

## Re-measures

None. Every arm was served and left a `run.json`, a clean VRAM reading (`pollution`
CLEAN, whisper-writer resident as the baseline consumer) and a full set of command
records. Under step 14's triage rules nothing was invalid, so nothing was re-screened.

One instrument artifact is recorded rather than re-measured. phi-4-14b's
`calls.jsonl` ends with a status-502 record. The tap synthesized it: "upstream
127.0.0.1:8412 failed: RemoteDisconnected". The call started at 20:23:34. The run
stage returned at 20:24:14 with that call still being served for an episode the harness
had already abandoned. `lms_ctl stop` (20:24:20–20:24:33) then stopped the server under
it, after 57.4 s. Under the rule as implemented, any own-model call without a reported
prompt length makes the context-fit gate UNMEASURED, so phi-4-14b's context-fit gate is
UNMEASURED. It does not change phi-4-14b's survival, because its throughput gate fails
on its own. The report gives the numbers.

## The usage tap: a deviation from θ's direct path

θ's runs talk to the arm directly. Here every call went through the in-process usage
tap, `fused_memory.arm_harness.usage_tap`, so that gate 3 could read the server's own
`usage.prompt_tokens` per call. The pinned harness records only per-episode token
sums. The tap forwards bytes unchanged and refuses streamed requests. It logs one
`CallRecord` per call, and the record is written before the reply is sent.

The tap's per-call `duration_ms` times the whole upstream exchange as the tap saw it.
The shortest phi-4-14b call took 518 ms end to end and its median call 2507 ms, against
an episode-latency p50 of 29050 ms. Whatever the tap adds is therefore a small fraction
of even the shortest call. Tap overhead was not measured separately.

When the harness cancels a client mid-call, the tap's reply finds a closed socket. The
unit log then shows `BrokenPipeError` tracebacks from the tap's handler threads. The
call's record was already written. A call still in flight when its tap session closes
completes on its daemon thread and still appends its record. That is why
`calls.jsonl` can hold calls that finished after the run stage returned.

## After the sweep

- Teardown: each arm's `teardown` record exited 0 (`deleted 'evalmem_lme_eta_<arm>'`),
  and `GRAPH.LIST` on FalkorDB listed no `lme_eta` graph afterwards.
- Frozen reference: `harness.py topology --graph evalmem_lme_ref_incumbent_a`, run from
  this branch afterwards, was byte-identical to
  `plans/local-memory-models-eval-controls/frozen-reference.json` (1834 nodes, 3101
  edges).
- `lms_ctl active` was empty, and the card held only whisper-writer (4050 MiB).
- `screening-verdict.json` was derived with:

```bash
cd fused-memory && uv run python scripts/local_memory_models_eval/harness.py screen \
  --evidence-root ../plans/local-memory-models-eval-screening \
  --arms-manifest ../scripts/local-model-serving/arms.yaml \
  --preregistration-inputs ../plans/local-memory-models-eval-controls/preregistration-inputs.json \
  --reference-outcomes ../plans/local-memory-models-eval-controls/runs/incumbent-generic-a/20261007T011000Z/outcomes.jsonl \
  --out ../plans/local-memory-models-eval-screening/screening-verdict.json
```

It exited 0 with zero survivors: a negative verdict is a result, not an error.

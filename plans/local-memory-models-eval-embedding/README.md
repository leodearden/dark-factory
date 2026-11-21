# LME embedding-axis runs (task 3722, ι)

The live embedding-arm runs behind `plans/local-memory-models-eval-embedding-report.md`,
for PRD `plans/local-memory-models-eval-prd.md`. This file is provenance only: what ran,
where, at which pins, at what cost. The rule they are judged by is
`plans/local-memory-models-eval-preregistration.md` §6, and the verdict is the report's.

`fused-memory/tests/arm_harness/test_lme_embedding_artifacts.py` re-derives
`embedding-preregistration-inputs.json` from the two committed control runs and requires
it byte-identical. It also requires each committed spec to equal the slate's reading of
`scripts/local-model-serving/arms.yaml`, and pins the six runs' shared code and corpus
shas, their instrument checks, the controls-first order and the probe set's frozen
reference. It carries no integration marker, so the merge lane runs it.

## Layout

| Path | What it is |
|---|---|
| `specs/<arm>.json` | The six `EmbeddingArmSpec`s as run, written by `harness.py embed-specs` |
| `probe-set.json` | `harness.py probe-set`'s output; its sha256 is every spec's `corpus_sha` |
| `runs/<arm>/<stamp>/` | Each run's `run.json` (the `EmbeddingRunManifest`, written last) and `metrics/` |
| `embedding-preregistration-inputs.json` | `harness.py embed-preregister` over the two control runs |

Not committed, and why:
- the Mem0 snapshot `mem0-source.jsonl` (31237 records, 27.06 M characters of live
  memory text); `probe-set.json` pins it by sha256 and point count;
- the transcript corpus (10 MB, mined from the agent-transcript archive);
  `probe-set.json` pins its sha256 and carries the 500 queries the latency probe used;
- the scratch graphs and Qdrant replicas, all torn down after the runs.

## Pins shared by every run

- code_sha `8bb8c8893b0f58957753954ab0338d47f4b609b9`: the clean task-branch commit that
  added the contract test (plan step 27), with the tree clean, from the transcript mining
  through the teardown. See "The code sha and the rebase" below.
- corpus_sha `1eb3ed883afb5396fcba47a240e41f4b501d5b678789c48541325eb7c811b287`: the bytes
  of `probe-set.json`.
- preregistration_sha `be4a44d6243b58a3293d66eb73f0ba56fa206efd` on the four candidates:
  the erratum commit Leo's esc-3722-3 ruling (2026-10-08) binds ι runs to. The controls
  carry none.
- The probe set: query words K = 10, 199 known items (one of the 200 reference episodes is
  cited by no edge and is left out), 500 transcript queries, 42 Mem0 known items over 14
  topics.
- The Mem0 snapshot: `fused_dark_factory`, 31237 points, sha256
  `7114411c67aaec1e60eea176465c167413723e2712e48e9ff3d5834c6f0359b7`, taken read-only on
  2026-10-09.
- The transcript corpus: `corpus-20261009T091224Z.jsonl`, 6191 searches mined from 18648
  archived transcripts across 2228 tasks, exit 0.
- Settings, from every `run.json`: embed batch 64 at concurrency 4, query concurrency 3,
  search k 10, search timeout 30.0 s (`fused-memory/config/config.yaml`, read through the
  default `CONFIG_PATH` from `fused-memory/`).

## The code sha and the rebase

The specs, the probe set and every run were written to `/tmp/lme-iota-3722/` while HEAD
stood at `8bb8c8893b`, because the harness refuses a dirty tree. After the teardown and
before this copy was committed, the orchestrator rebased the task branch from main
`03644d01d9` onto `38de0a21e7`, which rewrote the step-27 commit as `c7a8f7626d`. A
task-branch sha does not survive a rebase, so `8bb8c8893b` may not be reachable from main.

The code under test did not change. `git diff --stat 8bb8c8893b c7a8f7626d` names four
files, none of them read by a run: `orchestrator/src/orchestrator/agents/roles.py`,
`skills/merge-queue/SKILL.md`, `skills/unblock-low-risk/SKILL.md` and
`skills/unblock/SKILL.md`. Git tree hashes survive a rebase, so they are the durable pin.
These four were identical at both commits:

| subtree | git tree hash |
|---|---|
| `fused-memory/` | `1c57e3be6af89b306e870f4c4517d61f9d314bff` |
| `shared/` | `4746159f1be095f00761b486d58769879f008e91` |
| `scripts/local-model-serving/` | `a657f83c0cb5b597e5cbfc2ac8f35a454b4b60c8` |
| `plans/` | `c4cae89f7348ad57978d9f6db693b03eaf14f604` |

`git rev-parse <commit>:fused-memory` prints the first row at any commit whose
`fused-memory/` is the code that ran.

## The runs

All six ran one at a time, never concurrently, each in its own transient
`systemd --user` unit, and the two controls ran before any candidate. Every run exited 0
with every instrument check PASS. Wall-clock is `run.json`'s `started_at` → `finished_at`,
on 2026-10-09.

| arm | role | model @ dims | serving | harness unit | stamp | started → finished (UTC) | wall-clock |
|---|---|---|---|---|---|---|---|
| incumbent-embed-a | control | text-embedding-3-small @ 1536 | openai | `lme-iota-incumbent-embed-a` | `20261009T091930Z` | 09:19:34 → 09:25:41 | 6 min 06 s |
| incumbent-embed-b | control | text-embedding-3-small @ 1536 | openai | `lme-iota-incumbent-embed-b` | `20261009T092727Z` | 09:27:30 → 09:33:31 | 6 min 00 s |
| qwen3-embedding-0.6b | candidate | qwen3-embedding-0.6b @ 1024 | vllm, `lms-arm@qwen3-embedding-0.6b.service` | `lme-iota-qwen3-embedding-0.6b` | `20261009T094148Z` | 09:41:51 → 09:47:05 | 5 min 13 s |
| granite-embedding-english-r2 | candidate | granite-embedding-english-r2 @ 768 | vllm, `lms-arm@granite-embedding-english-r2.service` | `lme-iota-granite-embedding-english-r2` | `20261009T095139Z` | 09:51:42 → 09:54:21 | 2 min 38 s |
| qwen3-embedding-4b | candidate | qwen3-embedding-4b @ 2560 | vllm, `lms-arm@qwen3-embedding-4b.service` | `lme-iota-qwen3-embedding-4b` | `20261009T100100Z` | 10:01:03 → 10:22:40 | 21 min 37 s |
| gte-modernbert-base | candidate | gte-modernbert-base @ 768 | vllm, `lms-arm@gte-modernbert-base.service` | `lme-iota-gte-modernbert-base` | `20261009T102653Z` | 10:26:56 → 10:30:16 | 3 min 20 s |

The pre-registration was fixed before any candidate ran. `embed-preregister` first wrote
`embedding-preregistration-inputs.json` at 09:35 UTC, over the scratch copies of the
control runs. The committed file was re-derived over the committed copies and is
byte-identical to that one.

Spend. The four candidates ran on local vLLM units and spent nothing metered. The
manifests record no token counts, so the controls' spend is an estimate. Each control
embedded the snapshot's 27.06 M characters, plus the graph's 2721 names and facts and the
probe queries. At about 4 characters per token that is about 6.8 M tokens per run, or
$0.14 at the $0.02 per 1M tokens the PRD records for text-embedding-3-small. That is
about $0.27 for the pair.

## Commands

Everything ran from `fused-memory/` in this worktree, at the code sha with the tree clean,
with `Z=/tmp/lme-iota-3722`. `harness.py` is
`uv run --no-sync python scripts/local_memory_models_eval/harness.py`. The transient unit
for steps (a) to (c) was `lme-iota-transcripts`, `lme-iota-mem0-snapshot` and
`lme-iota-probe-set` respectively.

```bash
# (a) the transcript corpus, read-only
uv run --no-sync python scripts/memory_eval_transcript_corpus.py --out-root $Z/transcript-corpus

# (b) the Mem0 snapshot, read-only
harness.py mem0-snapshot --collection fused_dark_factory --out $Z/mem0-source.jsonl

# (c) the probe set
harness.py probe-set \
  --reference-json ../plans/local-memory-models-eval-controls/frozen-reference.json \
  --control-a-outcomes ../plans/local-memory-models-eval-controls/runs/incumbent-generic-a/20261007T011000Z/outcomes.jsonl \
  --transcript-corpus $Z/transcript-corpus/transcript-corpus/corpus-20261009T091224Z.jsonl \
  --mem0-snapshot $Z/mem0-source.jsonl \
  --registry tests/fixtures/memory_eval_topic_registry.json --out $Z/probe-set.json

# (d) the six specs
harness.py embed-specs --arms-manifest ../scripts/local-model-serving/arms.yaml \
  --probe-set $Z/probe-set.json --code-sha 8bb8c8893b0f58957753954ab0338d47f4b609b9 \
  --preregistration-sha be4a44d6243b58a3293d66eb73f0ba56fa206efd --out-dir $Z/specs

# (e)/(f) each run, one at a time; <arm> per the table above
systemd-run --user --unit=lme-iota-<arm> --collect --wait \
  --working-directory=<worktree>/fused-memory --setenv=OPENAI_API_KEY \
  -p StandardOutput=append:$Z/logs/<arm>.log -p StandardError=append:$Z/logs/<arm>.log \
  /home/leo/.local/bin/uv run --no-sync python scripts/local_memory_models_eval/harness.py embed-run \
    --arm-spec $Z/specs/<arm>.json --probe-set $Z/probe-set.json \
    --mem0-snapshot $Z/mem0-source.jsonl --out-root $Z/runs

# (e) after both controls, before any candidate
harness.py embed-preregister --run-a $Z/runs/incumbent-embed-a/<stamp> \
  --run-b $Z/runs/incumbent-embed-b/<stamp> --out $Z/embedding-preregistration-inputs.json

# (f) each candidate's serving unit, around its run (from the repo root)
LMS_BASELINE_DIR=$Z/lms-baselines uv run --no-sync --project shared python scripts/local-model-serving/lms_ctl.py start <arm>
uv run --no-sync --project shared python scripts/local-model-serving/lms_ctl.py wait-ready <arm>
uv run --no-sync --project shared python scripts/local-model-serving/lms_ctl.py stop <arm>

# (g) the comparison, the teardown of all six graphs and replicas, the frozen-reference re-check
harness.py embed-compare --preregistration $Z/embedding-preregistration-inputs.json \
  --run $Z/runs/<candidate>/<stamp>   # once per candidate
harness.py teardown --arm-spec $Z/specs/<arm>.json --collection
harness.py topology --graph evalmem_lme_ref_incumbent_a   # byte-identical to frozen-reference.json
```

`lms_ctl start` refuses an exclusive start while any other arm unit is running, and every
start here was exclusive. So no LLM arm was resident beside qwen3-embedding-4b, and its
VRAM baseline (09:56:44 UTC) records no co-resident arm. `start` writes that baseline file
before it starts the unit. Its default home, `$XDG_RUNTIME_DIR/lms-baselines`, is not
writable from the agent sandbox, so the runs set `LMS_BASELINE_DIR` to the scratch dir,
the override `lms_vram.py` documents for exactly that. It moves where the reading is
stored, not what passes.

A bare `--setenv=OPENAI_API_KEY` imports the caller's value, so the key never appeared on
a command line.

After the teardown, `GRAPH.LIST` held no `evalmem_lme_emb_*` graph and Qdrant no
`evalmem_lme_emb_*` collection. The frozen reference re-hashed byte-identical to
`plans/local-memory-models-eval-controls/frozen-reference.json`: 1834 nodes, 3101 edges,
`56de651d10af…`.

## Self-check

The report is the one carrier of its PRD marker:

```bash
git grep -cE 'PRD[-]MARKER:local-memory-models-eval embedding[-]report' -- plans/
```

The expected output names exactly one file,
`plans/local-memory-models-eval-embedding-report.md`. This README spells the marker only
in its bracketed form, so it does not satisfy the check it documents.

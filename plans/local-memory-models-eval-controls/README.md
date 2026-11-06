# LME incumbent control runs (task 3719, ζ)

The live incumbent control runs that `plans/local-memory-models-eval-preregistration.md`
is derived from, for PRD `plans/local-memory-models-eval-prd.md`. This file is
provenance only: what ran, where, at which pins, at what cost. The rules built on
these runs live in the preregistration doc, and the numbers it quotes come from
`preregistration-inputs.json` here.

`fused-memory/tests/arm_harness/test_lme_control_artifacts.py` re-derives
`preregistration-inputs.json` from the committed runs A and B and requires it to be
byte-identical. It also pins the four runs' completeness, code sha and corpus sha, the
parity deltas and the frozen reference. It carries no integration marker, so the merge
lane runs it.

## Layout

| Path | What it is |
|---|---|
| `specs/<arm>.json` | The four control `ArmSpec`s as run |
| `runs/<arm>/<stamp>/` | Each run's `run.json`, `outcomes.jsonl` and `metrics/`; B also has `graph_sameness_details.json`. The run-local `journal/` (sqlite) is deliberately not committed |
| `runs/incumbent-generic-20/<stamp>/parity/incumbent-openai-20/metrics/` | Client-class parity deltas, GenericClient − OpenAIClient, from `harness.py parity-check` |
| `preregistration-inputs.json` | `harness.py preregister` over the committed A and B |
| `frozen-reference.json` | `harness.py topology` over A's graph, `evalmem_lme_ref_incumbent_a`, when it was frozen |

## Pins shared by every run

- code_sha `055e9c15a0d756704017b0d77ffbe4185155402f`: the clean branch base, already on
  `main` ("Merge task/3718 into main"), so θ can check it out.
- corpus_sha `850cf7c937c745b0d224a0ba975efc1c69080ad61d84c5369d06287cdd16f2e8`: the bytes of
  δ's committed `fused-memory/scripts/local_memory_models_eval/corpus_manifest.json`,
  N = 200. `build_corpus.py --verify` exited 0 before the runs.
- model `gpt-4o-mini` on `https://api.openai.com/v1` (`OPENAI_API_URL` was unset),
  `structured_output_mode` `json_schema`, temperature 0.0, max_tokens 4096.
- `--concurrency 3` (production's queue `semaphore_limit` default) and
  `--index-configuration with-indices`. The episode timeout was 120 s
  (`queue.backend_write_timeout_seconds`), and graphiti's semaphore limit was 20.
- Pricing: $0.15 per 1M input tokens and $0.60 per 1M output tokens, gpt-4o-mini Standard
  tier, read from https://developers.openai.com/api/docs/pricing (where
  platform.openai.com/docs/pricing redirects) on 2026-10-07.

## The runs

All four ran sequentially, never concurrently, each in its own transient
`systemd --user` unit. Every run exited 0 with every instrument check PASS, and every
episode was ok. Wall-clock is `run.json`'s `started_at` → `finished_at`. Spend is
`usd-per-episode` × its `n`.

| arm | client class | scratch graph | stamp | episodes | unit | wall-clock | spend |
|---|---|---|---|---|---|---|---|
| `incumbent-generic-a` | `openai_generic` | `evalmem_lme_ref_incumbent_a` (FROZEN) | `20261007T011000Z` | 200 | `lme-zeta-incumbent-generic-a` | 1490.3 s | $0.5473 |
| `incumbent-generic-b` | `openai_generic` | `evalmem_lme_ctl_incumbent_b` (torn down) | `20261007T013800Z` | 200 | `lme-zeta-incumbent-generic-b` | 1531.8 s | $0.5529 |
| `incumbent-generic-20` | `openai_generic` | `evalmem_lme_ctl_generic_20` (torn down) | `20261007T020600Z` | first 20 | `lme-zeta-incumbent-generic-20` | 128.1 s | $0.0427 |
| `incumbent-openai-20` | `openai` | `evalmem_lme_ctl_openai_20` (torn down) | `20261007T020900Z` | first 20 | `lme-zeta-incumbent-openai-20` | 131.6 s | $0.0432 |

Measured spend was $1.19 across the four runs. The two endpoint smokes that preceded them
were one request each.

For a θ arm on this corpus at the same concurrency, the worst case inside the envelope is
N × p95 bound / concurrency = 200 × 60 s / 3 = 4000 s, about 67 min per arm. The
incumbent took about 25 min.

## Commands

Everything ran from `fused-memory/` in this worktree, with the tree clean at the code
sha. `harness.py run` refuses otherwise, counting untracked files, so specs and run
outputs were written outside the repo, to `/tmp/lme-zeta-3719/`, and copied here
afterwards. The plan named `$HOME/.local/share/lme-eval/zeta/`, but the agent sandbox
could not write there; nothing else differs.

```bash
# endpoint smokes: one schema-constrained request plus the negative control, no graph written
uv run --no-sync python scripts/local_memory_models_eval/harness.py smoke --arm-spec $Z/specs/incumbent-generic-a.json
uv run --no-sync python scripts/local_memory_models_eval/harness.py smoke --arm-spec $Z/specs/incumbent-openai-20.json

# each replay, one at a time; <arm>, <stamp> and [extra] per the table above
systemd-run --user --unit=lme-zeta-<arm> --collect --wait \
  --working-directory=<worktree>/fused-memory \
  --setenv=OPENAI_API_KEY --setenv=MEMORY_EVAL_RUN_STAMP=<stamp> \
  -p StandardOutput=append:$Z/logs/<arm>.log -p StandardError=append:$Z/logs/<arm>.log \
  /home/leo/.local/bin/uv run --no-sync python scripts/local_memory_models_eval/harness.py run \
    --arm-spec $Z/specs/<arm>.json \
    --manifest scripts/local_memory_models_eval/corpus_manifest.json \
    --out-root $Z/runs --concurrency 3 --index-configuration with-indices [extra]
#   incumbent-generic-b:  [extra] = --reference-outcomes $Z/runs/incumbent-generic-a/20261007T011000Z/outcomes.jsonl
#   incumbent-generic-20, incumbent-openai-20:  [extra] = --limit 20

# instrument and parity checks, both exit 0
harness.py control-check --run <A> --run <B> --reference-outcomes <A>/outcomes.jsonl
harness.py parity-check --run-a <generic-20> --run-b <openai-20>

# derived artifacts, over the COMMITTED copies so the contract test re-derives the same bytes
harness.py preregister --run-a runs/incumbent-generic-a/20261007T011000Z \
  --run-b runs/incumbent-generic-b/20261007T013800Z --out preregistration-inputs.json
harness.py topology --graph evalmem_lme_ref_incumbent_a > frozen-reference.json

# the three non-frozen graphs, then a re-check that the frozen hash did not move
harness.py teardown --arm-spec $Z/specs/incumbent-generic-b.json    # likewise -20 and openai-20
harness.py topology --graph evalmem_lme_ref_incumbent_a             # byte-identical to frozen-reference.json
```

A bare `--setenv=OPENAI_API_KEY` imports the caller's value, so the key never appeared
on a command line.

control-check passed `arm-config-symmetry`, `single-code-sha`, `token-cost-accounting`
for both arms, and `reference-nonempty` (200 ok reference episodes holding 1997
entities).

## The frozen reference graph

`evalmem_lme_ref_incumbent_a` is ι's reference graph. From the moment run A finished,
nothing may write to it, index-probe it, re-index it or tear it down. `frozen-reference.json`
records its node count, edge count and topology hash. ι re-runs
`harness.py topology --graph evalmem_lme_ref_incumbent_a` before re-embedding and requires
output byte-identical to that file. A mismatch means the reference has moved, and ι
escalates rather than proceeding.

## Self-check

The preregistration doc is the one carrier of its PRD marker:

```bash
git grep -cE 'PRD[-]MARKER:local-memory-models-eval preregistration' -- plans/
```

The expected output names exactly one file, `plans/local-memory-models-eval-preregistration.md`.
This README spells the marker only in its bracketed form, so it does not satisfy the
check it documents.

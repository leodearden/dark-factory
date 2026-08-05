# E1 retrieval-health — first live run (2026-08-05)

Frozen provenance for the **first ever live run** of the E1 retrieval-health
probe (`docs/prds/memory-eval-program.md` leaf β, task 3208). Laid out as the
real runtime tree — `<root>/<eval_id>/metrics-<STAMP>.json` — so this
directory is itself a valid artifact ROOT that
`shared.memory_eval_metrics.eval_dir` / `metrics_artifact_path` resolve
against unchanged, and any artifact-only consumer can be pointed at it as-is.

| | |
|---|---|
| eval_id | `e1-retrieval-health` |
| run_stamp | `20260805T093831Z` |
| schema_version | 1 (M1 metric series) |
| project scope | `dark_factory+reify` (the probe's default `--project-id` set) |
| mode | read-only — the probe searches, it never writes to memory |
| files | `e1-retrieval-health/metrics-20260805T093831Z.json`, `e1-retrieval-health/report-20260805T093831Z.txt` |

Produced by the probe at its defaults (default `--project-id`, `--registry`,
`--out-root` and `--k`), from the repo root:

```bash
uv run python fused-memory/scripts/memory_eval_retrieval_probe.py
```

Contract-tested by `fused-memory/tests/test_memory_eval_e1_first_live_run.py`.

## Why it is committed here

The probe's `--out-root` defaults to `fused-memory/data/memory-evals/`, and
`fused-memory/.gitignore:9` ignores `data/`. **Nothing under that root is or
can be tracked.** So this run came to rest on exactly one host, on a path that
no fresh clone, no second host, and nothing surviving `git clean -xdf` can
reach — while three PRD provenance citations already pointed readers at it
(`docs/prds/memory-eval-program.md`, `docs/prds/memory-briefing-and-fusion.md`
and its capability manifest; the latter two rest their whole evidence base on
this run). Merging task 3208 preserved the *probe*; it preserved none of the
*measurement*.

This frozen copy is what makes 3208's user-observable signal — "a live
read-only run produces the M1 artifact plus an initial-state report" —
independently checkable after the fact, by someone who was not there.

`plans/` is this repo's existing home for committed real-measurement
provenance: the probe already reads one from here
(`DEFAULT_CENSUS_PATH = _REPO_ROOT / 'plans' / 'memory-metadata-census-report.json'`),
alongside `plans/bug-hotspot-survey-2026-07-06-full-findings.json` and the
confusion censuses. The probe points back at this directory as
`COMMITTED_EXEMPLAR_ROOT`, one line away from `DEFAULT_CENSUS_PATH`, so the
two provenance pointers read as one convention.

## It is frozen provenance — NOT a test fixture, NOT live data

Both of the obvious places to "tidy" this into are actively harmful. Neither
move is a matter of taste:

1. **Do NOT move it under `fused-memory/data/memory-evals/`.** Beyond being
   gitignored (so it could not be committed there at all), un-ignoring it
   would be *worse than the bug it fixes*. `is_initial_run()`
   (`memory_eval_retrieval_probe.py`) globs `metrics-*.json` under exactly
   that tree, so a committed artifact there would make **every fresh clone
   look like it had already run** and would permanently burn the one-shot D1
   initial-state snapshot for the next genuine first run. The probe's
   `COMMITTED_EXEMPLAR_ROOT` is deliberately disjoint from `DEFAULT_OUT_ROOT`,
   and that disjointness is asserted in the test module named above.

2. **Do NOT fold it into `shared/tests/fixtures/memory_eval/`.** That corpus is
   hand-authored *synthetic* exemplars whose pooled anchors are hard-coded
   downstream (proportion 72/90, λ=5, superseded 6/90, derived α = 1/360 from
   exactly 4 alarm-eligible metrics). `shared/tests/test_memory_eval_boundary.py`
   collects the tree by glob and asserts `len(_SERIES_FILES) == 8`, and
   `dashboard/tests/test_memory_evals_data.py` mtime-guards it read-only.
   Dropping a real 6-metric run (n = 98/32/490, a different metric set) into
   any `e1-*` directory there breaks the count guard and perturbs the α
   anchors.

## The contents are KNOWN-BAD by design

Per the report's own D1 preamble, this is the **initial-state snapshot**: no
prior run existed under the artifact root, so what it records is what the
corpus looked like on 2026-08-05, inherited from the 3111/3112 fix lineage
(canonical pinning, consolidation, curator gates). It is **not a finding and
not a regression anyone introduced.** The headline numbers, so a skimmer is
not alarmed:

| metric | kind | value |
|---|---|---|
| `topic-canonical-present` | tripwire | **32/32 items failing** |
| `canonical-in-top-5` | proportion | 0.0204 (2/98) |
| `canonical-in-top-10` | proportion | 0.0306 (3/98) |
| `canonical-in-top-5-held-out` | proportion | 0.0 (0/32) |
| `claim-recall` | proportion | 0.0625 (2/32) |
| `contamination-share` | proportion | 0.0 (0/490) — see the gaps below |

Read the failing-topic list together with the report's "which store served the
query" section before concluding anything about findability: a phrasing the
read router sent to a store the canonical does not live in cannot hit however
healthy retrieval is, and the rate alone cannot tell a routing fact from a
corpus one.

## Known instrument gaps — out of scope here

This run exposed three limitations **of the instrument**, recorded as a
follow-up leaf in `docs/prds/memory-eval-program.md` §9:

- `contamination-share` counts only registry-topic-foreign results, and
  489/490 scored results were un-topiced and therefore excluded — the 0.0
  above is near-vacuous, not a clean bill of health.
- the canonical matcher matched only 6/196 trials by content-hash (wants a
  fuzzy content-prefix fallback).
- `superseded-above-successor` went unmeasured — 0 comparable pairs, because
  the registry has no `supersedes_pairs` populated.

Task 3694 preserves this measurement; it does not improve the instrument.
Fixing the above belongs to that follow-up leaf.

## Do not hand-edit

`test_memory_eval_e1_first_live_run.py` asserts that
`serialize_metric_series` re-emits the committed metrics file **byte for
byte** — the same anti-drift guard the synthetic corpus uses
(`shared/tests/fixtures/memory_eval/README.md` §"Regenerating — do not
hand-edit"). Both files here are verbatim copies of what the probe wrote; the
metrics file was already in exact canonical form
(`indent=2, sort_keys=True, ensure_ascii=False` plus a trailing newline).

If something here genuinely must change, regenerate it through
`shared.memory_eval_metrics.write_metric_series` rather than reconciling the
bytes by hand. Note that the run itself **cannot be reproduced**: it needs a
live Qdrant and an `OPENAI_API_KEY`, the corpus has moved on, and because the
live artifact root is no longer empty `is_initial_run()` now returns False —
so a re-run would emit no initial-state section at all. These bytes are the
only copy of that evidence.

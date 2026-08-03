# Startup-completion artifact corpus

The empirically-derived evidence for the two-regime watchdog startup grace
(PRD `plans/server-side-api-error-handling-prd.md`, contract **C5**, consumer
task **3326**). Every row describes one real observation of a `CLAUDE_CONFIG_DIR`
at a point during CLI startup, in enough detail to be rebuilt on disk and run a
predicate against.

| File | What it is |
|---|---|
| `startup_completion_healthy.json` | Curated rows for the `healthy` regime — a real `claude --print` run, sampled from spawn through first token. |
| `startup_completion_wedge.json` | Curated rows for the `wedge` regime — the three PRD-named wedge shapes plus the reader-side `transcript_unreadable` degrade. |
| `startup_completion_probe_raw.jsonl` | The raw capture: one redacted observation per line, 37 samples across 5 runs. Every curated row is a distillation of one of these lines. |

Loaded by **`shared/tests/startup_completion_fixtures.py`**
(`load_startup_completion_corpus()`), asserted by
**`shared/tests/test_startup_completion_fixtures.py`**, and consumed by task
3326's watchdog tests.

**Read the report first:** `docs/startup-completion-artifact-matrix.md` — the
artifact matrix these rows summarise, the named predicate
(`SESSION-TRANSCRIPT-MATERIALIZED`) with its rejected alternatives, and the
failure-mode table. This file documents the *schema*; the report documents the
*findings*.

## Record schema

Each corpus file is `{task, prd, consumer, report, schema, raw_capture, regime, rows: [...]}`.

`load_startup_completion_corpus()` runs `validate_row()` over **every** row before
returning it, so the schema check runs in the consumer's own test path — a malformed row
appended in a downstream branch fails at load with a row-id-prefixed `AssertionError`,
not as a `KeyError` deep inside a watchdog test. (`validate=False` skips the gate, for
debugging a row *because* it fails.) `validate_row()` enforces every per-row rule below;
each rule has a matching negative test in
`TestValidateRowRejects`, so a dead or inverted assertion cannot silently stop firing.

**Row-id uniqueness is the one documented rule `validate_row()` cannot enforce** — it is
handed one row at a time and cannot see the other file. It is asserted at corpus level by
`test_row_ids_are_unique_across_both_files`.

```
{
  "id":                   "<unique slug, unique across BOTH files>",
  "regime":               "healthy" | "wedge",
  "wedge_shape":          null | "from_source_build" | "uv_resolving"
                               | "mcp_init_hang" | "transcript_unreadable",
  "sample_offset_secs":   <float>,          // monotonic offset from spawn — PROVENANCE, not a bound
  "session_id":           "<uuid>",         // the session the watchdog is watching
  "config_dir_tree":      [ <tree entry>, ... ],
  "transcript_relpath":   null | "projects/<slug>/<session-id>.jsonl",
  "transcript_records":   null | [ <record projection>, ... ],
  "transcript_raw_lines": [ "<literal line>", ... ],   // OPTIONAL — see below
  "proc":                 { "alive", "pid", "state", "comm", "cmdline", "children" },
  "expected_startup_complete": true | false | null,
  "substrate_returns":    { "transcript_exists", "read_transcript_records_is_none",
                            "record_count", "count_transcript_turns" },
  "provenance":           { "probe_run_id", "mode", "cli_version", "capture_method", ... }
}
```

A tree entry is `{"relpath": str, "kind": "file"|"dir"|"symlink"|"vanished",
"size": int|null, "mtime_delta_secs": float|null}`, optionally with
`"pruned_descendants": int` where a subtree was collapsed to a count.
`vanished` is a real observation of a live directory (an entry that disappeared
mid-walk), not an error. **Contents are never inlined** — see *Secret hygiene*.

`source_path` is added by the loader at load time (the corpus file's basename);
it is not stored in the files.

### Field meanings

**`regime`** — `healthy` means the invocation was a real, working CLI run;
`wedge` means it was (or models) a startup that does not complete. `wedge_shape`
is `null` **iff** `regime == "healthy"`, and otherwise names which shape.

**`wedge_shape`** — the three PRD-named CLI shapes are `from_source_build` (the
wrapper is still compiling), `uv_resolving` (uv is still resolving packages),
and `mcp_init_hang` (an MCP server that never answers `initialize`; **note it
did not reproduce as a wedge on CLI 2.1.220** — see the report §6).
`transcript_unreadable` is the fourth, reader-side degrade case: the watchdog
cannot read the artifacts at all. It has two variants that must both stay
covered, because they produce *different* substrate returns:

| Variant | `transcript_relpath` | `read_transcript_records` | predicate |
|---|---|---|---|
| nothing resolves for the watched session | `null` | `None` | `None` |
| file resolves, every line truncated | set, with `transcript_raw_lines` | `[]` (tolerant parsing) | `False` |

**`expected_startup_complete`** — the recorded verdict of the predicate.
**Tri-state; `None` is a real value, not a missing one**: `true` = startup
proven, `false` = not proven, `null` = artifacts unreadable, cannot prove
either way. A row whose `transcript_relpath` is non-null must record a `bool`
(a locatable transcript is never the unreadable sentinel). Eight of the fourteen
rows carry `null`.

**`substrate_returns`** — what the three already-committed
`shared.cli_invoke` calls return against this row's materialized tree, so a
consumer can check a production port against the *substrate* and not only
against the verdict. Asserted per-row by
`TestPredicateDiscrimination::test_committed_substrate_returns_match_the_row`,
so these cannot drift from reality.

**`transcript_records`** — a safe field projection of the observed records
(record `type` plus the minimal fields a predicate might read), not the raw
records. `null` when no transcript resolved; `[]` when the file resolves but
parses to zero records.

**`transcript_raw_lines`** — optional; when present, `materialize_config_dir`
writes these literal lines instead of serialising `transcript_records`. This is
how the truncated/unparseable degrade variant is expressed. Requires a non-null
`transcript_relpath`.

**`provenance`** — `probe_run_id` and `mode` identify the run in
`startup_completion_probe_raw.jsonl`; `capture_method` is `live_spawn` for a
direct observation. Any other `capture_method` **must** also carry
`derived_from` (the raw sample it transforms) and `derivation` (what the
transform was) — enforced by
`TestWedgeShapeCoverage::test_derived_rows_declare_their_derivation`. Both
`transcript_unreadable` rows are `derived_from_live_capture`, because they are
failure modes of the *reader* and cannot be produced by spawning a CLI.

## Provenance

Every row traces to a line in `startup_completion_probe_raw.jsonl`, and
`test_every_row_is_linked_to_a_raw_probe_run` enforces it — a curated row whose
`probe_run_id` is not in the raw capture fails the suite.

| `probe_run_id` | Mode | Samples |
|---|---|---|
| `healthy-171d92bec337` | `healthy` | 7 |
| `healthy-e52685462d20` | `healthy` | 7 (second independent run — confirms the artifact set is stable) |
| `build_wedge-721c2ab8ebb1` | `build_wedge` | 8 |
| `uv_wedge-d178e0084890` | `uv_wedge` | 8 |
| `mcp_wedge-7182760110c3` | `mcp_wedge` | 7 |

Captured 2026-07-31 01:33:35Z – 01:37:09Z against CLI `2.1.220 (Claude Code)`,
main `11df885c73d8`. All five are `capture_method: live_spawn`.

## Secret hygiene

The healthy observations come from a config dir that really holds a live OAuth
access token (`TaskConfigDir.write_credentials`), so redaction is not optional
here. It is enforced twice:

- **capture time** — `startup_completion_probe.py` scans every assembled
  observation before emitting it, and never inlines file contents;
- **commit time** — `assert_no_credential_material()` is asserted over the full
  text of both corpus files and the raw capture by `TestCorpusSecretHygiene`,
  so a later hand-edit cannot reintroduce what the probe would have refused to
  write.

Credential-bearing paths (`.credentials.json`) are recorded by **presence and
size only**; a tree entry for one carrying inline `content` is a schema error.

## Regenerating the raw capture

```bash
cd shared
uv run python tests/startup_completion_probe.py --mode healthy \
    --out tests/fixtures/startup_completion/fresh.jsonl
# modes: healthy | build_wedge | uv_wedge | mcp_wedge | replay
```

`healthy` and `mcp_wedge` spawn the real `claude` (haiku + a one-word prompt,
~$0.002/run) and need an OAuth token in the environment. `build_wedge` and
`uv_wedge` spawn stub wrappers and need nothing. `replay` samples an existing
on-disk `CLAUDE_CONFIG_DIR` read-only (`--source-config-dir` + `--session-id`) and is
the documented fallback when a live spawn is impossible — it stamps
`capture_method: replayed_from_live_config_dir` so a replayed row is never
mistaken for a fresh observation.

## Appending new rows

**Append a probe-backed row, never a hand-written one.** When the CLI's startup
artifacts change — the transcript moves, appears at a different point in
startup, or the record vocabulary shifts — re-run the probe, append the new
observations to `startup_completion_probe_raw.jsonl`, and distil a curated row
from a specific sample. Set `provenance.probe_run_id` to that run; the suite
rejects a row that names a run the raw capture does not contain.

A row that cannot be produced by spawning a CLI (any reader-side degrade) is
still probe-backed: derive it from a real sample, set `capture_method` to
something other than `live_spawn`, and fill in `derived_from` + `derivation`
naming the exact transform. That is what keeps "empirically observed" an honest
claim rather than a label.

**Record what the predicate actually returns.** If a new shape is not
discriminated, set `expected_startup_complete` to the observed verdict and
document it as a known false positive in
`docs/startup-completion-artifact-matrix.md` §5–6. Do not tune the predicate to
manufacture a separation the artifacts do not support.

**Keep superseded rows.** Do not delete a row when a CLI upgrade changes the
artifacts, unless the old shape is confirmed gone from every supported version —
the old rows are the regression coverage that proves a predicate still handles
what it used to. Add the new row alongside it with a `provenance.note` saying
which CLI version each describes.

`TestWedgeShapeCoverage` enforces the floor: at least one row per wedge shape,
both `transcript_unreadable` variants, and at least one healthy pre-first-token
row (`substrate_returns.count_transcript_turns == 0`) — the incident shape the
whole two-regime grace exists for.

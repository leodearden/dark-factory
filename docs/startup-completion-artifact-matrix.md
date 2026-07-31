# Startup-Completion Artifact Matrix (task 3324)

| | |
|---|---|
| **Task** | 3324 — substrate validation: startup-completion artifacts |
| **PRD** | `plans/server-side-api-error-handling-prd.md` §"Unverified substrate (G3, scoped to ν)" |
| **Consumer** | task 3326 (ν) — watchdog two-regime startup grace, contract **C5** |
| **Date probed** | 2026-07-31 (captures 01:33:35Z – 01:37:09Z) |
| **CLI version** | `2.1.220 (Claude Code)` |
| **Probed main sha** | `11df885c73d8` |
| **Capture method** | `live_spawn` — 5 real CLI/stub invocations, 37 samples, no synthesised observations |
| **Probe harness** | `shared/tests/startup_completion_probe.py` |
| **Raw capture** | `shared/tests/fixtures/startup_completion/startup_completion_probe_raw.jsonl` |
| **Curated corpus** | `shared/tests/fixtures/startup_completion/{startup_completion_healthy,startup_completion_wedge}.json` (+ `README.md`) |
| **Loader / reference predicate** | `shared/tests/startup_completion_fixtures.py` |

The question this task was gated on: **which config-dir / transcript artifacts reliably
distinguish "startup complete, awaiting first token" from a from-source-build / uv / MCP wedge
at t < 120 s?** The answer, measured rather than assumed, is below — including two findings
that change what C5 is actually buying.

---

## 1. Headline findings

**F1 — The transcript file is the discriminator, and it is unambiguous.** A healthy
`claude --print` materialises `projects/<slug>/<session-id>.jsonl` at ~4.6–5.3 s, populated with
4–5 records (`queue-operation` ×2, `attachment`(SessionStart hook), `user`, sometimes a second
`attachment`) **before** any `assistant` record. A wrapper still compiling from source or
resolving `uv` packages never reaches CLI session init, so nothing under `projects/` ever
appears — at t = 60 s the config dir is still empty of everything the CLI writes.

**F2 — Today's 120 s startup kill cannot fire on a from-source-build or uv wedge.** The kill
guard is `not seen_turn and live_turns == 0 and elapsed >= startup_grace_secs`
(`shared/src/shared/cli_invoke.py:2111`), and `live_turns` is assigned **only** when
`count_transcript_turns` returns non-`None` (`:2088-2091`). Those wedges have no transcript, so
`count_transcript_turns` returns `None`, `live_turns` stays `None`, and `live_turns == 0` is never
true. Measured over all 14 corpus rows (§5), today's kill fires on **exactly three**: the two
healthy pre-first-token rows and the truncated-transcript degrade. The guard's own comment says it
"catches genuine from-source-build / uv / MCP-startup wedges" — on this evidence it does not; those
wedges are already carried through to the per-role ceiling by the `None`-degrade. What the 120 s
kill actually fires on is the *healthy* pre-first-token state, which is precisely the 2026-07-29
incident shape.

**F3 — The induced MCP-init hang did not reproduce as a wedge.** Spawning the real CLI with
`--mcp-config` pointing at a stub stdio server that accepts the connection and never answers
`initialize`, under `--strict-mcp-config`, did **not** wedge CLI 2.1.220: it delayed startup by
~2.2 s (transcript at 7.50 s vs 4.56/5.29 s healthy), then reached first token at 7.78 s and
exited normally at 8.42 s with `num_turns: 1`. Its pre-first-token sample is artifact-identical to
healthy. So this shape is **unvalidated as a wedge**, not "discriminated" — see §6 and the
`risk_identified` note filed to ν.

Consequence for C5: the predicate's verdict differs from today's behaviour on **one** observed
state — a materialised transcript with zero assistant turns — and on the measured evidence that
state is the healthy pre-first-token one. C5 is a targeted fix for the incident shape, not a
broad re-arming of the startup watchdog.

---

## 2. Methodology

`shared/tests/startup_completion_probe.py --mode {healthy,build_wedge,uv_wedge,mcp_wedge,replay}`.

**Config dir.** Each run builds a per-invocation `CLAUDE_CONFIG_DIR` through the production
`shared.config_dir.TaskConfigDir` — same seed files (`.credentials.json` written,
`settings.json` / `settings.local.json` symlinked), same layout the orchestrator hands a
dispatched agent. Nothing about the directory is mocked.

**Spawn.** `_build_argv` mirrors `cli_invoke._run_subprocess`'s shape: `claude --print
--output-format json --model haiku --system-prompt-file <tmp> --session-id <uuid4>
--permission-mode bypassPermissions --max-turns 1 --disallowed-tools '*' -p ok`, with
`start_new_session=True` and `CLAUDE_CONFIG_DIR` in the env. Haiku + a one-word prompt keeps a
run at ~$0.002, matching `shared/tests/test_cli_invoke_integration.py`'s stated practice.

**Modes.**

| Mode | How the condition was produced |
|---|---|
| `healthy` | The real `claude` binary, run twice independently (`healthy-171d92bec337`, `healthy-e52685462d20`) to confirm the artifact set is stable across runs. |
| `build_wedge` | A stub wrapper script that emits `Building claude-code from source (this may take a while)... Compiling cli v2.1.220` on stderr and sleeps, never `exec`ing the CLI — reproducing "the wrapper is still compiling". |
| `uv_wedge` | A stub wrapper emitting `Resolved 214 packages in 1.24s / Downloading numpy (18.2MiB)` and sleeping — reproducing "uv is still resolving". |
| `mcp_wedge` | The real CLI with `--mcp-config` → a stub stdio server that accepts the connection and never answers `initialize`, plus `--strict-mcp-config`. **Did not wedge** — see F3. |
| `replay` | Read-only sampler over an existing on-disk `CLAUDE_CONFIG_DIR`. Available as the documented fallback if a live spawn is impossible; **not used** — every committed observation is a live spawn. |

**Sample schedule.** Monotonic offsets ≈ 0.25 s, 1 s, 2 s, 5 s, 15 s, 30 s, 60 s, plus three
event-anchored samples: `pre_first_token` (the last sample before an `assistant` record appears),
`first_token`, and `after_exit`. The pre-first-token candidate is re-taken only when the watched
transcript's size/mtime moves — an unchanged transcript yields an identical observation, and
re-sampling it on every 0.2 s tick made the probe's own filesystem churn compete with the CLI
startup it is timing. A run that never reaches session init therefore carries its candidate at
t≈0; its late state is the `deadline` / `after_exit` sample. Per sample the probe records the
content-free config-dir tree
(relpath / kind / size / mtime-delta), whether `projects/*/<sid>.jsonl` resolves, a safe field
projection of the transcript records, the three already-committed `shared.cli_invoke` substrate
returns, and `/proc/<pid>` state (`stat` state char, cmdline, direct children `comm`).

**Redaction.** Built in at capture time, not bolted on afterwards. File contents are never
inlined — credential-bearing paths are recorded by presence and size only — and every assembled
observation is scanned for credential-shaped material before it is emitted. The same pattern set
is re-applied as a commit-time assertion over the committed artifacts
(`assert_no_credential_material`, `TestCorpusSecretHygiene`), because capture-time alone is not
safe under later hand-editing. This is load-bearing: the healthy observations come from a config
dir that really does hold a live OAuth access token.

**No wall-clock assertions.** Every offset in this document is provenance. No test asserts a
timing threshold — the observed ~5 s to transcript is a property of this machine and this CLI
build, not a bound anyone can guarantee.

---

## 3. Artifact matrix

Cells are from the raw capture. `turns` = `count_transcript_turns`; `recs` =
`len(read_transcript_records)`; `—` = the call returned `None`. "CLI-written dirs" lists which of
`.claude.json` / `sessions` / `session-env` / `plugins` / `projects` had appeared.

| Artifact | healthy @0.4 s | healthy @2.0 s | healthy pre-first-token @4.6–5.3 s | healthy first-token @5.2–5.8 s | build wedge @5 s / @60 s | uv wedge @5 s / @60 s | mcp wedge @2 s | mcp wedge pre-first-token @7.5 s |
|---|---|---|---|---|---|---|---|---|
| `.credentials.json`, `settings*.json` | ✅ (seeded by `TaskConfigDir`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `.claude.json` | ✗ | ✅ | ✅ | ✅ | ✗ | ✗ | ✅ | ✅ |
| `sessions/`, `session-env/`, `plugins/` | ✗ | ✅ | ✅ | ✅ | ✗ | ✗ | partial | ✅ |
| `projects/<slug>/` | ✗ | ✗ | ✅ | ✅ | ✗ | ✗ | ✗ | ✅ |
| `projects/*/<sid>.jsonl` resolves | ✗ | ✗ | ✅ | ✅ | ✗ | ✗ | ✗ | ✅ |
| `transcript_exists` | False | False | **True** | True | False | False | False | **True** |
| `read_transcript_records` | `None` | `None` | 4–5 recs | 6–7 recs | `None` | `None` | `None` | 4 recs |
| record types, in order | — | — | `queue-operation`, `queue-operation`, `attachment`, `user` (+`attachment`) | …+ `assistant`, `last-prompt` | — | — | — | `queue-operation`, `queue-operation`, `attachment`, `user` |
| `count_transcript_turns` | `None` | `None` | **0** | 1 | `None` | `None` | `None` | **0** |
| `/proc` state | `R` | `R` | `S` | `S` | `S` | `S` | `D` | `R` |
| direct children | — | `node` / `npm exec @playw`, `sh` | `ssh` / `git` | `ssh` / `git` | `sleep` | `sleep` | `claude` | `python3`, `git` |
| stderr | — | — | — | — | `Building claude-code from source…` | `Resolved 214 packages…` | — | — |

Reading across the row that matters: **`projects/*/<sid>.jsonl` resolving is the only artifact
that separates a started CLI from a never-started one**, and it does so cleanly — absent for the
whole 60 s life of both never-started wedges, present from ~5 s for every real CLI run.

The config-dir *seed* files prove nothing: `TaskConfigDir` writes `.credentials.json` and the
`settings*.json` symlinks **before** the process is spawned, so they are present at t = 0 in every
mode including the wedges. `/proc` state and children are informative for a human reading an
incident (`sleep` under a `claude` cmdline is a smoking gun) but are not a usable predicate —
a healthy CLI is variously `R`, `S`, and even `D`, and its children (`node`, `git`, `ssh`,
`npm exec`) are incidental.

---

## 4. Chosen predicate — `SESSION-TRANSCRIPT-MATERIALIZED`

Reference implementation:
`shared/tests/startup_completion_fixtures.evaluate_startup_completion_predicate`.

```python
records = read_transcript_records(config_dir, session_id)   # shared.cli_invoke
None   if records is None      # cannot locate/read — cannot prove either way
True   if len(records) >= 1    # session init reached; prompt enqueued
False  otherwise               # file resolves but parses to zero records
```

Built exclusively on substrate already public on main — `read_transcript_records`
(`cli_invoke.py:330`), which the watchdog's own `count_transcript_turns` already delegates to — so
the discrimination is proven against today's code and the port inherits that function's tolerant
parsing and `None`-on-unreadable semantics for free. λ deliberately does not touch
`cli_invoke.py`; the production predicate and the `server_error_startup_grace_secs` knob are ν's.

**Tri-state, deliberately.** `None` means "unreadable, cannot prove", which is what C5's
conservative degrade needs. A two-valued predicate would have to fold unreadable into `True`
(extending the bound for a possible wedge) or `False` (killing a possibly-healthy retry cycle).
The tri-state also matches the existing house convention at the kill site, which fires only on an
explicit `live_turns == 0` and never on `None` (`cli_invoke.py:2109-2111`).

**Why `>= 1 record` and not something narrower.** The observed leading record types —
`queue-operation` (prompt enqueue), `queue-operation`, `attachment` (SessionStart hook), `user` —
are all written before any `assistant` record, and their presence already proves the CLI reached
session init and accepted the prompt. Keying on a specific record type would pin the predicate to
one CLI version's record vocabulary for no gain in discrimination: no observed sample anywhere in
the capture has a resolvable transcript with zero records.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **Config-dir file presence** (`.credentials.json`, `settings.json`) | Written by `TaskConfigDir` *before* the spawn. Present at t = 0 in every mode including both wedges — proves nothing about CLI startup. |
| **`.claude.json` / `sessions/` / `plugins/` presence** | Genuinely CLI-written and appears ~1–2 s in, earlier than the transcript. But it is a *different* file per CLI version with no stability contract, it appeared partially in the mcp run, and it buys ~3 s of earliness on a 120 s bound. Not worth the version coupling. |
| **`/proc` state or child-process shape** | Not discriminating: healthy runs were observed `R`, `S`, and `D`, with children `node`, `git`, `ssh`, `npm exec`. |
| **`transcript_exists` alone** (drop the record count) | Would be `True` for a resolvable-but-empty file, which the tri-state deliberately distinguishes as `False`. Also loses the `None` state — `transcript_exists` is a total `bool`, so "unreadable" and "absent" collapse together. |
| **`count_transcript_turns >= 1`** | That is the *working*-regime signal the watchdog already latches as `seen_turn`. Using it here would make the predicate `True` only after turn 1, i.e. never during the pre-turn-1 window C5 is about. |
| **Record-type prefix match** (`queue-operation` first, etc.) | Pins to CLI 2.1.220's record vocabulary. The prefix was stable across all five runs and is recorded in the corpus as a drift signal, but is not load-bearing in the predicate. |

---

## 5. Failure-mode table

Computed over every corpus row against the real substrate, at `elapsed >= startup_grace_secs`
(reproduce with the loader + `count_transcript_turns`; `today` applies `cli_invoke.py:2111`'s
guard, `C5` applies "kill only when the predicate is FALSE").

| Row | Shape | `count_transcript_turns` | predicate | today @120 s | C5 @120 s | Verdict correctness | Blast radius if wrong |
|---|---|---|---|---|---|---|---|
| `healthy_t0_4_no_transcript`, `healthy_t2_0_no_transcript` | healthy, pre-transcript | `None` | `None` | no kill | no kill | **correct** — the CLI genuinely has not finished starting | none; behaviour unchanged |
| `healthy_pre_first_token_5rec`, `healthy_pre_first_token_4rec` | healthy, awaiting first token | 0 | `True` | **KILL** | no kill | **correct, and this is the fix** — the incident shape | if wrong (a wedge that somehow materialised a transcript), it lives to `server_error_startup_grace_secs` = 900 s instead of 120 s, capped by the per-role ceiling. Bounded, never unbounded. |
| `healthy_first_token`, `healthy_after_exit` | healthy, turn 1 landed | 1 | `True` | no kill | no kill | **correct** — `seen_turn` has latched; the working regime owns it | none; the predicate is not consulted once `seen_turn` is True |
| `wedge_from_source_build_t5/t60` | `from_source_build` | `None` | `None` | no kill | no kill | `None` is **correct** (the artifacts genuinely cannot prove startup) but note F2: today's kill is *already* inert here | none from C5. The pre-existing gap — a build wedge runs to the per-role ceiling, not 120 s — is unchanged by C5 and is ν's to decide whether to close. |
| `wedge_uv_resolving_t5/t60` | `uv_resolving` | `None` | `None` | no kill | no kill | as above | as above |
| `wedge_mcp_init_hang_t2` | `mcp_init_hang`, pre-transcript | `None` | `None` | no kill | no kill | correct for the observed state | none |
| `wedge_mcp_init_hang_pre_first_token` | `mcp_init_hang`, transcript materialised | 0 | `True` | **KILL** | no kill | **`True` is what the artifacts say** — and on this capture the run was not actually wedged (F3), so `True` was also *factually* right. Recorded as observed, not tuned. | if a real MCP-init hang does materialise a transcript and then stall, C5 extends it 120 s → 900 s (per-role ceiling capped). Bounded. See §6. |
| `wedge_transcript_unreadable_session_mismatch` | degrade: nothing resolves for the watched session | `None` | `None` | no kill | no kill | **correct degrade** — "predicate unreadable" → today's behaviour exactly | none; this *is* the conservative degrade. The config dir here is fully populated and even holds another session's transcript, which isolates the glob's anchor as the session id, not directory presence. |
| `wedge_transcript_unreadable_truncated` | degrade: file resolves, every line unparseable | 0 | **`False`** | **KILL** | **KILL** | **correct** — nothing is proven, so the bound is not extended | none; C5 matches today. Note this is a *different* unreadable return: `read_transcript_records` parses tolerantly and yields `[]`, not `None`. 3326 must not assume unreadable always means `None`. |

**"Predicate unreadable" — the conservative degrade.** When the transcript cannot be located or
read, `read_transcript_records` returns `None` and the predicate returns `None`. C5 must then fall
back to today's behaviour, which is already the conservative one: the watchdog's startup kill
requires an explicit `live_turns == 0` and never fires on `None`
(`shared/src/shared/cli_invoke.py:2109-2111`, with the assignment guard at `:2088-2091` and the
rationale comment at `:1982-1986`). Concretely — `None` must be treated as "not FALSE", so the
startup kill does not fire, and the invocation is carried to the per-role ceiling exactly as it is
today. Nothing new is needed for this case; it is a matter of not accidentally folding `None` into
`False`.

**The one thing C5 changes.** Across all 14 rows, C5's verdict differs from today's on exactly the
two `healthy_pre_first_token_*` rows: KILL → no kill. That is the whole delta, and it is the
incident.

---

## 6. Known false positive / unvalidated shape

`mcp_init_hang` is recorded as `expected_startup_complete: true` at its pre-first-token sample
because that is what the predicate returns on the observed artifacts — but the honest reading is
narrower than "the predicate handles MCP wedges":

- The induced condition **did not wedge the CLI** on 2.1.220 (F3). The run completed. So this
  capture is evidence that a stub MCP server hanging at `initialize` is not a startup wedge on
  this version — not evidence about how the predicate behaves on a real one.
- If a real MCP-init hang *does* stall after the transcript materialises, the predicate returns
  `True` and C5 extends that invocation's pre-turn-1 bound from 120 s to
  `server_error_startup_grace_secs` (900 s), capped by the per-role ceiling. **Bounded, never
  unbounded** — the invocation still dies, just later, and the existing timeout classification
  path is unchanged.
- Per F2, the alternative is not "today it dies at 120 s": if such a hang produces no transcript,
  today's kill is inert for it too.

The predicate was **not** tuned to manufacture a separation the artifacts do not support. A
non-blocking `risk_identified` note is filed to ν recording this gap so 3326 can decide whether a
900 s bound on an unvalidated shape is acceptable, or whether it wants a narrower extension.

---

## 7. Consumption contract for ν (task 3326)

`shared/tests/conftest.py` already puts `shared/tests/` on `sys.path`, so the loader imports as a
top-level module from any test in the `shared` package:

```python
import startup_completion_fixtures as scf

for row in scf.load_startup_completion_corpus():          # list[StartupCompletionRow]
    config_dir, session_id = scf.materialize_config_dir(row, tmp_path / row['id'])
    assert my_production_predicate(config_dir, session_id) is row['expected_startup_complete']
```

| Symbol | Signature / meaning |
|---|---|
| `load_startup_completion_corpus(*, validate=True)` | `-> list[StartupCompletionRow]` — every row from both corpus files, each stamped with `source_path`, each passed through `validate_row()` on the way out. The gate therefore runs in ν's test path too, not only in λ's suite. |
| `materialize_config_dir(row, dest)` | `-> tuple[Path, str]` — rebuilds the observed tree under `dest` as a **real filesystem** and returns `(config_dir, session_id)`. `_resolve_transcript_path`'s `projects/*/<session_id>.jsonl` glob resolves against it exactly as against a live config dir, so a production predicate — or the real `_run_subprocess` watchdog — can be pointed at it directly. |
| `snapshot_config_dir(config_dir, ...)` | `-> list[dict]` — the sampler shared with the probe, so probe output and materialized trees are describable by one function. |
| `evaluate_startup_completion_predicate(config_dir, session_id)` | `-> bool \| None` — the reference implementation. Diff a production port against it. |
| `validate_row(row)` / `assert_no_credential_material(text, *, source)` | Per-row schema gate (already applied by the loader — call it directly only when validating a row you built by hand) and secret-hygiene guard; reusable if ν appends rows. Row-id uniqueness is corpus-level and is *not* checked by `validate_row`. |
| `row['expected_startup_complete']` | `bool \| None` — the recorded verdict. **`None` is a real value**, not a missing one: it is the unreadable sentinel, and it is what eight of the fourteen rows carry. |
| `row['substrate_returns']` | `{transcript_exists, read_transcript_records_is_none, record_count, count_transcript_turns}` — the three committed substrate calls' returns, so a port can be checked against the substrate and not just the verdict. |

**Tri-state convention.** `True` → startup proven, extend the pre-turn-1 bound to
`server_error_startup_grace_secs`. `False` → not proven, keep today's `startup_grace_secs` kill.
`None` → cannot prove; treat as *not* `False` (no kill) and carry to the per-role ceiling, which
is today's behaviour. Folding `None` into `False` would newly kill invocations whose transcript is
merely unreadable — a behaviour change C5 does not authorise.

---

## 8. Known limits

- **CLI-version sensitivity.** Everything here is CLI `2.1.220`. The predicate is deliberately
  keyed on *transcript existence and non-emptiness* rather than on record types, so a vocabulary
  change does not break it — but a change to *where* the transcript is written, or to whether it
  is created before the first token, would. `TestLiveReprobe`
  (`shared/tests/test_startup_completion_fixtures.py`, `@pytest.mark.integration`) re-runs the
  probe live and fails loudly with an observed-vs-committed diff when a CLI upgrade moves the
  artifacts. It is deselected by `shared/pyproject.toml`'s `addopts = "-m 'not integration'"`, so
  CI stays hermetic; run it deliberately after a CLI bump:
  `uv run pytest tests/test_startup_completion_fixtures.py -m integration`.
- **`mcp_init_hang` is unvalidated as a wedge** (§6).
- **Single host, single filesystem.** All captures are from one machine. The ~5 s to transcript is
  not a contract; nothing asserts on it.
- **The healthy runs exited with `is_error: true`** in the result envelope (a property of the
  probe's minimal prompt and `--disallowed-tools '*'`, not of startup). Irrelevant to the
  question asked: the transcript materialised and an `assistant` record landed in every run.
- **Two wedge shapes are stub-driven.** `build_wedge` / `uv_wedge` are wrapper scripts that
  reproduce the *observable* condition (a `claude`-named process that never reaches session init,
  emitting build/resolve stderr). They are faithful to what the watchdog can see, which is what
  the predicate is defined over, but they are not a real cargo build.

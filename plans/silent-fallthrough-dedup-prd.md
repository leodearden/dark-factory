# PRD: Silent-fall-through-on-error — unify-and-eliminate

**Status:** deferred → ready to decompose · **Date:** 2026-06-18 · **Approach:** B+H · **Project:** dark_factory
**Companion to:** task **1799** (in-flight: `Scheduler.get_external_statuses` silent strand).
**Source analysis:** `plans/silent-fallthrough-dedup-remediation-2026-06-18.md` (full 48-site inventory + family map).

## Goal

An operator can no longer be left blind when a backend/file/agent payload comes back malformed, a journal is
corrupt, or a resolver fails: every such failure now emits a `WARNING+` log at production level **and** is
distinguishable from a legitimate empty/negative result (an error slot, a sentinel, an `offline`/`degraded`
marker, or a fail-closed gate). The 48 verified silent-fall-through sites are remediated **by unifying the
duplicated logic into shared primitives and routing every divergent copy through them** — not by per-site patches.

**User-observable end state:** a previously-silent strand (e.g. reify 4635: a ready cross-project task stranded
for hours with zero events) now produces a visible WARNING + a fail-safe-wait/escalation signal; a corrupt
`b3-state.json` no longer silently resets the auto-merge cap; a broken integrity DB renders a red badge on the
dashboard instead of a healthy 0-backlog.

## Background

A whole-codebase sweep (10 finder agents → 10 adversarial verifiers) found **48** instances of the
silent-fall-through-on-error anti-pattern — the exact shape task 1799 fixes. **Root cause is untreated code
duplication:** nearly every site has a twin (same file or sibling) that already handles the identical failure
loudly; one branch of a copy-paste pair forgot to mirror its loud sibling. Severity: 4 critical, 15 high, 12
medium, 17 low. Concentrated in the orchestrator scheduler dispatch path, fused-memory reconciliation/backend,
and the dashboard signal layer. The full inventory + the loud↔silent twin pairs are in the source analysis doc.

This directly violates the project directive **"prefer loud escalation over silent degradation."**

## Approach (B+H — high stakes: scheduler dispatch seam, blast radius ≥5 packages)

Extract one **loud-and-safe** implementation per duplication family into the `shared` workspace package
(a verified dependency of orchestrator/fused-memory/dashboard/sampler), then migrate every divergent copy onto
it. Each migration is single-package (hard rule: no task spans packages — cross-package architect budget) and
**RED-test-first**: reproduce the silent strand through the real code path, confirm it fails, then fix.

Duplication families → unification target (site lists in the analysis doc):
- **A** MCP/tool-result envelope parsing → `shared.mcp_envelope.parse_tool_result`.
- **B** resolver error-slot discipline (`x, _ = …` discards the error) → `shared` resolver-guard + fail-safe-wait.
- **C** load-or-warn JSON state files → `shared.safe_io.load_json_or_warn`.
- **D** corrupt-metadata-blob (sqlite backend) → extract `_row_to_task`'s deduped-WARNING handler; never clobber.
- **E** agent structured-output verdict extraction → `shared.agent_result` guard (distinguishable ERROR sentinel).
- **F** dashboard offline/degraded marker propagation + visible logging (unify on the existing `{'offline':True}`).
- **G** degrade-suppresses-escalation → mirror loud sibling; fail-toward-escalate.
- **H** observability/offline tail → add-WARNING + mirror in-package twin.

## Contract section (the shared primitives — H)

```python
# shared/mcp_envelope.py
def parse_tool_result(result, key, expected_type) -> tuple[Any | None, Exception | None]:
    """Parse an MCP/tool text-result envelope. Returns (value, None) ONLY when value isinstance expected_type.
    For EVERY abnormal shape (no text block / key absent / inner-not-dict / wrong-type) and for a raised
    exception: log a DISTINCT logger.warning naming the shape, and return (None, <error>) where <error> is a
    non-None Exception (EnvelopeParseError subtype carrying the shape + a payload prefix). NEVER returns
    (benign-default, None) on failure — a non-None error slot is the invariant the caller branches on."""

# resolver-guard (shared.mcp_envelope or shared.result)
def resolver_failed(value, err) -> bool:
    """True iff err is not None OR value is falsy-due-to-failure; callers fail-safe-wait + WARNING on True."""

# shared/safe_io.py
def load_json_or_warn(path, *, default, on_corrupt="warn") -> tuple[Any, bool]:
    """Returns (parsed, ok). FileNotFoundError/first-run-absent → (default, True) SILENTLY (legit).
    JSONDecodeError/ValueError (present-but-corrupt) → logger.warning(path, exc); on_corrupt in
    {"warn"→(default,False), "fail_closed"→raise, "quarantine"→rename .corrupt then (default,False)}.
    Uses a deduped-warning set keyed by path (mirrors sqlite_task_backend._warned_malformed_task_ids)."""

# shared/agent_result.py
def extract_agent_verdict(result, *, default_verdict, error_summary) -> AgentVerdict:
    """When the agent ran but produced no parseable structured output / no expected key (the
    {'warning': ...} / unparseable case): logger.warning(warning, output_prefix) and return a verdict whose
    summary is a DISTINGUISHABLE sentinel (f'agent-failed:{warning}'), never a neutral default conflatable
    with a real result. Mirrors workflow._run_reviewer:4476 (ERROR verdict on unparseable output)."""

# shared/timestamps.py
def parse_timestamp_or_warn(raw, *, fallback=None, context="") -> tuple[datetime, bool]:
    """Parse an ISO-8601 timestamp. Success → (dt, True). Malformed/None/non-str → logger.warning(context, raw)
    and return (fallback or datetime.min(UTC), False) — a SORTABLE sentinel so the record is NEVER silently
    dropped (mirrors escalation queue.py:636 / watcher.py:97). 'Oldest-first' callers get a deterministic fold
    target; age-gating callers get a visible signal instead of a silent skip."""
```

**Invariants:** (1) a non-raising malformed payload is NEVER collapsed to a bare `None`/`{}`/`[]` without a
non-None error slot + a WARNING; (2) the benign-absent branch (file not found, optional key) stays silent —
only the *fault* branch is loud; (3) safety-bearing gates (caps, dep gates, terminal-state guards, metadata
writes) fail **closed**; resolvers fail-safe-**wait** with a WARNING + grace/escalation counter; observability
sites are **WARNING-only**; (4) no degradation path may suppress an escalation it should have fired.

## Resolved design decisions

1. **Shared primitives live in `shared/`** (verified workspace dep of all Python consumers). **Escalation
   ADOPTS `shared`** (new `dark-factory-shared = { workspace = true }` dep — escalation is already a uv
   workspace member; `shared` is foundational with no first-party app deps, so no import cycle). Justified by
   genuine cross-package duplication: the *parse-ISO-timestamp-or-fall-back-safely* logic exists in **4 copies** —
   escalation `dedupe.py:291` (buggy silent drop), `queue.py:636` + `watcher.py:97` (correct, `datetime.min`
   fallback), and dashboard `redux_api.py:243` (#39, also silent) — so a 4th shared primitive `shared.timestamps`
   unifies all four and fixes the two buggy ones. (Reviewed + adopted 2026-06-18.)
2. **One task per package** (cross-package scopes blow the architect budget — feedback memory). The shared
   primitives are foundation tasks; each package's migration depends on the primitive(s) it uses.
3. **P1 depends on task 1799** and folds 1799's `get_external_statuses` fix onto the shared `parse_tool_result`
   primitive (dedup the 5th sibling) AFTER 1799 lands — sequenced, no concurrent edit on scheduler.py.
4. **The lint gate (σ) is a pytest AST-scan**, not a ruff custom plugin (ruff has no stable custom-rule support).
   It walks first-party source for two signatures and fails CI; lands LAST (depends on all migrations) so it
   passes clean.
5. **Fail-closed vs fail-safe-wait policy** is per-family per the contract invariants above (not implementer
   discretion). `b3_gate` gets a module logger added (it currently has none) and treats corrupt state as
   cap-**exhausted**, not full-cap-reset. `_merge_metadata` **refuses/quarantines** rather than clobbering
   `external_deps`/`memory_hints`.
6. **Standing dedup directive on every task:** unify onto the shared/local primitive (don't duplicate a WARNING
   into both branches); and **file a follow-up task** for any *other* duplication discovered in the touched
   files — do not fix it opportunistically beyond scope.

## Pre-conditions for activating

- Task **1799** done (P1 `δ` depends on it).
- Shared foundation tasks `α`/`β`/`γ` done before their dependent migrations dispatch.

## Cross-task / cross-PRD relationship (G4)

| Other work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| task **1799** | this-PRD extends | `scheduler.py` resolver loud-fail pattern + `get_external_statuses` | 1799 owns initial fix; `δ` owns unifying all 5 resolvers onto the shared primitive | `δ` depends_on 1799 |

No other live PRDs touch these files. No reciprocal-ownership ambiguity.

## Decomposition plan (17 tasks; labels α…σ)

**Foundation (shared) — intermediate, unlock the migrations:**
- **α** [shared] `mcp_envelope.parse_tool_result` + resolver-guard. *Signal:* unit test — malformed envelope →
  `(None, EnvelopeParseError)` + a captured WARNING; clean envelope → `(value, None)`. *Unlocks:* δ,ζ,θ,ν.
- **β** [shared] `safe_io.load_json_or_warn`. *Signal:* unit test — corrupt file → WARNING + `(default, False)`;
  absent file → `(default, True)` silently; `fail_closed` raises. *Unlocks:* ε,κ,ν.
- **γ** [shared] `agent_result.extract_agent_verdict`. *Signal:* unit test — `{'warning':...}` (no verdict) →
  WARNING + sentinel summary `agent-failed:*`. *Unlocks:* ζ,ι.
- **φ** [shared] `timestamps.parse_timestamp_or_warn`. *Signal:* unit test — malformed/None ts → WARNING +
  `(datetime.min, False)` sortable fallback; valid ts → `(dt, True)`. *Unlocks:* ξ, ν(#39).

**Orchestrator:**
- **δ** [orchestrator] dep **α, 1799**. Unify scheduler resolvers: migrate `_parse_tool_text_result`,
  `get_tasks`, `get_statuses`, `get_status`, and (folded from 1799) `get_external_statuses` onto
  `parse_tool_result`; fix `acquire_next` local-dep backfill to honor `_backfill_err` with fail-safe-wait + a
  per-(task,dep) grace counter mirroring the external-dep path. *Signal (e2e RED):* a scheduler tick fed a
  SUCCESS envelope that parses to a non-list/non-dict no longer dispatches nothing silently — it emits a WARNING
  and a ready dependent is treated fail-safe-wait, not idle.
- **ε** [orchestrator] dep **β**. JSON-state loaders onto `load_json_or_warn` + fail-closed:
  `merge_queue_store._load_raw`, `b3_gate._load_state`/`_resolve_cap` (add module logger; corrupt→cap-exhausted),
  `harness._reconcile_one_stranded` plan.lock. *Signal:* corrupt journal → WARNING + recovery distinguishes
  corrupt from fresh; corrupt b3-state → cap not silently reset.
- **ζ** [orchestrator] dep **α, γ**. Resolver-guard + agent-result + misc: `harness._reconcile_stranded_in_progress`
  /`_scan_for_terminal_active_tasks`, `workflow._build_train_state`, `substrate_gate.extract_probe_set`
  (distinguish malformed-present → fail-closed), `merge_queue._main_health_fingerprint`,
  `git_ops.get_merge_diff_files`, `evals/metrics._git_diff_stats`, `review_checkpoint._run_review`. *Signal:*
  each site emits a WARNING + distinguishable result where it was silent; startup stranded-sweep guards on `err`.

**Fused-memory:**
- **θ** [fused-memory] dep **α**. Recon + middleware envelope/resolver sites: `targeted._sweep_cancelled_descendants`,
  `reconciliation/harness._fetch_task_count_census`, `task_knowledge_sync._apply_post_flight_guards`,
  `queue_health.summarize_graphiti_queue_health`, `memory_consolidator` results-key,
  `task_interceptor._check_escalation_idempotency` (WARNING not DEBUG + fail-safe), `interceptor_write_succeeded`
  (bare `{}`→failure), `_extract_metadata_files`. *Signal:* malformed payload → WARNING + the diagnostic/guard
  no longer self-disables silently.
- **ι** [fused-memory] dep **γ**. Agent-result sites: `verify.CodebaseVerifier.verify`,
  `agent_loop._call_claude_cli`. *Signal:* a failed/unparseable verify → WARNING + `agent-failed:*` sentinel,
  not a neutral `inconclusive` that silently skips knowledge capture.
- **κ** [fused-memory] dep **β**. Recon consolidator + degrade + service: `memory_consolidator` episodes/mem0
  bare-except (WARNING + report stat flag), `reconciliation/harness._finding_persistence_count`
  (fail-toward-escalate), `stages/base._find_fused_memory_server` (load_json_or_warn), `memory_service.search`
  (in-band `degraded`/`failed_stores` channel + `success=False` journal on store outage). *Signal:* store
  outage → degraded flag surfaced + WARNING, not a silently-halved corpus read as sparse.
- **λ** [fused-memory] (indep). Task-backend metadata clobber-guard: extract `_row_to_task`'s deduped-WARNING
  corruption handler; `_merge_metadata` refuses/quarantines a corrupt existing blob (never clobbers
  external_deps/memory_hints); `remove_dependency` stops claiming a clean removal over an unparseable blob.
  *Signal (RED):* `update_task` over a corrupt-metadata row emits a deduped WARNING and does NOT silently
  overwrite the blob.

**Dashboard:**
- **ν** [dashboard] dep **α, β, φ**. Signal layer: `db.with_db`/`DbPool.get` (DEBUG→WARNING + red-badge error
  sentinel), `discover_orchestrators`/`fetch_external_statuses` (propagate the existing `{'offline':True}`
  marker), `read_task_artifacts` (load_json_or_warn + split corrupt from absent), `_shape_wal_status` (#39 —
  route ts parse through `parse_timestamp_or_warn`), `app.api_curator` accounts_summary (DEBUG→WARNING),
  `_split_queue_stats`, `tab_overview.jsx` HostLoadCard (stale badge). *Signal:* a corrupt/locked DB or
  MCP-unreachable orchestrator renders a red/offline badge, not a benign zero.

**Tail (per-package):**
- **ξ** [escalation] dep **φ**. Add `dark-factory-shared` workspace dep (+ pyright extraPaths) to escalation,
  then route ALL THREE escalation timestamp-parse copies — `dedupe.find_dedupe_parent:291` (the buggy silent
  one), `queue.py:636`, `watcher.py:97` — through `shared.timestamps.parse_timestamp_or_warn`, deleting the two
  hand-rolled `datetime.min` fallbacks. *Signal:* corrupt-ts parent → WARNING + candidate folds (not silently
  re-filed as new); `import shared.timestamps` resolves in escalation; the 3 local copies are gone.
- **ο** [shared] (indep). `pytest_jobserver.pytest_configure` — WARNING on the OSError/FIFO branch (mirror the
  timeout branch). *Signal:* broken FIFO → WARNING, distinguishable from intentional-off.
- **π** [sampler] (indep). `metrics.collect_process_metrics` (let proc_iter failure propagate to `__main__`'s
  visible-degrade, or WARNING+sentinel) and `parse_pressure_file` (sentinel on total parse miss). *Signal:*
  psutil failure no longer writes a healthy `0.0` sample; it's distinguishable.
- **ρ** [scripts] (indep). `reviewer_redundancy_diagnostic.load_review` — WARNING + count dropped/corrupt files
  in the report. *Signal:* corrupt review JSON → reported as skipped, not silently under-counted.

**Enforcement (lands last):**
- **σ** [shared/tests] dep **all migration tasks (δ,ε,ζ,θ,ι,κ,λ,ν,ξ,ο,π,ρ)**. A pytest AST-scan over first-party
  source that FAILS on either signature: (a) tuple-unpack of a known `(value, error)`-returning resolver into
  `_` for the error position; (b) `except (...): return <empty-literal>` with no `logger.warn/error/exception`
  in the handler body. *Signal (RED):* the test flags a deliberately-planted bad sample and passes on the
  migrated tree; encodes "loud over silent" as an enforced invariant.

## Boundary-test sketch (H — both sides of each seam)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Producer: `parse_tool_result` on malformed envelope | success-status envelope, text parses to wrong type | returns `(None, EnvelopeParseError)`; a WARNING naming the shape is emitted |
| Producer: `load_json_or_warn` on corrupt file | file present, non-JSON bytes | WARNING; `(default, False)`; `fail_closed` raises; absent file stays silent `(default, True)` |
| Producer: `extract_agent_verdict` on `{'warning':...}` | agent ran, no verdict key | WARNING; summary == `agent-failed:<warning>` |
| Producer: `parse_timestamp_or_warn` on malformed ts | raw is None / non-ISO string | WARNING; `(datetime.min(UTC), False)`; valid ts → `(dt, True)` silently |
| Consumer ξ: dedupe parent with corrupt ts | a pending parent matches category+key, corrupt timestamp | parent still considered (sorted oldest); WARNING; candidate folds, not re-filed |
| Consumer δ: scheduler tick, non-list `get_tasks` | resolver returns non-list, deps actually done | tick does NOT silently idle; WARNING; dependent fail-safe-wait, not dropped |
| Consumer ε: orchestrator restart, corrupt merge journal | truncated journal on disk | recovery distinguishes corrupt from fresh; WARNING; pending merges not silently dropped |
| Consumer λ: `update_task` over corrupt metadata row | row's metadata column is non-JSON | deduped WARNING; existing blob NOT overwritten |
| Consumer ν: panel reads corrupt/locked DB | sqlite file corrupt/locked, INFO log level | red/error badge surfaced; WARNING at production INFO (not DEBUG) |
| Gate σ: planted bad sample | a file with `x, _ = await resolver()` or silent `except: return {}` | pytest AST-scan FAILS; passes once removed |

## Out of scope

- The `mem0`/`graphiti` git submodules (third-party).
- Test files (test_*.py / tests/) — sweep excluded them; the lint may optionally cover them later.
- Broader refactors beyond the 48 sites + the shared primitives — captured as follow-up tasks per the standing
  directive, not pre-queued here.

## Open questions (tactical)

1. **σ dependency breadth.** σ depends on all 12 migrations; if one blocks, σ waits. *Suggested:* keep σ-last for
   a clean gate; if a migration stalls long-term, land σ warn-only first. Decide during ζ/ν.
2. **`memory_service.search` return-shape change (κ).** Widen to a dataclass vs attach a `degraded` meta field.
   *Suggested:* attach `degraded`/`failed_stores` to the existing MCP response payload (non-breaking). Decide in κ.
3. **`load_json_or_warn` quarantine naming** (`.corrupt` suffix vs timestamped). Decide in β.

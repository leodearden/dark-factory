# Capability manifest — resume-charter-loss-remediation-prd

Binds each new leaf's signal capabilities to evidence, per
`skills/prd/references/gates.md` §Capability Manifest. All evidence
verified against `main` @ `03ff70c5dd` on 2026-08-10 (line numbers are
of that SHA; the YAML sidecar's delivered_checks are pattern-anchored,
never line-anchored). The absorbed batch 3983-3994 is **not**
re-manifested — those tasks were filed pre-PRD and 3983 is already
merged; their substrate claims were re-verified this session and the
material corrections are recorded as amendments in PRD §7c.

## α — escalation drain durability

- `_COMPACT_ESCALATION_FIELDS` exists and currently DROPS
  `root_cause`/`members` → capability→producer (wired):
  `escalation/src/escalation/server.py:285-296` (tuple), applied
  `:1097-1101`; the drop is the RED premise. **PASS**
- `add_members_to_l2` write path exists and preserves only `members` →
  wired: `escalation/src/escalation/queue.py:866-925`; docstring
  `:889-892` states the discard policy verbatim; caller returns only
  `{'id','status','members'}` (`server.py:1401-1405`). **PASS**
- DAG-direction: α is upstream of β, γ, δ. **PASS**

## β — root-cause canonicalisation

- Exact-match dedup exists → wired:
  `escalation/src/escalation/queue.py:825-864`
  (`esc.root_cause.strip() != candidate`); server docstring
  `server.py:1340-1342` "exact-string dedup key". RED premise (~155
  duplicate L2s ≈ 30%) is a measured investigation figure, not a leaf
  assertion. **PASS**
- DAG-direction: depends on α (same-file serialization + projection
  contract). **PASS**

## γ — archive-inclusive drain loop

- Drain procedure section exists →
  `skills/escalation-watcher-auto/SKILL.md:260-272` ("### Draining
  pending escalations"; step 3 unions members of PENDING L2s only). **PASS**
- Archive-inclusive read exists → wired:
  `escalation/src/escalation/server.py:1105-1117`
  (`get_task_escalations`, "ARCHIVE-INCLUSIVE by default");
  `get_pending_escalations` documented pending-only `:1076-1079`. **PASS**
- Producer of the enriched projection is α, upstream. **PASS**

## δ — exit-on-drain + guard change (same task)

- Guard knobs exist → `orchestrator/src/orchestrator/defaults.yaml:767-774`
  (`watcher_crashloop_window_secs: 600`,
  `watcher_misconfigured_min_rotation_secs: 120`,
  `watcher_max_misconfigured_clean_exits: 5`) +
  `orchestrator/src/orchestrator/config.py:3783-3792`. **PASS**
- Free relaunch gate exists → wired:
  `orchestrator/src/orchestrator/harness.py:11343`
  (`_watcher_has_actionable_l1`), called `:11416`. **PASS**
- Supervisor split exists → `harness.py:11492` (degenerate-clean
  branch feeding the guard) vs `:11538-11548` (healthy-clean; sole
  outage-L2 clearer). **PASS**
- The `exit_reason=drained` marker is NEW substrate delivered by δ
  itself (named in PRD §9 C4) — no false pre-existence claim. **PASS**
- Prerequisites α/β/γ are upstream (D5 ordering). **PASS**

## ε — quiet-project watcher cost containment

- `watcher_daily_cost_ceiling_usd` exists and is enforced →
  `orchestrator/src/orchestrator/harness.py:9017` (+ defaults.yaml:788).
  RED premise: enforcement counts a store missing ~40% of watcher
  rotations (brief §8, measured). **PASS**
- Launch-side empty-queue skip landed as task 2629 (done 2026-07-16);
  ε's deployed-code check (is it running on the quiet four?) is the
  task's own first deliverable, per the fleet-staleness lesson —
  not assumed here. **PASS**

## ζ — unblock_auto envelope + governing cap-wait bound

- `_DRY_RUN_CAP_WAIT_SANITY_SECS: float = 1800.0` →
  `orchestrator/src/orchestrator/dry_run_unblock.py:112`, passed `:355`. **PASS**
- The bound's checker is sampled only inside the cap-hit retry loop →
  `shared/src/shared/cli_invoke.py:1387` (`_check_cap_wait` closure),
  called `:1896`/`:1988`; the governing wait sits in
  `usage_gate.invoke_slot`, unbounded (RED premise; overshoot 11×
  observed). **PASS**
- `permission_mode` never passed at the call site → zero occurrences in
  `dry_run_unblock.py`; inherits `bypassPermissions` default
  (`cli_invoke.py:1143`), emitted at `:2220`. Denylist plumbing exists
  (`--disallowed-tools` in `build_claude_argv`). **PASS**
- DAG-direction: depends on 3987 (upstream, pending, dep on 3983 done). **PASS**

## η — recon Landlock fix

- `sandbox_recon_writable_extras: list[str] = Field(default_factory=list)`
  → `fused-memory/src/fused_memory/config/schema.py:1116`; consumed
  fail-closed at
  `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py:448-453`;
  NOT set in `fused-memory/config/config.yaml` (verified zero hits) —
  so `[]` in practice. **PASS**
- `recon_config_base_dir` → `cli_stage_runner.py:369-377`
  (`data_dir / 'recon-config'`); creator
  `reconciliation/stages/base.py:244`. **PASS**
- Stale claim to correct → wired:
  `orchestrator/src/orchestrator/agents/landlock_exec.py:20-23`
  ("redirected to a per-task CLAUDE_CONFIG_DIR inside the worktree").
  **PASS**
- RED premise (no recon transcript since 2026-07-18; probe DENIED) —
  measured by the investigation; η's pinning test re-proves it. **PASS**
- DAG-direction: depends on 3983 (done — D1 gate, wired for the record).
  Downstream consumer 3972 gains the edge (PRD §7c-4). **PASS**

## θ — 3727 deployment gate (+ polarity dispute)

- `durable_archive_path` on main →
  `shared/src/shared/transcript_archive.py:232`. **PASS**
- `archive_available` instrumentation on main →
  `orchestrator/src/orchestrator/event_store.py:255-270`,
  `orchestrator/src/orchestrator/harness.py:1641-1649`. **PASS**

  > **CORRECTION 2026-08-11:** the `harness.py:1641-1649` anchor is
  > wrong. Those lines are only the *fault-log rate limiter*
  > (`self._archive_available_fault_logged`, task 3727) — they emit
  > nothing. The actual instrumentation is:
  > - the emit — `orchestrator/src/orchestrator/harness.py:8069-8081`
  >   (the `session_resume_fallback` `event_store.emit(...)`; the
  >   `'archive_available'` data field itself at `:8076-8079`);
  > - the helper — `_archive_available` defined at
  >   `orchestrator/src/orchestrator/harness.py:3179`.
  >
  > Verified by reading all three ranges. Both new anchors are
  > byte-identical at this manifest's stated baseline `03ff70c5dd`
  > (`harness.py` is unchanged between `03ff70c5dd` and `376b10cc5c`),
  > so they are valid at the SHA the rest of this file is pinned to.

- RED premise: `data.archive_available` NULL on all 260 historical
  fallbacks despite the code being on main — the deployed-fleet gap is
  exactly what θ measures (deployed-code check, `ExecMainStart`-style). **PASS**

  > **CORRECTION 2026-08-11:** "NULL on all 260" reports the **NULL ARM
  > as if it were the total**. 260 is the count of NULL events, not the
  > population. Re-measured directly against
  > `data/orchestrator/runs.db` (`event_type='session_resume_fallback'`,
  > dark-factory): the total was **264** (260 NULL + 4 non-NULL) when
  > the θ gate report was written on task 4005, and is **268** (260 NULL
  > + 8 non-NULL, every one of them `true`) as of 2026-08-11T15:12Z.
  >
  > The asymmetry is the point: the **NULL arm is frozen at 260** while
  > the non-NULL arm grows. That is not a rounding quibble — it is the
  > evidence that the field is live and emitting, which is what
  > dissolves the fleet-staleness reading recorded in PRD §5. See the
  > correction there.
- DAG-direction: 3578 depends on θ (edge wired at decompose); θ has no
  upstream deps (3727 already done). **PASS**

## ι — post-3578 eligibility reassessment

- Subjects exist: 3728/3729/3730/3733 all `pending` (verified via
  `get_statuses` 2026-08-10); 3731 pending (owned by the eligibility
  PRD; in scope of the reassessment sweep only if that PRD's owner
  agrees — ι's text scopes it to the four the brief names). **PASS**
- DAG-direction: depends on 3578 (upstream, pending). Decision task —
  `execution_class: decision`, no code capability asserted. **PASS**

## FAIL bindings

None. The batch queues clean.

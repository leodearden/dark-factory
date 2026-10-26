# INV-12 day-one rulings

The operator's rulings on the INV-12 day-one docket, from the sitting for
**esc-5603-1** held on 2026-10-18/19. The docket is step δ of
`plans/inv12-exceptions-owned-or-ratified-prd.md` (task 5603), Appendix A parts
1–3. Every figure was measured against main `79f93433e6`.

**ζb (task 5609) applies this file mechanically.** Each `Policy` below names a
row of `docs/legibility/exception-ratifications.yaml`, and all of those rows are
committed alongside this file. ζb does not edit the ratification table.

## How to read the tables

- **list_id**: for a list that is already declared, the id that
  `uv run --project shared python scripts/exception_register.py --report`
  prints. For a list not yet declared (Part 2, and the two missed lists at the
  end of Part 1), it is the `path::symbol` the report's unregistered-candidates
  section prints. ζb declares such a list and mints its id following the
  existing convention.
- **key-or-default**: `default` means the declaration's default disposition.
  A key is spelled as the declaration spells it.
- **disposition literal**: one of D2's literals, `Debt(TaskRef(N))` or
  `Policy('<row id>')`, or `design` for a Part 2 list ruled not governed.

## Part 1: governed lists

| list_id | key-or-default | disposition literal |
|---|---|---|
| `scripts.inline_suppression_baseline.json` | `default` | `Policy('inv12-shrink-only-baselines')` |
| `orchestrator.tests.merge_lane_ratchet_baseline.json` | `default` | `Policy('inv12-shrink-only-baselines')` |
| `fused_memory_scripts.check_bare_magicmock_config._WALL_CLOCK_DEADLINE_DEBT` | `default` | `Debt(TaskRef(6201))` |
| `shared.task_metadata._LEGACY_RETRY_LEDGER_COUNTER_KEYS` | `default` | `Debt(TaskRef(7135))` |
| `scripts.check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST` | `default` | `Policy('inv12-dashboard-unit-divergence')` |
| `scripts.tests.test_design_invariants_consistency.PINNED_SITES` | `default` | `Debt(TaskRef(7136))` |
| `scripts.tests.test_merge_state_vocabulary_consistency.PINNED_SITES` | `default` | `Debt(TaskRef(7136))` |
| `shared.task_metadata._BLESSED_METADATA_KEYS` | `default` | `Policy('inv12-task-metadata-blessed-keys')` |
| `fused_memory.memory_metadata.BLESSED_METADATA_KEYS` | `default` | `Policy('inv12-memory-metadata-blessed-keys')` |
| `workspace.pyproject_toml.ruff_ignores` | `tool.ruff.lint.ignore::E501` | `Policy('inv12-ruff-e501-line-length')` |
| `workspace.pyproject_toml.ruff_ignores` | `orchestrator/pyproject.toml::tool.ruff.lint.per-file-ignores::tests/**/*.py::F811` | `Debt(TaskRef(7137))` |
| `dashboard.config._LEGACY_CONFIG_NAMES` | `default` | `Debt(TaskRef(7138))` |
| `shared.tests.silent_fallthrough_allowlist.ALLOWLIST_ENTRIES` | `default` | `Debt(TaskRef(7139))` |
| `shared.tests.loop_blocking_allowlist.AUDITED_SITES` | `default` | `Debt(TaskRef(5270))` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `shared/src/shared/usage_gate.py::UsageGate.__init__` | `Policy('inv12-config-dir-unarchived-by-design')` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `shared/src/shared/usage_gate.py::UsageGate.__init__#2` | `Policy('inv12-config-dir-unarchived-by-design')` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `scripts/legibility/session_runner.py::SessionRunner.__init__` | `Policy('inv12-config-dir-unarchived-by-design')` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._resume_guard_reason` | `Debt(TaskRef(3972))` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `fused-memory/src/fused_memory/reconciliation/stages/base.py::BaseStage.run` | `Debt(TaskRef(3972))` |
| `shared.tests.config_dir_archival_allowlist.AUDITED_SITES` | `fused-memory/src/fused_memory/middleware/task_curator.py::TaskCurator._transcript_config_dir` | `Debt(TaskRef(6008))` |
| `scripts.merge_lane_metrics.SIZE_CEILING_EXEMPT` | `default` | `Debt(TaskRef(7140))` |
| `fused-memory/tests/test_timeout_marker_inversion_guard.py::_GRANDFATHERED` | `default` | `Debt(TaskRef(6337))` |
| `shared/tests/test_timeout_marker_inversion_guard.py::_GRANDFATHERED` | `default` | `Debt(TaskRef(6337))` |

Notes for ζb:

- **Loop-blocking ledger.** Only the default changes. It covers the 28
  `accepted` rows, and their owner 5270 was widened in this sitting to offload
  them. The 52 existing per-row overrides stand unchanged.
- **Config-dir archival.** The six keys above replace the single default of 6.
  Represent them however the declaration reads best, for example a `Policy`
  default covering 3 plus three `Debt` overrides.
- **Ruff ignores.** The two keys are ruled differently, so neither can be the
  whole default.
- **Wall-clock deadline debt.** 6201 (group 3) covers 4 of the 46 files. Each
  campaign architect files the next group as a follow-up. If that chain breaks,
  the owner sweep reports the dead owner.
- **The two `_GRANDFATHERED` lists** were missed by the docket and are ruled
  governed here. Their owner 6337 is the task their own comments' follow-up
  ticket resolved to. Declare them from the fused-memory and shared declaration
  modules.

## Part 2: borderline lists

The operator accepted the PRD's split under D1: a suppressor of a detector is
governed, a grant of a capability is design. The governed lists include the
third `_EXEMPT_EXECUTION_CLASSES`, in `gitignored_deliverable_guard.py`, which
the docket omits. All three copies are the same derived set, every execution
class except `code_tdd`, so a newly added class silently joins each exemption.
Each governed list gets its own `Policy` row.

### Governed: declare each list (a YAML-backed list is declared from a fused-memory test that loads its keys)

| list_id | key-or-default | disposition literal |
|---|---|---|
| `orchestrator/src/orchestrator/harness.py::_BY_DESIGN_SESSION_RESUME_REASONS` | `default` | `Policy('inv12-by-design-session-resume-reasons')` |
| `scripts/sweep_toolcall_markup.py::NEVER_TOUCH` | `default` | `Policy('inv12-toolcall-markup-never-touch')` |
| `fused-memory/config/cancelled_premise_blocklist.yaml` | `default` | `Policy('inv12-cancelled-premise-blocklist')` |
| `fused-memory/config/recon_code_fix_premise_registry.yaml` | `default` | `Policy('inv12-recon-code-fix-premise-registry')` |
| `fused-memory/src/fused_memory/reconciliation/flag_dedup.py::_SUPPRESSED_SNAPSHOT_CATEGORIES` | `default` | `Policy('inv12-flag-dedup-snapshot-categories')` |
| `fused-memory/src/fused_memory/reconciliation/flag_dedup.py::_SUPPRESSED_CEILING_CATEGORIES` | `default` | `Policy('inv12-flag-dedup-ceiling-categories')` |
| `fused-memory/src/fused_memory/reconciliation/internal_writers.py::INTERNAL_WRITER_AGENT_ID_PREFIXES` | `default` | `Policy('inv12-internal-writer-agent-id-prefixes')` |
| `fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py::PROTECTED_AUDIT_KINDS` | `default` | `Policy('inv12-mem0-tombstone-protected-audit-kinds')` |
| `fused-memory/src/fused_memory/middleware/operational_suggestion_guard.py::_WEAK_MARKER_LABELS` | `default` | `Policy('inv12-operational-suggestion-weak-markers')` |
| `fused-memory/src/fused_memory/middleware/recon_claim_verification_guard.py::_EXCLUDED_PATHSPECS` | `default` | `Policy('inv12-claim-verification-excluded-pathspecs')` |
| `fused-memory/src/fused_memory/middleware/routing_intent_guard.py::_EXEMPT_EXECUTION_CLASSES` | `default` | `Policy('inv12-routing-intent-exempt-classes')` |
| `fused-memory/src/fused_memory/middleware/operational_suggestion_guard.py::_EXEMPT_EXECUTION_CLASSES` | `default` | `Policy('inv12-operational-suggestion-exempt-classes')` |
| `fused-memory/src/fused_memory/middleware/gitignored_deliverable_guard.py::_EXEMPT_EXECUTION_CLASSES` | `default` | `Policy('inv12-gitignored-deliverable-exempt-classes')` |

### Design: seed `scripts/exception_register.py::NOT_EXCEPTION_LISTS` (the last column is each row's reason)

| list_id | disposition literal | reason |
|---|---|---|
| `escalation/src/escalation/authority.py::ROLE_LEVEL_ALLOWLIST` | `design` | Grants each identity the escalation levels it may resolve or park. |
| `escalation/src/escalation/authority.py::L2_AUTO_CLOSE_ALLOWLIST` | `design` | Grants which L2 escalations may be closed automatically. |
| `escalation/src/escalation/authority.py::PROMOTE_ALLOWED` | `design` | Grants which identities may mint an L2 via promote_to_l2. |
| `escalation/src/escalation/models.py::BORN_AT_L2_SEVERITIES` | `design` | Defines which severities are filed directly at L2, routed to a human. |
| `orchestrator/src/orchestrator/merge_lane/worker.py::AUTO_HEAL_MECHANICAL_CATEGORIES` | `design` | Grants which merge failure categories the lane may auto-heal. |
| `orchestrator/src/orchestrator/recovery_pins.py::MERGE_REMEDIABLE_ESC_CATEGORIES` | `design` | Grants which escalation categories a landed merge remediates. |
| `orchestrator/src/orchestrator/verify_categories.py::ARCHIVE_DENY_LIST` | `design` | A projection of CATEGORY_POLICY, the verify failure-category policy table. |
| `orchestrator/src/orchestrator/verify_categories.py::PREEXISTING_BREAK_SKIP_CATEGORIES` | `design` | A projection of CATEGORY_POLICY, the verify failure-category policy table. |
| `orchestrator/src/orchestrator/verify_categories.py::INFRA_TRANSIENT_CATEGORIES` | `design` | A projection of CATEGORY_POLICY, the verify failure-category policy table. |
| `orchestrator/src/orchestrator/verify_categories.py::INDETERMINATE_VERDICT_CATEGORIES` | `design` | A projection of CATEGORY_POLICY, the verify failure-category policy table. |
| `fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS` | `design` | Grants which fused-memory config fields hot-reload. |
| `orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` | `design` | Grants which orchestrator config fields hot-reload. |
| `fused-memory/src/fused_memory/server/entity_mint_authz.py::_DEFAULT_ALLOWED_PREFIXES` | `design` | Fail-closed default grant of which agent_id prefixes may mint entities. |
| `fused-memory/src/fused_memory/server/mem0_update_authz.py::_DEFAULT_ALLOWED_PREFIXES` | `design` | Fail-closed default grant of which agent_id prefixes may update Mem0 records. |
| `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_BRIEFING_REFRESH_PROJECT_ALLOWLIST` | `design` | Grants which projects may use the briefing-refresh hook. |
| `scripts/systemd_unit_parity.py::ENV_POLICY` | `design` | The unit environment policy table; grants which names are host-local and preserved by the renderer. |

`ENV_POLICY` replaces the docket's `scripts/render_dashboard_unit.py::HOST_LOCAL_ENVIRONMENT`.
That constant no longer exists, because task 6757 folded it into the table. Only
the table's divergence projection is governed, through
`scripts.check_dashboard_unit_parity.DIVERGENCE_ALLOWLIST` in Part 1.

## Part 3: questions

1. **Class table.** Ruled EMPTY. `scripts/inline_suppressions.py::RATIFIED_SUPPRESSION_CLASSES`
   gets no rows, so each new suppression carries its own inline disposition. The
   docket projected about 310 new markers per 30 days for the four candidate
   classes. Since the 2026-10-10 seed, 152 merges added 5 owned markers and
   caused 0 gate violations. The valve stays available if inflow rises.
2. **Is a mechanical rename a touch?** CONFIRMED (D7). A rename that rewrites a
   marker-bearing line is a touch. The exit is one follow-up ticket cited on
   every rewritten marker. No re-keying mechanism is built.
3. **Burn-down.** CONFIRMED NONE (D10). Grandfathered suppressions convert when
   their line is touched, and the baseline only shrinks. This matches
   `Policy('inv12-shrink-only-baselines')` in Part 1.
4. **Dead markers.** Ruled a separate operator task after θ: task **7141**, held
   behind 5611. It runs the RUF100 autofix per workspace member under a
   merge-queue halt, like the κ1 seed, then `--tighten`. Measured at the repo
   root on 2026-10-18: 2,169 unused-noqa against the docket's 1,499. Whether
   the two counts used the same method is not confirmed. D8 already rejects new
   dead markers.

## Actions taken in the sitting

- **New owners filed**, each depending on ζb (5609) except 7141:
  - 7135: migrate the 21 legacy ledger blobs and delete the list.
  - 7136: shrink both PINNED_SITES maps. Restatements that can go are
    removed; the rest are escalated for a per-key ruling.
  - 7137: fix the 322 F811 sites and drop the per-file ignore.
  - 7138: retire the legacy config-name fallback.
  - 7139: burn down the 12 silent-fallthrough sites.
  - 7140: author the git_ops.py split PRD.
  - 7141: the RUF100 operator task, after θ 5611.
- **5270 widened** to own and offload the 28 `accepted` loop-blocking rows.
  `accepted` is no longer a resting state in that ledger.
- **Docket drift corrected:**
  - Five lists are retired and need no ruling: `_KNOWN_LOAD_FLAKE_NODEID_RES`
    (5578), the orchestrator `_GRANDFATHERED` (5149), both 5034 allowlists, and
    `_DATACLASS_DOUBLE_DEBT` (4354).
  - Counts are now: silent-fallthrough 12, task-metadata blessed keys 54
    (52 at the sitting; task 4031 then added `plan_text_fp` and
    `plan_authored_at`, and the whole list is ruled `Policy`), loop-blocking
    80 rows.
  - Paths moved: `AUTO_HEAL_MECHANICAL_CATEGORIES` is now in
    `merge_lane/worker.py`, `MERGE_REMEDIABLE_ESC_CATEGORIES` in
    `recovery_pins.py`, and `HOST_LOCAL_ENVIRONMENT` became `ENV_POLICY`.
  - The curator archival gap's owner is 6008 (ticket `tkt_0RV5C7TGH7495ZW43KCPSE38KH`).
  - `SIZE_CEILING_EXEMPT` is redundant with the ratchet baseline, so the debt
    it records is the split itself.

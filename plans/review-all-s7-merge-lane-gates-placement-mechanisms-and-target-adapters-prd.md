# S7 — Merge-lane gates, placement mechanisms and target adapters (PRD)

**Status:** draft, authored 2026-11-01 by the S7 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S7, agent mode, size M, risk medium). The
critic's findings (`critique/S7.critic.md`) were folded in on 2026-11-02. The sitting's rulings of
2026-11-10 are applied, and every claim was re-checked against main on 2026-11-10 (§Rulings). The
fold-verifier pass of 2026-11-10 (`critique/S7.foldverify.md`) applied the lead's sequencing rulings
(δ first) and re-measured at `bb1c659562`. The alignment pass of 2026-11-10 applied the lead's
rulings after the fold-verifiers: S6 α lands first on every file it shares with this plan. B + H:
contracts and a two-way boundary-test sketch, and ζ is the integration gate. G5 fired on blast radius
(gates, worker, workflow, event store, config, the escalation server's `merge_request`, skills) and on
the S6 consumer. Seven leaves, plus one external reify block; no held leaf unless S5 has not filed by
decompose (γ). Ruling R11 governs, as amended twice on 2026-11-10 (§Rulings). R22 item 6 (the
invariant) and R23 (weakened keys re-verified here) also apply.

**Code anchors** verified against main `bb1c659562` (2026-11-10). Main moves fast — cite by symbol;
re-locate lines at implementation time. Provenance: authored against `ce0f71a384` (2026-11-01),
re-checked at `0935e63d88` (critic) and `d40ba048e9` (fold, 2026-11-02); the run pinned `6551b75aba`
(2026-10-08); the ruling fold re-checked at `e634cfe658`. Between `d40ba048e9` and `bb1c659562` main
took about 2,600 commits. On the paths this plan touches:

- `merge_lane/gates.py` changed only in `ALREADY_LANDED_CITATION_PATTERN` (task 6221); every gate
  function this plan converts is byte-identical, and the file is byte-identical from `e634cfe658` to
  `bb1c659562`.
- `merge_lane/worker.py` took +710/−652 (tasks 5290, 5302, 8523, 6670) and then task 8524 (done
  2026-11-10, the re-merge re-anchor). 8524 split the re-merge path: `classify_and_merge` is now called
  from `_merger_loop`, `_remerge_on_tip` and `_remerge_on_main` (×2), not from `_remerge_inner` (D3).
  No other symbol this plan edits moved; every span below was re-measured.
- `workflow.py`'s plan-files and already-landed call sites now measure from `own_work_base` (task 7668).
- `git_ops.py` gained `GitQueryFailed` (task 7485).

Every count below is labelled `measured at bb1c659562`, or names the date and store it was read from.

## Goal

Today a merge gate that cannot measure looks the same as a gate that passed. After this PRD it says
so: as a typed value, in an event, on the merge's own `merge_attempt` row, and under a storm counter.
Whether each failure arm fails open or closed is decided in one table. Two dead placement mechanisms
leave the lane's hot path, and the live one, deep chain, stays. The dormant reify retry narrowing is
deleted end to end. The resource audit stops parsing its own strings.

What a user observes when this lands:

- **A failing git probe is recorded where you look.** When a git probe inside a gate fails (for
  example a `merge-base` that cannot resolve across a shallow boundary):
  - `data/orchestrator/runs.db` gains one `merge_gate_unmeasured` event naming the gate, the probe,
    `rc` and the stderr tail;
  - that merge's `merge_attempt` row carries `gates_unmeasured: [<gate>]`;
  - the merge's fate is the probe's row in `UNMEASURED_POLICY`, which reproduces today's direction
    probe by probe;
  - a persistent failure files one L1 per gate under `merge_gate_unmeasured_storm`.
- **The plan-files gate stops misreporting a `git log` failure.** It no longer reports every declared
  file as "not touched by branch" (fk-689227abfab1) and no longer sends the task to a "restart from
  scratch" triage.
- **The dead sentinels and placement code are gone, and deep chain keeps working.**
  `git grep -E '_OVERLAP_GIT_ERROR_SENTINEL|_RENAME_PROBE_UNMEASURABLE|AUTO_CHAIN_GENERATIONS_ENABLED|class ProbePlacement' orchestrator/src`
  is empty. `merge_verify` rows with `chain_items > 1` and `merge_finalized` rows with
  `landed_via_chain` keep appearing in both projects.
- **The reify retry narrowing is gone.** `merge_request` no longer takes `retry_failed_only`, and
  `merge_verify` events no longer carry `retry_scope` or `retry_subset_sizes`.
  `git grep -E 'retry_failed_only|_REIFY_ATTEMPT_SIDECAR_RELPATH|retry_scope_event_fields' orchestrator/src escalation/src`
  is empty. Nothing observed in production changes: 0 of 5,431 `merge_verify` events since 2026-07-20
  were narrowed (3,433 dark-factory, 1,998 reify; both run stores read 2026-11-10).
- **A design invariant enforces the rule.** `merge-gate-unmeasured-typed` (INV-17 under the lead's
  order) has a guard that flags seeded fixtures and passes the post-γ tree.

## Background

The run found four encodings of "could not measure" in one file:

- an empty pass value (`DropGuardResult()`, `[]`, `PostMergePyrightResult()`);
- the in-band string list `_OVERLAP_GIT_ERROR_SENTINEL`;
- the typed singleton `_RENAME_PROBE_UNMEASURABLE`;
- `None` overloaded in `_resolve_already_landed_branch`.

Each is recorded only by `logger.warning`, which INV-11 `no-silent-fail-soft` forbids. The run also
found three speculative-placement mechanisms in the hot path believed off in every config, about 730
reify-specific lines in the generic worker, and a regex that re-parses the resource audit's own
f-strings.

### Freshness (re-measured at `bb1c659562`)

**Fail-open pass-value returns: 12, reproduced.** This is an AST enumeration of returns guarded by
`rc != 0`, by `except`, or by an unmeasured sub-probe:

- `_check_plan_targets_in_tree`: 5 `return DropGuardResult()` (no second parent, merge-base, branch
  diff, deletion diff, `real_drops is None`);
- `_check_post_merge_equivalence`: 6 `return []` (rev-parse HEAD, merge-base, branch diff, main diff,
  `compare_set is None`, scoped diff);
- `_check_post_merge_pyright`: 1, its `except Exception` arm.

Its legitimate empty returns are excluded: `allow_worktree_head_fallback=False`, an empty
`branch_touched` and an empty `compare_set`.

**The other encodings are all still present.**

- `_rebase_delta_touched_overlap` returns `_OVERLAP_GIT_ERROR_SENTINEL` at 5 sites: 4 `rc != 0` arms
  and the falsy-`rebased_from` precondition. `_decide_reverify_gate` compares to it by `==`. The name
  has 12 matches in gates.py (the definition, the 5 returns, the compare and 5 prose mentions) and 1 in
  `skills/merge-queue/references/two-layer-model.md`.
- `_rename_pair_for` returns `_RENAME_PROBE_UNMEASURABLE` at 2 sites (4 matches with the definition and
  a docstring mention).
- `_resolve_already_landed_branch` returns `None` on 2 rc arms and 1 except arm, the same `None` as
  "not landed".
- Its never-raises wrapper `resolve_already_landed_branch`, which is what `workflow.py` calls, returns
  `None` from an `except Exception` arm.
- gates.py has 3 `_emit_merge_attempt` call sites (auto-chain, and two in `_finalize_advanced_merge`).
  None is on these paths, and 2 remain after β.

**Sub-probe directions are mixed by design.** This is the critic's B1, measured at `0935e63d88`; the
functions are unchanged at `bb1c659562`.

Fail-CLOSED arms:

- `_branch_delta_survives` (4 returns; "EVERY failure mode returns False, i.e. 'keep flagging'"; called
  by both the drop guard and the equivalence compare set);
- `_commit_is_linear` (2; "Fail-closed … never masks a drop");
- the plan-files gate's `git ls-tree` rc arm (entry UNCLASSIFIED → `not_touched` → block);
- its nested `_tree_paths` arm ("Fail closed");
- `_rename_pair_for` ("so the resolver can fail CLOSED");
- `_path_existed_in_branch_history` ("Fails CLOSED").

Fail-OPEN arms:

- `_rename_aware_real_drops` and `_rename_aware_compare_set` (`None` → the gate's pass value);
- `_resolve_second_parent` ("caller falls back to the next tier").

The drop guard is therefore fail-closed *by blocking* today on its delta-survival arm, through
`worker.py::classify_and_merge` → `MergeOutcome('blocked', DROPPED_PLAN_TARGETS…)`.

**Exit codes that are answers, not failures.** These were also measured by the critic.

- `_resolve_second_parent`'s `rc != 0` on a linear commit means "no second parent".
  `git rev-parse <linear>^2` exits 128. It runs on every landing (`_finalize_advanced_merge`,
  `worker.py::_land_chain_prefix`).
- `_branch_delta_survives`'s `rev-parse --verify merged:path` rc means "no blob": a measured absence.
- `_resolve_already_landed_branch`'s `ahead != 0` is a measured "did real work".

**fk-689227abfab1 is still present.**

- The docstring of `gates.py::_check_plan_files_touched_in_branch` promises "Fail-open on git error",
  but `git_ops.py::GitOps.get_files_touched_in_branch` still returns `[]` on `rc != 0`, so the gate puts
  every declared file in `not_touched`.
- The `workflow.py::TaskWorkflow` merge phase has two silent caller arms. Both were present at
  `ce0f71a384`, though the draft named only the first:
  - `if rc == 0 and branch_head.strip():` skips the gate entirely when `git rev-parse HEAD` fails;
  - after a successful narrowing pass, `if rc2 == 0 and branch_head2.strip():` skips the re-check, so
    the pre-narrowing `not_touched` stands and blocks.
- Task 7668 (done) made both gate calls and the already-landed probe measure from
  `TaskWorkflow._own_work_base` (`branch_base.py::own_work_base`). That helper falls back to the
  stored base with a WARNING when its merge-base fails. It is shared with review and the architect and
  is not a gate probe, so it stays out of scope.
- Task 7485 (done) added `git_ops.py::GitQueryFailed`, raised by `git_ops.py::_git_query`. It is a
  message-only `RuntimeError` with no `returncode` or `stderr` field (D6).

**Placement mechanisms: ruled 2026-11-10 (S7-1 = A).**

- **Probe.** `select_probe_depth`, `ProbePlacement`, `_probe_verify_placement`, `_available_built_depth`,
  `_built_depth_tip`, `_recent_verify_fail_rate` and `_record_verify_outcome` span 275 lines (measured
  at `bb1c659562`). `probe_fraction` is `0.0` in dark-factory's and reify's
  `dark-factory-orchestrator.yaml` (both read 2026-11-10). It was 0.5 for four days in July and has
  delivered nothing usable: its depth labels were declared wrong and excluded from calibration.
  `_record_verify_outcome`'s single feed is in `_finalize_inflight`, and `_recent_verify_fail_rate` is
  its window's single reader (measured at `bb1c659562`). `_note_chain_outcome`'s docstring says it
  "does not append to `_recent_verify_outcomes`".
- **Auto-chain.** `_maybe_auto_chain_generation` (141 lines), plus `gates.py::_GenerationChainContext`
  (22) and `_auto_chain_on_equivalence_blocked` (31), total 194 lines (measured at `bb1c659562`).
  `AUTO_CHAIN_GENERATIONS_ENABLED` is still a `False` module constant. Neither run store has a
  `merge_attempt` row with outcome `post_merge_generation_chained` (read 2026-11-10).
- **Deep chain is live and kept.** `merge_deep.chain_cap: 6` has been live in reify since `349aa26775`
  (2026-10-20) and in dark-factory since `2389803498` (2026-10-27, task 7619, under Leo's ruling B of
  2026-10-20). Its definitions span 908 lines in worker.py (measured at `bb1c659562`). Leo ruled on
  2026-11-10 to keep it (S7-1 = A). It is owned by `plans/deep-merge-ahead-prd.md` and
  `plans/merge-lane-policy-prd.md`. Their pending tasks build on it: 7720, 7721, 7824, 7825, 7810,
  7828, 8527 and 8528 (all high) and 7922.

**Retry narrowing: ruled deleted (S7-3 = B).**

- The producer section runs from its banner after `_CROSS_CHECK_SENTINEL` through
  `_build_attempt0_payload`: 733 lines. The AST span from `_RETRY_FILTER_SUBDIR` to
  `_build_attempt0_payload` is 676 of them.
- worker.py holds 9 `verify.sh:NNNN` pins, and `grep -ciE 'reify|nextest'` counts 179 worker.py lines
  and 86 shadow.py lines. All measured at `bb1c659562`.
- 0 of 5,431 `merge_verify` events since 2026-07-20 carry `retry_scope='failed_only'` (read
  2026-11-10).
- No in-repo caller passes `merge_request(retry_failed_only=True)`. The producer runs only when
  `runner is None` (a local dispatch), and `verify_runner.py::LocalRunner.run_merge_verify` never
  reads `spec.verify_env` (6576).
- shadow.py holds eight retry-only functions and two constants with no caller outside the producer
  (measured at `bb1c659562`): `build_fail_fast_map`, `did_not_pass_subset`, `nextest_filter_ids`,
  `_nextest_case_is_planned`, `parse_nextest_list_planned`, `parse_failed_run_all_members`,
  `select_failed_run_all_members`, `merge_retry_shadow_baseline` (342 lines of definitions),
  `_RUN_ALL_FAILED_MARKER_RE` and `_RUN_ALL_PARTIAL_MARKER_TOKEN`.
  - The warm and cold shadow use only `parse_per_test_results`, which stays.
  - The sitting's discussion text said "the nextest and run_all parsers stay, because the warm shadow
    uses them". Measured, that holds for `parse_per_test_results` alone, so δ deletes the rest under
    the ruling's "end to end".
- reify needs no change: its `scripts/verify.sh` reads `REIFY_VERIFY_RETRY_SCOPE` only when set.
- Task 6575 (pending, low, filed under Leo's 2026-10-09 ruling) re-measures the retry rate after a
  30-day delay, which elapsed on 2026-11-08, and then either escalates or removes the dormant
  producer. Leo's 2026-11-10 ruling removes the narrowing unconditionally, and more of it (the flag,
  the MCP parameter, the event keys), so δ subsumes 6575 (§Existing tasks).

**Resource audit: weakened under R23, kept.**

- `_RESOURCE_AUDIT_AGE_FIGURE_RE` and `_resource_audit_fingerprint` still re-parse the
  `(age Ns > grace Ms)` wording of `SpeculativeMergeWorker.worktree_ledger_violations`.
- The skeptic's point stands: `test_the_volatile_age_figure_is_stripped` drives the real producer, so a
  rewording breaks a test rather than failing silently.
- What remains is the cost the heuristic *structured data instead of meaningful strings* names: a regex
  over self-produced text, plus 24 test references to the regex or the fingerprint (measured at
  `bb1c659562`).
- The leaf is small and real, so it stays at low priority.

## Findings disposition

| Key | Sev | Verdict | Disposition | Note |
|---|---|---|---|---|
| `fk-88e3c228b649` | medium | confirmed | leaf α1 | α1 lands the type, the policy and the reporter. α2 converts the 12 fail-open sites, γ converts the remaining encodings and ζ enforces the rule; all three also carry the key. |
| `fk-689227abfab1` | medium | confirmed (prior row, re-observed) | leaf γ | `get_files_touched_in_branch` still returns `[]` on `rc != 0`, and the gate docstring still promises fail-open (measured at `bb1c659562`). γ wraps S5/C4's typed query failure (D6) and depends on 3918, which rewrites the same query. |
| `fk-7d1580bb3f03` | medium | weakened | leaf β | β deletes the probe and auto-chain, which closes the key. The deep-chain third is dropped under G6 because its cost is gone, not accepted: the cost was dead placement code, and deep chain has been live since 2026-10-20 (reify) and 2026-10-27 (dark-factory). Leo kept it on 2026-11-10 (S7-1 = A, an amendment to R11), owned by the deep merge-ahead and merge-lane policy plans. No acceptance task is filed. |
| `fk-dd411ada7c29` | low | confirmed | leaf δ | Deleted end to end under S7-3 = B (an amendment to R11): about 1,300 source lines leave `orchestrator/src` and `escalation/src` (measured at `bb1c659562`). δ subsumes 6575 and replaces 6576 (lead actions). |
| `fk-5d4406afeade` | low | weakened | leaf ε | Re-verified under R23. The h12 smell holds, but the silent-storm cost is mostly covered by a real-producer test, so priority stays low. |

## Resolved design decisions

- **D1 — one gate-measurement module.** `orchestrator/src/orchestrator/merge_lane/gate_measurement.py`
  owns `GateId`, `ProbeId`, `Measured`, `Unmeasurable`, `OnUnmeasured`, `UNMEASURED_POLICY`,
  `StormEscalationSink` and `GateUnmeasuredReporter`.
  - R11 names gates.py as the owner. But gates.py is 3,668 lines (measured at `bb1c659562`), and both
    the worker and the workflow must import the type without importing every gate. So the type gets
    its own module and gates.py imports it, per the heuristics *no file too large* and *files make
    internal sense in isolation*.
  - The module imports nothing from `merge_lane/ports.py` at runtime. `ports.py` imports
    `gates.PostMergePyrightResult` at runtime (still true at `bb1c659562`), so a runtime import would
    close the cycle ports → gates → gate_measurement → ports. Escalation types are named only under
    `TYPE_CHECKING`.
  - File placement was ruled by the lead (OQ-5): `gate_measurement.py` and `resource_audit.py` under
    `merge_lane/`. Under S7-3 = B no target-verify module is created.
- **D2 — decided in one place, per probe (S7-4, ruled as recommended).**
  - `UNMEASURED_POLICY` is keyed by a closed `ProbeId` enum: one member per (gate, git step), one per
    rc or except arm a gate or its probes has. Its values are directional:
    - `pass_` treats the arm as no finding;
    - `flag` treats it as a finding (the closed direction);
    - `recheck` takes the gate's re-measure path.
  - A call site never chooses a direction. It executes the disposition `settle` hands back. Each `ProbeId`
    declares which dispositions its call site can execute (`SUPPORTED`), and the table is validated
    against `SUPPORTED` at import. Flipping one arm is then a one-row edit that cannot name an action the
    site does not implement.
  - `GateId` remains the event's `gate` field and the storm key.
  - Every row reproduces today's direction, with one change (S7-2, ruled as recommended). The rows are:

  | ProbeId | Today | Row |
  |---|---|---|
  | `drop_guard.second_parent` (a parent-count failure, or a one-parent input: the guard's input is `classify_and_merge`'s own `--no-ff` merge), `.merge_base`, `.branch_diff`, `.deletion_diff`, `.rename_pairs` | fail-open (`DropGuardResult()`) | `pass_` |
  | `drop_guard.delta_survival` | fail-closed ("keep flagging") | `flag` |
  | `equivalence.rev_parse_head`, `.merge_base`, `.branch_diff`, `.main_diff`, `.rename_pairs`, `.scoped_diff` | fail-open (`[]`) | `pass_` |
  | `equivalence.delta_survival` | fail-closed | `flag` |
  | `finalize.second_parent` | fail-open (next tier) | `pass_` |
  | `finalize.linearity` | fail-closed (never masks a drop) | `flag` |
  | `pyright.infra`, `pyright.timeout` | fail-open (clean) | `pass_` |
  | `overlap.rebased_from_missing`, `.rev_parse_head`, `.merge_base`, `.branch_diff`, `.onto_diff` | fail-closed (`ReverifyGateReason.overlap_unknown`) | `recheck` |
  | `already_landed.count`, `.count_parse`, `.subject`, `.wrapper` | "not landed" | `pass_` (no finding = not landed) |
  | `plan_files.rev_parse_head` (the workflow's `git rev-parse HEAD` before the gate) | gate skipped, merge proceeds | `pass_` |
  | `plan_files.touched_set` | accidental fail-closed (every file `not_touched`) | `pass_` (S7-2) |
  | `plan_files.ls_tree`, `.tree_listing`, `.rename_pair`, `.history` | fail-closed (entry stays `not_touched`, no basename resolve) | `flag` |
  | `plan_files.recheck_rev_parse_head` (the post-narrowing re-check's `git rev-parse HEAD`) | re-check skipped, the pre-narrowing `not_touched` stands and blocks | `flag` |

  The exact member spellings are tactical. They must stay one per arm, which ζ's guard checks:
  every `Unmeasurable(...)` construction names a `ProbeId`, and every `ProbeId` has a row.
- **D3 — the worker owns the reporter, and S6 injects the policy.**
  - `SpeculativeMergeWorker` constructs one `GateUnmeasuredReporter` from its event store, its raw
    `escalation_queue` (as a `StormEscalationSink`) and `UNMEASURED_POLICY`, and exposes it as the
    read-only property `gate_reporter`.
  - Gate runners receive it through `_PostAdvanceContext` and through `classify_and_merge`'s and
    `_finalize_advanced_merge`'s callers. The finalize callers are `_do_train_merge`,
    `_land_chain_prefix`, `_finalize_landed_by_successor` and `_land_advanced_candidate`. The classify
    callers are `_merger_loop`, `_remerge_on_tip` and `_remerge_on_main` (×2), after task 8524. Counts
    measured at `bb1c659562`: 4 `_finalize_advanced_merge` call sites and 4 `classify_and_merge` call
    sites in worker.py.
  - `TaskWorkflow` already holds `merge_worker`, and its pre-merge gates use
    `self.merge_worker.gate_reporter`. When `merge_worker` is `None` (tests, CLI), it constructs its own.
  - A lane therefore has one storm window, and no harness wiring changes.
  - The policy argument is how S6 "injects the fail-open policy" (G4 row). If task 5040 has landed by
    then, `VerifyDispatcher` receives the same instance and never constructs a second.
- **D4 — storm escape per the house pattern (OQ-CQ4, ruled by the lead).**
  - The pattern is `merge_lane/landing_evidence.py::LandingTally` plus
    `file_landing_git_error_storm_escalation`: one `shared.storm_counter.StormCounter` per `GateId`, and
    strict exceedance of `GATE_UNMEASURED_STORM_THRESHOLD` per `GATE_UNMEASURED_STORM_WINDOW_SECS` files
    one L1.
  - The L1 is filed against the per-gate sentinel `__merge_gate_unmeasured_storm__:<gate>`, never the
    settling task's id: a real task id would be read by the recovery predicates as a hold. Dedup is
    `has_open_l1(<that sentinel>, category='merge_gate_unmeasured_storm')`, so an open drop-guard storm
    does not suppress an equivalence storm.
  - The filing goes through `StormEscalationSink`, a three-method Protocol in `gate_measurement.py`
    (`make_id(key)`, `has_open_l1(task_id, *, category)` and `submit(escalation)`), because
    `merge_lane/ports.py::EscalationPort` exposes only `file(record)`. `make_id` is needed because the
    house pattern mints the escalation id from the queue
    (`landing_evidence.py::file_landing_git_error_storm_escalation` builds
    `Escalation(id=escalation_queue.make_id(...))`). The three signatures match
    `escalation/queue.py::EscalationQueue` at `bb1c659562`. S1's `EscalationPort` widening (R8) absorbs
    the sink later.
  - The L1 does not resolve itself. While it is open, further storms of that gate file nothing.
  - Thresholds are module constants (tactical, decided in α1; default more than 5 per gate in
    3,600 s), not config leaves. Every unmeasured result emits the event whether or not the storm
    fires.
- **D5 — fail-closed today means `flag`, not a new outcome.**
  - Several arms are fail-closed by blocking today (the delta-survival, linearity and plan-files
    per-entry arms above). They block through the gate's own existing finding (the dropped file, the
    `not_touched` entry), so `flag` needs no new `MergeOutcome` kind.
  - A future "block with an honest *could not measure* reason" disposition would need a new kind, which
    S6 owns under R1 (`MergeOutcome.kind`). That is out of scope.
- **D6 — the git query carries its failure; S5 owns how.**
  - S5's draft C4 (its leaf ε) makes every `GitOps` read query, `get_files_touched_in_branch` included,
    return the parsed value or raise `GitQueryError(argv, cwd, returncode, stderr)`. Its G4 row says
    "S7's `GateMeasurement` wraps it".
  - Task 7485 (done) has since landed `git_ops.py::GitQueryFailed`, a message-only `RuntimeError`. γ
    needs `returncode` and `stderr` as fields, so S5's C4 widens that class or replaces it. That is S5's
    call (xref returned).
  - γ consumes the C4 contract and does not edit `git_ops.py`. Each gates.py caller catches the C4
    exception and returns `Unmeasurable(probe, reason, rc=exc.returncode, stderr=exc.stderr)`.
  - γ depends on S5/C4 and on task 3918, which rewrites the same query
    (`--diff-merges=dense-combined`, base re-derivation) and the same gate. If S5 has not filed when S7
    decomposes, γ becomes a held leaf on S5/C4 (`report-format.md` §Artefact 3).
  - If 3918 lands a new `plan_delivery.py` (it is in its declared files; absent at `bb1c659562`), γ
    converts the measurement where 3918 left it, and ζ's guard scope includes that module.
  - Gates' own direct git probes (`_run(` today, `run_in_worktree(` / `query(` after S5's C1) are not
    `GitOps` queries. They convert their rc branches in place. α2's fault fixtures use real git.
- **D7 — exit codes that are answers are tested before the failure arm [B2].**
  - A probe whose exit code carries the answer tests the answering code explicitly first, and only then
    treats `rc != 0` as unmeasurable.
  - `_resolve_second_parent` first counts parents with `rev-list --parents -n 1` (the call
    `_commit_is_linear` already makes). A missing `^2` is then `Measured(None)`, and only a failing
    count is `Unmeasurable`. Without this, every linear landing would emit `merge_gate_unmeasured` and
    raise a false storm L1.
  - `_branch_delta_survives` reads "no blob" as `Measured(False)` once a preceding object-type check
    succeeded.
  - `_resolve_already_landed_branch`'s `ahead != 0` stays a `Measured` answer.
- **D8 — probe and auto-chain deleted, deep chain kept (S7-1 = A).** β deletes:
  - the probe: config model `SpeculationProbeConfig`, its `RELOADABLE_FIELDS` leaves, the dark-factory
    yaml stanza, the dispatch branch and the `probe_base` threading;
  - the probe's verify-outcome window: `_record_verify_outcome`, `_recent_verify_fail_rate`,
    `_recent_verify_outcomes`, `RECENT_VERIFY_OUTCOME_WINDOW`, `_probe_round_counter` and the
    `_finalize_inflight` feed (none has a reader outside the probe, measured at `bb1c659562`);
  - auto-chain: `AUTO_CHAIN_GENERATIONS_ENABLED`, `_GenerationChainContext`,
    `_auto_chain_on_equivalence_blocked`, `_maybe_auto_chain_generation`, `chain_ctx` on
    `_finalize_advanced_merge` and the counter cleanup;
  - the skill prose that cites auto-chain as a live path.

  Other probe helpers are deleted only once `git grep` proves they have no caller outside the probe.
  `OutcomeKind.post_merge_generation_chained` is kept. No run store holds a row with it (read
  2026-11-10), but the outcome vocabulary is S6's under R1, and `test_outcome_kind.py` and
  `scripts/merge_lane_throughput.py` mirror it. Retiring the member is S6's call (xref).

  Deep chain is untouched apart from what the deletions force: β removes the `chain_ctx` kwarg from the
  finalize call inside `_land_chain_prefix`, plus two prose references to the probe window (the
  `_chain_halving_state` init comment and the sentence of `_note_chain_outcome`'s docstring naming
  `_recent_verify_outcomes` and `_probe_round_counter`). Merge-lane policy task 8528 replaces
  `_chain_halving_state` with `ChainPolicyState`. If it lands first, the comment edit is moot, and β's
  absent checks hold either way.

  **Retiring the yaml key (OQ-CQ2, ruled by the lead).**
  - The orchestrator's unknown-key census (`config.py`'s `ConfigUnknownKey` census, and
    `harness.py::Harness._file_config_unknown_keys_escalation`, which files a severity-critical L2 and
    self-heals once the config is clean) already reports a leftover key. β adds no bespoke WARNING.
  - reify's live yaml still carries `speculation_probe:` (`probe_fraction: 0.0`,
    `suppress_flake_rate: 0.40`; read 2026-11-10), so a reify-side change removes that stanza before β
    deploys. The lead files it in reify, and β carries it as `metadata.external_deps`.
  - The 9 throwaway reify fix, recovery and unblock checkouts are left alone: a critical L2 there
    self-resolves once their yaml is clean.
  - β edits one call inside `_land_chain_prefix`. That function is deep-chain code that 7720 is
    rewriting, so β depends on 7720 (OQ-CQ7, ruled by the lead).
- **D9 — reify retry narrowing deleted end to end (S7-3 = B).** δ deletes every part. The counts are
  measured at `bb1c659562`:
  1. *The producer in worker.py*: the 733-line section (sidecar reader, nextest list probe, subset
     builder, `_REIFY_ATTEMPT_SIDECAR_RELPATH`, `_KNOWN_REIFY_PROFILES`, the 9 `verify.sh:NNNN` pins);
     the narrowing arm in `_run_post_merge_verify` (`if req.retry_failed_only and not narrowed and runner is None:`)
     with its `shadow_baseline_sink` hand-off; and the attempt-0 verdict union at the warm-shadow
     store.
  2. *The narrowed-retry budget* (task 2835): `_run_post_merge_verify`'s `max_narrowed` and
     `narrowed_retries` parameters, `MAX_POST_MERGE_VERIFY_NARROWED_RETRIES` (the class constant and its
     Protocol attribute), and `_post_merge_verify_narrowed_retries` (its init and both call sites).
  3. *The shadow helpers* in `merge_lane/shadow.py`: the eight retry-only functions and two constants
     listed under Freshness, plus `build_warm_shadow_results`' `attempt0_verdicts` parameter.
     `parse_per_test_results` stays.
  4. *The flag*: `MergeRequest.retry_failed_only` in `merge_lane/types.py`.
  5. *The tool parameter*: `retry_failed_only` on `merge_request` in `escalation/src/escalation/server.py`,
     with its docstring paragraph and the threading kwarg. This lands before the escalation-server
     plan's (S3) `merge_request` move, so that move carries one parameter fewer.
  6. *The event marker*: `verify_runner.py::retry_scope_event_fields` and the always-present
     `retry_scope` and `retry_subset_sizes` keys on `merge_verify` events, with their field note in
     `event_store.py`. Historical rows keep their `None`. Readers outside tests: none (measured).
  7. *Docs*: "narrowed-scope retries" in `ARCHITECTURE.md` §5.3, and the
     `verify_runner.py::MergeVerifySpec` docstring paragraph naming `_build_retry_verify_env`.

  8. *Fixtures*: `orchestrator/tests/fixtures/reify_verify_retry/` goes, except one file with a
     reader outside the narrowing. `test_verify_event_channel.py`'s D3 corpus reads
     `reify_verify_retry/run_all-failed-marker.txt` at import time (measured at `bb1c659562`), as a
     verify-classifier input, not a narrowing one. δ moves that file and its `PROVENANCE.md` section
     into `orchestrator/tests/fixtures/verify_classify_run_all/` and repoints the corpus entry.
     Deleting the directory whole would break that module's collection.

  The retry loop itself stays: attempt counting and the legacy `max_enospc` budget are unchanged, and
  every infra-transient retry runs the full gate, as every retry has since 2026-07-20. No module and no
  adapter registry are created, and reify is not touched. Leo's 2026-10-07 ruling on 6285 ("dormant")
  is superseded: dormant becomes removed.
- **D10 — violations as records (OQ-CQ6, ruled by the lead as a clarification of R11).**
  - `ResourceAuditViolation(kind, subject, age_secs, grace_secs, detail)` lives in a small
    `merge_lane/resource_audit.py`, with `render()` and `fingerprint()`.
  - The fingerprint is the record minus `age_secs`: (kind, subject, grace_secs, detail).
  - R11 says "(kind, subject)". The two extra static fields keep task 3203's pinned behaviour: a grace
    move and a growing permit count re-log, as `TestResourceAuditFingerprint` asserts.
  - The producers stay on the worker. `snapshot()['resource_audit']` and the escalation body render
    byte-identical strings, because `test_merge_queue_invariant_integration_gate.py` pins that key by
    exact dict equality.
- **D11 — the invariant's scope is R11's [B2].**
  - R11 fixed the checkable form as "every Return inside an except or rc != 0 branch of a `_check_*`
    function". ζ's guard covers the `_check_*` functions and their nested functions, plus every function
    whose return annotation names `GateMeasurement`, which is every probe α2 and γ convert.
  - It does not cover every git-calling function in gates.py. Measured at `0935e63d88`, R11's form has
    zero false positives, while the wider form flags three measured negatives (D7). The functions are
    unchanged at `bb1c659562`.
- **D12 — the merge's own record carries it (S7-5, ruled as recommended).** Every `_emit_merge_attempt`
  that closes a merge whose lane gates settled at least one `Unmeasurable` adds
  `gates_unmeasured: [GateId, …]` to `data`. The key is additive. The reporter holds no task state, so
  the list has an explicit carrier:
  1. Each lane-side `settle` caller appends `m.probe.gate` to a per-merge `list[GateId]`.
  2. The drop guard settles in `worker.py::classify_and_merge`, at merge time. Its list rides on
     `MergedOk` and on the in-flight item (`merge_lane/types.py`), and is passed to
     `_finalize_advanced_merge` as a new `gates_unmeasured` kwarg, which seeds
     `_PostAdvanceContext.unmeasured`.
  3. The post-advance gates append to `_PostAdvanceContext.unmeasured`. The closing `_emit_merge_attempt`
     in gates.py emits it, and so do `classify_and_merge`'s own blocked emits.
  4. Workflow-side gates (`plan_files_touched`, `already_landed`) settle in `TaskWorkflow` before
     `merge_request`. They are recorded only on their own `merge_gate_unmeasured` events, never on a
     `merge_attempt` row.

  Carrying the list on `GateVerdict`/`MergeOutcome` is S6's type (xref).

## Contract

```python
# orchestrator/src/orchestrator/merge_lane/gate_measurement.py   (C1, C2 land in α1; S7 owns)
class GateId(StrEnum):
    drop_guard = 'drop_guard'; plan_files_touched = 'plan_files_touched'; already_landed = 'already_landed'
    post_merge_equivalence = 'post_merge_equivalence'; post_merge_pyright = 'post_merge_pyright'
    rebase_delta_overlap = 'rebase_delta_overlap'; finalize = 'finalize'

class ProbeId(StrEnum):            # one member per rc/except arm (D2 table); .gate -> GateId
    ...

@dataclass(frozen=True)
class Measured(Generic[T]):
    value: T
    measured: ClassVar[Literal[True]] = True

@dataclass(frozen=True)
class Unmeasurable:
    probe: ProbeId        # closed vocabulary; .gate derives the GateId
    reason: str           # one human sentence
    rc: int | None        # None for an exception arm
    stderr: str           # tail, <= 400 chars
    measured: ClassVar[Literal[False]] = False

GateMeasurement = Measured[T] | Unmeasurable

class OnUnmeasured(StrEnum):
    pass_ = 'pass'        # treat the arm as no finding (the direction is the gate's: see the D2 table)
    flag = 'flag'         # treat as a finding (fail-closed)
    recheck = 'recheck'   # take the gate's re-measure path (re-verify)

SUPPORTED: Mapping[ProbeId, frozenset[OnUnmeasured]]
UNMEASURED_POLICY: Mapping[ProbeId, OnUnmeasured]   # THE one place; validated against SUPPORTED at import

class StormEscalationSink(Protocol):     # OQ-CQ4, ruled by the lead: three methods, fed the raw queue
    def make_id(self, key: str) -> str: ...
    def has_open_l1(self, task_id: str, *, category: str | None = None) -> bool: ...
    def submit(self, escalation: 'Escalation') -> str: ...

class GateUnmeasuredReporter:
    def __init__(self, *, event_store: EventStore | None, escalations: StormEscalationSink | None,
                 policy: Mapping[ProbeId, OnUnmeasured] = UNMEASURED_POLICY,
                 clock: Callable[[], float] = time.monotonic) -> None: ...
    def settle(self, m: GateMeasurement[T], *, task_id: str | None) -> T | OnUnmeasured: ...

# event: EventType.merge_gate_unmeasured (orchestrator/src/orchestrator/event_store.py)
#   data = {gate, probe, reason, rc, stderr_tail, disposition, window_count}
# merge_attempt data (additive, α2): gates_unmeasured: list[GateId]   (lane gates only; D12)
#   carrier: MergedOk / in-flight item field -> _finalize_advanced_merge(gates_unmeasured=...) ->
#            _PostAdvanceContext.unmeasured -> closing _emit_merge_attempt; workflow gates: events only
# storm L1: task_id '__merge_gate_unmeasured_storm__:<gate>', category 'merge_gate_unmeasured_storm';
#   > THRESHOLD in WINDOW per gate; deduped by has_open_l1(that sentinel, category=...)

# orchestrator/src/orchestrator/merge_lane/resource_audit.py   (C4 lands in ε; S7-internal)
class ViolationKind(StrEnum):
    speculation_slot_conservation = ...; merge_ahead_cap_conservation = ...; merge_ahead_cap_handoff = ...
    speculation_check_raised_slot = ...; speculation_check_raised_cap = ...; speculation_check_raised_handoff = ...
    worktree_unregistered_scheduled = ...; worktree_unregistered_overdue = ...; worktree_check_raised = ...

@dataclass(frozen=True)
class ResourceAuditViolation:
    kind: ViolationKind; subject: str; age_secs: float | None; grace_secs: float | None; detail: str
    def render(self) -> str: ...            # byte-identical to today's strings
    def fingerprint(self) -> tuple: ...     # every field except age_secs
```

`ViolationKind` covers today's 8 producer shapes. The worktree-unregistered shape splits in two by
reclaim disposition, so the enum has 9 members. The three speculation "check raised" members map to
today's three `except` arms in `speculation_accounting_violations`; their exact names are tactical, one
per arm. Every "check raised" kind carries `str(exc)` in `detail`, and that is fingerprinted, as today.

**Invariants and error semantics.**

- **C1.** A probe returns `Measured` only when every git step it depends on answered: it exited 0, or it
  returned an exit code that is itself the answer (D7). Every typed query failure (S5 C4) it catches
  becomes an `Unmeasurable` carrying `returncode` and `stderr`. `Unmeasurable` is never constructed for
  a measured negative.
- **C2 `settle`.**
  - For `Measured`, it returns `m.value` and emits nothing.
  - For `Unmeasurable`, it emits exactly one `merge_gate_unmeasured`, charges `m.probe.gate`'s storm
    counter, files at most one L1 per open per-gate storm, and returns `policy[m.probe]`.
  - It never raises. An emit or filing failure is logged at WARNING, and the disposition is still
    returned.
  - It is synchronous and runs on the event loop, with no `await` between the counter check,
    `make_id`, `has_open_l1` and `submit`.
  - The reporter holds no task state. A process restart drops the window, which is the same lifetime as
    `LANDING_TALLY`.
- **C3 — withdrawn.** The target-verify adapter (`build_retry_env`) is not built: ruling S7-3 = B
  deletes the narrowing it would have held (D9). The number is kept so cross-plan references stay
  stable.
- **C4.** `render()` is byte-identical to the f-strings it replaces. Two violations with an equal
  `fingerprint()` describe the same unchanged leak.
- **C5.** The invariant `merge-gate-unmeasured-typed`; see §Design invariant.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | gates (α2) | drop guard whose merge-base cannot resolve (real git, `git clone --depth 2` of a merge tip: `^2` resolves, `merge-base ^1 ^2` exits 1, measured by the critic) | shallow repo, merge commit | `drop_guard.merge_base` → `pass_`; merge proceeds; one `merge_gate_unmeasured` row `{gate: drop_guard, probe: drop_guard.merge_base, rc: 1}`; the `merge_attempt` row carries `gates_unmeasured: [drop_guard]` |
| 2 | reporter (α1) | `THRESHOLD + 1` unmeasured drop-guard results in one window, then one more, then an equivalence storm | injected clock, fake `StormEscalationSink` | exactly one L1 on `__merge_gate_unmeasured_storm__:drop_guard`; the extra drop-guard result files nothing; the equivalence storm files its own L1; every result emitted an event |
| 3 | gates (α2) | equivalence: `rev-parse HEAD` fails, vs a measured divergence | broken worktree HEAD; a separate real divergence | first → `pass_` + event; second still blocks `post_merge_equivalence_failed` with no event |
| 4 | gates (α2) | pyright: worktree creation raises; separately, a type-check times out with nothing broken | injected failing `create_merge_worktree`; a timed-out module | both → `pass_` + event (`pyright.infra`, `pyright.timeout`); a genuine failure still blocks |
| 5 | policy (α1) | a row naming a disposition outside `SUPPORTED[probe]`; a `ProbeId` with no row | fixture tables | both raise at validation; the shipped table validates |
| 6 | gates (α2) | delta-survival probe git error in the drop guard | rename fixture with a failing `rev-parse` | `drop_guard.delta_survival` → `flag`: the drop stays and the merge blocks `DROPPED_PLAN_TARGETS`; event emitted |
| 7 | gates (α2) | linear landing; a merge commit whose parent count cannot be read; a one-parent commit handed to the drop guard | linear `advanced_sha`; corrupt object; one-parent `merge_commit_sha` | linear → `Measured(None)`, **no event**; corrupt → `finalize.second_parent` `pass_` + event; drop guard → `drop_guard.second_parent` `pass_` + event (`rc: None`, "merge commit has one parent") |
| 8 | gates (γ) | overlap probe git error | rebased worktree, failing `merge-base` | `ReverifyGateReason.overlap_unknown` (`recheck`) + event; no sentinel compare |
| 9 | gates (γ) | plan-files per-entry arms: `ls-tree` fails / rename-pair probe fails / history probe fails | declared path absent | each entry stays `not_touched` (`flag`), no basename fallback resolves it; one event per arm |
| 10 | workflow (γ) | the touched-set query raises S5 C4's typed failure; `git rev-parse HEAD` fails before the gate; `git rev-parse HEAD` fails before the post-narrowing re-check | declared files present on the branch; broken HEAD; a narrowing pass that then meets a broken HEAD | no blanket `not_touched` listing and no `PLAN_FILES_NOT_TOUCHED` block (`plan_files.touched_set` → `pass_`, S7-2); the first rev-parse failure emits (`plan_files.rev_parse_head` → `pass_`) instead of skipping silently; the re-check failure emits (`plan_files.recheck_rev_parse_head` → `flag`) and the block names the pre-narrowing entries, as today |
| 11 | workflow (γ) | already-landed: an rc failure, the wrapper's `except` arm, and a measured "not landed" | failing `rev-list --count`; raising probe; genuinely unlanded branch | all three read "not landed"; only the first two emit |
| 12 | consumer (α1/γ) | the worker builds one reporter; a `TaskWorkflow` with `merge_worker` set uses `merge_worker.gate_reporter` | real `SpeculativeMergeWorker` + `TaskWorkflow` | identity holds; with `merge_worker=None` the workflow constructs its own |
| 13 | guard (ζ) | seeded fixtures: `return []` under `if rc != 0:` and `DropGuardResult()` under `except` in a `_check_*`; an exit-code-as-answer probe testing `rc == 1` first; `if ahead != 0: return Measured(None)` inside a `GateMeasurement`-annotated function; an `Unmeasurable` naming a `ProbeId` with no row | fixture source text | the first two and the last flag; the answer-code probe and the `ahead != 0` fixture pass; post-γ gates.py passes; one fault-injection row exists per `GateId` |
| 14 | worker + server (δ) | a post-merge verify whose attempt 0 fails infra-transient and whose retry passes; the `merge_request` tool's input schema | real `SpeculativeMergeWorker` with a fail-then-pass verifier; the registered tool | exactly one retry, charged to the legacy `max_enospc` budget; the retry's `verify_env` equals attempt 0's; neither `merge_verify` event carries a `retry_scope` or `retry_subset_sizes` key; the tool schema has no `retry_failed_only` property |
| 15 | audit (ε) | an unregistered worktree observed at age 120 s, then 150 s, then crossing the reap floor; a growing permit leak; a grace change | real producers | equal fingerprints at 120/150 s; a disposition flip, a grace change and a permit-count change each change the fingerprint; `snapshot()['resource_audit']` strings are byte-identical to the pre-change rendering |

## Design invariant (ζ performs the four-site lockstep edit)

Minted here: the invariant **`merge-gate-unmeasured-typed`**. The slug is the stable id, and every
check anchors on it. Its number is **INV-17** under the lead's revised order of 2026-11-10: S6 α
(`lane-budgets-exceed-verify-timeouts`) lands INV-15 first, then S10 θ (the static metadata-slice
registry) INV-16, then this leaf. The highest invariant on main is INV-14 (`sampler-vocabulary-one-table`,
measured at `bb1c659562`). The edges S10 θ → S6 α and S7 ζ → S10 θ serialize the three invariant
files, so they land S6 α, then S10 θ, then S7 ζ. The
FILED row records the number that landed.

Sites, as listed in `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`:

- `docs/legibility/design-invariants.md`: the section, the header dating paragraph and the `## Fixtures`
  date list;
- `docs/legibility/design-invariants-fixtures.md`: the `-PRD` and `-CODE` fixtures, plus a dated
  addendum walk with its rehearsal rows and the cumulative match count;
- `skills/prd/references/gates.md`: the family-inventory row and the `inv-trigger-shapes` span;
- `CONTRIBUTING.md`: no change (pinned as an absence);
- `orchestrator/src/orchestrator/agents/code_quality.md`: optional. Its mapping is partial and pinned
  pair-wise.

Draft normative text:

> **Rule**: Every gate function `_check_*` in `orchestrator/src/orchestrator/merge_lane/gates.py`, every
> probe it calls, and every function in that module annotated to return a `GateMeasurement`, returns a
> `GateMeasurement`; a return reached from an `except` arm or an
> `rc != 0` branch is an `Unmeasurable` naming its `ProbeId`, never the gate's pass value, `None` or a
> sentinel; an exit code that is itself the answer is tested explicitly before the failure arm. The
> caller settles it through `GateUnmeasuredReporter.settle`, which emits `merge_gate_unmeasured` and
> applies the probe's row in `UNMEASURED_POLICY`, the one place a direction is chosen.
>
> **Checkable design question(s)**: Does this feature add or change a git probe in a merge gate? On its
> failure arm, what does it return, does that value reach `settle`, and does its `ProbeId` have a row? Is
> an answering exit code tested before the failure arm? Does anything choose fail-open or fail-closed
> outside `UNMEASURED_POLICY`?
>
> **Evidence**: at `6551b75aba`, gates.py encoded "could not measure" four ways, with 12 fail-open returns
> equal to the pass value, logged only (`fk-88e3c228b649`, run `review-all-dark_factory-20261008`).
>
> **House pattern**: `merge_lane/gate_measurement.py`; the guard in `orchestrator/tests`.
>
> **Family boundary**: INV-11 `no-silent-fail-soft` states the general rule; this is its checkable form
> for the merge lane's gates, with a single policy home. INV-4 `storm-escape-required` governs the
> aggregate and is met by the reporter's per-gate storm counter; this invariant governs each occurrence.

The trigger shape for gates.md: *a merge gate's git probe returning a value, `None`, a sentinel or an
empty result from an `except` or `rc != 0` arm, or choosing fail-open/closed outside
`UNMEASURED_POLICY` (`merge-gate-unmeasured-typed`)?*

The guard is `orchestrator/tests/test_gate_measurement_invariant.py`, with its scanner in
`orchestrator/tests/gate_measurement_scan.py`. Every part is exercised against fixtures (INV-10):

1. **AST scan.**
   - Scope (D11): every `_check_*` function in gates.py, its nested functions, and every function whose
     return annotation names `GateMeasurement`. This adds `plan_delivery.py` if 3918 creates it.
   - Rule: a `Return` inside an `ExceptHandler`, or inside the body of an `If` whose test compares a
     name or attribute ending in `rc` or `returncode` with `!= 0`, or truth-tests one, must be a call to
     `Unmeasurable(...)`. Re-raises are allowed. A `!= 0` on any other name (for example
     `_resolve_already_landed_branch`'s `ahead != 0`, a measured "did real work") is out of the rule.
   - The scope is derived from the code, not listed by name, and does not depend on which git runner the
     code calls (`_run`, `run_git`, `run_in_worktree`, `query`).
2. **Policy completeness.** Every `Unmeasurable(...)` construction in scope names a `ProbeId` member,
   and every `ProbeId` member has a `UNMEASURED_POLICY` row.
3. **Sentinel absence.** `_OVERLAP_GIT_ERROR_SENTINEL` and `_RENAME_PROBE_UNMEASURABLE` are absent from
   `orchestrator/src`. These are γ's two delivered checks.
4. **Behavioural completeness.** A parametrized fault-injection test has one row per `GateId` member,
   enumerated from the enum, driven through the real worker and workflow. Each row asserts one emitted
   `merge_gate_unmeasured`.

There are no hand-listed exemptions, so INV-12 has nothing to own.

## Cross-PRD relationship

| Other PRD / stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S6 lane lifecycle (R1) | produces (to) | C1/C2. The injection point is `SpeculativeMergeWorker.gate_reporter`, built from the policy S6's composition passes. If 5040 has landed, `VerifyDispatcher` receives the same instance. A block disposition, or `gates_unmeasured` on `MergeOutcome`, needs S6's `MergeOutcome.kind`. | S7 owns the type and policy; S6 constructs (G4 row) | Same-file order (lead, after the fold-verifiers, 2026-11-10): S6 α (the INV-15 timing-budget validator) lands first on every file it shares with this plan and takes no edge on any S7 leaf. S7 δ → S6 α and S7 β → S6 α (worker.py, shadow.py, types.py, config.py, `test_merge_queue.py` and the ratchet baseline), and S7 ε → S6 α on the resource-audit file (`test_merge_queue_resource_audit.py`); α1 … ζ follow S6 α through the chain. S6's remaining worker.py leaves depend on S7 β (OQ-CQ5), so β sits between S6 α and them. S6-2 = B: S6's lifecycle and limits tasks wait on 5040, which is S6's edge. Under S7-1 = A and S7-3 = B, about 470 lines (probe 275, auto-chain 194) plus the 733-line reify section and about 370 shadow.py lines leave the lane, not 1,450, so S6 re-measures. The invariant edges are S10 θ → S6 α and S7 ζ → S10 θ. Xref returned. |
| S5 GitOps query contract (R5), draft C4 | consumes | C4's typed failure raised by `get_files_touched_in_branch` (and by S5 ζ's `count_commits` in `_resolve_already_landed_branch`), wrapped into `Unmeasurable` by γ (D6). γ needs `returncode` and `stderr` fields, which `git_ops.py::GitQueryFailed` (task 7485, done) does not have. | S5 | γ depends on S5 ε. S5 ε must depend on 3918 (same query), and its consumer obligations should name the two gates.py callers as propagate-until-S7-γ, because otherwise a failing `git log` raises uncaught out of the workflow merge phase in the window. Xref returned. |
| S5 git runner (R5), draft C1 | rebase-only | S5 α repoints gates.py's `_run` import; S5 β injects `GitOps(runner=…)` | S5 | no dependency. ζ's guard scope does not depend on the runner spelling. Task 6432 becomes cheap after S5 β and α2. |
| S3 escalation server (R10), ε2 `merge_request` move | precedes (δ first) | δ deletes the `retry_failed_only` parameter from `escalation/server.py::merge_request` before S3 ε2 moves `merge_request` into `orchestrator/mcp/merge_tools.py`, so the move carries one parameter fewer | S3 owns the surface; S7 owns the deletion | S3 ε2 depends on S7 δ (the lead wires it at decompose; both ids are unfiled). S3 α's `EventType.merge_deduplicated` and S7 α1's `EventType.merge_gate_unmeasured` are adjacent `event_store.py` edits, serialized in the lead's per-file order. Xref returned. |
| S4 verify context (R4) | none | The `build_retry_env(…, attempt0: VerifyResult, …)` consumer is never built (S7-3 = B), so S4 drops its consumer row. After δ, nothing in the merge lane reads `VerifyResult.run_all_failed_members` (verify.py). | S4 | no dependency; xref returned (S4's call whether its `VerifyResult` rework drops that field) |
| S9 workflow (R2) | edits (call sites) | γ settles at the plan-files (gate and post-narrowing re-check), already-landed and both rev-parse sites in `workflow.py` | S9 owns workflow; S7 owns the call's meaning | S9 records a same-file edge at decompose |
| S1 filing entry point (R8) | consumes | the storm L1 goes through `StormEscalationSink` over the raw queue; S1's `EscalationPort` widening absorbs it | S1 | xref returned |
| S10 metadata registry (R3), θ | ordering | the invariant docs (`design-invariants.md`, the fixtures, `gates.md`) | lead | ζ depends on S10 θ (INV-16), which depends on S6 α (INV-15) |
| `plans/merge-lane-quality-prd.md` κ 5040 / λ 5041 | rebase-only (S6-2 = B) | 5040 (pending, high, half-built branch `task/5040` with 5 commits) extracts `_dispatch_item`, `_run_inflight_verify` and `_run_post_merge_verify`; its declared files include `merge_lane/retry_verify_env.py` for the reify section. λ 5041 moves deep chain into `speculation.py` unchanged. | that PRD | 5040 waits on no S7 task (S6-2 = B), and no S7 leaf waits on 5040. Whichever lands second rebases. If 5040 is done when S7 decomposes, β's and δ's declared files are re-measured onto 5040's extracted modules, and δ deletes the section wherever 5040 put it. δ's absent checks scan all of `orchestrator/src`, so they hold either way. |
| `plans/deep-merge-ahead-prd.md`: 7720 (high), 7721, 7824, 7825, 7810, 7828, 7922 | ordering (7720); rebase-only (the rest) | β removes the `chain_ctx` kwarg inside `_land_chain_prefix` and two prose references to the probe window. α2 converts `_resolve_second_parent`, which `_land_chain_prefix` reads, and threads the reporter through the same finalize call. 7720 (in progress since 2026-11-10T21:02Z, re-planned that day) rewrites `_land_chain_prefix`. It declares `merge_lane/worker.py`, `config.py`, `defaults.yaml`, new `chain_reservation.py` and `lookahead_hold.py` modules, `test_merge_queue_deep_dispatch.py`, `test_merge_queue_deep_landing.py`, `test_merge_queue_deep_integration_gate.py` and the ratchet baseline. It no longer declares `test_merge_queue.py` or `test_merge_queue_resource_audit.py`. | that PRD | β and ε depend on 7720 (OQ-CQ7, lead). δ does not: δ is first by the lead's sequencing ruling and only rebases over 7720 on worker.py and the baseline. 7824 depends on 7720 and is ordered with it. |
| `plans/merge-lane-policy-prd.md`: 8527 (ε), 8528 (ζ), both high | same-file neighbours (rebase-only) | 8527 rewrites `_deep_chain_placement` guard 1 and `_adopt_head_on_tip_authority`. 8528 replaces `_chain_halving_state` with `ChainPolicyState`, deletes `next_halving_state` and adds `merge_deep.depth_table`. β and α2 edit `_land_chain_prefix`, the `_chain_halving_state` comment and `config.py` in the same files. | that PRD | No edge (S7-1 = A). The lead finalises the per-file serial order at decompose. 8527 waits on 7720, 7721, 7824 and 8526. |
| task 3918 (plan-files measurement, high, pending, dispatchable) | precedes | same query and gate | 3918 | γ depends on 3918 |
| task 7384 (retire the probe) | subsumed | probe deletion | S7 β | lead action: cancel → β, unconditionally |
| task 4767 (retire the `depth` label) | file overlap | `test_merge_queue_speculative_probe.py` (β deletes it); `probe_base` projections in the deep tests | 4767 | lead action: 4767 depends on β |
| task 6242 (stale PASS strands commits) | narrows | its option (b), "finish γ3 and enable auto-chain", is removed by β | 6242 | lead actions: amend, and depends on β |
| task 6576 (LOCAL runner ignores the retry env) | moot | it describes a defect in code δ deletes | δ | lead action: cancel, naming δ (ruled S7-3 = B) |
| task 6575 (30-day re-measure, then escalate or remove the narrowing) | subsumed | its branch B is a narrower δ, and Leo's 2026-11-10 ruling removes the measure-then-decide step | δ | Parked since 2026-11-10: it depends on 5040 and 8528, so it cannot dispatch. Lead action: the S7 decompose seat cancels it, naming δ, and removes nothing else (lead sequencing ruling). |
| reify (cross-project) | precedes β's deploy | removing `speculation_probe:` from reify's live yaml (OQ-CQ2) | reify | lead files a reify block; β's `external_deps` |

Out-of-batch pending tasks that share files without a real dependency are rebase-only overlaps: 7129 and
7768 (gates.py, workflow.py), 7295 and 7922 (worker.py), and 8299 and 4806 (the plan-files gate in
gates.py).

## Decomposition plan

There are seven leaves and no `high`/`critical` leaf, so no leaf is a dependency of gate 6698.

Every leaf rewrites `merge_lane/worker.py`, `gates.py` or `orchestrator/tests/merge_lane_ratchet_baseline.json`,
and 3918's record measures that the baseline "ALWAYS conflicts" on sibling rebases. So the leaves form
one serialized chain: **δ → β → α1 → α2 → γ → ε → ζ**. ζ is the integration gate.

δ is first by the lead's sequencing ruling of 2026-11-10 (`digests/wave2-lead-rulings.md`, last
block): β depends on δ, so δ does not wait on task 7720. S6 α lands before this plan on every file
they share (the lead's rulings after the fold-verifiers, same file): δ, β and ε each depend on S6 α,
and the rest of the chain follows it through δ. δ's only dependency is S6 α. S3 ε2 depends on δ.

Each leaf that touches the ratchet cluster runs `python scripts/merge_lane_metrics.py --check`. It lowers
measures where it can, and records any raise with `--write-baseline … --authorize-raise <task-id>
--reason …`, as the baseline's `_README` sanctions for net-additive work.

Every leaf carries `metadata.x_finding_key` (list), `x_finding_run = "review-all-dark_factory-20261008"`
and `source = "review-all"`.

**δ — Delete reify retry narrowing end to end** (S7-3 = B)
(`task_kind=normal`, priority low, est. ~4,500 changed LOC, at least 97% deletion: about 1,300 source
lines, about 2,900 test lines and about 280 fixture lines)

- **Modules:**
  - source: `merge_lane/worker.py`, `merge_lane/shadow.py`, `merge_lane/types.py`,
    `orchestrator/src/orchestrator/verify_runner.py`, `orchestrator/src/orchestrator/event_store.py`,
    `escalation/src/escalation/server.py`, `ARCHITECTURE.md`;
  - deleted: `orchestrator/tests/test_merge_queue_verify_retry.py` (1,828 lines) and
    `orchestrator/tests/fixtures/reify_verify_retry/` (5 files), less `run_all-failed-marker.txt`;
  - moved: `run_all-failed-marker.txt` and its provenance section into
    `orchestrator/tests/fixtures/verify_classify_run_all/` (D9 item 8), with the corpus entry in
    `orchestrator/tests/test_verify_event_channel.py` repointed;
  - tests: `orchestrator/tests/test_merge_shadow.py` (38 tests, 534 lines, on the deleted helpers),
    `orchestrator/tests/test_merge_queue.py` (6 `TestRunPostMergeVerify` narrowing tests, 325 lines,
    holding the 10 private patch strings), `orchestrator/tests/test_verify_runner.py`
    (`TestRetryScopeEventFields`, 4 tests, plus 2 event-shape assertions),
    `orchestrator/tests/test_verify_events_units.py` (1 assertion), `escalation/tests/test_server.py`
    (2 `TestMergeRequestDedup` cases, 108 lines);
  - ratchet: `orchestrator/tests/merge_lane_ratchet_baseline.json` (its shadow.py rows for the deleted
    helpers go).

  Every count here is measured at `bb1c659562`.
- **Size:** 17 paths (each fixture directory counted once), above the >15 review trigger. The task
  is one deletion under one ruling. Two whole-file
  deletions (the test module and the fixture directory) account for about 2,100 of the lines, so the
  line count overstates the review cost. Splitting the flag and event-key half from the producer half
  would leave a dead flag on `MergeRequest` between the two, so it is not split.
- **Keys:** `fk-dd411ada7c29`. Subsumes 6575; replaces 6576.
- **Deps:** S6 α (same-file order: `merge_lane/worker.py`, `shadow.py`, `types.py`,
  `test_merge_shadow.py`, `test_merge_queue.py` and the ratchet baseline; S6 α lands first, lead
  ruling after the fold-verifiers, 2026-11-10). δ is first in this plan's chain (lead sequencing
  ruling, 2026-11-10), and β depends on it. S3 ε2 depends on δ (lead edge at decompose). There is no
  edge on 7720. δ and 7720 both edit `merge_lane/worker.py` (different sections) and the ratchet
  baseline, so whichever lands second rebases. 7720 has been in progress since 2026-11-10T21:02Z.
  There is no edge to 5040 either way. If 5040 has landed by filing, δ's file list names 5040's
  extracted modules instead of the worker sections.
  S6 α waits on no S7 leaf, so δ → S6 α closes no cycle. δ must not depend on S3 ε2, on any S6 leaf
  that depends on S7 β (S6's remaining worker.py leaves), or on any leaf that waits on them (for
  example S3 ε1 through S6 μ, and S3 ζ through ε2). Each of those waits on δ, through S3 ε2 → δ or
  through S6's worker.py leaves → S7 β → δ, so such an edge would close a cycle.
- **DELETES:** everything D9 lists: the 733-line producer section with its 9 pins, the narrowing arm and
  its shadow hand-off, the narrowed budget, the shadow helpers, the flag, the tool parameter, the
  event marker and the prose.
- **Keeps:** the retry loop, attempt counting, the legacy `max_enospc` budget and
  `shadow.py::parse_per_test_results`.
- **Observable signal:**
  - boundary row 14 passes;
  - `orchestrator/tests/test_verify_event_channel.py` still collects, and its D3 corpus reads the
    moved `run_all-failed-marker.txt`;
  - `git grep -E 'retry_failed_only|_REIFY_ATTEMPT_SIDECAR_RELPATH|retry_scope_event_fields' orchestrator/src escalation/src`
    is empty, and so is `git grep -E 'verify\.sh:[0-9]' orchestrator/src/orchestrator/merge_lane`;
  - after deploy, new `merge_verify` rows in `runs.db` have no `retry_scope` key.
- **G7:**
  - INV-13: the deleted producer had no live reader, so the deletion removes the INV-13 note rather
    than ratifying it.
  - INV-9: one retry budget, not two.
  - INV-1: the captured reify bytes go with the contract they pinned; reify's own tests keep its side.

**β — Delete the variable-depth probe and γ2 auto-chain; adjacent placement plus deep chain**
(`task_kind=normal`, priority medium, est. ~1,600 changed LOC, mostly deletion)

- **Modules:**
  - source: `merge_lane/worker.py`, `merge_lane/gates.py`, `merge_lane/types.py` (comment),
    `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/event_store.py` (probe
    prose), `dark-factory-orchestrator.yaml`;
  - deleted: `orchestrator/tests/test_merge_queue_speculative_probe.py`;
  - tests: `test_merge_queue.py`, `test_merge_gates.py`, `test_merge_gates_equivalence.py`,
    `test_merge_queue_depth_telemetry.py`, `test_merge_queue_deep_dispatch.py`,
    `test_merge_queue_deep_integration_gate.py` and `test_merge_queue_deep_landing.py` (drop the dead
    `probe_base` projection);
  - ratchet: `orchestrator/tests/merge_lane_ratchet_baseline.json`;
  - skills: `skills/merge-queue/SKILL.md`, `skills/unblock/SKILL.md`,
    `skills/merge-queue/references/two-layer-model.md` (auto-chain rows).
- **Size:** 18 files, above the >15 review trigger. They are all one deletion: one test module goes,
  seven test files lose cases, and three skill docs lose prose. Coherent, so not split.
- **Keys:** `fk-7d1580bb3f03`. Subsumes 7384.
- **Deps:** δ (chain; lead sequencing ruling, 2026-11-10); task 7720 (OQ-CQ7); S6 α (same-file
  order: worker.py, types.py, config.py, `test_merge_queue.py`, `test_merge_queue_deep_integration_gate.py`
  and the ratchet baseline; S6 α lands first, lead ruling after the fold-verifiers); the reify yaml
  block, `external_deps` (OQ-CQ2). No edge to 5040 either way (S6-2 = B). S6's remaining worker.py
  leaves depend on β (OQ-CQ5).
- **DELETES:**
  - 275 probe lines and 194 auto-chain lines (measured at `bb1c659562`);
  - the 1,162-line probe test module;
  - `SpeculationProbeConfig`;
  - the two deep-chain prose references to the probe window: the `_chain_halving_state` init comment
    in `SpeculativeMergeWorker.__init__` ("matching `_recent_verify_outcomes`' empty-at-construction
    rationale above") and the sentence of `_note_chain_outcome`'s docstring that names
    `_recent_verify_outcomes` and `_probe_round_counter`. β's absent check needs both gone; if 8528
    has already replaced `_chain_halving_state`, only the docstring sentence remains;
  - the probe's outcome window (D8);
  - `chain_ctx` threading;
  - the `probe_base` arm.
- **Keeps:** deep chain (S7-1 = A) and `OutcomeKind.post_merge_generation_chained`.
- **Observable signal:**
  - `git grep -E 'AUTO_CHAIN_GENERATIONS_ENABLED|class ProbePlacement|SpeculationProbeConfig|_maybe_auto_chain_generation|_recent_verify_outcomes' orchestrator/src`
    is empty;
  - the deep-chain tests are green, and `merge_verify.chain_items > 1` rows keep appearing in
    `runs.db`;
  - `orchestrator check-config --config /home/leo/src/reify/dark-factory-orchestrator.yaml` reports no
    `speculation_probe` unknown key.
- **G7:**
  - INV-12: the probe's `except` arms are removed, not ratified.
  - INV-11: the existing unknown-key census covers a leftover key.

**α1 — `GateMeasurement`, `ProbeId`, `UNMEASURED_POLICY`, `StormEscalationSink` and `GateUnmeasuredReporter`**
(`task_kind=normal`, priority medium, est. ~600 LOC)

- **Modules:** `orchestrator/src/orchestrator/merge_lane/gate_measurement.py` (new),
  `orchestrator/src/orchestrator/event_store.py` (`EventType.merge_gate_unmeasured` and its field
  contract), `merge_lane/worker.py` (reporter construction over the raw `escalation_queue`, and the
  `gate_reporter` property), `orchestrator/tests/test_gate_measurement.py` (new),
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- **Keys:** `fk-88e3c228b649`.
- **Deps:** β (worker.py and baseline serialization; the chain is δ → β → α1).
- **Produces:** C1 and C2, with every `ProbeId` row of D2. The rows exist before the conversions, so the
  table is complete from day one.
- **Observable signal:** boundary rows 2, 5 and the worker half of 12 pass; the shipped table validates
  at import.
- **G7:**
  - INV-4: the per-gate storm counter.
  - INV-9: one policy home.
  - INV-2: structured `rc`, `stderr` and `probe`.

**α2 — Convert the 12 fail-open returns and the sub-probes they own; `gates_unmeasured` on `merge_attempt`**
(`task_kind=normal`, priority medium, est. ~1,200 LOC)

- **Modules:**
  - source: `merge_lane/gates.py`, `merge_lane/worker.py` (threading the reporter through the 4
    `_finalize_advanced_merge` callers and the 4 `classify_and_merge` call sites; the drop-guard settle),
    `merge_lane/types.py` (the `gates_unmeasured` field on `MergedOk` and the in-flight item, D12),
    `merge_lane/ports.py` (the `VerifyPort`/`ProductionVerifier` annotations);
  - new test: `orchestrator/tests/test_merge_gate_unmeasured_boundary.py` (real git);
  - tests: `test_merge_gates.py`, `test_merge_gates_equivalence.py`, `test_merge_gates_drop_guard_rename.py`,
    `test_merge_queue.py` (15 direct `_check_plan_targets_in_tree` calls and 1 string patch target,
    measured at `bb1c659562`), `test_merge_guard_pipeline.py`, `test_merge_gates_rename_family.py`,
    `test_merge_gates_equivalence_rename.py`, `test_atomic_train_merge.py`,
    `test_merge_queue_equivalence.py`, `test_workflow_dry_run_hook.py`,
    `orchestrator/tests/_merge_lane_verifier_doubles.py`;
  - ratchet: `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- **Size:** 17 files, above the review trigger. The fan-out is one return-type change, followed
  mechanically by its callers and fakes. α1 was split off for that reason.
- **Keys:** `fk-88e3c228b649`.
- **Deps:** α1.
- **Converts:**
  - drop guard (5), equivalence (6), pyright (1, plus the timeout arm, as `pyright.timeout`);
  - `_rename_pairs`, `_rename_aware_real_drops`, `_rename_aware_compare_set`, `_resolve_second_parent`
    (parent count first, D7), `_commit_is_linear` and `_branch_delta_survives` (answer test first, D7);
  - in `_check_plan_targets_in_tree`, a `Measured(None)` second parent returns
    `Unmeasurable(drop_guard.second_parent, rc=None, reason='merge commit has one parent')`, because a
    one-parent input there means the gate cannot measure, not that nothing was dropped;
  - adds `gates_unmeasured` to the closing `_emit_merge_attempt` (D12).
- **DELETES:** the 12 pass-value returns and the "failing open" log-only arms.
- **Observable signal:**
  - boundary rows 1, 3, 4, 6, 7 and the lane half of 12 pass. Row 1 is real git (INV-13);
  - after deploy, `sqlite3 data/orchestrator/runs.db "select count(*) from events where event_type='merge_gate_unmeasured'"`
    runs, and a non-zero count names its gate and probe.
- **G7:**
  - INV-11 is the core of this leaf.
  - INV-12: the pyright `except Exception` stays and is now owned by its settle.

**γ — Remaining encodings onto `GateMeasurement`: overlap, rename pair, already-landed, plan-files**
(`task_kind=normal`, priority medium, est. ~1,100 LOC)

- **Modules:**
  - source: `merge_lane/gates.py`, `merge_lane/gate_measurement.py` (policy-row adjustments, if any),
    `orchestrator/src/orchestrator/workflow.py` (the plan-files gate and re-check, already-landed and
    both rev-parse settle sites), plus `orchestrator/src/orchestrator/plan_delivery.py` if 3918 created
    it;
  - prose: `skills/merge-queue/references/two-layer-model.md` (its `_OVERLAP_GIT_ERROR_SENTINEL` row);
  - tests: `test_merge_gates.py` (imports the sentinel), `test_merge_queue.py`,
    `test_merge_gates_plan_files_rename.py`, `test_merge_gates_rename_family.py`,
    `test_convert_to_blocked.py`, `test_branch_work_landed.py`, `test_merge_gates_git_probes.py`,
    `test_workflow.py`, `test_workflow_narrowing_exit_reports.py`, `test_plan_tools_server.py`;
  - new test: `orchestrator/tests/test_plan_files_gate_unmeasured.py`;
  - ratchet: `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- **Size:** 16–17 files, above the review trigger. One gate family, one conversion.
- **Keys:** `fk-689227abfab1`, `fk-88e3c228b649`.
- **Deps:** α2; S5/C4 (S5 leaf ε, the typed query failure; id at decompose; held on S5/C4 if S5 has not
  filed); task 3918.
- **DELETES:**
  - `_OVERLAP_GIT_ERROR_SENTINEL`, `_RenameProbeUnmeasurable` / `_RENAME_PROBE_UNMEASURABLE`, and every
    prose mention of both sentinels (6 in gates.py and 1 in the skill reference, measured at
    `bb1c659562`);
  - the `None`-overload in `_resolve_already_landed_branch`'s rc and except arms, and in the
    `resolve_already_landed_branch` wrapper's `except` arm;
  - the false "Fail-open on git error" docstring claim (now true);
  - the gate's reading of an empty touched set as "touched nothing" when the query failed;
  - the workflow's two silent rev-parse skips.
- **Observable signal:**
  - boundary rows 8–11 and the workflow half of 12 pass;
  - `git grep -E '_OVERLAP_GIT_ERROR_SENTINEL|_RENAME_PROBE_UNMEASURABLE' orchestrator/src` is empty;
  - a `git log` failure in the plan-files gate produces a `merge_gate_unmeasured` row, not a
    `PLAN_FILES_NOT_TOUCHED` block naming every declared file.
- **G7:**
  - INV-11 and INV-9.
  - INV-3: the gate still measures the branch ref it was given.

**ε — Resource-audit violations as records fingerprinted on identity, not regex** (OQ-CQ6)
(`task_kind=normal`, priority low, est. ~450 LOC)

- **Modules:** `orchestrator/src/orchestrator/merge_lane/resource_audit.py` (new), `merge_lane/worker.py`
  (`speculation_accounting_violations`, `worktree_ledger_violations`, `_check_resource_audit`,
  `_alarm_resource_audit`, the deleted fingerprint), `orchestrator/tests/test_merge_queue_resource_audit.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- **Keys:** `fk-5d4406afeade`.
- **Deps:** γ (chain); task 7720 (OQ-CQ7: the same worker.py and ratchet baseline; 7720's plan of
  2026-11-10 no longer declares `test_merge_queue_resource_audit.py`); S6 α (the resource-audit file
  `test_merge_queue_resource_audit.py`, which S6 α also edits; S6 α lands first, lead ruling after the
  fold-verifiers).
- **DELETES:** `_RESOURCE_AUDIT_AGE_FIGURE_RE` and `_resource_audit_fingerprint`. The 24 test references
  to them (measured at `bb1c659562`) move onto `ResourceAuditViolation.fingerprint`, a public seam.
- **Observable signal:**
  - boundary row 15 passes;
  - `git grep _RESOURCE_AUDIT_AGE_FIGURE_RE orchestrator/src` is empty;
  - a live leaked worktree's heartbeat log line still coalesces on the `ExponentialLogCoalescer`
    schedule.
- **G7:**
  - INV-9: one rendering home.
  - INV-4: the coalescer schedule is unchanged.

**ζ — The `merge-gate-unmeasured-typed` invariant (INV-17), its guard, and the PRD's integration gate**
(`task_kind=normal`, priority medium, est. ~600 LOC)

- **Modules:** `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`,
  `skills/prd/references/gates.md`, `orchestrator/tests/gate_measurement_scan.py` (new),
  `orchestrator/tests/test_gate_measurement_invariant.py` (new; guard parts 1–4, part 4 driving the
  real worker and workflow).
- **Keys:** `fk-88e3c228b649`.
- **Deps:** γ, ε; S10 θ (INV-16, cross-stream, wired by the lead at decompose).
- **Observable signal:**
  - `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py` passes with the
    slug in every pinned site;
  - the guard flags its seeded fixtures and passes gates.py at the landing tree (row 13);
  - boundary rows 1–15 pass together at ζ's tree;
  - the program doc's `## Triage briefing` line describes real behaviour (the lead applies the xref).
- **G7:**
  - INV-10: every guard part is exercised against fixtures.
  - INV-12: the scope is derived, with no hand-listed entries.

### Existing tasks (contract §8: step 1 and step 2 re-run on 2026-11-10)

Step 1 (`find_tasks_by_metadata(key="x_finding_key")`) returns no task for any of the five keys. Step 2
(`search_tasks`, threshold 0.5, every hit confirmed with `get_task`) found the following:

- **7384** (pending, low, "retire the inert probe"): subsumed by β. Cancel it unconditionally, with
  `x_acceptance_reason = "subsumed by S7 leaf β (<id>): probe and auto-chain deletion, plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.md"`.
  7384 depends on 4767, so it cannot dispatch first.
- **6575** (pending, low; Leo's 2026-10-09 ruling; its 30-day milestone elapsed on 2026-11-08): its
  branch B removes the dormant producer and keeps the flag and event keys. Leo's 2026-11-10 ruling
  removes all of them, without the measurement step, so δ subsumes it.
  - The lead parked it on 2026-11-10 with a temporary edge on 5040. `get_task` shows dependencies
    [5040, 8528], so it cannot dispatch before decompose.
  - The S7 decompose seat cancels it with `x_acceptance_reason = "subsumed by S7 leaf δ (<id>), the
    retry-narrowing deletion task: Leo ruled on 2026-11-10 to delete the narrowing end to end (S7-3 =
    B)"`, and removes nothing else (lead sequencing ruling).
  - If it was nonetheless claimed, decompose gotcha (a) applies: δ depends on 6575 and deletes the
    remainder.
- **6576** (pending, low): describes a defect in code δ deletes. Cancel with a reason naming δ (ruled
  S7-3 = B). It has no dependents.
- **3918** (pending, high, dispatchable since 5036 is done): precedes γ (D6).
- **7720** (in progress since 2026-11-10T21:02Z, high, re-planned 2026-11-10): precedes β and ε
  (OQ-CQ7). δ does not wait on it (lead sequencing ruling).
- **8524** (done 2026-11-10, the re-merge re-anchor): it moved the `classify_and_merge` re-merge call
  sites, which D3 and α2 now name.
- **4767 and 6242**: ordering edges, listed under lead actions. 6242 is also amended.
- **5040** (pending, high, half-built branch `task/5040` with 5 commits; waits on file locks): no edge
  either way (S6-2 = B).
- **8527 and 8528** (pending, high, merge-lane policy plan): same-file neighbours, no edge (S7-1 = A).
- **6432** (pending, low): not subsumed. It is noted to S5.
- **8299 and 4806** (pending, plan-files gate glob support and message wording): rebase-only overlaps
  with γ.
- **Phase-5 mechanical tasks**: none carries a key of this stream.

Priority inversions, recorded for the FILED row (contract §4):

- S3 ε2 (high) waits on δ (low) and on no other S7 leaf, because δ is first in the chain.
- β and the medium leaves after it wait on δ (low), and so do S6's remaining worker.py leaves, some
  of them high (through their edge on S7 β, then β → δ). S6 α does not: it waits on no S7 leaf.
- In both cases the scheduler applies priority inheritance through the dependency graph, so δ is not
  hand-raised.
- δ (low), β (medium) and ε (low) wait on S6 α (high), and β and ε wait on 7720 (high). Those are
  the benign direction.

## Out of scope

- Deep-chain semantics, depth policy and `chain_cap` (`plans/deep-merge-ahead-prd.md`,
  `plans/merge-lane-policy-prd.md`), and moving deep chain into `speculation.py` (λ 5041). Deep chain
  is kept untouched, apart from the call edit in `_land_chain_prefix` and two prose references to the
  probe window (S7-1 = A). If the policy readout (8532) or the 14-day measurement (5059) shows chains
  do not pay, deep chain is removed by its own task under the merge-lane policy plan, not here.
- A "block, could not measure" disposition, and any `MergeOutcome` kind or field (S6, R1; D5, D12).
- reify's side of the retry contract (its `scripts/verify.sh` reads the variables only when set, and
  its own `--confirm-failed` mode keeps using them).
- `VerifyResult.run_all_failed_members` in `verify.py`, which loses its merge-lane reader with δ (S4's
  type; xref).
- `branch_base.py::own_work_base`'s merge-base fallback (shared with review and the architect; not a
  gate probe).
- Repointing `git_ops._run` to `shared.git_async.run_git`, and the typed query failure contract itself
  (S5 C1/C4).
- The 9 throwaway reify fix, recovery and unblock checkouts' yamls.

## Rulings (sitting of 2026-11-10)

Leo ruled every design question on 2026-11-10 (`digests/wave2-rulings-applied.md` §S7), and the lead
ruled the lead-rulable ones (`digests/wave2-lead-rulings.md`, with its revised block). No design question
remains open.

| Question | Ruling | Ruled by |
|---|---|---|
| OQ-1 (S7-1) deep chain | Option A: keep deep chain; delete the variable-depth probe and auto-chain. **Amends R11 (2026-10-10)**, whose deletion of deep chain rested on "every project config has them off", false for deep chain since 2026-10-20 (reify) and 2026-10-27 (dark-factory). Deep chain is owned by the deep merge-ahead and merge-lane policy plans; 8527 and 8528 join the same-file neighbours; 6242 is amended. | Leo |
| OQ-2 (S7-2) plan-files touched-set disposition | As recommended: `pass_` for the touched-set query only; the per-entry ls-tree, rename and history arms keep `flag`. | Leo |
| OQ-3 sequencing against 5040 | The lead's ruling (5040 depends on β and δ; δ depends on β only) is **void for 5040** under Leo's S6-2 = B: 5040 finishes first and waits on no S7 task. Its "δ depends on β only" half is superseded by the lead's sequencing ruling below (δ first). | lead, then Leo (S6-2) |
| OQ-4 (S7-3) retry narrowing | Option B: delete it end to end (the recommendation was to move it). The adapter task becomes the deletion task δ; no `target_verify.py`; 6576 cancelled; S3's `merge_request` move waits on δ; S4 drops its `build_retry_env` consumer row. **Amends R11's remedy** "reify retry narrowing behind `build_retry_env` in a target-verify adapter module with a contract pointer": the reify-specific lines still leave the worker, by deletion. Supersedes Leo's 2026-10-07 ruling on 6285 (dormant becomes removed). | Leo |
| OQ-5 file placement | `merge_lane/gate_measurement.py` and `merge_lane/resource_audit.py`; the third module, `target_verify.py`, is dropped by S7-3 = B. | lead |
| OQ-6 storm thresholds | Tactical: decided in α1 (default more than 5 per gate in 3,600 s). | — |
| OQ-CQ1 (S7-4) policy granularity | As recommended: rows keyed by `ProbeId`, values `pass_`/`flag`/`recheck`. | Leo |
| OQ-CQ2 retiring `speculation_probe` | A reify-side stanza removal before β deploys; no bespoke WARNING. | lead |
| OQ-CQ3 (S7-5) the merge's own record | As recommended: additive `merge_attempt.data.gates_unmeasured` now for lane gates; `MergeOutcome` via S6 later. | Leo |
| OQ-CQ4 storm filing before S1 | `StormEscalationSink` in `gate_measurement.py`, fed the raw queue; S1 absorbs it. Three methods (`make_id`, `has_open_l1`, `submit`), per the fold of 2026-11-02. | lead |
| OQ-CQ5 S6/S7 same-file order | β before S6's worker.py leaves, which depend on β. Unchanged by S6-2 = B, except that S6's lifecycle and limits tasks also wait on 5040. Since the lead's sequencing ruling β itself follows δ. Superseded for S6 α by the lead's rulings after the fold-verifiers (row below): S6 α takes no edge on β. | lead |
| OQ-CQ6 fingerprint fields | (kind, subject, grace_secs, detail); a **clarification of R11's** "(kind, subject)". | lead |
| OQ-CQ7 β/ε against 7720 | β and ε depend on 7720. | lead |
| Invariant order (cross-stream) | Landing order S6 α INV-15, then S10 θ INV-16, then S7 ζ INV-17, with slug `merge-gate-unmeasured-typed`; edges S10 θ → S6 α and S7 ζ → S10 θ; checks anchor on the slug. | lead (revised 2026-11-10; slugs confirmed in the sequencing block) |
| S7 chain order (sequencing, after the ruling folds) | δ goes first and β depends on it, so δ does not wait on 7720. S3 ε2 depends on δ. The chain is δ → β → α1 → α2 → γ → ε → ζ. | lead (2026-11-10) |
| S6 α first on shared files (after the fold-verifiers) | S6 α takes no edge on S7 δ or S7 β. S7 δ → S6 α and S7 β → S6 α where they share a file; S7 ε → S6 α on the resource-audit file; S6's remaining worker.py leaves keep their edge on S7 β. Supersedes the part of OQ-CQ5 and of the sequencing ruling that held S6 α behind δ and β. | lead (2026-11-10 late) |
| 6575 (parked on 5040 by the lead, 2026-11-10) | The S7 decompose seat cancels 6575 with `x_acceptance_reason` naming δ and Leo's S7-3 ruling, and removes nothing else. | lead (2026-11-10) |

**Tactical (decide in the leaf):**
- `ProbeId` member spellings (α1).
- `Unmeasurable.stderr` cap: 400 vs 200 characters (γ).
- PEP 695 `type` alias vs `Union` for `GateMeasurement` (α1).
- Whether `shadow.py::build_warm_shadow_results` survives as a one-line wrapper once its
  `attempt0_verdicts` parameter is gone (δ).

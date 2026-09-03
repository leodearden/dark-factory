# Capability manifest — `plans/merge-lane-quality-prd.md`

Mechanizes G3 (assumed-substrate verified) and G6 (premise validity) for the 29-task
decomposition of the merge-lane quality PRD. Every binding below was resolved at decompose time
against **main `4811d62883`** (2026-09-03) — the commit that carries the PRD itself. Main moves
fast: these are dated provenance, not live counts.

Machine-readable twin: `plans/merge-lane-quality-prd.capability-manifest.yaml` (schema
`shared/src/shared/capability_manifest.py::CapabilityManifestDoc`). `commit_planning` stamps that
file's `task_id` fields and copies each label's **mechanical** (`grep`/`script`) `delivered_check`
into the producer task's `metadata.delivered_checks`; `manual` checks stay sidecar-only and are
excluded from the dispatch gate.

## Verdict summary

| | Count |
|---|---|
| Task blocks | 29 |
| Capabilities bound | 93 |
| PASS | 93 |
| FAIL (would block the batch) | 0 |
| OPEN | 0 |
| Mechanical `delivered_check`s copied to producers | 29 |
| `manual` checks (recorded, not gated) | 64 |

## Gate walk — what was checked and what it cost

**G1 (consumer named).** Every mechanism has a present consumer. The two seams built partly for a
future PRD — `verify_dispatch.py::VerifyDispatcher` (κ) and `speculation.py::ChainPlanner` /
`Speculator` (λ) — each have a consumer TODAY (`worker.py`, task ο); the
speculation/dispatch-policy follow-up PRD is their second consumer, which is G1 resolution (a)
correctly applied rather than the "future consumer in an unfiled PRD" the gate rejects. The
weakest binding in the batch is **ε**, whose consumer is a human reading a committed measurement
report plus a decision the PRD explicitly defers (Open question 4). It introduces no mechanism, so
the producer-orphan failure mode does not apply — recorded here rather than resolved.

**G2 (user-observable leaf signal).** All 29 tasks carry one. None rests on "a unit test passes
against synthetic input": α's is an executed self-test that must go red, β's and γ's are ratchet
measures over real code plus a green suite, σ's is twelve boundary scenarios against real git
fixtures, τ's is a `done_provenance` kind plus the systemd journal.

**G3 (assumed substrate).** Re-verified item by item on main — see the per-task blocks. The PRD's
own G3 note is confirmed correct in both directions: `MergeVerifySpec`, `VerifyResult`,
`GateVerdict`, `VerifyRunnerPool` and all seven wrapped functions EXIST; `DiskGuardOutcome`,
`DryRunProposal`, `EscalationRecord` and `MergeFailureKind` DO NOT, and are owned by ζ1/β and π
respectively. All 18 Appendix A source modules, the 3 test helpers, all 120 sidecar test files,
`scripts/restart-all-orchestrators.sh` (present, mode 0775) and the
`orchestrator-dark-factory.service` unit were verified present. `radon`, `complexipy` and `mutmut`
are confirmed ABSENT from `uv.lock`, as the PRD's pre-conditions state.

**G4 (cross-PRD seams).** The PRD's relationship table names an owner for every seam and asserts
no reciprocal ambiguity; re-read at decompose, confirmed. One prose correction, no design impact:
the escalation-server row says "six names" and then lists nine; §Contract says nine; **nine** is
what the code imports (measured). Recorded in η's task text so the implementer is not hunting a
tenth or stopping at six.

**G5 (B+H).** B+H, as the PRD declares. The integration-gate leaf σ exists in the decomposition
and names the §Boundary-test sketch rows 1–12 as its signal.

**G6 (premise validity).** Three numeric bounds carry stated achievability bases (1,500 lines/file;
cognitive 40/function; ≤15 for new functions) and one asserts zero by construction (the four
coupling counts). The 1,500-line bound on `worker.py` is the batch's highest-risk premise and is
flagged as such in ο's block. Two measured corrections to PRD prose are recorded in the blocks
below (routing-site count; escalation-server name count); neither changes a signal.

**G7 (design invariants, `docs/legibility/design-invariants.md`).** Walked over all 29 tasks. **No
hits, no waivers.** Four invariants are load-bearing *by design* here rather than at risk:
INV-9 `one-fact-one-home` is ρ's entire purpose; INV-10 `guards-exercise-behaviour` is why α's
ratchet must be proven by a seeded fixture and why σ asserts ceilings through the metrics script
instead of ad-hoc greps; INV-11 `no-silent-fail-soft` is stated explicitly in α (an unparseable
file is a hard failure, never a skipped measure) and extended to ε (an incomplete mutmut module
must be named INCOMPLETE); INV-2 `structured-facts-at-failure` is π. INV-7
`holds-owned-and-bounded` was examined closely because decision 10 places 120 existing tasks
behind ζ2: the hold has a named owner (ζ2) and a bound (ζ2's own terminal status under the
standard steward/escalation ladder), and the PRD deliberately does NOT wire ζ1 behind pending
no-claimant branches for exactly this reason. Recorded, not waived.

## Decompose-time gate questions resolved without the author

1. **Baseline-JSON contention across the parallel γ wave.** The PRD marks γ1–γ10 `[medium;
   parallel]` AND requires each to lower its group's contribution in
   `orchestrator/tests/merge_lane_ratchet_baseline.json` in the same commit. `lock_depth: 12` in
   `dark-factory-orchestrator.yaml` makes module locks file-granular, so declaring that one shared
   file on all ten leaves would serialize the wave through `ModuleLockTable`, and leaving it
   undeclared risks ten concurrent branches conflicting on one `totals` line. Resolved by
   constraining the format in α (PRD Open question 3 is α's to decide anyway): per-path keys must
   be line-local, and cluster **totals must be DERIVED** from the stored per-path map rather than
   stored as independently-edited numbers. The anti-rename-gaming property the PRD asks of totals
   survives, because a total derived from stored per-path baselines still catches lines moved to a
   new path. Each γ leaf declares only its own group's test files, per the PRD's "Files: the
   group's list", and its task text says to touch only its own keys.
2. **Who owns the `archaeology_blocks` measure.** α's task text does not list it; the PRD calls it
   "the metrics script's *new* measure" in ρ. Bound to **ρ** (self-produced), and ρ's task text and
   file set say so — otherwise ρ's only signal would depend on a capability no task delivers.
3. **`γ` groups vs. the >15-file review trigger.** Largest γ group is 14 files; ζ2 declares 19.
   ζ2 was re-examined for coherence rather than split: it is one atomic `git mv` of the package
   plus the reach-back/shim deletion that must land with it — a half-moved package with surviving
   reach-backs is exactly the state the batch exists to remove.

## Per-task bindings

### α — Merge-lane metrics script + ratchet test + committed baseline

- **`appendix-a-paths-all-exist-and-parse`** — PASS
  - binding: capability→producer (wired) — all 18 Appendix A source modules + 3 test helpers verified present on main 4811d62883 (merge_queue.py 21,550 lines; git_ops.py 14,721; verify_runner.py 3,531)
  - delivered_check: manual — the script is the measurer; its own coverage is proven by its seeded-fixture self-test, not by a grep
- **`complexipy-and-radon-installable`** — PASS
  - binding: capability→producer (upstream=self) — neither is in uv.lock today (verified at decompose); α adds both to orchestrator/pyproject.toml [dependency-groups] dev
  - delivered_check: `grep complexipy` in `orchestrator/pyproject.toml` — expect **present**
- **`ratchet-test-reads-a-committed-baseline`** — PASS
  - binding: capability→producer (upstream=self) — α writes orchestrator/tests/merge_lane_ratchet_baseline.json and the test that consumes it
  - delivered_check: `grep merge_lane_ratchet_baseline` in `orchestrator/tests/test_merge_lane_ratchet.py, scripts/merge_lane_metrics.py` — expect **present**
- **`whole-tree-scan-timeout-marker-exists`** — PASS
  - binding: capability→producer (wired) — WHOLE_TREE_SCAN_TEST_TIMEOUT defined at orchestrator/tests/_orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT (5x PYPROJECT_DEFAULT_TIMEOUT); already consumed by test_eval_boundary_suite.py
  - delivered_check: `grep WHOLE_TREE_SCAN_TEST_TIMEOUT` in `orchestrator/tests/test_merge_lane_ratchet.py` — expect **present**
- **`ratchet-goes-red-on-a-seeded-increase`** — PASS
  - binding: rejection-mechanism — the asserted rejection is built AND bound by α itself: a fixture adding +1 to each measure must be observed to turn the test red (INV-10 tier 1). No mechanism exists today; α is its producer.
  - delivered_check: manual — rejection is proven by an executed seeded fixture inside the task, not by a pattern

### ζ1 — merge_lane package skeleton: facade + ports

- **`merge-lane-package-absent-today-so-zeta1-creates-it`** — PASS
  - binding: capability→producer (upstream=self) — orchestrator/src/orchestrator/merge_lane/ verified ABSENT on main at decompose
  - delivered_check: `grep MergeLane` in `orchestrator/src/orchestrator/merge_lane/__init__.py` — expect **present**
- **`ports-protocols-defined`** — PASS
  - binding: capability→producer (upstream=self) — VerifyPort/ClockPort/EscalationPort are new; the PRD sketches them in §Contract→Ports
  - delivered_check: `grep class VerifyPort` in `orchestrator/src/orchestrator/merge_lane/ports.py` — expect **present**
- **`seven-wrapped-functions-exist-to-adapt`** — PASS
  - binding: capability→producer (wired) — all seven verified on main: run_scoped_verification (verify.py), _run_unscoped_typechecks (merge_queue.py), _check_post_merge_pyright + _check_post_merge_equivalence (merge_gates.py), _ensure_verify_disk_space (merge_queue.py), _run_cold_shadow_verify (merge_shadow.py), run_dry_run_unblock (dry_run_unblock.py); VerifyRunnerPool (verify_runner.py)
  - delivered_check: manual — production entry points verified at decompose; the adapters wrapping them are ζ1 output, checked by the ports grep above
- **`three-result-types-are-defined-not-assumed`** — PASS
  - binding: capability→producer (upstream=self) — DiskGuardOutcome / DryRunProposal / EscalationRecord verified ABSENT on main; the PRD names ζ1/β as their producer and forbids any task assuming them earlier. MergeVerifySpec (verify_runner.py::MergeVerifySpec), VerifyResult (verify.py::VerifyResult) and GateVerdict (merge_gates.py::GateVerdict) DO exist and are imported, not redefined.
  - delivered_check: `grep class (DiskGuardOutcome|DryRunProposal|EscalationRecord)` in `orchestrator/src/orchestrator/merge_lane/` — expect **present**

### β — Worker accepts injected ports; fakes; conftest autouse fixture

- **`autouse-dotted-path-patch-removed`** — PASS
  - binding: rejection-mechanism (expect absent) — the string 'orchestrator.merge_queue.run_scoped_verification' is present in orchestrator/tests/conftest.py::_mock_merge_queue_verification on main today; β's signal is that it is gone, replaced by fixture injection
  - delivered_check: `grep orchestrator\.merge_queue\.run_scoped_verification` in `orchestrator/tests/conftest.py` — expect **absent**
- **`merge-lane-fakes-module-exists`** — PASS
  - binding: capability→producer (upstream=self) — orchestrator/tests/_merge_lane_fakes.py is new in β; γ1..γ10 and σ all consume it
  - delivered_check: `grep class FakeVerifier` in `orchestrator/tests/_merge_lane_fakes.py` — expect **present**
- **`exercise-merge-verify-optout-preserved`** — PASS
  - binding: capability→producer (wired) — the marker is registered in orchestrator/pyproject.toml and honoured at orchestrator/tests/conftest.py (request.node.get_closest_marker); σ boundary row 2 depends on it surviving β
  - delivered_check: `grep exercise_merge_verify` in `orchestrator/tests/conftest.py` — expect **present**
- **`ports-available-upstream`** — PASS
  - binding: DAG-direction — VerifyPort/ClockPort/EscalationPort are produced by ζ1, which is UPSTREAM of β (α → ζ1 → β)
  - delivered_check: manual — DAG-direction only; ζ1 carries the mechanical check for the ports themselves

### γ1 — Migrate merge-lane test group γ1

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 1 files of sidecar group [γ1] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ2 — Migrate merge-lane test group γ2

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 12 files of sidecar group [γ2] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ3 — Migrate merge-lane test group γ3

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 12 files of sidecar group [γ3] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ4 — Migrate merge-lane test group γ4

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 13 files of sidecar group [γ4] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ5 — Migrate merge-lane test group γ5

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 13 files of sidecar group [γ5] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ6 — Migrate merge-lane test group γ6

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 13 files of sidecar group [γ6] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ7 — Migrate merge-lane test group γ7

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 14 files of sidecar group [γ7] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ8 — Migrate merge-lane test group γ8

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 14 files of sidecar group [γ8] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ9 — Migrate merge-lane test group γ9

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 14 files of sidecar group [γ9] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### γ10 — Migrate merge-lane test group γ10

- **`group-files-all-exist`** — PASS
  - binding: capability→producer (wired) — all 14 files of sidecar group [γ10] verified present on main at decompose; the sidecar's per-group file counts and line totals were re-derived and match
  - delivered_check: manual — file existence verified at decompose against plans/merge-lane-quality-prd.test-groups.txt
- **`fakes-available-upstream`** — PASS
  - binding: DAG-direction — FakeVerifier/FakeClock/FakeEscalations are produced by β, which is UPSTREAM of this leaf (ζ1 → β → γ)
  - delivered_check: manual — DAG-direction only; β carries the mechanical check for the fakes module
- **`group-scoped-patch-targets-and-private-reads-reach-zero`** — PASS
  - binding: numeric floor — the asserted bound is ZERO, and it is reachable by construction: every dotted-path patch and private-attribute read in this group is rewritten or the test is deleted. The measure is α's AST-based ratchet, not a text grep — a grep for 'orchestrator.merge_queue.' cannot distinguish a patch-target string literal from a legitimate module reference, so no mechanical check is bound here.
  - delivered_check: manual — the measure is the ratchet's AST patch-target/private-read count for this group's paths; a text grep would false-FAIL on legitimate module references

### δ — Discard the serial worker; retire the reach-back patch guard

- **`serial-worker-copy-removed`** — PASS
  - binding: rejection-mechanism (expect absent) — orchestrator/tests/_serial_merge_worker.py (393 lines) exists on main today and anchors ~89 test constructions; δ deletes it and re-homes the behaviours
  - delivered_check: `grep _serial_merge_worker` in `orchestrator/` — expect **absent**
- **`reachback-patch-guard-retired`** — PASS
  - binding: rejection-mechanism (expect absent) — orchestrator/tests/test_merge_queue_reachback_patch_guard.py (632 lines) exists today; δ deletes it ONLY once the ratchet patch-target measure is 0, so α's count supersedes the allowlist (plans/merge-queue-reliability-prd.md scope ε)
  - delivered_check: `grep test_merge_queue_reachback_patch_guard` in `orchestrator/` — expect **absent**
- **`all-ten-groups-upstream`** — PASS
  - binding: DAG-direction — δ depends on γ1..γ10; the batch-wide zero it asserts cannot be produced by δ itself
  - delivered_check: manual — DAG-direction only

### ε — Mutation-score baseline (measurement, no threshold)

- **`mutmut-installable`** — PASS
  - binding: capability→producer (upstream=self) — mutmut is not in uv.lock today (verified at decompose); ε adds it to the dev group
  - delivered_check: `grep mutmut` in `orchestrator/pyproject.toml` — expect **present**
- **`report-committed-with-per-module-counts`** — PASS
  - binding: capability→producer (upstream=self) — plans/merge-lane-quality-prd.mutation-baseline.md is new in ε
  - delivered_check: `grep survived` in `plans/merge-lane-quality-prd.mutation-baseline.md` — expect **present**
- **`no-threshold-asserted`** — PASS
  - binding: numeric floor — N/A by construction: ε asserts NO bound. Whether a mutation score becomes a ratchet measure is PRD Open question 4, decided after ε lands and outside this batch. A guessed threshold here would be exactly the bare-guess G6 rejects.
  - delivered_check: manual — deliberately unbounded — measurement only

### ζ2 — Move the lane into the package; delete reach-backs and shims

- **`worker-module-lands-in-the-package`** — PASS
  - binding: capability→producer (upstream=self) — git mv merge_queue.py → merge_lane/worker.py. Pattern is rename-robust ON PURPOSE: §Contract keeps MergeLane as the facade name with SpeculativeMergeWorker as an alias only until η, so a check pinned to the old class name alone would start FAILING after η — and this check gates all 120 externally-wired tasks (PRD decision 10).
  - delivered_check: `grep SpeculativeMergeWorker|MergeLane` in `orchestrator/src/orchestrator/merge_lane/` — expect **present**
- **`re-export-shims-gone`** — PASS
  - binding: rejection-mechanism — 8 "# noqa: F401 re-export shim" blocks measured in merge_queue.py on main; ζ2 deletes every one. Deliberately NOT a mechanical delivered_check: ζ2 gates 120 external tasks, and a stray occurrence of that string in a comment would hold and then escalate all 120. The ratchet already enforces re-export = 0 as a hard test on every verify leg, which is the stronger guard (INV-10).
  - delivered_check: manual — enforced by α's ratchet measure on every verify leg, not by a text grep amplified across 120 dependents
- **`function-local-reachback-imports-gone`** — PASS
  - binding: numeric floor — bound is ZERO and structurally reachable: the 23 reach-back sites in 6 satellites exist only to break import cycles that the one-way import rule dissolves, and the string-path patches that froze them are already removed by Phase 1 (γ/δ, upstream)
  - delivered_check: manual — measured by α's AST function-local-import count over the package; a text grep cannot see import scope
- **`phase-1-seams-unfrozen-upstream`** — PASS
  - binding: DAG-direction — ζ2 depends on δ, which depends on γ1..γ10; the move is only safe once the tests no longer pin 79 dotted paths into the monolith
  - delivered_check: manual — DAG-direction only

### η — Migrate external importers; delete merge_queue.py

- **`nine-escalation-server-names-stay-in-the-facade`** — PASS
  - binding: capability→producer (wired) — measured on main: escalation/src/escalation/server.py imports NINE names from orchestrator.merge_queue across four lazy blocks (InFlightMergeRegistry; MergeOutcome, MergeRequest, QueuedBranch, WaiterRecord, coalesce_or_enqueue_merge_request, patch_content_contained; _resolve_dispatch_time_merge_base; retire_cancelled_merge_request). The PRD's cross-PRD table prose says "six" and then lists nine; §Contract says nine — NINE is correct.
  - delivered_check: `grep from orchestrator\.merge_lane import` in `escalation/src/escalation/server.py` — expect **present**
- **`old-import-path-unreachable`** — PASS
  - binding: rejection-mechanism (expect absent) — after η, orchestrator/src/orchestrator/merge_queue.py is deleted and no first-party module imports it. NOTE dashboard/data/merge_queue.py is an unrelated module with a colliding name and is deliberately out of the checked paths.
  - delivered_check: `grep orchestrator\.merge_queue` in `orchestrator/src/, escalation/src/` — expect **absent**
- **`package-exists-upstream`** — PASS
  - binding: DAG-direction — η depends on ζ2; the facade it re-points importers at is ζ2 output
  - delivered_check: manual — DAG-direction only

### θ — Replace _WipHaltMixin with a composed HaltState object

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — _WipHaltMixin exists at orchestrator/src/orchestrator/merge_queue.py::_WipHaltMixin and SpeculativeMergeWorker inherits it (verified on main); the four direct-read bypasses of _lane_halt/_operator_halt are the measured reason composition is needed
  - delivered_check: `grep class HaltState` in `orchestrator/src/orchestrator/merge_lane/halt.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### ι — Extract merge_lane/intake.py

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — coalesce_or_enqueue_merge_request exists in merge_queue.py at cognitive complexity 103 (measured 45bf/4811d62883) and is imported by the escalation server; ι moves and decomposes it
  - delivered_check: `grep def coalesce_or_enqueue_merge_request` in `orchestrator/src/orchestrator/merge_lane/intake.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### κ — Extract merge_lane/verify_dispatch.py (VerifyDispatcher)

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — _run_post_merge_verify (175) / _run_inflight_verify (101) / _dispatch_item (49) all exist in merge_queue.py; VerifyPort is produced upstream by ζ1/β
  - delivered_check: `grep class VerifyDispatcher` in `orchestrator/src/orchestrator/merge_lane/verify_dispatch.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### λ — Extract merge_lane/speculation.py (ChainPlanner, Speculator)

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — build_chain / select_chain_depth / _deep_chain_placement / _land_chain_prefix / MergeDeepConfig.chain_cap all verified present on main; merge_speculation_controller.py (586 lines) exists and is absorbed
  - delivered_check: `grep class ChainPlanner` in `orchestrator/src/orchestrator/merge_lane/speculation.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### μ — Extract merge_lane/landing.py (Lander); decompose advance_main in place

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — _finalize_inflight (102, 69 commits/12wk) exists in merge_queue.py; GitOps.advance_main (113) exists in git_ops.py, which stays OUTSIDE the package per PRD decision 9
  - delivered_check: `grep class Lander` in `orchestrator/src/orchestrator/merge_lane/landing.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### ν — Extract merge_lane/worktrees.py

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — _owned_merge_worktrees / _owned_merge_wt_keys are worker attributes on main; the worktree-conservation audit already exists and must stay unchanged
  - delivered_check: `grep _owned_merge_worktrees` in `orchestrator/src/orchestrator/merge_lane/worktrees.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### ξ — Extract merge_lane/telemetry.py

- **`source-symbols-exist-to-move`** — PASS
  - binding: capability→producer (wired) — snapshot (58) exists in merge_queue.py and its keys are read by the dashboard; PRD §Contract freezes the key set as additive-only
  - delivered_check: `grep def snapshot` in `orchestrator/src/orchestrator/merge_lane/telemetry.py` — expect **present**
- **`package-and-ports-upstream`** — PASS
  - binding: DAG-direction — every Phase 3 task depends transitively on ζ1 (ports), ζ2 (package) and η (importers); none of them assumes a module a downstream task produces
  - delivered_check: manual — DAG-direction only
- **`cognitive-ceiling-40-per-function`** — PASS
  - binding: numeric floor — bound 40 against a demonstrated basis: verify_runner.py, same authors and domain, has total cognitive 275 with max 48 across 3,531 lines (measured 45bf). 40 sits inside a band the codebase already sustains; it is not a guessed threshold. Sonar's published 15 is the separate NEW-function bound α enforces.
  - delivered_check: manual — asserted by α's ratchet per function and re-asserted by σ against PRD §Ceilings

### ο — Residual worker.py: composition root + two loops + stop

- **`all-concern-modules-upstream`** — PASS
  - binding: DAG-direction — ο depends on ξ ← ν ← μ ← λ ← κ ← ι ← θ; every concern module it composes is produced upstream
  - delivered_check: manual — DAG-direction only
- **`worker-file-under-1500-lines`** — PASS
  - binding: numeric floor — bound 1,500 lines against a stated basis: one default agent Read call is 2,000 lines, and the cluster's healthiest module (verify_runner.py) already sustains 3,531 lines at max cognitive 48. 1,500 is STRICTER than what demonstrably works, and is reachable only because θ..ξ have removed ~8 concerns and ρ relocates archaeology prose (11,876 of merge_queue.py's 21,550 lines are comment/docstring). This is the batch's highest-risk bound; it is asserted only at σ, and the ratchet permits equality throughout so no intermediate task is blocked by it.
  - delivered_check: manual — measured by scripts/merge_lane_metrics.py; asserted at σ against PRD §Ceilings
- **`frozen-constructor-keyword-set-still-accepted`** — PASS
  - binding: capability→producer (wired) — the keyword set harness.py passes is enumerated in PRD §Contract and verified against merge_queue.py::SpeculativeMergeWorker.__init__ on main (git_ops, queue, speculation_depth, event_store, on_merge_landed, escalation_queue, train_callback_factory, merge_store, scheduler, mcp, usage_gate, cost_store, provenance_conflict_sink)
  - delivered_check: `grep SpeculativeMergeWorker|MergeLane` in `orchestrator/src/orchestrator/merge_lane/worker.py, orchestrator/src/orchestrator/merge_lane/__init__.py` — expect **present**

### π — MergeFailureKind: route on structured data

- **`reason-prefix-constants-exist-and-are-frozen`** — PASS
  - binding: capability→producer (wired) — 18 constants measured across merge_gates.py and merge_queue.py: 17 public *_REASON_PREFIX / *_REASON plus the private _MERGE_CANCEL_RETIRE_REASON. The PRD's "seventeen" is the public set.
  - delivered_check: `grep class MergeFailureKind` in `orchestrator/src/orchestrator/merge_lane/types.py, orchestrator/src/orchestrator/merge_lane/__init__.py` — expect **present**
- **`workflow-routing-sites-exist-to-convert`** — PASS
  - binding: capability→producer (wired) — MEASURED CORRECTION: nine `reason.startswith` sites over eight distinct prefixes in orchestrator/src/orchestrator/workflow.py at 4811d62883 (POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX appears twice). PRD decision 7 prose says "ten of seventeen"; neither the design nor π's signal depends on the count.
  - delivered_check: `grep reason\.startswith` in `orchestrator/src/orchestrator/workflow.py` — expect **absent**
- **`routing-equivalence-is-provable-from-pre-pi-behaviour`** — PASS
  - binding: end-to-end capability — the expectation table is built from the PRE-π code so a self-consistently wrong mapping still fails; every prefix is in π's own dependency set (they exist today, unchanged)
  - delivered_check: manual — equivalence is a parametrised test property, not a pattern

### ρ — Relocate archaeology prose; add the archaeology_blocks measure

- **`archaeology-blocks-measure-is-produced-by-rho-itself`** — PASS
  - binding: capability→producer (upstream=self) — α's task text does NOT list an archaeology_blocks measure; the PRD calls it "the metrics script's NEW measure". ρ therefore owns adding it. This is the one capability in the batch whose producer could have been mistaken for an upstream task; it is bound to ρ.
  - delivered_check: `grep archaeology_blocks` in `scripts/merge_lane_metrics.py` — expect **present**
- **`decisions-doc-is-the-single-home`** — PASS
  - binding: capability→producer (upstream=self) — docs/merge-lane/decisions.md verified ABSENT on main; ρ creates it. INV-9: one home, dated pointers elsewhere. Dispatched agents hold no memory-write surface, so the home is git-tracked (PRD decision 8).
  - delivered_check: `grep ^#` in `docs/merge-lane/decisions.md` — expect **present**
- **`count-is-the-contract-not-a-percentage`** — PASS
  - binding: numeric floor — the bound is a COUNT of 0 archaeology blocks in merge_lane/, not a prose-reduction percentage. A percentage would be an unbacked target; a count is measurable and reachable by relocating every block.
  - delivered_check: manual — measured by the new archaeology_blocks measure ρ itself adds

### σ — B+H integration gate (boundary rows 1-12)

- **`every-boundary-row-capability-is-upstream`** — PASS
  - binding: DAG-direction — rows 1-12 exercise the facade (ζ1/ζ2), the ports and fakes (β), each concern module (θ..ο), the structured failure kind (π) and the ceilings (α + ρ). All produced upstream; σ produces no mechanism of its own.
  - delivered_check: manual — DAG-direction only — σ IS the check suite
- **`real-local-runner-path-is-exercisable`** — PASS
  - binding: capability→producer (wired) — row 2 needs the REAL LocalRunner against a fixture project, reached via the exercise_merge_verify marker; the marker is registered in orchestrator/pyproject.toml and honoured in conftest.py today, and β is required to preserve it
  - delivered_check: manual — row 2 is proven by an executed real-LocalRunner land, not by a pattern; the gate module filename is not a stable anchor
- **`ceilings-asserted-through-one-encoding`** — PASS
  - binding: INV-5 — σ asserts PRD §Ceilings by calling scripts/merge_lane_metrics.py, not by re-implementing the measures in test code; rows 10/11 use the script's AST measures rather than an ad-hoc grep (INV-10)
  - delivered_check: manual — asserted by σ calling the metrics script; the gate module filename is not a stable anchor
- **`rollback-runbook-recorded`** — PASS
  - binding: capability→producer (wired) — the runbook line is real: CLAUDE.md §"Working in the main checkout" documents the direct-commit path and the git-stash prohibition; the blast radius (a broken lane blocks its own revert) is why it is on the record in σ's task text
  - delivered_check: manual — a task-text provenance requirement, not a code capability

### τ — Deterministic fleet redeploy

- **`restart-script-exists-and-is-executable`** — PASS
  - binding: capability→producer (wired) — scripts/restart-all-orchestrators.sh verified present and mode 0775 on main at decompose; before_done validation requires it AT FILING TIME, so this had to be checked here, not at dispatch
  - delivered_check: `grep restart` in `scripts/restart-all-orchestrators.sh` — expect **present**
- **`target-unit-exists`** — PASS
  - binding: capability→producer (wired) — orchestrator-dark-factory.service is a live systemd --user unit (loaded active running) with a unit file at ~/.config/systemd/user/. It equals the dispatching orchestrator's own unit, so the DeterministicRunner takes the detached systemd-run path and closes with done_provenance kind deterministic-deploy-scheduled — exactly the PRD's signal.
  - delivered_check: manual — systemd unit liveness is host state, not a repo pattern
- **`not-gated-behind-the-paused-fleet-deploy-work`** — PASS
  - binding: DAG-direction — τ is deliberately NOT wired behind tasks 3730 / 3733 / 4755 / 5020 (Leo, 2026-09-03). PRD decision 11: if 5020 lands first, τ is redundant and harmless.
  - delivered_check: manual — an explicit non-dependency; recorded so a later reader does not "fix" it

## What the orchestrator does and does not read

`metadata.delivered_checks` (copied from this manifest's mechanical checks by `commit_planning`)
IS read — by the scheduler's dispatch gate, per dependent/dep pair, with `grace_cycles` FAILED
ticks before a born-at-L2 escalation. Everything else stamped from this decomposition —
`user_observable_signal`, `consumer_ref`, `substrate_confirmed`, `prd_task_label` — is inert
today: it is substrate for a future tracking-infra session, not a live contract.

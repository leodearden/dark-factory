# Capability manifest — S25 Scripts instruments

PRD: `plans/review-all-s25-scripts-instruments-prd.md`. The machine-readable twin is
`plans/review-all-s25-scripts-instruments-prd.capability-manifest.yaml`.

**Code anchors** were verified against main `ae3047ec3a` (2026-10-10) and are cited by symbol.

Every grep and path `delivered_check` in the sidecar was measured at that tree with the expected
polarity:
- each `expect: present` pattern is absent today;
- each `expect: absent` pattern is present today.

Each leaf's checks were then run through `shared.delivered_check_polarity.lint_delivered_checks`
with the leaf's declared `metadata.files`. The result was zero findings for every leaf.

## Verdict summary

No binding is FAIL.

Two bindings are OPEN. Each is the cross-stream producer of a held leaf, and the release protocol in
the PRD's Pre-conditions binds it. Neither held leaf is committed to `pending` until its producer's
dependency is wired.

- ι: shared sidecar surface (C3), producer is S20.
- ξ: `duplicate_of` (C5), producer is S2.

## Per-leaf bindings

**α — run_parity driver**
- `run_parity` defined: produced by α. The parts it composes are wired today:
  `systemd_unit_parity.environment_map`, `find_dropins`, `Drift`. **PASS**
- `_ABSENT` private sentinel retired (absorbs task 6447). It is present today. **PASS**
- The `environment_map as _environment_map` shim is retired. It is present today. **PASS**
- The setup-host and watchdog contract is unchanged. Consumers are wired today:
  - `setup-host.sh::_orch_skip_reason` reads `[orchestrator_unit_parity] verdict` lines and
    `VERDICT_KINDS`;
  - `orchestrator-watchdog.py::unit_parity_pass` reads exit codes.
  
  The guard is the golden-fixture test (B3/B4). **PASS (manual)**

**β — EnvPolicy table**
- The renderer derives from the table. Producer α is upstream.
  - The renderer may import `systemd_unit_parity`. The section-8 repo builder in
    `tests/scripts/test_check_dashboard_unit_parity.py` copies `render_systemd_unit.py` and
    `systemd_unit_parity.py` unconditionally, before `write_checker`.
  - This was verified at `ae3047ec3a`, and it is the G3 premise the weakened key needed. **PASS**
- An impossible row is rejected: the rejection mechanism is built by β (B5). **PASS (manual)**

**γ — `_checkout` module**
- The module exists. The workspace member list is the root `pyproject.toml`
  `[tool.uv.workspace].members` (7 members), read with stdlib `tomllib`. **PASS**
- The bootstrap idiom is lint-clean.
  - Probed under the repo's `[tool.ruff]` with the scripts rule set.
  - `import _checkout as _checkout` raises no F401, being a redundant-alias re-export.
  - It sorts as its own section ahead of first-party.
  
  **PASS**
- `merge_lane_metrics.py` binds its checkout. It imports `shared` with no bootstrap today
  (recounted at main: 9 unbootstrapped shared importers). **PASS**

**δ — top-level sweep**
- Member-`src` inserts are gone from the 17 named files. Producer γ is upstream, and the named
  bootstrap variables are present in those files today. **PASS**

**ε — subdirectory sweep and guard**
- `_SHARED_SRC` and `_ORCH_SRC` are gone from `scripts/legibility/` and `scripts/sitting/`. They are
  present today. **PASS**
- The guard fires on a fixture script that lacks the bootstrap (B2). **PASS (manual)**

**ζ — public decoders**
- No private-name imports from the audit. They are present today in `census_tagger_debris.py` and
  `repair_wiped_metadata_files.py`. **PASS**
- `TERMINAL_STATUSES` is replaced by `shared.task_statuses.TERMINAL`, which exists today. **PASS**

**η — exit-code enum**
- An enum is defined. None exists today. **PASS**
- The adopters' hand restatements of "never treat 3 as a clean" are gone. They are present today.
  **PASS**

**θ — `audit_project` split**
- One coverage builder serves both branches. Two hand-built `AuditCoverage` constructions exist
  today. The guard is the completeness test on the git-failure branch. **PASS (manual)**

**ι — sidecar surface consumers (HELD)**
- Shared sidecar surface: the producer is S20's C3 leaf, which is not yet filed. **OPEN**
- The stamper calls `sidecar_for`. It derives the path inline in `_prd_bound_tasks` today. **PASS**
- The three consumer suffix spellings are gone. They are present today. **PASS**
- DAG direction: θ and 6037 are upstream. S20's leaf is upstream once the release protocol wires it.
  **PASS**

**κ — report stats and budget policy**
- `scripts/_report_stats.py` exists. **PASS**
- The three `_percentile` copies are gone. They are present today. **PASS**
- The census policy no longer comes from the test tree: `module_budget_family` and `REFUSAL_CEILING`
  are present in `verify_budget_census.py` today. **PASS**

**λ — structured rejections**
- `shared/route_rejection.py` exists. **PASS**
- The producer stops composing f-strings. There are 4 `rejected.append(f'…')` sites in `routing.py`
  today. **PASS**
- The audit vocabulary copies and the `rpartition` parse are gone. They are present today. **PASS**
- The event carries `rejections`. Both emitters are wired today:
  - `routing_dispatch` builds `{**mirror.model_dump(), 'inputs_digest': …}`;
  - `TaskWorkflow`'s routing record builds `{**decision.model_dump(), 'inputs_digest': …}`.
  
  λ adds the key through one builder. **PASS (manual, B8)**
- Field population: the `model` field is non-sentinel on the production path for the three
  `MODEL_REJECTIONS`, because `resolve_route` holds the candidate when it calls
  `_model_rejection_reason`. It is `None` only for `NOT_IN_LADDER`, where no candidate exists by
  construction (D7). **PASS**

**μ — sweep classifier**
- `FileVerdict` is defined. It is absent today. **PASS**

**ν — ratchet measures**
- `lines` and `prose_lines` leave the ratcheted set. `_SUMMED_FILE_MEASURES` contains both today.
  The guard is the fixture test (lines-only rise passes, cognitive rise fails). **PASS (manual)**

**ξ — sitting structured facts (HELD)**
- Pending legacy pins are migrated (lead ruling, 2026-10-11).
  - This is ξ's step 1, with its own check, not a decompose-time action.
  - `declare_pin` exists, and `queue.declare_pin` is its sole writer.
  - One pending prefix record exists today: `esc-3169-2`.
  - The check is a read-only scan of pending records that finds zero unmigrated ones.
  
  **PASS (manual; reads the live queue, not the tree)**
- `duplicate_of`: the producer is S2's C5 leaf, which is not yet filed. `pin_declared_by` exists and
  is enforced today: `Escalation.pin_declared_by`, refused by `resolve_issue` through
  `escalation/declared_pins.py::blocking_pin_declarations`. **OPEN**
- The string-protocol gates are gone. `VETO_PIN_PREFIX` and `DEDUP_MARKER` are present in
  `scripts/` today. **PASS**

## Decompose record (2026-10-11)

`commit_planning` stamped 12 labels with no errors and no missing labels:

| Label | Task |
|---|---|
| α | 6756 |
| β | 6757 |
| γ | 6758 |
| δ | 6759 |
| ε | 6760 |
| ζ | 6761 |
| η | 6762 |
| θ | 6763 |
| κ | 6765 |
| λ | 6766 |
| μ | 6767 |
| ν | 6769 |

Two leaves are held. They were filed in planning mode and are `deferred` outside S25's
`commit_planning`, with `metadata.x_seam_wait` set:

| Label | Task | Waits on | Who stamps it |
|---|---|---|---|
| ι | 6764 | S20, contract C3 | the session that runs `commit_planning` on it |
| ξ | 6770 | S2, contract C5 | the session that runs `commit_planning` on it |

The stamper copied each leaf's mechanical checks into that producer's
`metadata.delivered_checks`. θ and ν carry manual checks only.

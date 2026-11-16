# S25 — Scripts instruments: shared locators and parity checkers

**Status:** active — authored 2026-10-10 by the S25 seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S25, wave 1, agent mode, approved as
proposed under R21). Remediates run `review-all-dark_factory-20261008`.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10); the run's as_of is `6551b75aba`.
Anchors are cited by symbol; re-locate at implementation time.

## Goal

The next agent who changes an operator instrument under `scripts/` edits one place, and a mistake
fails a test instead of shipping a quiet wrong answer. Concretely, after this PRD lands:

- A script run from a task worktree measures that worktree's first-party code, whatever
  `VIRTUAL_ENV` the shell inherited. Every script binds its checkout the same way, through one
  module, and a guard test fails when a script imports a workspace package without it.
- The four systemd unit-parity checkers share one driver: one per-unit loop, one set of buckets, one
  exit ladder. Adding a verdict class or changing what exit 1 means is a one-file edit.
  `setup-host.sh` and `orchestrator-watchdog.py::unit_parity_pass` see byte-identical verdict lines
  and exit codes.
- Which `Environment=` variables are host-local is declared once. The renderer's preserve sets, the
  dashboard allowlist and the fused-memory required directives are derived from that declaration.
- The capability-manifest sidecar path is derived in one function that the `commit_planning` stamper
  and both manifest audits call. A git failure during sidecar discovery shows up in the audit report.
  It no longer reads as a clean, empty corpus.
- Routing rejections reach the admission audits as structured `{layer, reason, model}` records. The
  audits filter on fields, can attribute a rejection to the model that was refused, and no longer
  parse strings.
- The sitting's close-safety gates read only enforced structured fields, never punctuation inside
  free text.
- Reporting scripts share one stats/window module. The verify-budget census takes its policy from a
  production module rather than from the test tree. The merge-lane ratchet stops failing on raw line
  counts.

## Consumers and user-observable surfaces

| Mechanism | Consumer (named surface) |
|---|---|
| `scripts/_checkout.py` (C1) | Every non-test script under `scripts/` that imports a workspace member, for example `merge_lane_metrics.py`, run by `check_staged_ratchet_raise.py` and the commit gate, and `audit_delivered_checks.py`, run by operators and `/review-all`. Observable: running a script from a task worktree imports that worktree's `shared`. |
| `run_parity` driver (C2) | `scripts/setup-host.sh` sections 7/8 (exit codes plus `[orchestrator_unit_parity] verdict` lines) and `scripts/orchestrator-watchdog.py::unit_parity_pass` (exit codes). Operators run the four `check_*_unit_parity.py` CLIs. |
| Host-local env policy table (β) | `scripts/render_systemd_unit.py` (`--fix`/install path in setup-host section 8) and the dashboard and fused-memory parity checkers |
| `sidecar_for` / `tracked_sidecar_paths` (C3, S20 implements) | `commit_planning`'s stamper (`fused_memory.server.manifest_stamping`), `scripts/audit_manifest_descriptor_drift.py`, `scripts/audit_delivered_checks.py`, `scripts/audit_combine_gate_marker_loss.py`, `shared.delivered_check_polarity` |
| `shared.route_rejection` (C4) | the `routing_decision` event payload (emitted by `orchestrator.routing_dispatch` and `TaskWorkflow`), read by `scripts/audit_model_admission.py` and `scripts/review_model_admission.py` (D6-style admission milestones) |
| `Escalation.duplicate_of` (C5, S2 implements) | `scripts/sitting/gates.py::classify_member`, `scripts/member-chain-sweep.py`, the escalation-watcher dedup procedure (`skills/escalation-watcher/SKILL.md`) |
| `scripts/_report_stats.py`, verify-budget policy module (κ) | `verify_budget_census.py`, `merge_lane_throughput.py`, `review_model_admission.py`, `audit_model_admission.py`, `mint_hard_v2_fixtures.py`, `escalation_ladder.py`. The policy module is also consumed by `tests/scripts/test_module_verify_budgets.py` (the gate). |
| Exit-code enum per `_task_db_scan` tier (η) | the `--help` epilogs and docstrings of every Tier-1/Tier-2 adopter, and the operator reading exit 3 |

## Finding keys

Every key in the S25 row of the program doc, plus the one mechanical key this PRD absorbs.

| Key | Sev | Verdict | Anchor | Closed by |
|---|---|---|---|---|
| `fk-3042baa5983e` | medium | confirmed | `scripts/systemd_unit_parity.py` | α |
| `fk-cb9f382ab2bd` | medium | weakened (re-verified, kept) | `scripts/render_systemd_unit.py::HOST_LOCAL_ENVIRONMENT` | β |
| `fk-638eb09e52b1` | medium | confirmed | `slug:scripts-checkout-binding-bootstrap` | γ, δ, ε |
| `fk-9a99057011f3` | medium | confirmed | `scripts/audit_wiped_metadata_files.py` | ζ |
| `fk-679f3dc5ca48` | medium | confirmed | `scripts/_task_db_scan.py` | η |
| `fk-f70004144a3d` | low | confirmed | `scripts/audit_manifest_descriptor_drift.py::audit_project` | θ |
| `fk-b16470a6e0d2` | medium | confirmed | `scripts/audit_manifest_descriptor_drift.py::_manifest_binding` | ι |
| `fk-bb8a584a8237` | medium | confirmed (Phase-5 mechanical, `filed:6037`) | `scripts/audit_delivered_checks.py::manifest_paths` | ι takes over the discovery half from 6037 (see Related tasks) |
| `fk-915ca17415b7` | low | confirmed | `scripts/verify_budget_census.py::_percentile` | κ |
| `fk-30cd73379093` | low | confirmed | `scripts/verify_budget_census.py::_census_budget_floor` | κ |
| `fk-7fee08736560` | medium | confirmed | `scripts/audit_model_admission.py::_is_model_rejection` | λ |
| `fk-195f86e31fea` | low | confirmed | `scripts/sweep_toolcall_markup.py::run_sweep` | μ |
| `fk-f4af080fcd25` | medium | re-observed (prior row, hotspot refresh) | `scripts/merge_lane_metrics.py::build_report` | ν |
| `fk-9bb2996a620a` | medium | confirmed | `scripts/sitting/gates.py::classify_member` | ξ |

No key is dropped. `fk-6d9e5b64770a` (`quality_metrics_snapshot._import_graph`) is not an S25 key.
Its Phase-5 task **6612 is done** (merged `5ceeeae6a3`, the hidden/typing cycle sets and the
per-entry `closes_cycle` flag the layering seat asked for). Nothing here depends on it or reopens it.

### Freshness and G3/G6 re-verification

`git diff 6551b75aba..main` is empty for every anchor file except these four:
- `scripts/systemd_unit_parity.py`: one docstring pointer changed.
- `scripts/audit_delivered_checks.py`: 287 lines changed, `manifest_paths` unchanged.
- `fused-memory/src/fused_memory/server/manifest_stamping.py`: reworked by 5276, but the step-1
  derivation `re.sub(r'\.md$', '', prd_path) + _SIDECAR_SUFFIX` in `_prd_bound_tasks` is unchanged.
- `shared/src/shared/capability_manifest.py`: docstrings only.

Each finding's core claim was re-read at `ae3047ec3a`. Two numbers moved:
- **fk-638eb09e52b1 grew.** At main there are 39 files and 54 `sys.path.insert` sites (was 37/50).
  Shared importers split 13 with a bootstrap and 9 without (was 7 without): the new
  `inline_suppression_classify.py` and `inline_suppressions.py` landed without one. The hazard is
  live and spreading.
- **fk-9bb2996a620a's data shrank.** A scan of every project's `data/escalations` (archive included)
  on 2026-10-10 finds 14 dark-factory records whose `root_cause` starts with
  `veto-pin-do-not-close:`. Exactly one is pending (`esc-3169-2`), with an empty `pin_declared_by`.
  The dedup marker's only on-disk occurrence is a quotation inside `esc-3781-6`'s resolution,
  which cites `esc-3781-5`'s dismissal.
  The 40 in the finding is not reproducible. The structural claim holds: the gate reads protocol
  text, and the only pending prefix record is outside `resolve_issue`'s protection.

**Weakened key `fk-cb9f382ab2bd` (R23).** Re-verified under G3/G6.
- `render_systemd_unit.py` is unchanged since as_of. Its `HOST_LOCAL_ENVIRONMENT`,
  `FUSED_MEMORY_HOST_LOCAL_ENVIRONMENT` and `_ENVIRONMENT_SECTION` still restate a policy the
  checkers hold.
- The proposal's premise is verified. `tests/scripts/test_check_dashboard_unit_parity.py`'s
  section-8 repo builder copies `systemd_unit_parity.py` UNCONDITIONALLY, before `write_checker`, so
  the renderer may import from it without breaking the stub-checker and `with_checker=False` tests.
- The skeptic is right that the four relations are test-held. The remaining cost is that the policy
  spans three modules' prose plus a 600-character allowlist value. Adding a host-local variable is a
  three-file edit whose missing step only some relation test catches.
- β is kept and scoped to that cost: one declaration, derived sets, and a constructor that rejects
  an impossible row.

## Sketch of approach

Six workstreams, each owning the files it touches:

1. **Checkout binding (γ, δ, ε).** A stdlib-only `scripts/_checkout.py` puts this checkout's
   workspace-member `src/` directories at the front of `sys.path` (C1). Every script that imports a
   member imports it first, under the lint-clean idiom `import _checkout as _checkout`. Ruff treats
   that redundant alias as an explicit re-export: probed at main, it raises no F401 and sorts as a
   third-party section ahead of first-party, so no inline suppression is added. The per-file
   paragraphs and inserts go. A behavioural test proves the binding from a copied checkout layout,
   and a static guard keeps it uniform.
2. **Parity family (α, β).** `systemd_unit_parity.run_parity(units, compare, probes)` owns the loop,
   the buckets, the report blocks and the exit ladder (C2). Each checker declares its `UnitSpec`
   table, `compare_unit` and remedy strings. The re-export shims retire, the suites import
   `systemd_unit_parity` directly, and the absent sentinel becomes public. β then moves the
   host-local, allowlisted and required `Environment=` policy into one table in
   `systemd_unit_parity`, and the renderer and both checkers derive from it.
3. **Audit library hygiene (ζ, η, θ, ι).**
   - ζ: the wiped-metadata audit's decoders move to a public module; importers name the definition
     (`discover_project_roots` from where it is defined, `TERMINAL` from `shared.task_statuses`).
   - η: `_task_db_scan` exit codes become one `IntEnum` per tier that epilogs render from; the
     docstring keeps the API, and the history becomes a pointer to its record.
   - θ: `audit_project` splits into per-direction passes over one coverage accumulator.
   - ι: the stamper and all three manifest audits call S20's `sidecar_for` and
     `tracked_sidecar_paths` (C3), and `audit_delivered_checks` records discovery failure in coverage.
4. **Typed routing rejections (λ).** A new shared leaf `shared.route_rejection` (C4) holds the
   rejection vocabulary as enums plus a frozen `RouteRejection(layer, reason, model)`.
   - `routing.resolve_route` builds these values instead of f-strings.
   - Both `routing_decision` emitters add an event-only `rejections` list through one builder in
     `routing_dispatch`.
   - The metadata mirror's `rejected: list[str]` stays as it is (it is S10's model) and is rendered
     from the same values.
   - The audits filter on fields. Pre-change rows go through the single legacy decoder that lives
     beside `render`, and the report says how many rows were legacy.
5. **Reporting helpers (κ, μ, ν).**
   - κ: `scripts/_report_stats.py` holds one percentile per method (None on an empty series), one
     `--window` grammar and one instant parser. `census_budget_floor` and the refusal ceiling move to
     a production module that the census and the gate test both import.
   - μ: `run_sweep` becomes a pure per-file classifier plus one accumulator.
   - ν: `lines` and `prose_lines` leave the ratcheted measure set but stay in `--report`.
6. **Sitting structured facts (ξ).** The close-safety gates read `pin_declared_by` and S2's
   `duplicate_of` (C5) only. The `VETO_PIN_PREFIX` and `DEDUP_MARKER` checks go, and the watcher
   skill's dedup and pin instructions name the structured route.

## Resolved design decisions

- **D1 — Locators are S14's, not ours.** S25 defines no path or root locator.
  - The scripts' `runs.db` spellings and hard-coded roots belong to S14's key `fk-954f6787b07c` and
    its leaf, which repoints `scripts/` onto `shared.project_layout`. The "7+ runs.db spellings and
    16 hard-coded roots" deletable in S25's remedy is delivered there.
  - S25 leaves that touch the same scripts depend on that S14 leaf (file serialisation). They import
    any locator they need from `shared.project_layout` and never re-spell one.
- **D2 — `_checkout` binds every workspace member, from `pyproject.toml`.**
  - The member list is read from the root `pyproject.toml` `[tool.uv.workspace].members` with
    `tomllib` (structured data, one home) and is not hard-coded.
  - All members' `src/` directories are prepended, so a script in checkout X runs against checkout
    X's first-party code for every package it imports, not just `shared`.
  - An unreadable or memberless pyproject raises at import. A script that cannot bind its checkout
    must not run against another one (INV-11).
- **D3 — Scripts in a subdirectory bind `scripts/` in exactly one line.**
  - `scripts/sitting/*` and `scripts/legibility/*` run with their own directory at `sys.path[0]`, so
    `import _checkout` is unreachable without help. Their one permitted `sys.path` mutation is the
    scripts-dir line `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`, followed by
    `import _checkout as _checkout`.
  - The guard (ε) accepts exactly this form in a subdirectory, plus `_checkout.py` itself.
  - Inserts that bind something other than a workspace member are outside the rule by construction,
    so no allow-list exists (INV-12). Examples: `check_write_triage_attach_*`'s extracted historical
    trees, and the scripts-dir inserts in `orchestrator-watchdog.py`, which S24 owns.
- **D4 — Parity exit codes and verdict lines are a frozen external contract.** `run_parity` keeps the
  0/1/2 ladder exactly: drift dominates absence, and zero units compared never reports parity. It
  also keeps `check_orchestrator_unit_parity.VERDICT_KINDS` and the
  `[orchestrator_unit_parity] verdict <unit> <kinds>` line byte for byte, because `setup-host.sh`'s
  `_orch_skip_reason` and `orchestrator-watchdog.py::unit_parity_pass` branch on them. Checker-specific
  verdict classes (lms `[unverifiable]`/`[effective]`) enter through the `probes` hook, not as forks
  of the loop.
- **D5 — The host-local policy is a table of rows, and impossible rows do not construct.**
  - Each row is `EnvPolicy(unit_family, name, host_local, compared_cross_copy, required)`, defined in
    `systemd_unit_parity`, which is stdlib-only and copied unconditionally by the section-8 harness.
  - A row both `host_local` and `required` raises at module import, because `--fix` would append a
    last-wins line that clobbers the preserved value.
  - The checker's `DIVERGENCE_ALLOWLIST`, the renderer's preserve sets and the `Environment=` part of
    `REQUIRED_SERVICE_DIRECTIVES` become projections of the table.
  - The section name `Service` is declared once.
  - Rationale prose moves out of the allowlist's value strings into this PRD.
- **D6 — Sidecar derivation belongs to `shared.capability_manifest`, which S20 owns (program seam
  table, synthesis precedent).** S25 states the interface (C3) and consumes it, and the consuming
  leaf ι is held until S20's implementing leaf exists (see Pre-conditions). Two parts stay with each
  consumer: the stamper's label-admission buckets (`missing_labels`, `near_miss_*`) and the audits'
  "both prd_path and prd_task_label" admission. They are consumer policy, and they differ on purpose.
- **D7 — Structured rejections travel on the event, not the metadata mirror.**
  - `shared.task_metadata.RoutingDecision.rejected: list[str]` is S10's wire model, and S10 is moving
    slice models (R3), so it is not changed. The event payload gains the `rejections` list beside it,
    as it already carries the event-only `inputs_digest` and `applied_budget_usd`.
  - Both emitters (`routing_dispatch` and `TaskWorkflow`'s routing record) call one payload builder in
    `routing_dispatch`. S9's later consolidation onto `routing_dispatch.resolve_and_record_route`
    (R2) absorbs the workflow site unchanged.
  - The `model-not-in-ladder` reason carries `model=None` (no candidate was formed). The audit
    excludes it by enum member, not by an omitted string.
- **D8 — Legacy routing rows are decoded once, beside the encoder.** `RouteRejection.parse_legacy(str)`
  lives in `shared.route_rejection` next to `render()`, and a round-trip test pins it. It is the only
  string decoder of the vocabulary. The audits report `legacy_rows` (attributed to no model) as their
  own count, so "no structured rows yet" never reads as "zero rejections" (INV-13).
- **D9 — Empty series → `None`, everywhere.** `_report_stats` percentiles return `None` on an empty
  input and renderers print `n/a`. `mint_hard_v2_fixtures`'s `0.0`-on-empty is the misreading
  `merge_lane_throughput` warns about; κ moves it to `None` and makes its callers handle it.
  - The `--window` grammar is the union `Nh | Nd | <iso>..<iso>`.
  - Interpolated percentile and nearest-rank percentile are two named functions. Method is a real
    dimension of variability, so neither replaces the other.
- **D10 — The merge-lane ratchet measures structure, not size.** `lines` and `prose_lines` stay in
  `build_report` and `--report` but leave the ratcheted set. Both the per-file check and the
  `derive_totals` sum stop gating them. The append-only authorized-raises ledger keeps its history.
  The new-file line ceiling, a heuristic-14 soft ceiling applied to files created inside the cluster,
  is unchanged and out of scope.
- **D11 — The sitting reads only enforced fields.**
  - Pin: `pin_declared_by`, which `resolve_issue` already enforces (task 4377).
  - Dedup survivor: S2's `duplicate_of` (C5).
  - The one live legacy pin, `esc-3169-2`, is migrated through `declare_pin` before ξ removes the
    prefix read. The migration is a decompose-time action recorded in the FILED row, not code.
  - The 13 archived prefix records are terminal, and a pin protects only an open record, so they
    need no migration.
- **D12 — Reviewer-answering history leaves the code, with a pointer.** Each history block is
  replaced by a one-line pointer to the record that already holds it (a task id, a plans file, or
  this PRD), per the Comments stance. This covers `systemd_unit_parity`'s "what is not shared"
  history (obsolete once α lands), `_task_db_scan`'s task-3817 narrative, and `verify_budget_census`'s
  false "COPIED, not imported" header. Nothing load-bearing is deleted without a home.

### D5 policy rows

The rows live in `scripts/systemd_unit_parity.py::ENV_POLICY`. Each column has one reader. The
renderer reads `host_local` for both families, the dashboard checker reads `compared_cross_copy` for
dashboard rows, and the fused-memory checker reads `required_value` for fused-memory rows. A row
states only the columns it changes from their defaults, and a name with no row takes the defaults:
not preserved, value-compared across copies, not required. Each row is what it is for these reasons,
which used to live in the allowlist's value strings and the renderer's and fused-memory checker's
comments.

**dashboard `DASHBOARD_KNOWN_PROJECT_ROOTS`: host-local, not compared.** This is the gate's declared
hole. Cost/burndown aggregation roots beyond this repo are a local setting: the committed unit's own
comment says extra roots are added to the installed unit, not committed. Measured 2026-08-01, the
installed unit carried 9 roots and the committed one carried 1. Value-comparing it would make the
gate red on every correctly configured host, and a gate that is always red gets switched off. It is
preserved because the old `sed ... > "$UNIT_DIR/..."` re-render collapsed the 9 roots to 1 (task
4793), and the allowlist hid the loss. Presence is still compared, so a dropped line is still drift.

**dashboard `DASHBOARD_PROJECT_ROOT`: not compared, and not host-local.** It is allowlisted because
its value embeds the host's repo root: the committed unit hardcodes `/home/leo/src/dark-factory`, and
setup-host.sh renders the installed copy with that host's `$REPO_ROOT`. It is still checked, within
one copy, against that copy's `WorkingDirectory=`
(`scripts/check_dashboard_unit_parity.py::UnitSpec.env_matches_directive`). It is rendered from
`__REPO_ROOT__`, so preserving it would pin the data root at the previous checkout while
`WorkingDirectory=` moved to the new one, which is exactly the intra-copy drift that check reports.

**fused-memory `DASHBOARD_KNOWN_PROJECT_ROOTS`: host-local, not required.** On this
unit the variable is `fused_memory/models/scope.py`'s `KNOWN_PROJECT_ROOTS_ENV`, and reconciliation
raises `UnknownProjectError` for a project outside the set. Collapsing it therefore de-registers
every other project from reconciliation (task 4796). The fused-memory checker cannot see that
damage, because it checks only host-invariant safety lines.

**fused-memory `MEM0_TELEMETRY` and `PATH`: required.** These are host-invariant safety lines (`%h`
expands per user), so the checker exact-matches them and `--fix` may synthesize them. Rule B5 keeps
them from being host-local.

**No row.** `CONFIG_PATH`, `PROJECT_ROOT` and `TASKMASTER_DIR` are rendered from `__REPO_ROOT__`,
so preserving them would pin the server at the old checkout, the `DASHBOARD_PROJECT_ROOT` hazard
again. `FUSED_MEMORY_PREDONE_HOOK_REIFY` looks host-local because it carries a literal host path,
but it is committed verbatim with no sentinel, so a re-render reproduces it byte for byte.
Preserving it would pin a possibly stale hook path over the committed one. The default policy is
right for all four.

**The impossible rows.** `EnvPolicy.__post_init__` rejects them, so a module declaring one fails at
import.
- B5, `host_local` with a `required_value`. The fused-memory checker tests exact whole-line
  membership, so a preserved multi-root line reads as the required line missing. `--fix` then
  appends the required line after the last `[Service]` line, systemd's last-wins makes it beat the
  preserved value, and the checker exits 0.
  `tests/scripts/test_check_fused_memory_unit_parity.py::test_a_required_known_project_roots_line_would_reclobber`
  demonstrates the clobber.
- `host_local` with `compared_cross_copy` on a dashboard row. The dashboard checker would
  value-compare a value the renderer deliberately keeps divergent, so the gate would be red on every
  configured host. This rule is what makes "the renderer's dashboard preserve set is a subset of the
  allowlist" hold by construction.
- A column changed from its default on a family whose checker never reads it: `required_value` on a
  dashboard row, or `compared_cross_copy=False` on a fused-memory row. The setting would do nothing,
  and nothing would notice an author flipping it.

## Contracts (B+H)

G5 fires: the stream reaches scripts, shared, fused-memory and orchestrator, it has six other
streams as consumers or producers, and it carries more than eight mechanisms. The contracts below
are the seams. C3 and C5 are interface requests to their owners.

**C1 — `scripts/_checkout.py` (S25 owns).**
```python
# stdlib only; importing it is the whole API.
ROOT: Final[pathlib.Path]            # this checkout's root (parent of scripts/)
MEMBER_SRC: Final[tuple[pathlib.Path, ...]]  # <ROOT>/<member>/src for each pyproject workspace member that has one
# On import: for each p in reversed(MEMBER_SRC) not already at its target position, sys.path.insert(0, str(p)).
# Raises CheckoutBindingError(RuntimeError) if pyproject.toml is unreadable or lists no members.
```
Invariants:
- After `import _checkout`, `importlib.util.find_spec(m).origin` lies under `ROOT` for every
  workspace member package `m`, even when `VIRTUAL_ENV`/`sys.path` already resolve `m` elsewhere.
- Importing it twice is a no-op.
- No other non-test script mutates `sys.path` toward a workspace member's `src`.

**C2 — `systemd_unit_parity.run_parity` (S25 owns).**
```python
@dataclass(frozen=True)
class ParityRun:            # what a checker hands the driver
    tool: str               # report prefix, e.g. 'orchestrator_unit_parity'
    units: Sequence[UnitRef]          # name, repo path, installed path
    compare: Callable[[UnitRef, str, str], list[Drift]]
    report: ReportText                # per-bucket BlockText wording; only drift is required
    probes: Sequence[Probe] = ()      # extra per-unit verdict classes (lms): Probe.run -> ProbeOutcome
    print_verdicts: bool = False      # orchestrator's `verdict <unit> <kinds>` lines
def run_parity(run: ParityRun, *, out: TextIO) -> ParityExit: ...
# ParityExit: 0 parity, 1 drift or could-not-verify (vanished/unreadable/override/probe/zero compared),
# 2 not-installed only
ABSENT: Final = '<absent>'           # public; _ABSENT retired
```
Invariants:
- The returned code follows the ladder (drift > absence/override > parity), and a run that compared
  zero units never returns 0.
- Bucket blocks print in a fixed order, so per-unit verdict lines are emitted unchanged when a
  checker declares them (orchestrator).
- The driver opens no file the checker did not name.

**C3 — `shared.capability_manifest` sidecar surface (S20 owns and implements; S25 requests).**
```python
SIDECAR_SUFFIX: Final = '.capability-manifest.yaml'
def sidecar_for(prd_path: object, project_root: str | os.PathLike[str]) -> str | None:
    """Root-relative POSIX path of the sidecar commit_planning would open for prd_path.
    None when prd_path is not a non-empty str, cannot be resolved (embedded NUL), or resolves
    outside project_root. Derivation: re.sub(r'\\.md$', '', prd_path) + SIDECAR_SUFFIX, then
    (root / derived).resolve() containment-checked against root.resolve()."""
class ManifestDiscoveryUnavailable(RuntimeError): ...
def tracked_sidecar_paths(project_root: str | os.PathLike[str], *, timeout: float = 30) -> list[str]:
    """Sorted root-relative paths of every git-TRACKED sidecar (git ls-files -z -- '*'+SUFFIX).
    Raises ManifestDiscoveryUnavailable on OSError, timeout or non-zero rc — never returns []
    for a failure."""
```
Invariants:
- `sidecar_for` behaves exactly like the stamper's current step 1 plus step 2 for every input in the
  boundary table below.
- The suffix is spelled once in production code.
- Discovery failure is an exception, never an empty list.

**C4 — `shared.route_rejection` (S25 owns; new stdlib-only leaf).**
```python
class RejectionLayer(StrEnum): CONFIG='config'; POLICY_RULE='policy_rule'; METADATA_OVERRIDE='metadata_override'
class RejectionReason(StrEnum):
    NOT_IN_ALLOWLIST='model-not-in-allowlist'; CEILING_EXHAUSTED='model-ceiling-exhausted'
    CAPACITY_EXHAUSTED='model-capacity-exhausted'; NOT_IN_LADDER='model-not-in-ladder'
MODEL_REJECTIONS: Final[frozenset[RejectionReason]]  # the three a candidate model was refused for
@dataclass(frozen=True)
class RouteRejection:
    layer: RejectionLayer; reason: RejectionReason; model: str | None
    def render(self) -> str: ...            # '<layer>:<reason>' — the legacy/mirror spelling
    def to_dict(self) -> dict[str, str | None]: ...
    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> 'RouteRejection': ...   # raises on unknown enum
    @classmethod
    def parse_legacy(cls, entry: str) -> 'RouteRejection | None': ...      # model=None; None if unparseable
```
Invariants:
- `parse_legacy(r.render())` returns `r` with `model=None` for every enum combination.
- `routing._model_rejection_reason` returns `RejectionReason | None`.
- A `routing_decision` event written after λ carries `rejections` with one entry per `rejected`
  entry, in the same order.

**C5 — `Escalation.duplicate_of` (S2 owns and implements; S25 requests).**
- A persisted `duplicate_of: str | None` on `escalation.models.Escalation`. It is the id of the
  survivor record that a dedup dismissal folded this record into.
- It is written at resolve time by `resolve_issue` (a parameter accepted with a dismissing action),
  and its zero-migration default is `None`.
- S2 chooses the name and shape (R9's `record_disposition` on `ACTION_EFFECTS` is a natural home).
  The constraints are fixed: it is persisted on the record, written by the dedup dismissal itself,
  and readable from `escalation.models` without parsing `resolution`.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions | Leaf |
|---|---|---|---|---|
| B1 | Worktree binding beats an inherited venv | tmp copy of `pyproject.toml`, `scripts/_checkout.py`, a stub `shared/src/shared/__init__.py`; `PYTHONPATH`/cwd pointing at a different `shared` | `python -I tmp/scripts/probe.py` prints a `shared.__file__` under tmp | γ |
| B2 | Guard catches a missing bootstrap | a fixture script importing `shared` with no `_checkout` import | guard test reports that path; the live tree reports none | ε |
| B3 | Four checkers, one ladder | fixture unit dirs per checker: vanished, missing, drop-in override, drift, zero-compared | each checker's exit code and report blocks equal its pre-α golden output on the same fixture, and the ladder is the same across all four; orchestrator verdict lines are byte-identical | α |
| B4 | setup-host contract holds | existing section-7/8 slicer tests | unchanged and green; `VERDICT_KINDS` arm-coverage test green | α |
| B5 | Impossible policy row | an `EnvPolicy` row with `host_local=True, required=True` | import raises; renderer preserve set ⊆ allowlist holds by derivation | β |
| B6 | Stamper and audits agree on every sidecar path | table of `prd_path` values: plain, `./`-prefixed, `..`-bearing in-root, escaping root, absolute in-root, `.prd.md` stem, non-str, NUL-bearing | `manifest_stamping`'s admitted path == `audit_manifest_descriptor_drift`'s binding == `sidecar_for(...)` for each row | ι |
| B7 | Discovery failure is visible | project root that is not a git checkout | `audit_delivered_checks` report carries a discovery-failed coverage row and a non-clean exit; drift audit unchanged | ι |
| B8 | Structured rejection end to end | resolver config rejecting a candidate on ceiling | the emitted `routing_decision` event's `rejections` names `{layer, reason, model}`; `audit_model_admission` attributes it to that model | λ |
| B9 | Legacy rows still counted | runs.db fixture with pre-λ rows (`rejected` strings only) | audit counts them as `legacy_rows` with model unattributed; no row silently dropped | λ |
| B10 | Survivor read from the record | member record with `duplicate_of` set and a free-text resolution lacking the marker | `classify_member` → dedup survivor; a record whose resolution says `survivor; stays open` but has no field → `ruled` | ξ |

## Pre-conditions for activating

- **S14's scripts-locator leaf** (the leaf closing `fk-954f6787b07c`, filed in S14's wave-1 batch)
  lands before γ, so the locator repoint and the bootstrap sweep do not collide in the same files.
- **S20's sidecar-surface leaf** implementing C3 exists before ι is released. S20 is a wave-3 spawn
  stream. ι is filed `planning_mode` and held out of S25's `commit_planning` (`deferred`, carrying
  `metadata.x_seam_wait` naming S20 and C3). S20's decompose session wires
  `add_dependency(ι, <its C3 leaf>)` and includes ι in its own `commit_planning`. Owner: the S20
  stream (program seam table). Bound: the trigger-chain gate 6698 is committed only when the FILED
  table is complete (INV-7).
- **S2's `duplicate_of` leaf** implementing C5: the same arrangement for ξ, released by S2's decompose
  session.
- `declare_pin` on `esc-3169-2` (D11), done at S25 decompose time or by the lead, before ξ lands.

## Cross-PRD relationship

| Other stream / PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S14 shared leaves | consumes | `shared.project_layout` locators; the scripts repoint (`fk-954f6787b07c`) | S14 | γ depends on S14's scripts leaf |
| S20 CLI invocation and usage gate | consumes | `shared.capability_manifest` `SIDECAR_SUFFIX`, `sidecar_for`, `tracked_sidecar_paths`, `ManifestDiscoveryUnavailable` (C3) | S20 | requested here; ι held on S20's leaf |
| S2 typed escalation record facts | consumes | `Escalation.duplicate_of` (C5); `escalation.models` vocabulary | S2 | requested here; ξ held on S2's leaf |
| S16 config and routing | produces → | `shared.route_rejection` (C4), new; S16's resolver move to shared imports it | S25 | λ lands first; S16 depends on λ when it files |
| S10 task-write contract | none (deliberate) | `RoutingDecision.rejected: list[str]` untouched (D7) | S10 | no edit |
| S9 workflow spine | shares a file | `TaskWorkflow` routing-record emit calls `routing_dispatch`'s payload builder; S9's consolidation absorbs it | S9 | λ edits one call site; S9 files after |
| S6 merge-lane lifecycle | consumes later | `MergeOutcome.kind` replaces `review_model_admission`'s `DROPPED_PLAN_TARGETS_REASON_PREFIX` match; until then λ imports the constant from `orchestrator.merge_lane.gates` (the copy and its stale `merge_gates.py` pointer go) | S6 | note for S6's session |
| S4 verify run context | shares a file | `attempt-*` grammar helper consumed by `verify_budget_census` | S4 | κ lands first; S4's leaf depends on κ |
| S21 legibility pipeline | shares files | `scripts/legibility/*` headers | S21 | ε depends on S21's leaves touching those files (wired at decompose) |
| S24 fleet watchdog | none | `orchestrator-watchdog.py` scripts-dir inserts stay S24's (D3) | S24 | out of rule scope |

## Related tasks (contract §8)

- **6037** (pending, low; curator-combined carrier of `fk-bb8a584a8237`). Its sidecar-discovery
  bullet would add discovery to `shared` next to `load_capability_manifest`, which is S20's surface
  under the seam table. Decompose amends 6037: that bullet is superseded by ι (via C3), and the
  remaining bullets stand. ι depends on 6037 because both edit `scripts/audit_delivered_checks.py`,
  and ι's `x_finding_key` carries `fk-bb8a584a8237`.
- **6612** (done): see Finding keys. No action.
- **6447** (pending, low): make `_ABSENT` public. Wholly inside α's scope. Cancelled at decompose with
  a pointer to α, and α's description names it.
- **6381** (pending, low): dashboard parity is blind to `ExecStart` switches. This is a behavioural fix
  in α's files. Wired to depend on α, so it lands on the driver.
- **6422** (pending, low): split `merge_lane_metrics`'s verdict half. Wired to depend on ν, so the
  measure set is settled first.
- **6208** (pending, low): touches `manifest_stamping` and `audit_manifest_descriptor_drift`. No
  dependency: file locks serialise dispatch, and the scopes are disjoint.

## Decomposition plan

Sizes are expected changed LOC. Every leaf is `task_kind="normal"`, its priority is its keys' highest
severity, and no leaf is `high` or `critical`, so none becomes a dependency of gate 6698.
Same-file leaves are chained by real edges.

- **α — Lift the unit-parity driver into systemd_unit_parity (run_parity)**
  - Size and keys: ~1,200 LOC, net negative. Key `fk-3042baa5983e`. Priority medium.
  - Files: `scripts/systemd_unit_parity.py`, `scripts/check_dashboard_unit_parity.py`,
    `scripts/check_orchestrator_unit_parity.py`, `scripts/check_lms_unit_parity.py`,
    `scripts/check_fused_memory_unit_parity.py`, `tests/scripts/test_check_dashboard_unit_parity.py`,
    `tests/scripts/test_check_orchestrator_unit_parity.py`, `tests/scripts/test_check_lms_unit_parity.py`,
    `tests/scripts/test_check_fused_memory_unit_parity.py`, `tests/scripts/test_render_systemd_unit.py`.
  - Work:
    - C2.
    - The three `main()` loops and the fourth `[override]` block become one driver.
    - `ABSENT` becomes public, absorbing 6447.
    - The `_join_continuations` and `_environment_map` re-export shims are retired, and the suites
      import `systemd_unit_parity` (the identity asserts go).
    - The module docstring keeps the API; the "not shared" history becomes a pointer (D12).
  - Signal: B3 and B4. On a fixture matrix the four CLIs exit identically, the
    `[orchestrator_unit_parity] verdict` lines are byte-identical to pre-change golden output, and
    the setup-host section-7/8 tests pass unchanged.
  - Consumer: `setup-host.sh`, `orchestrator-watchdog.py::unit_parity_pass`.
  - Deps: none in batch. Out of batch: 6381 depends on α.
- **β — Declare host-local Environment= policy once (EnvPolicy table)**
  - Size and keys: ~500 LOC. Key `fk-cb9f382ab2bd`. Priority medium.
  - Files: `scripts/systemd_unit_parity.py`, `scripts/render_systemd_unit.py`,
    `scripts/check_dashboard_unit_parity.py`, `scripts/check_fused_memory_unit_parity.py`, and the
    three matching test files.
  - Work: D5. The renderer imports the table from `systemd_unit_parity`. The relation tests become
    derivation tests plus B5.
  - Signal: rendering a unit with a host-local variable in a tmp installed copy preserves it, and the
    dashboard checker reports it as allowlisted, both from one table row. B5 passes.
  - Consumer: setup-host section 8 render/install, the dashboard and fused-memory checkers.
  - Deps: α.
- **γ — Add scripts/_checkout.py and bind the unbootstrapped shared importers**
  - Size and keys: ~450 LOC. Key `fk-638eb09e52b1`. Priority medium.
  - Files: `scripts/_checkout.py`, `tests/scripts/test_scripts_checkout_binding.py`,
    `scripts/gen_dashboard_task_vocab.py`, `scripts/inline_suppression_classify.py`,
    `scripts/inline_suppressions.py`, `scripts/merge_lane_metrics.py`,
    `scripts/quality_metrics_snapshot.py`, `scripts/suite_census.py`,
    `scripts/trial_module_tagger_haiku.py`, `scripts/check_staged_ratchet_raise.py` (its "run from a
    shell whose python3 can import shared" instruction is replaced).
  - Work: C1 and D2.
  - Signal: B1. Also, `python3 scripts/merge_lane_metrics.py --report` run from a task worktree with
    the main checkout's `VIRTUAL_ENV` exported imports the worktree's `shared` (the test runs the
    real script under `-I` from a temp copy).
  - Consumer: δ, ε, and every operator CLI run.
  - Deps: S14 scripts-locator leaf.
- **δ — Replace per-file checkout bootstraps in top-level scripts with _checkout**
  - Size and keys: ~400 LOC, net negative. Key `fk-638eb09e52b1`. Priority medium.
  - Files: `scripts/_eval_results_io.py`, `audit_combine_gate_marker_loss.py`,
    `audit_delivered_checks.py`, `audit_manifest_descriptor_drift.py`,
    `backfill_decision_queue_stamp.py`, `check_consolidation_closure.py`,
    `inline_suppression_refusal.py`, `migrate_metadata_modules_to_files.py`, `mint_hard_v2_fixtures.py`,
    `repair_wiped_metadata_files.py`, `reviewer_redundancy_diagnostic.py`,
    `run_fable_trial_v2_campaign.py`, `run_judge_ofat_pilot.py`, `run_vllm_eval.py` (its hard-coded
    `/home/leo/src/dark-factory/...` member path goes), `scan_plan_decision_pairing.py`,
    `scan_task_toolcall_leaks.py`, `sweep_toolcall_markup.py`. That is 17 files, all under `scripts/`.
  - On the size: this crosses the 15-file review trigger deliberately. It is one uniform header
    substitution with no behaviour change, so it is coherent as a unit.
  - Signal: none of the 17 files contains a member-`src` `sys.path` insert any more (delivered check),
    and each CLI's existing suite passes.
  - Consumer: ε (the guard needs the tree clean).
  - Deps: γ.
- **ε — Bind subdirectory scripts through _checkout and add the uniform-binding guard**
  - Size and keys: ~450 LOC. Key `fk-638eb09e52b1`. Priority medium.
  - Files: `scripts/legibility/{account_pool,census,check_transcript_persistence,check_trickle_health,check_trickle_progress,codebook,coder,digest,inventory,nightly,sampling,session_ledger,session_runner}.py`,
    `scripts/sitting/{nightly_prepare,prepare_sitting,return_brief,ledger,preparation}.py`,
    `tests/scripts/test_scripts_checkout_binding.py`. 19 files, the same uniform-edit justification
    as δ.
  - Work: D3 plus the static guard. Every non-test script importing a workspace member imports
    `_checkout` first, and member-`src` inserts exist only in `_checkout.py`.
  - Signal: B2. The guard is green over the live tree and red on the fixture.
  - Consumer: every future script author (the guard).
  - Deps: δ, plus S21 leaves that touch `scripts/legibility/*` (wired at decompose).
- **ζ — Publish the wiped-metadata audit's shared decoders; importers name definitions**
  - Size and keys: ~400 LOC. Key `fk-9a99057011f3`. Priority medium.
  - Files: `scripts/audit_wiped_metadata_files.py`, a new public module
    (`scripts/_plan_scope.py`, name tactical), `scripts/census_tagger_debris.py`,
    `scripts/repair_wiped_metadata_files.py`, `scripts/tests/test_audit_wiped_metadata_files.py`,
    `tests/scripts/test_census_tagger_debris.py`, `tests/scripts/test_repair_wiped_metadata_files.py`.
  - Work:
    - `_coerce_file_list`, `_decode_files`, `_EVENT_PLAN_SOURCES` and the coverage renderer move to
      public names.
    - `discover_project_roots` is imported from its definition, and the `noqa` re-export shim goes.
    - `TERMINAL_STATUSES` is replaced by `shared.task_statuses.TERMINAL`.
  - Signal: on the existing fixture stores the census, repair dry-run and audit outputs are
    byte-identical, and `git grep` finds no `from audit_wiped_metadata_files import _` and no
    `TERMINAL_STATUSES =` in scripts (delivered checks).
  - Consumer: `census_tagger_debris.py`, `repair_wiped_metadata_files.py`.
  - Deps: δ (`repair_wiped_metadata_files.py`), S14 scripts-locator leaf.
- **η — One exit-code enum per _task_db_scan tier; adopters render from it**
  - Size and keys: ~500 LOC. Key `fk-679f3dc5ca48`. Priority medium.
  - Files: `scripts/_task_db_scan.py`, `scripts/audit_wiped_metadata_files.py`,
    `scripts/audit_combine_gate_marker_loss.py`, `scripts/audit_delivered_checks.py`,
    `scripts/audit_manifest_descriptor_drift.py`, `scripts/scan_plan_decision_pairing.py`,
    `scripts/scan_task_toolcall_leaks.py`, `scripts/repair_wiped_metadata_files.py`,
    `scripts/tests/test_task_db_scan.py`.
  - Work: a `ScanExit(IntEnum)` per tier. The epilogs and docstrings that restate 0/1/2/3 render from
    it or point to it. The 127-line docstring is cut to the tier API, with the history pointer per
    D12.
  - Signal: each adopter's `--help` epilog shows the exit table generated from the enum. A test
    asserts that every adopter's epilog equals the enum rendering and that no adopter spells "never
    treat 3 as a clean" by hand.
  - Consumer: operators reading `--help`; every Tier-1/2 adopter.
  - Deps: ζ.
- **θ — Split audit_project into per-direction passes over one coverage accumulator**
  - Size and keys: ~450 LOC. Key `fk-f70004144a3d`. Priority low.
  - Files: `scripts/audit_manifest_descriptor_drift.py`,
    `scripts/tests/test_audit_manifest_descriptor_drift.py`.
  - Work: the drift comparison, external-block classification and label sweep each return a
    coverage slice. One builder constructs `AuditCoverage` in both the success branch and the
    git-failure branch, so the hand-zeroed 18-field literal goes.
  - Signal: on the git-failure fixture the report carries every coverage category the success
    report carries (test), and on the tracked corpus the audit's output is byte-identical.
  - Consumer: operators; ι.
  - Deps: η.
- **ι — Route stamper and manifest audits through shared sidecar_for / tracked_sidecar_paths**
  - Status: held on S20.
  - Size and keys: ~500 LOC. Keys `fk-b16470a6e0d2`, `fk-bb8a584a8237`. Priority medium.
  - Files: `fused-memory/src/fused_memory/server/manifest_stamping.py`,
    `scripts/audit_manifest_descriptor_drift.py`, `scripts/audit_delivered_checks.py`,
    `scripts/audit_combine_gate_marker_loss.py`, `shared/src/shared/delivered_check_polarity.py`,
    `shared/tests/capability_manifest_corpus.py`, `fused-memory/tests/test_manifest_stamping.py`,
    `scripts/tests/test_audit_delivered_checks.py`, `scripts/tests/test_audit_manifest_descriptor_drift.py`,
    `scripts/tests/test_audit_combine_gate_marker_loss.py`.
  - Work:
    - C3 consumers.
    - `_manifest_binding` and `_contained_sidecar_relpath` call `sidecar_for`, and the
      "mirrors … exactly" prose goes.
    - Both audits discover through `tracked_sidecar_paths`. `audit_delivered_checks.AuditCoverage`
      gains a discovery-failed field and a non-clean exit.
    - `audit_combine_gate_marker_loss`'s glob table uses `SIDECAR_SUFFIX`.
    - The stamper calls `sidecar_for`.
  - Signal: B6 and B7. Also, `python3 scripts/audit_delivered_checks.py --project-root <non-git dir>`
    prints a discovery-failed coverage row and exits non-zero instead of reporting a clean sweep.
  - Consumer: `commit_planning`, operators, `/review-all`.
  - Deps: θ, 6037, S20 C3 leaf (held; see Pre-conditions).
- **κ — One _report_stats module; verify-budget policy out of the test tree**
  - Size and keys: ~600 LOC. Keys `fk-915ca17415b7`, `fk-30cd73379093`. Priority low.
  - Files: `scripts/_report_stats.py`, `scripts/verify_budget_policy.py` (name tactical),
    `scripts/verify_budget_census.py`, `scripts/merge_lane_throughput.py`,
    `scripts/mint_hard_v2_fixtures.py`, `scripts/review_model_admission.py`,
    `scripts/audit_model_admission.py`, `scripts/escalation_ladder.py`,
    `tests/scripts/module_budget_family.py`, `tests/scripts/test_module_verify_budgets.py`,
    `tests/scripts/test_verify_budget_census.py`, `scripts/tests/test_report_stats.py`.
  - Work: D9. `census_budget_floor` and the refusal ceiling are imported at module scope by both the
    census and the gate test. The runtime `tests/scripts` insert goes, and so does the false "COPIED,
    not imported" header.
  - Signal: `verify_budget_census`'s refusal ceiling is the gate's constant by import (no
    `REFUSAL_CEILING = 7200` restatement; delivered check). `--window 6h` is accepted by all three
    window-taking CLIs. An empty series renders `n/a` in `mint_hard_v2_fixtures`'s table.
  - Consumer: operators of the three report CLIs; the verify-budget gate test.
  - Deps: δ (`mint_hard_v2_fixtures.py`). Out of batch: S4's `attempt-*` leaf depends on κ.
- **λ — Emit routing rejections as structured {layer, reason, model} values**
  - Size and keys: ~700 LOC. Key `fk-7fee08736560`. Priority medium.
  - Files: `shared/src/shared/route_rejection.py`, `shared/tests/test_route_rejection.py`,
    `orchestrator/src/orchestrator/routing.py`, `orchestrator/src/orchestrator/routing_dispatch.py`,
    `orchestrator/src/orchestrator/workflow.py` (the one emit site), `scripts/audit_model_admission.py`,
    `scripts/review_model_admission.py`, `orchestrator/tests/test_routing_dispatch.py`,
    `scripts/tests/test_audit_model_admission.py`, `scripts/tests/test_review_model_admission.py`.
  - Work: C4, D7 and D8. `MODEL_REJECTION_REASONS`, `CEILING_EXHAUSTED_REASON` and the
    `rpartition` parse go. `DROPPED_PLAN_TARGETS_REASON_PREFIX` is imported from
    `orchestrator.merge_lane.gates`.
  - Signal: B8 and B9. After deploy, `audit_model_admission --model <m>` shows rejections attributed
    to the refused model on rows emitted after λ, and reports pre-λ rows as `legacy_rows`.
  - Consumer: admission milestone checks (the D6 pattern); S16.
  - Deps: κ (same two audit files).
- **μ — Per-file classifier plus one accumulator in sweep_toolcall_markup.run_sweep**
  - Size and keys: ~300 LOC. Key `fk-195f86e31fea`. Priority low.
  - Files: `scripts/sweep_toolcall_markup.py`, `scripts/tests/test_sweep_toolcall_markup.py`.
  - Work: a pure `classify_file -> FileVerdict(refusal_reason | None, repairs, stalls, diff)` and one
    accumulator that applies the exit-code routing rule once. The duplicated refusal-banking block goes.
  - Signal: on the existing fixture corpus `--check` exit codes and the summary are unchanged. A new
    test feeds a dirty refused document to each write gate and sees it banked once, in
    `skipped_with_markup`.
  - Consumer: the periodic CHECK run and operators.
  - Deps: δ.
- **ν — Stop ratcheting raw lines/prose_lines in the merge-lane ratchet**
  - Size and keys: ~250 LOC. Key `fk-f4af080fcd25`. Priority medium.
  - Files: `scripts/merge_lane_metrics.py`, `scripts/check_staged_ratchet_raise.py`,
    `orchestrator/tests/test_merge_lane_ratchet.py`, `orchestrator/tests/test_merge_lane_ratchet_commit_gate.py`,
    `orchestrator/tests/merge_lane_ratchet_baseline.json`.
  - Work: D10.
  - Signal: `merge_lane_metrics.py --report` still prints the lines and prose columns. A fixture
    whose only rise is `lines`/`prose_lines` passes the ratchet, and one with a cognitive or
    `function_local_imports` rise still fails.
  - Consumer: every merge-lane change, through `test_merge_lane_ratchet` and the commit gate.
  - Deps: γ (`merge_lane_metrics.py`, `check_staged_ratchet_raise.py`). Out of batch: 6422 depends
    on ν.
- **ξ — Sitting close-safety gates read pin_declared_by and duplicate_of only**
  - Status: held on S2.
  - Size and keys: ~300 LOC. Key `fk-9bb2996a620a`. Priority medium.
  - Files: `scripts/sitting/gates.py`, `scripts/sitting/prepare_sitting.py`,
    `scripts/member-chain-sweep.py`, `skills/escalation-watcher/SKILL.md`,
    `scripts/tests/test_sitting_gates.py`, `scripts/tests/test_sitting_preparation.py`,
    `scripts/tests/test_sitting_brief.py`, `scripts/tests/test_sitting_inventory.py`.
  - Work: D11. `VETO_PIN_PREFIX` and both `DEDUP_MARKER` copies go. The watcher skill's dedup step
    sets `duplicate_of`, and its pin step uses `declare_pin`.
  - Signal: B10. `prepare_sitting` lists `esc-3169-2` as a pin from `pin_declared_by`, and no code
    under `scripts/` matches the two literals (delivered check).
  - Consumer: the human sitting and the escalation-watcher.
  - Deps: ε (`sitting/prepare_sitting.py`), S2 C5 leaf (held), and the D11 migration.

DAG: S14 → γ → {δ, ν}; δ → {ε, ζ, κ, μ}; ε → ξ; ζ → η → θ → ι; κ → λ; α → β. ι and ξ are held.

### G7 walk (advisory at author time; re-run at decompose)

- `contracts-machine-checked`: C1 has B1 and B2, C2 has B3, C3 has B6, C4 has its round trip, and
  D5 rejects impossible rows at construction. Each contract has a test, not prose.
- `no-silent-fail-soft`: D2's raise and ι's discovery row remove two silent fallbacks. κ's
  `None`-on-empty removes a 0.0 that read as data.
- `readers-prove-their-producer`: λ's `legacy_rows` keeps "no structured rows yet" distinct from
  "none".
- `exceptions-owned-or-ratified`: the binding guard has no allow-list by construction (D3), and the
  bootstrap idiom adds no inline suppression.
- `one-fact-one-home` and `no-lockstep-duplication`: β, ι, κ, λ, η.
- `holds-owned-and-bounded`: ι and ξ are held with an owning stream and a program-level bound (see
  Pre-conditions).
- `structured-facts-at-failure`: ξ and λ.
- No hits on `storm-escape-required`, `loop-thread-occupancy-bounded` or `status-matches-liveness`:
  the stream adds no detector, coroutine or status write.

## Out of scope

- Line-pin cleanup and `ruff format` cleanup (CLAUDE.md standing rulings). Neither is done
  incidentally in any leaf.
- Path and root locators, and the hand-rolled MCP clients in scripts: these are S14's.
- The merge-lane new-file line ceiling (D10).
- `orchestrator-watchdog.py` structure (S24).
- `audit_delivered_checks`' tasks.db half and the remaining 6037 bullets.
- Making the dropped-plan-targets outcome structured. That is S6's `MergeOutcome.kind`; λ only
  removes the copy.
- Migrating archived pin-prefix records (D11).

## Open questions (tactical)

1. **Module names.** `scripts/_plan_scope.py` (ζ), `scripts/verify_budget_policy.py` (κ) and
   `shared/route_rejection.py` (λ) are proposed names. The implementer may rename within the same
   tier, keeping the one-home rule. Decide in ζ, κ and λ.
2. **Subdirectory scripts may already pass through E402.** If an existing subdir script needs its
   import order kept for E402, ε may restructure the header, but it adds no `noqa`. Decide in ε.
3. **`orchestrator.merge_lane.gates` might not import cheaply under `review_model_admission`'s
   runtime.** If so, λ may pin the copy with a parity test against the definition instead of
   importing it, recording why. Decide in λ.
4. **Held-leaf mechanics.** The default: ι and ξ stay `deferred` outside S25's `commit_planning`, and
   the S20 and S2 decompose sessions release them. If the lead prefers that those sessions file a
   thin "C3/C5 contract" leaf of their own that ι/ξ depend on, nothing in this PRD changes. Decide
   at Phase B with the lead.

# S14 — Shared leaves for cross-package facts (PRD)

**Status:** active — authored 2026-10-10, stream S14 of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, run `review-all-dark_factory-20261008`), agent
mode, wave 1. Approved as proposed under R21; no open design choice was carried into this session.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10); the finding records were verified at
`6551b75aba`. Main moves fast — anchors here are by symbol; re-locate at implementation time.

## Goal

Every fact that more than one package needs — where a project's stores live, what a project id is,
what a landing-merge subject looks like, what the scheduler snapshot contains, how the override store
is written, what a recon-stage caller is, what an MCP error code is, how a streak escalates once, and
how a script talks to fused-memory — has exactly one definition, in a module that every package imports
downward. A change to any of these facts is then a one-file change that every consumer sees, and a
second spelling cannot reappear without a test going red.

Heuristics this PRD acts on (`docs/code-quality.md`, cited by name, not restated): **11 SPOT**,
**12 structured data instead of meaningful strings**, **9 deep modules with narrow interfaces**,
**10 invariants uniformly enforced**, **3 orthogonal dimensions** (the streak key axis), **13 files make
internal sense in isolation** (every new shared module is a leaf with no reach-back).

## Consumers and user-observable surface (G1)

| Mechanism (new home) | In-batch consumers | Other-stream consumers (program G4 table) | User-observable surface |
|---|---|---|---|
| `shared/project_layout.py` | β γ ε κ η θ | S12, S13, S17, S19, S22, S24, S25 | an operator who relocates a store edits one function; `EscalationConfig.queue_dir` set to a non-layout value is refused at config load with a message naming the layout |
| `shared/project_id.py` | β γ ε | S19 (session package carries the fold), S22 | a new project's `fused_memory.project_id` is derived from its root when the yaml omits it; `df`, `dark-factory`, `Dark_Factory` fold to one token in cockpit, fused-memory and scripts alike |
| `shared/merge_subject.py` | δ ε | S6 (merge lane reads landings), S13 (recon found-on-main evidence) | the already-landed gate, recon's landed-on-main probe, the eval miners and the warm-lane bash library recognise exactly the subject `git_ops` writes, for any configured `branch_prefix`/`main_branch` |
| `shared/scheduler_state.py::SchedulerStateSnapshot` | η | S17 (typed test seam), S22 (dashboard scheduler panel) | `get_scheduler_state` returns every key the producer writes, including on a cold start with no file, and says "no snapshot written yet" rather than an empty state |
| `shared/scheduler_overrides.py` (+ `shared/task_priority.py`) | θ κ | S17 | a pin set through the fused-memory MCP tool and one set by the orchestrator go through the same code; `tools.py`'s "intentional duplicate" block is gone |
| `shared/agent_identity.py::AgentIdentity` | μ | S12 (stage identity from `StageId`), S15 (dispatch wrapper parses identity once), S2 (`parse_caller_identity` for headers is S2's and stays separate) | a `reconciliation-stage-*` caller is held to the same recon-stage gates as `recon-stage-*`, and the transition authority agrees with every gate |
| `shared.mcp_envelope.McpErrorType` | ξ κ ι | S22 | renaming a fused-memory exception class leaves the wire code, and the dashboard's terminal-window fallback, `chronic_flake`'s not-found path and the repair script's absent-task verdict, unchanged |
| `shared/mcp_session.py::McpSession` + `shared.mcp_envelope.write_failure_reason` | ζ ι κ | S21 (census_trigger transport), S22 (mcp_fanout transport), S25 | every fused-memory MCP caller (orchestrator, dashboard, scripts) uses one handshake/decode/retry implementation; an `isError` reply or an undecodable reply is a typed failure, never `{}` |
| `shared/streak.py` consecutive-streak primitive | ο π | S12 (journal-recomputed streaks as declarations), S24 (watchdog streak/cap) | an INV-4 "N bad events then escalate once" path fires exactly once per run of failures and re-arms on success, the same way in every package |
| `fused_memory.memory_metadata.child_parent_id` | ν | S19 (moves grouped_read assembly) | a child-shaped memory is folded the same way by search anchoring and grouped reads |
| spelling guard | — (integration gate ρ) | every stream | `pytest` fails naming file and literal when a second spelling of any family above is added |

## Background

Synthesis stream S14 of `plans/review-all-dark_factory-20261008.md` (`## Candidate streams`). Its
remedy, deletable list and seam candidates are the record's `synthesis.streams[S14]`; the program doc's
G4 table makes S14 the owner of every leaf above. The critic's join error on the override store
("would reintroduce the fm→orchestrator edge S10 deletes") is resolved by this PRD's shared-leaf design:
fused-memory never imports orchestrator.

## Freshness re-verification (G3/G6, at `ae3047ec3a`)

722 commits landed between `6551b75aba` and `ae3047ec3a`. Every anchor was re-read:

| Key | Status at `ae3047ec3a` |
|---|---|
| fk-06de98fa328b (weakened) | **Dropped — cost gone.** `d6b1626681` (task 4618 step 2) added `shared.task_claimant.DEFAULT_CLAIMANT_HEARTBEAT_TTL`; the harness, `task_ground_truth`, dashboard and fused-memory constants and the false "no config field" comment are gone and every 600 s reader imports it. The 300 s `claimant_liveness_ttl_secs` is a deliberately distinct dispatch knob documented beside the shared constant, and `clear_stale_plan_lock`'s 600 s is documented as plan.lock's own window. The program's "claimant TTL constant" seam is therefore already landed; consumers import `DEFAULT_CLAIMANT_HEARTBEAT_TTL`. |
| fk-3644c6841b3e | present: `git_ops._merge_subject`, `landed_on_main._merge_subject` mirror, `DEFAULT_COMMIT_CITATION_PATTERN`, `ALREADY_LANDED_CITATION_PATTERN`, `task_sampler._MERGE_SUBJECT_RE`, `mining` `--grep`, `mint_hard_v2_fixtures._merge_subject_patterns`, `replay_deep_stack_study`, `lib_task_citation.sh` |
| fk-7a85d17a7c7e | present: 18 files spell `data/escalations`; 20 runs.db constructions; 12 tasks.db constructions; `EscalationConfig.queue_dir` honoured by the orchestrator only |
| fk-e367d6a83223 | present: `_to_underscore_canonical`, `sqlite_task_backend` inline fold, `memory_recall._canonical_project`, `session_registry.normalize_project_token` (+ `PROJECT_TOKEN_ALIASES`), two script resolvers; `FusedMemoryConfig.project_id` default `'dark_factory'` |
| fk-1b25c0b5d3a2 | present: `tool_errors` and ~15 sites render `type(e).__name__`; dashboard `tasks.py`, `chronic_flake`, `repair_wiped_metadata_files`, `flag_dedup` compare class-name literals |
| fk-43a75be2cd47 | present: `StreakRegistry` consumed by `scheduler.py` only; scheduler's two raw streaks; the hand-rolled attributes and private `_Streak`/`_DegradedStreak`/`_ShardFailureStreak`/`ConsecutiveFailureBreaker` classes all present |
| fk-f6ab6bfad0be | present: 13 produced keys vs a 10-key `_empty_skeleton`; path built in `scheduler.py` and `mcp_tools/scheduler_state.py` |
| fk-37f2baab3aec, fk-b84e0aa4e545, fk-f5e16692e170 | present: `tools.py` `_OVERRIDE_SCHEMA`, `_PRIORITY_TIERS`, `_VALID_CLEAR_FIELDS`, the four aiosqlite tool bodies and park-eviction copy, with line-pinned "Mirrors overrides.py:<n>" comments |
| fk-7aff55dd1465, fk-75fa713ba3d5 | present: 23 `startswith('recon-stage-')` sites (12 in `tools.py`); `derive_actor_class` also accepts `reconciliation-stage`; `internal_writers`, `standing_decision_writer`, `flag_record_contract`, `stats_verifier` hold their own prefix tables |
| fk-f01bf7bc23f9 | present: `memory_metadata.CHILD_KINDS`, `grouped_read.CHILD_KINDS` (+ its own kind literals, imported by 7 modules), `topic_anchor._CHILD_KINDS` and `_is_child_shaped` |
| fk-954f6787b07c | present: scripts' runs.db spellings, 13 scripts assigning the home-path default root, 4 `DASHBOARD_KNOWN_PROJECT_ROOTS` parsers, `AuditCoverage` without an absent-store field |
| fk-9cdaf62ed9af | present (no diff since pin): dashboard `memory.py` `McpSession`, `_parse_sse_response`, `_extract_tool_result` |
| fk-ccd26ad88810 | present (no diff since pin): `migrate_metadata_modules_to_files.FusedMemoryClient`/`write_failure_reason`; copies in `trial_module_tagger_haiku`, `strip_leaked_control_keys`, `cgl_eta_scheduler_gate` |

Dedup protocol step 1 (forensic `json_each(metadata,'$.x_finding_key')` over the live store) found no
task carrying any of the 16 keys; step 2 (`search_tasks` ≥ 0.55) found only done historical tasks and
pending **4709** (write-through runs.db events from the MCP override tools), which edits the same tool
bodies θ replaces — see §Decomposition, θ.

## Approach — B + H

G5 fires on every count: four packages plus scripts (blast radius ≥ 3), eleven mechanisms, a
load-bearing seam (two processes writing one SQLite file; the task-write authority's actor class), and
seven consumer streams. The contract below is what each consumer stream builds against; the boundary
tests are the integration gate's signal.

## Resolved design decisions

1. **Leaves are stdlib-only where bash or an un-synced interpreter reads them.** `project_layout`,
   `project_id`, `merge_subject`, `streak` and `agent_identity` import only the stdlib and `shared`
   siblings, so `python3 -c 'from shared.project_layout import …'` works from bash (S24's precedent).
   `scheduler_state` and `scheduler_overrides` may use pydantic/sqlite helpers already in `shared`.
2. **The escalation queue location is a layout fact, not a per-project knob.** No package except the
   orchestrator can read `EscalationConfig.queue_dir`, so a non-layout value would split the queue.
   The field stays (ten yamls set it, all to the layout value) with its default derived from
   `project_layout`, and a validator refuses any other value with an INV-2 message naming
   `shared.project_layout.escalation_queue_dir`.
3. **`FusedMemoryConfig.project_id` defaults to `project_id_for_root(project_root)`; the yaml value
   wins.** All 18 live yamls declare it (the `reify-fix*` clones must keep `reify`, which basename
   derivation would get wrong), so the change affects new projects only.
4. **One project-id rule plus one explicit fold.** `normalize_project_id` (lower-case, `-`→`_`) is the
   canonical rule fused-memory has used; `fold_project_token` (casefold, collapse `_` runs, strip edges,
   then `PROJECT_ID_ALIASES`) is the cockpit's tolerant fold, defined in terms of `normalize_project_id`
   in the same module. The alias table moves with it. fused-memory's path-shape guard
   (`canonicalize_project_id` raising `PathShapedProjectIdError`) stays fused-memory's boundary
   validation and calls the shared rule.
5. **The default project root is derived, never a literal.** `default_project_root()` is the MAIN
   checkout of the repository `shared` was imported from (worktree-aware: follows a worktree's
   `.git` file to the common dir), because `.taskmaster/` exists only in the main checkout.
   `known_project_roots()` is the one `DASHBOARD_KNOWN_PROJECT_ROOTS` parser.
6. **The override store moves whole, not just its schema.** `shared/scheduler_overrides.py` holds
   `OverrideStore`, `ParkEvictionRequestStore`, both schemas, `OverrideRow`, `PinOrderCollision`,
   `VALID_CLEAR_FIELDS` and the pin-order allocation; it is synchronous sqlite3 with the stricter of the
   two current connection setups (full durability pragmas, `BEGIN IMMEDIATE`, 30 s busy timeout).
   fused-memory calls it through `asyncio.to_thread`; `orchestrator/overrides.py` and
   `orchestrator/park_eviction_requests.py` are deleted and importers repointed (no shim). Moving only
   schema + SQL text would leave two transaction implementations of one store, the duplication the
   three keys name.
7. **Priority tiers get their own leaf.** `PRIORITY_TIERS`/`PRIORITY_RANK` are the task-priority
   vocabulary, not an override detail (heuristic 1), so they live in `shared/task_priority.py`, which
   `scheduler_overrides`, `orchestrator.config`, `pin_reservation`, `scheduler`, fused-memory tools and
   the dashboard boost validator import.
8. **One recon-stage grammar.** `parse_agent_identity` recognises `recon-stage-<stage>` and the
   `reconciliation-stage-<stage>` spelling (CLAUDE.md's write-tagging example, already accepted by
   `derive_actor_class` and `internal_writers`) as the same `RECON_STAGE` kind. Every gate then applies
   to both spellings — the stricter reading, since ids are self-reported. `recon_stage_agent_id(stage)`
   renders only the `recon-stage-` form. `ActorClass` stays in `shared.task_transitions`;
   `derive_actor_class` is deleted and callers read `parse_agent_identity(x).actor_class`.
   Operator-configured prefix allowlists (`deterministic_provenance_allowed_agent_prefixes` and the
   entity-mint / mem0-update allowlists) stay prefix lists by design, evaluated through
   `AgentIdentity.matches_prefix`.
9. **Wire error codes are declared, not derived from class names.** `McpErrorType` (StrEnum) in
   `shared.mcp_envelope` keeps today's wire strings as values (`'TaskNotFoundError'`,
   `'TaskmasterError'`, …), so nothing on the wire changes. Each fused-memory exception that crosses
   the wire declares `error_type: ClassVar[McpErrorType]`; one `error_envelope(exc)` renders every MCP
   error reply; an undeclared exception keeps `type(exc).__name__` (an open set nobody may route on).
10. **One MCP session client.** The orchestrator's `McpSession` (handshake, session id, retry,
    idempotency injection, SSE decode) moves to `shared/mcp_session.py` with the retry schedule and
    `clientInfo.name` as constructor parameters, since `clientInfo.name` is the identity fused-memory
    records. `parse_tool_result` gains `EnvelopeShape.IS_ERROR` (a `result.isError` reply is never a
    value) and prefers `structuredContent`; the scripts' `isError`/`_raw` reserved-key stamping is
    replaced by those typed shapes. `write_failure_reason(reply)` moves to `shared.mcp_envelope` and
    fused-memory's `interceptor_write_succeeded` becomes `write_failure_reason(resp) is None`.
11. **One consecutive-streak primitive keyed by any hashable.** `KeyedStreaks[K]` with explicit
    threshold, latch-once and reset-on-success, plus `forget(keys)`/`retain(pred)` for GC;
    `StreakCounter`/`StreakRegistry` become its task-id specialisation. Plain backoff counters
    (`background_service`, `offline_lane`) are not escalate-once triggers and are left alone.
12. **The snapshot is parsed at both ends.** The producer builds a `SchedulerStateSnapshot` and
    serialises it; readers parse into it. `snapshot_at is None` is the one way to say "no producer has
    written a snapshot" (INV-13); `SchedulerStateSnapshot.empty()` replaces the 10-key skeleton. Unknown
    keys from a newer producer are ignored on read (rolling restarts), so a new key is added in one
    place: the model.
13. **A spelling guard is the uniform enforcement (heuristic 10).** ρ adds one test that scans string
    constants (docstrings excluded) of every tracked non-test `.py`/`.sh` for the families above
    outside their home module and fails naming file and literal. Its allow-list starts empty except
    entries owned by a live task in another stream (INV-12 `Debt(owner)`).

## Contract

Signatures are the contract; bodies are the implementer's. Every function is total over its stated
domain and raises only where stated.

```python
# shared/project_layout.py — stdlib only
ESCALATION_QUEUE_RELPATH: str            # 'data/escalations'
def escalation_queue_dir(root: PathLike) -> Path
def orchestrator_data_dir(root: PathLike) -> Path         # <root>/data/orchestrator
def runs_db(root: PathLike) -> Path
def scheduler_state_json(root: PathLike) -> Path
def scheduler_overrides_db(root: PathLike) -> Path
def park_eviction_requests_db(root: PathLike) -> Path
def tasks_db(root: PathLike) -> Path                       # <root>/.taskmaster/tasks/tasks.db
def default_project_root() -> Path                         # main checkout of the repo `shared` came from
def known_project_roots(environ: Mapping[str, str] = os.environ) -> tuple[Path, ...]

# shared/project_id.py — stdlib only
PROJECT_ID_ALIASES: Mapping[str, str]                      # moved from session_registry.PROJECT_TOKEN_ALIASES
def normalize_project_id(raw: str) -> str                  # lower(), '-' -> '_'; idempotent
def project_id_for_root(root: PathLike) -> str             # normalize_project_id(basename)
def fold_project_token(value: object) -> str               # '' for empty/None; idempotent; aliases applied

# shared/merge_subject.py — stdlib only
@dataclass(frozen=True)
class MergeSubject: branch: str; main_branch: str
def format_merge_subject(branch: str, main_branch: str) -> str
def parse_merge_subject(subject: str) -> MergeSubject | None      # anchored; None for body quotes, Revert "…", suffixes
def merge_subject_pattern(*, branch_prefix: str, task_id: str | None = None,
                          main_branch: str | None = None, legacy_colon_form: bool = False) -> str
    # one pattern valid as both git --extended-regexp and Python re; legacy_colon_form adds the historical
    # 'Merge <branch>: <subject>' reader spelling (reify history) for miners only
def rendered_prefix(branch_prefix: str) -> tuple[str, str]       # (prefix, suffix) bash takes as arguments

# shared/scheduler_state.py
class SchedulerStateSnapshot(BaseModel):     # frozen; extra='ignore' on parse
    skip_counts, parks, park_stacks, pin_reservations, effective_priorities, pin_queue, overrides,
    current_holders, lock_depth, is_paused, pause_reason, requeue_cooldowns, snapshot_at: datetime | None
    @classmethod empty() -> SchedulerStateSnapshot          # snapshot_at None
    @classmethod from_json(data: Mapping) -> SchedulerStateSnapshot
    def to_json(self) -> dict

# shared/task_priority.py
PRIORITY_TIERS: tuple[str, ...]; PRIORITY_RANK: Mapping[str, int]

# shared/scheduler_overrides.py
class PinOrderCollision(Exception); class OverrideRow (frozen)
VALID_CLEAR_FIELDS: frozenset[str]
class OverrideStore:             # __init__(db_path: Path); set_override, clear_override, reorder_pin_queue,
                                 # clear_terminal, clear_expired, get_overrides, get_pin_queue, checkpoint
class ParkEvictionRequestStore:  # __init__(db_path: Path); enqueue, drain

# shared/agent_identity.py — stdlib only (+ shared.task_transitions.ActorClass)
class AgentKind(StrEnum): RECON_STAGE, ORCHESTRATOR, DETERMINISTIC, ESCALATION, TASK_AGENT, INTERACTIVE, OTHER, ABSENT
@dataclass(frozen=True)
class AgentIdentity:
    raw: str | None; kind: AgentKind; recon_stage: str | None
    @property actor_class -> ActorClass                    # today's derive_actor_class table, unchanged per kind
    @property is_recon_stage -> bool
    def matches_prefix(self, prefixes: Iterable[str]) -> bool
def parse_agent_identity(agent_id: object) -> AgentIdentity   # never raises
def recon_stage_agent_id(stage: str) -> str

# shared/mcp_envelope.py (additions)
class McpErrorType(StrEnum): TASK_NOT_FOUND = 'TaskNotFoundError'; TASKMASTER = 'TaskmasterError'; …
EnvelopeShape.IS_ERROR
def write_failure_reason(reply: object) -> str | None       # None iff a successful write reply

# shared/mcp_session.py
class McpSession:
    def __init__(self, base_url: str, *, client_name: str, retry_backoffs: Sequence[float] = ()) -> None
    async def initialize(self) -> None
    async def call_tool(self, name: str, arguments: Mapping, *, timeout: float = 30) -> dict   # JSON-RPC body
    async def call(self, method: str, params: Mapping | None = None, *, timeout: float = 30) -> dict

# shared/streak.py — stdlib only
@dataclass(frozen=True)
class StreakStep: count: int; fired: bool                   # fired True exactly once per run reaching threshold
class KeyedStreaks(Generic[K]):
    def __init__(self, *, threshold: int) -> None
    def failure(self, key: K) -> StreakStep
    def success(self, key: K) -> None                       # resets count and latch
    def count(self, key: K) -> int
    def forget(self, keys: Iterable[K]) -> None
    def retain(self, keep: Callable[[K], bool]) -> None
    @property tracked_keys(self) -> frozenset[K]     # keys in a live failure run (snapshot)
```

Invariants every implementation and consumer keeps:

- **I1** `parse_merge_subject(format_merge_subject(b, m)) == MergeSubject(b, m)` for every branch with
  no whitespace; `merge_subject_pattern` compiled in Python and run by `git log -E --grep` selects the
  same subjects.
- **I2** `normalize_project_id` and `fold_project_token` are idempotent; for every id `x` already in
  normal form with no alias, `fold_project_token(x) == x`.
- **I3** No package other than `shared.scheduler_overrides` issues SQL against `scheduler_overrides.db`
  or `park_eviction_requests.db`.
- **I4** `SchedulerStateSnapshot.from_json(s.to_json()) == s`; the producer cannot emit a key the model
  lacks.
- **I5** `parse_agent_identity(x).actor_class` equals today's `derive_actor_class(x)` for every input,
  except that `reconciliation-stage*` inputs and `recon-stage-*` inputs now agree on `is_recon_stage`.
- **I6** For every wire-crossing exception class, the rendered `error_type` equals its declared
  `McpErrorType`, independent of the class's Python name.
- **I7** `KeyedStreaks.failure` returns `fired=True` exactly once between two `success` calls on the
  same key, at the call where `count == threshold`.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Merge subject round trip, writer → every reader | `git_ops` writes a merge with `branch_prefix='feat/'`, `main_branch='trunk'` in a temp repo | `find_merge_marker`, `ALREADY_LANDED_CITATION_PATTERN`'s merge arm, `landed_on_main`, `task_sampler`, `mining`, `lib_task_citation.sh` (fed `rendered_prefix`) all identify the task; a `Revert "Merge …"` subject is recognised by none |
| B2 | Layout parity across packages | a temp project root | orchestrator config's runs.db / overrides / park / scheduler-state / escalation paths, fused-memory's escalators and recon paths, dashboard config paths and `_task_db_scan` all resolve to the `project_layout` function's value |
| B3 | Non-layout queue_dir refused | yaml with `escalation.queue_dir: data/esc2` | config load raises naming `shared.project_layout.escalation_queue_dir` and both values |
| B4 | Project id agreement | inputs `Know-Live`, `dark--factory`, `df`, `Dark_Factory`, `-home-leo-x` | fused-memory, orchestrator memory recall, cockpit fold and scripts agree per I2; the path-shaped input is rejected by fused-memory's guard |
| B5 | Override store, two writers | orchestrator `OverrideStore` and the fused-memory MCP tool on one temp file | a pin from each side gets distinct `pin_order`; a colliding explicit order raises `PinOrderCollision` on the orchestrator side and returns the tool's existing error dict on the fused-memory side; `get_pin_queue` agrees from both |
| B6 | Snapshot cold start | no `scheduler_state.json` | `get_scheduler_state` returns all 13 keys with `snapshot_at` null; after one scheduler write it returns the producer's values, including `pin_reservations`, `requeue_cooldowns`, `lock_depth` |
| B7 | Identity agreement | `recon-stage-memory_consolidator`, `reconciliation-stage-1`, `orchestrator-deterministic-x`, `claude-task-9`, `None` | every recon gate, `internal_writers`, `stats_verifier` and the transition authority classify each identically per I5 |
| B8 | Error-code rename resilience | test subclass of `TaskNotFoundError` with a different class name raised by a tool | the reply's `error_type` is `McpErrorType.TASK_NOT_FOUND`; dashboard, `chronic_flake` and the repair script take their not-found branch |
| B9 | MCP failure shapes | stub server answering `isError`, non-JSON text, a 202 empty body, a 307 | `McpSession` + `parse_tool_result` + `write_failure_reason` report a typed failure for each; none yields `{}` treated as success |
| B10 | Streak latch | threshold 3; failures ×5, success, failures ×3 | `fired` true at failure 3 and at the second run's failure 3 only |
| B11 | Guard red on a seeded literal | a fixture module spelling `'data/escalations'`, `'Merge task/'`, `startswith('recon-stage-')`, `== 'TaskNotFoundError'`, `'protocolVersion'` | the guard names each file and literal; the clean tree passes |

## Cross-PRD relationship (G4)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S12 recon run lifecycle | consumes | `shared/streak.py` (λ); `agent_identity.recon_stage_agent_id` (μ); `project_layout` (α) | S14 | S12 tasks depend on λ, μ, α |
| S13 recon sweeps / flag identity | consumes | `project_layout` (α); `merge_subject` for found-on-main evidence (α) | S14 | depends on α |
| S17 scheduler seams | consumes | `SchedulerStateSnapshot` (η); `scheduler_overrides` (θ) | S14 | depends on η, θ |
| S19 package layering | consumes | `project_id.fold_project_token` + alias table (α) for the shared session package; `memory_metadata.child_parent_id` (ν) — S19's "second CHILD_KINDS" deletable is delivered by ν | S14 | depends on α, ν |
| S21 legibility | consumes | `McpSession` (ζ) for census_trigger's transport — S21 migrates census_trigger; ρ's guard carries one `Debt(owner=<S21 leaf>)` entry for it until then | S14 owns client; S21 owns migration | ρ allow-list entry, id wired at decompose |
| S22 dashboard readers | consumes | `project_layout` (α); `McpSession` (ζ); `McpErrorType` (ξ) | S14 | κ does the dashboard adoption of S14's leaves; S22 builds on κ |
| S24 fleet watchdog | consumes | `project_layout`, `streak` | S14 | depends on α, λ; S24 owns the fleet-coordination leaf (lease/clock), not S14 |
| S25 scripts instruments | **produces for ε** and consumes | S25 owns `scripts/_checkout.py` (fk-638eb09e52b1); ε depends on that leaf. S14 owns `project_layout`; ε (not S25) migrates the scripts' runs.db/tasks.db/default-root/known-roots spellings — S25's deletable "7+ runs.db spellings and 16 hard-coded roots" is delivered by ε and S25 files no task for it | S14 (locators), S25 (bootstrap) | ε ← S25 bootstrap leaf, id wired at decompose |
| S10 task-write contract | adjacent | `write_failure_reason` is the MCP reply predicate (ζ); S10's `MetadataWriteFailed` is the scheduler's raise path and does not redefine it | S14 | no edge |
| S15 fm MCP server boundary | consumes | `AgentIdentity` (μ) and `error_envelope` (ξ) parsed once by S15's dispatch wrapper | S14 | S15 depends on μ, ξ |
| S1/S2/S3 escalation | consumes | `project_layout.escalation_queue_dir` (α) | S14 | S1 rewrites the escalators' filing on top of γ |
| S11 curator pipeline | same-file only | π edits `task_curator.py` streak state; S11 edits its candidate types | — | merge-lane rebase; no seam |
| task 4709 (pending) | consumes | the MCP override tool bodies θ rewrites | θ | decompose wires 4709 → θ |

## Decomposition plan

Sizing per `.claude/skills/prd/project.md`. Every leaf: `planning_mode=True`, `metadata.x_finding_key`
(list), `metadata.x_finding_run = "review-all-dark_factory-20261008"`, `metadata.source = "review-all"`,
file-level `metadata.files`, `priority` = highest severity among its keys. Same-file leaves are
serialized by real dependencies (noted per row). Heuristic 13 and 14's no-cheating rule bind every
leaf that creates or deletes a module: each new shared module is a leaf importing nothing above it, and
a deleted module's importers are repointed, never shimmed.

- **α — Shared layout, project-id and merge-subject leaves.** New `shared/project_layout.py`,
  `shared/project_id.py`, `shared/merge_subject.py` + tests (I1, I2, B4's shared half, worktree-aware
  `default_project_root`). ~500 LOC, 6 files. Prereqs: —. Intermediate: unlocks β γ δ ε η θ κ and
  S12/S13/S19/S22/S24/S25. Keys: fk-7a85d17a7c7e, fk-e367d6a83223, fk-3644c6841b3e, fk-954f6787b07c.
  Priority medium.
- **β — Orchestrator adopts layout and project id.** `config.py` (queue_dir default + refusal per
  decision 2; `FusedMemoryConfig.project_id` derivation; override/park path methods via layout),
  `harness.py`, `flake_ledger.py`, `sandbox_soak.py`, `b3_gate.py`, `cli.py`, `config_census_ignore.py`,
  `mcp/markup_sink.py`, `mcp/plan_tools.py`, `resume_age_bound.py`, `evals/prompt_opt/__main__.py`,
  `evals/reviewer_trial/__main__.py`, `agents/memory_recall.py`, `session_registry.py`
  (`normalize_project_token` and `PROJECT_TOKEN_ALIASES` become imports of the shared fold). ~350 LOC,
  14 files. Prereqs: α. Signal: B3 (config load refuses a non-layout queue_dir with the named message)
  and B2's orchestrator rows. Keys: fk-7a85d17a7c7e, fk-e367d6a83223. Medium.
- **γ — fused-memory adopts layout and project id.** Middleware escalators (`_folded_escalation`,
  `curator_escalator`, `dead_letter_escalator`, `entity_mint_storm_escalator`,
  `mem0_update_storm_escalator`, `scope_violation_escalator`, `ticket_janitor`), recon `backlog_policy`,
  `bulk_reset_guard`, `harness`, `targeted`, `backends/sqlite_task_backend.py` (tasks.db + project-id
  fold), `utils/target_store_preflight.py`, `utils/validation.py` (rule from shared, guard kept),
  `models/scope.py` (`resolve_project_id`, `known_project_roots_from_env` delegate),
  `fused-memory/scripts/audit_unverified_completion_claims.py`. ~350 LOC, 16 files (above the review
  trigger; one mechanical replacement of one spelling family per file, coherent as one leaf). Prereqs: α.
  Signal: B2's fused-memory rows — an escalation filed by the curator escalator in a temp project lands
  in `escalation_queue_dir(root)`. Keys: fk-7a85d17a7c7e, fk-e367d6a83223. Medium.
- **δ — Merge-subject readers derive from the shared pair.** `git_ops.py` (`_merge_subject` and
  `_merge_marker_pattern` call shared; the merge arm of `DEFAULT_COMMIT_CITATION_PATTERN` rendered from
  `merge_subject_pattern`), `merge_lane/gates.py` (`ALREADY_LANDED_CITATION_PATTERN`'s merge arm),
  `fused-memory/.../services/landed_on_main.py` (mirror deleted), `evals/task_sampler.py`,
  `evals/reviewer_trial/mining.py`, `orchestrator/scripts/warm-lane/lib_task_citation.sh` (takes the
  rendered prefix/suffix as arguments; its caller renders them via `python3 -c` from `rendered_prefix`;
  no `Merge`/`into` literal remains in bash). ~350 LOC, 7 files + tests. Prereqs: α. Signal: B1 in a
  temp repo with a non-default prefix and main branch. Keys: fk-3644c6841b3e. Medium.
- **ε — Scripts adopt the shared store locators.** `scripts/_task_db_scan.py` (`tasks_db_path`,
  `runs_db_path`, default/known roots delegate to `project_layout`), `audit_wiped_metadata_files.py`
  (plus an `AuditCoverage` field recording an absent runs.db, rendered in the report — INV-13),
  `audit_model_admission.py`, `merge-pytest-n-ab-analysis.py`, `merge_lane_throughput.py`,
  `mint_hard_v2_fixtures.py` (both subject spellings from `merge_subject_pattern(…, legacy_colon_form=True)`),
  `replay_deep_stack_study.py`, `sitting/fleet_state.py`, `sitting/inventory.py`,
  `suite_census_evidence.py`, `census_tagger_debris.py`, `audit_combine_gate_marker_loss.py`,
  `member-chain-sweep.py`, `sweep_toolcall_markup.py`, and every remaining script assigning the
  home-path default root. ~500 LOC, ~16 files. Prereqs: α; **S25's `scripts/_checkout.py` bootstrap
  leaf** (cross-stream; scripts must bind this checkout's `shared` before importing it). Signal:
  `audit_wiped_metadata_files` run against a temp project with no runs.db prints an absent-event-log
  coverage row instead of reporting every task `NO_MERGE_EVENT`; B2's scripts rows. Keys:
  fk-954f6787b07c, fk-7a85d17a7c7e, fk-e367d6a83223, fk-3644c6841b3e. Medium.
- **ζ — One MCP session client and write-success predicate.** New `shared/mcp_session.py` (the
  orchestrator's `McpSession` moved, retry and `client_name` injected), `shared/mcp_envelope.py`
  (`EnvelopeShape.IS_ERROR`, `structuredContent` preference, `write_failure_reason`),
  `orchestrator/mcp_lifecycle.py` (imports the shared class; passes `fm_retry_backoffs` and
  `ORCHESTRATOR_MCP_IDENTITY`), `fused-memory/.../middleware/task_interceptor.py`
  (`interceptor_write_succeeded` delegates) + tests (B9). ~600 LOC, 5 files + tests. Prereqs: —.
  Intermediate + signal: the orchestrator's fused-memory writes (completion memory, status writes) still
  land end-to-end through the shared class (existing mcp_lifecycle suites move to the shared tests).
  Unlocks ι κ ξ and S21/S22/S25. Keys: fk-9cdaf62ed9af, fk-ccd26ad88810. Medium.
- **η — Typed scheduler state snapshot.** New `shared/scheduler_state.py`; `orchestrator/scheduler.py`
  (`get_state_snapshot` builds the model, path from layout); `fused-memory/.../mcp_tools/scheduler_state.py`
  (skeleton deleted, parse via model, its runs.db join via layout); `services/live_workflow_detector.py`
  and `dashboard/.../data/scheduler.py` (typed reads); `server/tools.py` (`get_scheduler_state`
  returns `to_json()`). ~450 LOC, 6 files + tests. Prereqs: α. Signal: B6 through the
  `get_scheduler_state` MCP tool. Keys: fk-f6ab6bfad0be. Medium.
- **θ — One scheduler-override store in shared.** New `shared/scheduler_overrides.py` and
  `shared/task_priority.py`; delete `orchestrator/overrides.py` and `orchestrator/park_eviction_requests.py`;
  repoint `config.py`, `pin_reservation.py`, `scheduler.py`, `harness.py`, `cli.py` and the orchestrator
  tests that import them; `fused-memory/.../server/tools.py` (the override/park constants, aiosqlite
  openers, checkpoint helper and four tool bodies call the shared store via `asyncio.to_thread`, response
  shapes and audit writes unchanged); `reconciliation/stale_priority_override_edge_sweep.py` and
  `fused-memory/scripts/cleanup_pin_queue_edges.py` read through the store (I3). ~1,200 changed LOC
  (net negative), ~10 src files + ~14 test files whose only change is the import line (above the review
  trigger; the test edits are mechanical). Prereqs: α, β (`config.py`, `harness.py`, `cli.py`), η
  (`scheduler.py`, `tools.py`). Signal: B5 through the `set_task_priority_override`/`get_pin_queue` MCP
  tools against a running orchestrator store. Keys: fk-37f2baab3aec, fk-b84e0aa4e545, fk-f5e16692e170.
  Medium. Decompose also wires pending task **4709** to depend on θ.
- **ξ — Declared MCP error codes.** `shared/mcp_envelope.py` (`McpErrorType`),
  `backends/task_backend_errors.py` (class-level codes), `server/tool_errors.py` (`error_envelope`),
  the envelope-building sites in `server/tools.py`, `server/main.py`, `middleware/task_interceptor.py`,
  `reconciliation/flag_dedup.py`, `services/consolidation_ops.py`, `server/grouped_read.py`,
  `server/topic_cluster_store.py`; consumer `orchestrator/chronic_flake.py`. ~300 LOC, 11 files.
  Prereqs: ζ (`mcp_envelope.py`, `task_interceptor.py`), θ (`tools.py`). Signal: B8 through the
  `get_task` MCP tool for a missing id. Keys: fk-1b25c0b5d3a2. Low.
- **ν — The memory child rule in one home.** `memory_metadata.py` gains `AMENDMENT_KIND`,
  `SIGHTING_KIND`, `PARENT_ID_KEY` and `child_parent_id(meta)`; `server/grouped_read.py` and
  `services/topic_anchor.py` call it (the restated set, its cycle-avoidance comment and the parity test
  go); the seven importers of grouped_read's kind names repoint (`maintenance/link_heal.py`,
  `maintenance/link_heal_store.py`, `reconciliation/prompts/stage1.py`,
  `reconciliation/stages/memory_consolidator.py`, `server/tools.py`, `server/write_triage.py`,
  `server/write_triage_judge.py`). ~150 LOC, 10 files. Prereqs: ξ (`grouped_read.py`, `tools.py`).
  Signal: a `search` whose window holds a child-shaped record anchors and folds it identically through
  topic anchoring and grouped read (one fixture exercised through both MCP paths). Keys:
  fk-f01bf7bc23f9. Medium.
- **μ — Parsed caller identity.** New `shared/agent_identity.py`; `shared/task_transitions.py`
  (`derive_actor_class` deleted); `server/tools.py` (`_resolve_identity` returns `AgentIdentity`; the
  12 prefix sites read it), `middleware/task_interceptor.py`, `execution_class_guard.py`,
  `premise_lint_guard.py`, `recurring_gate_guard.py`, `reconciliation/stale_gate_citation_guard.py`,
  `stats_verifier.py`, `flag_record_contract.py`, `internal_writers.py`, `standing_decision_writer.py`,
  `targeted.py`; config-driven allowlists (`done_provenance_authz`, `entity_mint_authz`,
  `mem0_update_authz`) evaluate through `matches_prefix`. ~600 LOC, ~15 files. Prereqs: ν (`tools.py`),
  γ (`targeted.py`). Signal: B7 through the `add_memory` and `set_task_status` MCP tools — a
  `reconciliation-stage-2` caller gets the same `ReconSnapshotWriteRejected` / transition refusal as
  `recon-stage-2`. Keys: fk-7aff55dd1465, fk-75fa713ba3d5. **High** → decompose adds it as a dependency
  of trigger-chain gate **6698**.
- **κ — Dashboard adopts S14's leaves.** `app.py` (runs.db/escalation joins via layout; boost-tier
  validation from `task_priority`), `config.py` (paths and known roots via layout), `api/merge_queue.py`,
  `data/escalation_corpus.py`, `loops.py`, `project_dbs.py`, `data/memory.py` (private `McpSession`, SSE
  and envelope parsers deleted; `mcp_tool_call` keeps its per-URL session cache over the shared client),
  `data/tasks.py` (`McpErrorType`). ~450 LOC, 8 files + tests. Prereqs: α, ζ, ξ, θ. Signal: the
  dashboard Memory and Tasks panels against a live fused-memory render as before, and a stopped
  fused-memory renders as an outage, not an empty panel (B9 through `mcp_tool_call`). Keys:
  fk-9cdaf62ed9af, fk-7a85d17a7c7e, fk-1b25c0b5d3a2. Medium.
- **ι — Scripts adopt the shared MCP client.** `scripts/migrate_metadata_modules_to_files.py`
  (`FusedMemoryClient`, `write_failure_reason` and its roots parser removed), `scripts/repair_wiped_metadata_files.py`
  (function-local imports and the delegation monkeypatch go; a truth-table contract test of
  `write_failure_reason` replaces it; `McpErrorType` for the absent-task verdict),
  `scripts/trial_module_tagger_haiku.py`, `fused-memory/scripts/strip_leaked_control_keys.py`,
  `fused-memory/scripts/cgl_eta_scheduler_gate.py` + their tests. ~450 LOC, 5 files + tests. Prereqs:
  ζ, ξ. Signal: `repair_wiped_metadata_files --dry-run` against a stub server answering `isError` reports
  the write as failed (B9). Keys: fk-ccd26ad88810, fk-1b25c0b5d3a2. Medium.
- **λ — Shared consecutive-streak primitive.** New `shared/streak.py` + tests (I7, B10). ~350 LOC,
  2 files. Prereqs: —. Intermediate: unlocks ο π and S12/S24. Keys: fk-43a75be2cd47. Medium.
- **ο — Orchestrator streaks on the primitive.** `streaks.py` (`StreakCounter`/`StreakRegistry` as the
  task-id specialisation), `scheduler.py` (`_fm_read_failure_streak`, `_delivered_check_error_streak`
  as registered streaks), `git_ops.py`, `service_restart.py`, `agents/memory_recall.py`,
  `recovery_emission.py`, `zero_progress_requeue.py` (both private `_Streak` classes go),
  `harness.py` (both streaks; the `_recovery_streak_filed_at`/`_zero_progress_filed_at` latch memos go),
  `merge_lane/worker.py` (resource-audit streak). ~600 LOC, 9 files. Prereqs: λ, θ (`scheduler.py`,
  `harness.py`), δ (`git_ops.py`). Signal: the escalate-once events these paths emit
  (e.g. `fm_read_failure` storm escape, recovery-veto escalation) fire once per failure run and re-arm
  after success in the scheduler/harness suites driven through their public entry points. Keys:
  fk-43a75be2cd47. Medium.
- **π — fused-memory and dashboard streaks on the primitive.** `middleware/path_scope_adjudicator.py`,
  `middleware/task_curator.py` (`_DegradedStreak` and the zero-output counter),
  `reconciliation/judge.py`, `services/memory_service.py`, `maintenance/link_heal_executor.py`,
  `maintenance/link_adjudicator.py` (`_ShardFailureStreak` stops emulating "consecutive" on a latched
  `StormCounter`), `arm_harness/replay.py` (`ConsecutiveFailureBreaker`), `dashboard/.../data/mcp_fanout.py`
  (`_bypass_streaks`). ~450 LOC, 8 files. Prereqs: λ. Signal: the curator's degraded-mode escalation and
  the adjudicator's zero-output escalation each fire once per run through their MCP-driven paths. Keys:
  fk-43a75be2cd47. Medium.
- **ρ — Integration gate: the spelling guard.** One test module scanning tracked non-test `.py`/`.sh`
  string constants for: store-path literals outside `project_layout`; `Merge `+branch-prefix subject
  literals outside `merge_subject`; `startswith('recon-stage-'|'reconciliation-stage')` outside
  `agent_identity`; comparisons of `error_type` to an exception-name literal; hand-written
  `protocolVersion` handshakes outside `mcp_session`; `CREATE TABLE … overrides`/`park_eviction_requests`
  outside `scheduler_overrides`. Allow-list empty except `Debt(owner)` entries for other streams'
  pending migrations (census_trigger → S21's leaf). ~300 LOC, 2 files. Prereqs: β γ δ ε ζ η θ ι κ μ ξ.
  Signal: B11 — seeded-fixture red, clean tree green, run by the project verify. Keys:
  fk-7a85d17a7c7e, fk-3644c6841b3e, fk-75fa713ba3d5, fk-1b25c0b5d3a2, fk-ccd26ad88810, fk-b84e0aa4e545.
  Medium.

Same-file serialization (real edges above): orchestrator `config.py`/`harness.py`/`cli.py` β→θ→ο;
`scheduler.py` η→θ→ο; `git_ops.py` δ→ο; fused-memory `server/tools.py` η→θ→ξ→ν→μ;
`task_interceptor.py` ζ→ξ→μ; `targeted.py` γ→μ; `grouped_read.py` ξ→ν; `mcp_envelope.py` ζ→ξ.
Disjoint by construction: ε vs ι (scripts), κ vs π (dashboard: `mcp_fanout.py` is π's only).

## Finding keys

| Key | Severity | Leaves |
|---|---|---|
| fk-06de98fa328b | high (weakened) | dropped — fixed by `d6b1626681` (see Freshness) |
| fk-3644c6841b3e | medium | α, δ, ε, ρ |
| fk-7a85d17a7c7e | medium | α, β, γ, ε, κ, ρ |
| fk-e367d6a83223 | medium | α, β, γ, ε |
| fk-1b25c0b5d3a2 | low | ξ, κ, ι, ρ |
| fk-43a75be2cd47 | medium | λ, ο, π |
| fk-f6ab6bfad0be | medium | η |
| fk-37f2baab3aec | medium | θ |
| fk-b84e0aa4e545 | medium | θ, ρ |
| fk-f5e16692e170 | medium | θ |
| fk-7aff55dd1465 | high | μ |
| fk-75fa713ba3d5 | medium | μ, ρ |
| fk-f01bf7bc23f9 | medium | ν |
| fk-954f6787b07c | medium | α, ε |
| fk-9cdaf62ed9af | medium | ζ, κ |
| fk-ccd26ad88810 | medium | ζ, ι, ρ |

## G7 walk (advisory at author time; re-walked at decompose)

- INV-4 `storm-escape-required`: λ/ο/π are the house storm-escape mechanism; no leaf adds a fallback
  without one. ζ's retry schedule is bounded (injected schedule, null-sentinel single attempt kept).
- INV-5 / INV-9: the purpose of the batch; ρ enforces it.
- INV-8 `loop-thread-occupancy-bounded`: θ moves fused-memory's override SQL from aiosqlite to
  `asyncio.to_thread` over sync sqlite3 — single-row statements under a 30 s busy timeout, off the loop.
- INV-10 `guards-exercise-behaviour`: ρ scans code string constants (docstrings excluded), not prose,
  and proves its matcher with seeded fixtures (B11); every other leaf's tests exercise behaviour through
  public entry points (B1–B10).
- INV-11 `no-silent-fail-soft`: ζ removes the `{}`-as-success and silent `None` shapes (typed
  `EnvelopeShape`); decision 2 refuses a split queue loudly; ε records an absent runs.db.
- INV-12 `exceptions-owned-or-ratified`: ρ's allow-list entries are `Debt(owner)` with live owners only.
- INV-13 `readers-prove-their-producer`: η's `snapshot_at is None`; ε's absent-store coverage row.
- INV-1/2/3/6/7: no new contract in prose (the contract above is signatures + invariants enforced by
  B-tests), no held state, no action on uncorroborated snapshots. No waiver needed.

## Out of scope

- The claimant TTL (fk-06de98fa328b, fixed). The fleet lease/clock leaf (S24). `scripts/_checkout.py`
  (S25). census_trigger's transport migration (S21). The dashboard's own MCP fan-out cache split (S22).
  The escalation filing entry point the escalators call (S1). Stage identity as a typed `RunContext`
  value (S12; μ provides the parse/format it uses). Task priority ranks inside the curator
  (`task_curator._PRIORITY_RANK`, not a finding key; S11 may adopt `shared.task_priority`).
- Changing any wire string, stored project id, merge-commit subject or on-disk path. Every leaf moves
  where a fact is defined; none changes the fact.

## Open questions (tactical)

1. **Depth of nested typing in `SchedulerStateSnapshot`.** Top-level keys are typed; nested values may
   start as `dict[str, Any]` where the producer's docstring does not pin a shape. Decide in η.
2. **Whether `McpSession.call` stays public.** The orchestrator's `mcp_call` uses non-`tools/call`
   methods through `_raw_call`; ζ either exposes `call` or keeps a narrower name. Decide in ζ.
3. **How the warm-lane caller obtains `rendered_prefix`.** `python3 -c` against `shared` (S24's
   precedent) or an argument from its systemd unit's environment; either satisfies "no literal in bash".
   Decide in δ.
4. **Guard home.** `shared/tests/` or a top-level `tests/` module, whichever the project verify already
   collects repo-wide. Decide in ρ.

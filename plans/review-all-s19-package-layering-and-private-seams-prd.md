# S19 — Package layering and private seams (PRD)

**Status:** draft, authored 2026-11-11 by the S19 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S19, agent mode, wave 3, size L, risk
medium); the sitting's rulings of 2026-11-12 folded in the same day (§Rulings). B + H: contracts C1–C10
and a two-way boundary-test sketch. ε is the integration gate of the fused-memory cluster and promotes
the R22 item 3 invariant; θ2 is the integration gate of the session cluster. Sixteen leaves, one
operator gate (λ, Leo's addition to S19-2) and one acceptance record (ω). Two leaves are held on wave-3
siblings by lead rulings: ε on the reconciliation-lifecycle plan's stage-capabilities leaf (S12 ι1,
S19/Q-C4 (a)), and β2 on the reconciliation-sweeps plan's lexicon leaf (S13 δ2, S19/Q-C5 (a)); δ is
held on S12 ι1 conditionally (only if ι1 keeps its `server/grouped_read.py` edit). Each releases when
that sibling is filed and its task id is wired.
Rulings applied without relitigating: R17 (this stream), R22 item 3 (the invariant), R23 (weakened
keys: none in this stream), R51/R54/R55 (wave-2 ordering conventions), R57 (codemod leaves keep their
declared files, hand-carry trigger; extended to this plan's moves above 50 files by the lead's
S19/Q-C6 (a)).

**Code anchors** re-verified against main `1b106418da` (2026-11-12) by the ruling fold (scratch
`scratch/S19-rulingfold/`). Main moves fast — cite by symbol; re-locate lines at implementation time.
The run pinned `6551b75aba` (2026-10-08). The author measured at `1f240e6da2` and the critique folds at
`97be1d71fa` and `7baa5b997c` (all 2026-11-11). A count the ruling fold re-measured is labelled
`measured at 1b106418da`; a count it did not re-measure keeps its earlier sha label and rests on no
done-criterion or delivered check. §Freshness at 1b106418da lists what moved. The fold-verifier
re-checked the tree at `7848173a7d` (2026-11-12, 33 commits later, none touching a declared path):
§Fold-verify at 7848173a7d.

## Goal

After this PRD:

- **fused-memory has a declared stratum order and a guard that enforces it.** `fused_memory/layers.py`
  lists the order of fused-memory's second-level units, bottom first. A test fails, naming the importer,
  the target, the import kind and the two layers, when any import statement points upward. The test
  counts module-level, function-local and `TYPE_CHECKING` imports. The reconciliation stores,
  vocabularies and write-path policies that services and middleware read live in a new package
  `fused_memory.recon_state` below services. `reconciliation/` keeps only the engine above middleware.
  `server/` is imported only by the composition root `server/main.py` and the server's own modules.
- **fused-memory no longer imports orchestrator.** The Landlock and bwrap builders and the backend
  resolution live in a stdlib-only `shared.sandbox` package. Both packages import it at module level.
  The two `# type: ignore[import]` imports and the "venv install contract" prose are gone.
- **cockpit depends on `shared` only.** `session_registry.py` is split into modules by its partition,
  and the stdlib-only parts move to `shared.session`. `cockpit/pyproject.toml` no longer declares the
  orchestrator member. `orchestrator/src/orchestrator/session_registry.py` stays the command-line entry
  point, so `spawn-claude.sh`, the watcher skills and `OPERATIONS.md` keep their command lines.
- **Each fact named by the findings has one home.** pid liveness lives in `shared.proc_liveness`
  (task 6615, done, merged 2026-11-11 `f750373949`), the one liveness check every caller uses
  (S19/Q-C1, ruled by Leo 2026-11-13, and S19/OQ-2, both (a)). The wmctrl runner, line parse and window-id canonical form live in
  `shared.session.windows`. `encode_cwd` lives in `shared.session.records`. The escalation archive
  layout and the terminal escalation status names live in `shared.escalation_layout` (S19/OQ-3 (a),
  S19/Q-C8 (a)).
- **A later operator session decides whether decisions should close at the source.** About a week
  after the review-all capstone (gate task 6698) and this plan's session integration task are both
  done, gate λ files a level-2 escalation that opens an explore, assess and discuss session on moving
  cockpit decision closing into the escalation package (Leo's addition to S19-2).
- **No private name crosses a module boundary on the paths this stream owns.** `orchestrator.agents.invoke`
  calls a public runner in `shared.cli_invoke` and keeps no second runner. The dashboard's MCP egress is
  a module below both `memory.py` and `mcp_fanout.py`, with no cycle. The dashboard's cross-module
  underscore names become public, and the three "read the underscore as vestigial" paragraphs go.

What a user observes when it lands:

- `uv run --project fused-memory pytest fused-memory/tests/test_package_layering.py` passes on main.
  Each of its seeded fixture trees fails with the offending edge named.
- A fresh interpreter importing `fused_memory.reconciliation.sandbox_guard` has no `orchestrator.*`
  module in `sys.modules`. The next live reconciliation cycle on the systemd `fused-memory.service`
  still runs its stage agents confined (its sandbox log line names `landlock`).
- `uv tree --package cockpit` lists no `orchestrator`. The `cockpit` script starts and renders sessions
  and decisions from `~/.claude/fleet`. `skills/spawn/spawn-claude.sh` launches a session whose
  `record.json` appears, run from a bare shell with no `PYTHONPATH`, as today.
- The dashboard serves every panel as before. After an endpoint hangs, the next call opens a fresh
  session, and eviction keeps today's two rules: hangs in the MCP client, failed urls in the fan-out.
- A design invariant, `fused-memory-subpackage-dag`, is listed in `docs/legibility/design-invariants.md`
  and its guard flags a seeded fixture.
- The escalation watcher surfaces gate λ's level-2 escalation to Leo after its delay, with the session
  brief in the task description; resolving it with a ruling (and a `/prd` path if he picks B or D)
  marks λ done.

## Rulings (sitting of 2026-11-12)

Sources: `digests/wave3-rulings-applied.md` §S19 and its cross-stream block, with
`digests/discussion/S19.md` (part of the ruling where cited), and `digests/wave3-lead-rulings.md`
(later blocks win). None of these amends R17's remedy; S19/OQ-1 and S19/OQ-2 reword two of its clauses,
as stated.

- **S19/OQ-1** (S19-1): (a), as recommended. `models` and `utils` sit below `backends`; every other
  step of R17's order stands (D3, C3). A rewording of R17's order clause to match the code's imports
  (an R49-style reading). Ruled by Leo.
- **S19/OQ-2**: (a), follows S19/Q-C1. `shared.proc_liveness.pid_alive` as task 6615 built it is the
  single home; R17's seam row `shared.config_dir.pid_alive` is reworded to
  `shared.proc_liveness.pid_alive` (lead action on the program doc). Ruled by the lead (a duplicate of
  S19/Q-C1); amends R17's seam row.
- **S19/OQ-3** (S19-2): (a), as recommended, PLUS Leo's addition: "file a milestone to explore,
  assess, and discuss B in depth after the review-all capstone". Adds gate λ (an operator gate
  depending on gate task 6698, by the lead's decision, and on θ2); ω's acceptance reason names λ
  instead of "reopen when an escalation stream takes it". Nothing else changes. Ruled by Leo; the
  edge onto 6698 ruled by the lead.
- **S19/OQ-4**: (b), `fused_memory.escalators` (D5, C10). Ruled by the lead.
- **S19/OQ-5**: (a). An earlier program task edits a body first and the move carries its code; a
  task behind the verify-dispatch (5040) or merge-lane (8528) chains goes after the move (D8). Ruled
  by the lead.
- **S19/Q-C1** (S19-3): (a), as recommended. Leo, 2026-11-13, in his words: "Good. Consider the zombie detection negligable (until evidence to the contrary), and use the portable liveness detection throughout." The lead
  had ruled it on 2026-11-12 on his condition, measured minor (no zombie-held lease, claim or folder in
  60 days, every caller fails safe, task 6615 done). `shared.proc_liveness.pid_alive` (C1) is the one
  liveness check every caller uses; no zombie-aware variant is public; zombie detection is recorded as
  negligible until a recorded case of a zombie read as alive that cost something (C1). Leaf ν converts
  `shared/process_identity.py`'s zombie arm, and S24's lease leaf the watchdog copy. Ruled by Leo.
- **S19/Q-C2**: (a). ι's public runner lands first in `shared.cli_invoke`; the CLI-invocation plan's
  moves carry the public names (C7). Ruled by the lead.
- **S19/Q-C3**: (a). κ1 creates `dashboard/data/mcp_client.py`; the dashboard-readers plan builds on
  it (C8). Ruled by the lead.
- **S19/Q-C4**: (a). The reconciliation-lifecycle plan's stage-capabilities task (S12 ι1) removes the
  prompt builder's upward import; δ drops that cut and ε waits on S12 ι1. Ruled by the lead.
- **S19/Q-C5**: (a), as S13/OQ-4 (a). The sweeps plan's split lands first and its
  `recon_prose_grammar.py` joins β2's move; β2 never waits on S13's flag-relay task. Ruled by the lead
  (duplicate of S13/OQ-4).
- **S19/Q-C6**: (a). An edit that only rewrites an import line or a prose citation in a non-test file
  is rebase-only under file locks, and the 24-hour hand-carry trigger applies to every move in this
  plan above 50 files. Amends R54 (its per-file chaining rule) and extends R57's hand-carry trigger.
  Ruled by the lead.
- **S19/Q-C8**: (a), flagged to Leo in the glance list. The terminal escalation status names and the
  archive layout live in one stdlib module θ0 creates first (`shared/escalation_layout.py`); the
  typed-escalation-facts plan's vocabulary task builds its status type from it and its invariant
  names it. R9's invariant text gains that home. Ruled by the lead.

Cross-stream consequences applied: one public liveness check (S19/Q-C1 with S24: S24's lease task
imports C1 and deletes the watchdog copy and 6615's parity test; S24's edge onto 6615 is satisfied);
Leo's decision-partition principle, the design invariant `one-decision-partition-per-area` (INV-23 once
the scheduler plan's task mints it; §Decision partition below: this plan's one decision table
complies); evals hermetic (this plan adds no store, sink, filer, queue or archive an eval cell can
reach: its moves relocate modules unchanged, and λ is a human gate); the S2 identity-header, S12
stage-identity, S15 shutdown, S16 sweep and S24 restart consequences touch no S19 text.

## Background

The `/review-all` run `review-all-dark_factory-20261008` found twelve costs that share one root: a
lower layer reaching up, or a fact copied because the layer that should own it sat in the wrong
place. The synthesis grouped them as stream S19 and Leo ruled the remedy as proposed (R17), with one
choice settled: `session_registry.py` is split in place first, then only its stdlib-only parts move to
`shared`. R22 item 3 promotes "the fused_memory subpackages form a DAG in the declared order, counting
module-level, function-local and TYPE_CHECKING imports" to a design invariant owned by this stream.

Live work already carries parts of the ruling. Task 6615 (done, merged 2026-11-11, `f750373949`)
consolidated pid liveness into `shared.proc_liveness`. Task 6922 (`702445cca6`) gave the session hooks an injected `ProcessTree`. Task
6179 (pending, planned) extracts the decision registry from `session_registry.py`. S14 ν (task 6720)
moves the memory child vocabulary into `memory_metadata`. S14 κ (task 6722) puts the dashboard's MCP
session onto `shared.mcp_session`. This PRD builds on them by dependency and files no second task for
their work.

## Findings disposition

Every key of the stream, exactly once. "Leaf" names the leaf or leaves that close the key.

| Key | Sev | Disposition | Note (re-measured at `1b106418da` unless a row says otherwise) |
|---|---|---|---|
| `fk-4fb8ffbb0141` reconciliation holds two strata; no subpackage order | medium | leaf α + β1 + β2 + γ + ε | Still present, unchanged since the draft. 7 bidirectional unit pairs (6 at the run's pin). 48 import statements point upward against the declared order (§Freshness F1). 18 reconciliation modules are imported from below reconciliation (16 at the pin). The 15 write-path stores, vocabularies and policies, plus 5 modules they need, move to `recon_state`. The three engine modules keep their place and lose their lower importers (γ). ε promotes the invariant. |
| `fk-420bc8f9a425` server/ imported from lower subpackages | medium | leaf δ (the `recon_report` edge: S12 ι1, S19/Q-C4 (a)) | Vocabulary half lands in S14 ν (task 6720, dependency): it repoints `link_heal`, `link_heal_store`, `prompts/stage1` and `child_memory_link`. The storm_counter arm is `fixed:0f86a2cdcc`: task 6547 deleted the `server/storm_counter.py` shim (merged `d614ee408b`, 2026-11-11), so δ no longer deletes it. Still present: 13 import statements in 9 files outside `server/` name `grouped_read`, `consolidation` or `mem0_update_authz`. δ moves those three into services. It subsumes task 5983 (cancel). The deferred `prompts → server.recon_report` import is cut by the reconciliation-lifecycle plan's stage-capabilities leaf (S12 ι1, which carries the upward-import half of `fk-20cb93d17ff3`); ε's guard waits on it. |
| `fk-64e06774ee0e` fused-memory imports orchestrator for its sandbox | medium | leaf ζ | Unchanged (no sandbox module changed since the draft): 2 function-local `# type: ignore[import]` imports in `sandbox_guard.py::resolve_recon_sandbox_wrap`, plus the hand-mirrored resolution order. |
| `fk-d91317f88f21` session_registry is a fleet library inside orchestrator | medium | leaf θ1 + θ2 | Present. 6,682 lines (5,772 at the pin; task 6615 removed 42 net lines). cockpit's only orchestrator import is this module: 19 import statements in 17 src and test files. 14 scripts (11 of them import it; measured at `97be1d71fa`), 2 skill scripts and 1 shell script also use it. `encode_cwd` has 3 copies. |
| `fk-cfcf00669c46` session_registry is five subsystems in one file | medium | leaf η1 + η2 (with task 6179) | Present. Top-level partition in lines, measured at `1f240e6da2`: records 1,034, decisions 1,578, reaper 751, windows 277, leases 817, identity 232, CLI 1,508; task 6615 since removed `_pid_alive` from the reaper region (net −42 lines). Task 6179 extracts the decisions in place (dependency, key stamped). Task 6955 is subsumed by η1/η2 (cancel). |
| `fk-e8c835c3f412` DecisionRecord is a polled second copy of escalation lifecycle | medium | leaf θ0 + θ2 (S19/OQ-3 (a)) | Present: `read_escalation_status` (rglob of `archive/`), `DECISION_CLOSE_MAP`, `normalize_escalations_dir`, the project-token hints and `migrate_decision_project_tokens`. Grown since the draft: `scripts/check_l0_settle_outcomes.py` (the S9 settle check) adds a third hand copy of the layout (`ARCHIVE_SUBDIR`, `ARCHIVE_DAY_FORMAT`, "Mirrors escalation/src/escalation/archive.py"), and its test reads the layout through `escalation.archive`; θ2 takes both. Under S19/OQ-3 (a), θ2 gives the archive layout one home (C9) and carries the key (its one disposition, contract §7). ω, filed cancelled with no `x_finding_key`, names "the polled-join arm of fk-e8c835c3f412" in its `x_acceptance_reason`, as S6's acceptance task 9211 names the coalesce-train arm of `fk-ba531799522b`, and names gate λ, which explores closing decisions at the source after the capstone (Leo's addition). |
| `fk-57599c7fd907` wmctrl helpers copied and diverged | low | leaf η2 (cockpit half in θ2) | Present: 2 `_wmctrl_list` copies, 2 `_WMCTRL_TIMEOUT_SECS`, 3 canonicalisers with 2 output forms (`session_hooks._canonical_window_id`, `session_registry._normalize_wm_window_id`, `cockpit/backends/wm.py::_window_id_int`). |
| `fk-ea4463b19d86` session_hooks tests patch private /proc readers | medium | leaf η2 | The /proc half is fixed by `702445cca6` (task 6922): `session_ownership_probes.py` takes an injected `ProcessTree`, and patches of session_hooks privates dropped from 89 to 14. The residual named by the same claim is the window probe: 8 `_resolve_wm_window_id` patches. η2 retires it with the injected `WindowProbe` (C6). The other 6 patches (`__file__` 4, `_run_install` 1, `_env_slug_is_owned` 1) patch no process-identity reader and are out of this key's claim. |
| `fk-e22732500172` `_pid_alive` six hand copies | medium | `filed:6615` + leaf ν | Leo, 2026-11-13: zombie detection is negligible until evidence to the contrary, and C1 is used throughout. That evidence, which reopens the zombie question, is a recorded case of a zombie read as alive that cost something (C1). Census at `565d15a06f`: 5 liveness call sites outside C1; ν converts `process_identity`'s 2, S24 γ the watchdog's 2, and the warm-lane sweep's `kill -0` is the same portable test (C1's rule and table). Task 6615 is done (merged 2026-11-11, `f750373949`): `shared/proc_liveness.py::pid_alive` exists and is imported by `task_ground_truth`, `config_dir`, `fleet_drain`, `service_restart`, `orchestrator_detector`, `session_registry`, `agent_session_reaper` (task 3729), the review-all session watcher and the watchdog's parity test; `harness._pid_alive` is gone. One production copy remains: `scripts/orchestrator-watchdog.py::_pid_alive`, pinned to C1 by `tests/scripts/test_watchdog_pid_alive_parity.py`; S24's lease leaf imports C1 and deletes both (S19/Q-C1 (a), cross-stream). `shared/src/shared/proc_group.py::pid_running` (reads `/proc`, a zombie is not running) stays the process-group snapshot's own fact, with no production caller, not a second public liveness predicate (S19/Q-C1 (a)). R17's seam name `shared.config_dir.pid_alive` is reworded to `shared.proc_liveness.pid_alive` (S19/OQ-2 (a)). Tasks 7444 and 5768 are subsumed by 6615 (cancel now). |
| `fk-ad5323d27142` invoke.py imports 4 cli_invoke privates and keeps a second runner | low | leaf ι | Present and grown: `_materialize_stdin`, `_parse_claude_output`, `_run_subprocess`, `_SubprocessResult` under `noqa: F401`. `_run_subprocess_local` has 2 `_SubprocessResult` constructions and now 5 callers at `565d15a06f`: codex, gemini, pi, ZCode (task 5399, `invoke.py::_invoke_zcode`) and MiMo (task 7993, `invoke.py::_invoke_mimo`), with `agents/zcode_backend.py` and `agents/mimo_backend.py` importing `_SubprocessResult`. Code importers of a private name: 18 files at `565d15a06f` (16 at `1b106418da`, 14 at `97be1d71fa`; the ZCode files came with task 5399 and the MiMo files with task 7993). Tests patch `_run_subprocess` at 13 sites in 7 files and `_run_subprocess_local` at 19 sites in 1 file (rule in §Freshness at 1b106418da). |
| `fk-99f44f47a6cc` dashboard MCP client in memory.py; mcp_fanout reaches back | medium | leaf κ1 | Present, unchanged: 7 modules import `mcp_tool_call` from `data/memory.py` (`metrics.py` through a multi-line import), and `app.py` calls `memory_data.mcp_tool_call` and `memory_data.enclosing_deadline`. `memory.py` imports 8 names from `mcp_fanout` at top. `mcp_fanout.first_success` imports `invalidate_session` from `memory` inside the function. Eviction is decided at 4 sites. |
| `fk-3ddc73250c64` dashboard cross-module underscore names | low | leaf κ2 | Present, unchanged since the draft: 13 ImportFrom statements bring 22 underscore names into 8 files, 1 of them deferred. `active_tasks._all_project_roots` no longer crosses; `api/window._parse_window` now does (3 importers). There are now 3 "vestigial" paragraphs (`project_dbs.py`, `loops.py`, `api/window.py`). |

Contract §8 dedup: step 1 (`find_tasks_by_metadata(key="x_finding_key", value=<key>)`, all statuses)
returned no task for any of the twelve keys on 2026-11-11 and again on 2026-11-12. Step 2
(`search_tasks`, ≥ 0.6, confirmed with `get_task`) found 6615, 6179, 6955, 7444, 5768, 9100, 5983, 6603,
6635, 5345 and 4444; the fold adds 7902 (score 0.659: session_hooks' window-id resolver family is
η2's). For gate λ (2026-11-12) step 2 found no task owning options B or D: the nearest hits are done
cockpit work (2690, 2303, 2515) and S3 ι 9181 (resolve-gate tables), none of which closes decisions at
the source. Their handling is in §Existing tasks.

## Freshness (F1–F3 re-measured at `1b106418da`)

**F1 — fused-memory typed import edges.** Rule: one edge for each first-party target of each import
statement in `fused-memory/src`. The kind is `top`, `deferred` (function-local) or `typecheck` (inside
`if TYPE_CHECKING:`), classified by `scripts/source_measures.py::import_statements_in_tree`. Units are
second-level packages and top-level modules. Measured at `1b106418da`: 331 modules, 1,085 edges, 16
units (327 modules and 1,062 edges at `1f240e6da2`; the five new modules, `backends.paged_read`,
`config.falkordb_endpoints`, `server.tool_middleware_bridge`, `server.unknown_argument_guard` and
`services.queued_writes`, add no unit and no upward edge). Bidirectional unit pairs (edges each way),
the same counts as at `1f240e6da2`:

- backends 2 ↔ 19 middleware;
- maintenance 2 ↔ 1 server;
- middleware 16 ↔ 8 reconciliation;
- middleware 7 ↔ 11 services;
- reconciliation 5 ↔ 30 server;
- reconciliation 27 ↔ 10 services;
- server 22 ↔ 2 services.

Against the order in C3, 48 statements point upward (the same 48 statements as at `1f240e6da2`; only their lines moved):

- middleware→reconciliation 16;
- services→middleware 11;
- services→reconciliation 10;
- reconciliation→server 5;
- backends→middleware 2;
- maintenance→server 2;
- services→server 2.

Each one and its cut is listed in §Module placement.

**F2 — the write-path set.** Modules below reconciliation import 18 reconciliation modules. Three are
engine modules that stay: `context_assembler`, `targeted` and `cli_stage_runner`. Their lower importers
take a moved helper or a Protocol instead. The other 15 are write-path stores, vocabularies and
policies (12,866 lines). Their own reconciliation imports pull in 5 more modules: `coalescing`,
`throughput`, `dead_letter_writer`, `event_journal` and `graphiti_degradation_probe` (2,029 lines). So
20 modules (14,895 lines) move. Outside the set they depend on only `models`, `utils` and
`memory_metadata`, plus four edges each leaf cuts:

- `journal → services.write_journal` (TYPE_CHECKING): α;
- `audit_trail_rotation → context_assembler.HINT_QUERIES_EXECUTED`: α;
- `backlog_policy → services.orchestrator_detector`: β1;
- `dead_letter_escalator → services.durable_queue`, `services.queued_writes`: module-level since task 9143 (`204de188c4`, S26 α), so γ places `dead_letter_escalator` in services, not in `escalators` (D5).

Importer sets, src and tests (data fixtures and dated reports excluded; rule in scratch
`scratch/S19-rulingfold/importers_at_main.py`, measured at `1b106418da`):

- α's 9 modules: 146 files;
- β1's 6 modules plus `orchestrator_detector`: 85 files;
- β2's 5 modules: 83 files;
- γ's 9 modules: 48 files;
- δ's 3 modules: 70 files (the shim is gone, so the δ set no longer counts it as a move).

**F3 — server/.** Nine import statements outside `server/` reach four server modules (measured at
`1b106418da`, unchanged):

- `grouped_read` ×4;
- `consolidation` ×2;
- `mem0_update_authz` ×2;
- `recon_report` ×1, deferred (cut by S12 ι1, S19/Q-C4 (a)).

`CHILD_KINDS` has 3 definitions: `memory_metadata`, `grouped_read`, and `topic_anchor._CHILD_KINDS`.
`grouped_read` imports the private `services.memory_service._MEM0_CONTENT_KEYS`.

**F4 — sandbox.** The orchestrator sandbox modules import only the stdlib:

- `landlock.py` 108 lines;
- `landlock_exec.py` 260 lines;
- `sandbox.py` 150 lines;
- `sandbox_dispatch.py` 256 lines.

`fused-memory/pyproject.toml` declares no orchestrator dependency. Its pytest `pythonpath` and pyright
`extraPaths` keep `../orchestrator/src` for 11 cross-package test modules, and they stay.

**F5 — session registry.** Partition sizes are in the disposition table. `session_registry.py` already
appends this checkout's `shared/src` to `sys.path` at import, and it imports three stdlib-only shared
leaves (`process_identity`, `project_id`, `session_liveness`; θ1 moves the last into the package as
`shared.session.liveness`, C5). A `shared.session` package is therefore
reachable from the tier-1 bare shell with no new mechanism. The run-by-path callers are:

- `skills/spawn/spawn-claude.sh` (`SESSION_REGISTRY_PY`);
- `skills/escalation-watcher/SKILL.md`, `skills/recon-escalation-watcher/SKILL.md`,
  `skills/unblock/SKILL.md` and `skills/spawn/SKILL.md`;
- `OPERATIONS.md`;
- `scripts/sitting/payloads.py` (`session_registry.__file__`).

Consumers by partition (files, src and tests):

- records 56;
- decisions 33;
- reaper 11;
- identity 4;
- leases 4;
- windows 1;
- CLI 13.

**F6 — invoke and dashboard.** As in the disposition table.

### Freshness at 1b106418da (the ruling fold, 2026-11-12)

`git log 1f240e6da2..1b106418da` touches 80 of the paths this plan declares (743 commits on main). What
moved under the plan:

- **Task 6615 done** (merged 2026-11-11, `f750373949`). C1 exists on main. β1's and η1's edges onto it
  are satisfied; lead actions cancel 7444 and 5768 now. The files β1 and η2/θ1 had dropped because of
  6615 stay dropped: `harness._pid_alive` and `orchestrator/tests/test_pid_alive_contract.py` are gone,
  and `service_restart.py` has no `session_registry` reference.
- **Task 6547 done** (merged `d614ee408b`, 2026-11-11): it deleted `server/storm_counter.py` and its test
  and repointed the storm-counter prose to `shared/src/shared/storm_counter.py`. That arm of
  `fk-420bc8f9a425` is `fixed:0f86a2cdcc`. δ no longer deletes the shim; its sidecar row became
  substrate, and it drops four files that only carried the shim's prose
  (`tests/server/test_storm_counter.py`, `shared/src/shared/storm_counter.py`,
  `shared/tests/test_storm_counter.py`, `escalators/mem0_update_storm_escalator.py`).
- **Task 9143 done** (S26 α, merged `3983fa2265`): `middleware/dead_letter_escalator.py` now imports
  `services.durable_queue.POST_EXECUTE_DEAD_PREFIX` and `services.queued_writes.QueueOperation` at module
  level (`204de188c4`). The draft's plan (move the constant into the escalator, put the escalator in the
  low `escalators` unit) would leave an `escalators → services` upward edge. By C2's own rule (place a
  module by its measured edges), γ moves `dead_letter_escalator` to `services/` instead, beside the
  queue whose events it reports; the constant stays in `durable_queue`. The deferred import the draft
  had γ delete is already gone, so task 6635's amendment changes reason (§Existing tasks). γ's edge
  onto 9143 is satisfied; 9147 (S26 β) still edits the module first.
- **Task 6621 done** (merged `9c3f608c88`): β2's edge onto it is satisfied. It left two more test
  importers of `consolidation_gate` (`test_operational_routing_boundary_matrix.py`,
  `test_operational_routing_guard.py`), which β2 now declares.
- **Task 5399 (ZCode backend)** added a fifth `invoke_agent` arm on `_run_subprocess_local` and
  `agents/zcode_backend.py` importing `_SubprocessResult`; ι declares that module and its test and
  threads `runner` to it. Patch sites are counted as lines that call `patch(`, `patch.object(` or
  `setattr(` naming `_run_subprocess` or `_run_subprocess_local` (scratch `iota_measure.py`): 13 in 7
  files and 19 in 1 file at `1b106418da`, the same at `97be1d71fa`; the draft's "22 in 9" used a broader rule.
- **The S9 settle check** (`scripts/check_l0_settle_outcomes.py`) landed with a hand copy of the archive
  layout, and its test reads the layout through `escalation.archive`; θ2 declares and repoints both
  (C9 now lists 11 readers through `escalation.archive`).
- **New δ importer:** `fused-memory/tests/server/test_write_triage_completion_records.py` imports
  `server.grouped_read`; δ declares it.
- Unchanged and re-checked: F1–F3 counts, the sandbox modules (no commit), the session-registry
  consumer set (no new importer; three left with 6615), the cockpit import count, the dashboard counts,
  `EscalationQueue._locate_path` and `set_resolve_callback` (task 4567 changed only the listing memo),
  the four `shared.cli_invoke` private definitions, and `scripts/quality_metrics_snapshot.py::_targets`
  and `::_cycles` (still private, no commit). INV-14 is still the highest invariant on main.
- Task statuses (`get_task`, 2026-11-12): done 6615, 6621, 9143 (above), 3729, 4567, 6710; in progress
  8402 (rebase-only, ε); every other task this plan names is pending, and 3419 is still deferred.

### Fold-verify at 7848173a7d (2026-11-12)

- `git diff --name-only 1b106418da 7848173a7d` lists 11 files (wait-hygiene tests, a magic-mock
  config check, `CONTRIBUTING.md`); none is in this plan's 640 declared paths.
- Importer sets of the α, β1, β2, γ, δ and ζ move sets re-measured with the ruling fold's rule: the
  same counts as at `1b106418da` (146, 85, 83, 48, 70, 20 files), and every undeclared hit is a dated
  snapshot, a data fixture, a path a dependency moves first, or prose (D6).
- Sidecar: loads through `CapabilityManifestDoc`; the 38 mechanical checks all fail at `7848173a7d`.
- Task statuses (`get_statuses`): unchanged, except 8402, now pending.
- Sibling labels re-read on 2026-11-12: S12 δ (not ι1) → β1; S13 κ3 (not κ2) → β2; S15 κ → α and
  S15 ε5 → δ; S1 θ1a, ι1, ι2 → β2 and S1 θ1b, θ2 → γ. The cross-PRD rows carry them.

### Second fold-verify at 565d15a06f (2026-11-13)

- Leo's liveness ruling of 2026-11-13 applied (C1's rule and census; leaf ν).
- Three importers landed after `7848173a7d`: `orchestrator/tests/_trigger_chain_rig.py` (α, now 158
  files), `fused-memory/tests/test_task_created_at.py` (γ, `candidate_key`, now 69) and
  `orchestrator/tests/_landlock_required.py` (ζ, now 32). MiMo (task 7993) added a sixth backend arm and
  two private-name importers (ι, now 20). Every other undeclared hit is a snapshot, fixture or prose.
- The lead's test-file ruling applied: α, θ1 and ι no longer depend on ζ.

## Resolved design decisions

- **D1 — The declared order is data the guard reads, beside the code.** `fused_memory/layers.py::LAYERS`
  is a tuple of frozensets, bottom first. The guard reads it, and the package docstring of
  `fused_memory/__init__.py` points at it in one line. An agent adding a second-level package must add
  it to a layer, or the guard fails naming the unlisted unit. R22 item 3's "order a committed list the
  test reads" is this file.
- **D2 — Units are second-level packages and top-level modules.** `memory_metadata`, `topic_slug` and
  `canonical_marker` are units. `fused_memory/__init__.py` is not. A same-layer edge is allowed. The
  guard's SCC check forbids cycles inside a layer.
- **D3 — The R17 order, completed (S19/OQ-1 (a), ruled by Leo).** R17 names
  `backends < models/utils < recon_state < services < middleware < reconciliation < server`. At main,
  `backends` imports `models` (3 edges) and `utils` (5 edges), and neither imports `backends`, so the
  first `<` is inverted against the code. As ruled, `models`/`utils` sit below `backends` and every
  other R17 step stands. The units R17 does not name are placed by their measured
  edges (C3).
- **D4 — recon_state takes write-path modules; the engine stays.** The finding asked to "give
  context_assembler and targeted one side each". Both are engine modules (the context assembler for a
  stage, the targeted reconciler), and so is `cli_stage_runner`. They stay in `reconciliation/`. Their
  lower importers lose the edge in γ:
  - `task_curator`'s `recon_archive_root` moves to `recon_state`;
  - `estimate_tokens` moves to `utils`;
  - `task_interceptor`'s `TYPE_CHECKING` reference to `TargetedReconciler` becomes a Protocol in
    middleware that names the methods it calls.

  This puts each module on one side.
- **D5 — The write-path escalation filers get their own low unit `fused_memory.escalators` (S19/OQ-4
  (b)).** Services reach six middleware modules that file escalations in 9 statements, and backends in
  1: `_folded_escalation`, the four storm and dead-letter escalators (`dead_letter_escalator`,
  `entity_mint_storm_escalator`, `mem0_update_storm_escalator`, `referent_repair_storm_escalator`) and
  `candidate_key_escalation`. Five of them import nothing from fused_memory but each other; moving those
  below `backends` removes 9 upward statements without moving any service. The sixth,
  `dead_letter_escalator`, imports `services.durable_queue` and `services.queued_writes` at module level
  since task 9143 (measured at `1b106418da`), so by C2's placement rule it moves to `services/` beside
  the queue whose dead-letter events it reports, which removes its one statement too. The other 3 go
  with them: `candidate_key` (53 lines, no fused-memory imports) goes to
  `utils`, and `dependency_direction_check` and `recon_claim_verification_guard`, write-path checks over
  recon vocabulary, go to `recon_state`. `_folded_escalation` becomes `escalators/folded_escalation.py`:
  11 modules import it, so its underscore was false. The package is named for what it holds, escalation
  filers (heuristic 1): the detectors named `*_alarm` (`classification_fallback_alarm`,
  `journal_growth_alarm`, `mem0_retry_backlog_alarm`) stay in `services` and only import the filer.
  Two middleware escalators stay where they are: `curator_escalator` imports middleware modules, and
  `scope_violation_escalator` is imported only from middleware and above (measured at `97be1d71fa`).
  The one-escalation-filing-entry-point plan (S1) writes the dead-letter escalator at
  `services/dead_letter_escalator.py`, its post-move path (the lead corrected S1's earlier `escalators/`
  path at S1's ruling fold).
- **D6 — Moves are codemods: no re-export shim, dated snapshots untouched.** Every importer is repointed
  in the same commit as its move (heuristic 13: a shim kept so an old path resolves is a finding). The
  codemod also rewrites executable references:
  - `python -m` paths (`throughput`'s CLI: `fused-memory/config/config.yaml` comment and
    `tests/test_config_schema.py`);
  - `caplog` logger names;
  - path keys in `docs/legibility/exception-ratifications.yaml` (the INV-12 registry; one key,
    `recon_claim_verification_guard`, moves in γ);
  - live docs and skills (`docs/task-authoring.md`, `skills/curate-fused-memories/SKILL.md`,
    `docs/sandbox-fleet-status.md`).

  It does not rewrite dated snapshots: `plans/`, `review/reports/`, calibration notes, fixture
  `PROVENANCE.md`, data fixtures (`pack_pool_2026-11-04.json`, `mined_921.json`), and dated ruling docs
  (author-mode §Code anchors: never re-anchor a dated snapshot). The moved files keep their history
  (`git mv`). A move leaf changes no behaviour, and its signal is that the live service still runs on
  the moved tree.

  In-code `path::symbol` prose citations of a moved file inside `*/src` are repointed when the citing
  file is one the leaf already declares; the others are left to the next editor of that file
  (CLAUDE.md's citation rule: fix a pin only in prose your own change edits). Examples left to the next
  editor: `middleware/done_provenance_authz.py`, `reconciliation/consolidation_auto.py`,
  `reconciliation/consolidation_gate.py` (until β2 moves it), `services/read_telemetry.py` and
  `services/topic_anchor.py` citing `server/{grouped_read,consolidation,mem0_update_authz}.py`;
  `backends/sqlite_task_backend.py` and `models/memory.py` citing moved recon modules;
  `shared/src/shared/task_metadata.py`'s comment citing `reconciliation/audit_trail_rotation.py`.
- **D7 — Codemod leaves keep their declared files (R57 class; S19/Q-C6 (a), ruled by the lead).** α,
  β1, β2, γ, δ and θ1 declare every file they repoint, from 66 to 158 (measured at `565d15a06f`).
  Measured lists are below. An undeclared repoint would be a scope violation. R57 was ruled for S8 β0's
  73 files; the lead's S19/Q-C6 (a) extends its 24 h hand-carry trigger to every move in this plan above
  50 files: α (158), θ1 (105), β2 (96), β1 (91), γ (69), δ (66), η1 (59) and κ1 (58). The moves below 50
  files, ζ (32) and θ2 (31), and the renames κ2 (42), η2 (19), ι (20) and the edit ν (2) keep the ordinary 15-file review
  trigger. Each is one mechanical rewrite per importer, which is the R57 justification; splitting one
  would leave importers of a deleted path between two tasks. Under the same ruling, an edit that only
  rewrites an import line or a prose citation in a non-test file is rebase-only under file locks, like
  R54's test files (an amendment of R54), so the hot files `orchestrator/src/orchestrator/workflow.py`
  (η1, θ1: one `build_session_slug` import) and the fused-memory importer lists do not chain S19 behind
  task 5040's file chains.
- **D8 — Order against live tasks (S19/OQ-5 (a), ruled by the lead).** A program task that edits the body of a module a leaf
  moves lands first, and the move carries its code (R52's principle). There are two exceptions. A task
  that waits on 5040 or 8528 depends on the S19 leaf instead, so S19 does not inherit those chains. And
  a pending task outside the program is rebase-only: the clerk repaths its `metadata.files` from the
  §Moves table when the move lands. Measured: 49 live tasks declare a recon_state module path (at
  `1f240e6da2`), and 16 declare `session_registry.py` (at `1b106418da`; 22 at `1f240e6da2`). Overlaps
  limited to importer files (one import line) are rebase-only (S19/Q-C6 (a)); the lead's per-file order
  (R54 method) settles the rest at decompose. The fold measured the
  hot-file declarations the critic flagged against the wave-2 per-file chains
  (`digests/wave2-order.json`): β1's `orchestrator/src/orchestrator/harness.py` (a 50-task chain) was a
  docstring citation inside `harness._pid_alive`, and its `orchestrator/tests/test_pid_alive_contract.py`
  was deleted with it by task 6615 (done; neither matches at `1b106418da`), so β1 declares neither; α's `shared/src/shared/task_metadata.py` (a chain through S9 ξ
  and S3 ε1) was a comment, so α drops it and the comment is left to the next editor (D6). Only η1 and
  θ1 still declare a hot file, `workflow.py`, for one import line (rebase-only, S19/Q-C6 (a)).
- **D9 — Session registry: two moves, as R17 rules.**
  - Step 1 splits in place inside `orchestrator/`: 6179 (decisions plus fleet files), η1 (records and
    identity), η2 (leases, reaper, windows). `session_registry.py` keeps only the CLI. The new
    orchestrator modules use the dual-mode sibling import that 6179's plan established
    (`if __package__:`), which `TestStdlibOnlySelfContainment`'s tier-1 rule requires.
  - Step 2 (θ1) moves the seven stdlib-only modules into `shared/src/shared/session/`. The dual-mode
    imports become plain `from shared.session import …` through the existing `shared/src` bootstrap, and
    every consumer is repointed.
  - The CLI files (`session_registry.py`, 6179's `decision_cli.py`) stay in orchestrator. That keeps
    every documented command line valid across both moves.
- **D10 — One window-id form: wmctrl's padded `0x%08x`.** It is what the hooks persist and what
  `wmctrl -l` prints. `canonical_window_id` parses with base autodetection (`int(raw, 0)`), so a decimal
  `$WINDOWID` and a hex id agree. The reaper compares canonical forms on both sides.
  `_normalize_wm_window_id` (unpadded, base-16 only with a `0x` prefix) is deleted. Before deleting it,
  η2 confirms that nothing persisted its unpadded form (it is used only for comparison, measured at
  `1f240e6da2`). The hook entry points take `windows: WindowProbe | None` beside `process_tree`, which
  retires the 8 `_resolve_wm_window_id` patches.
- **D11 — Sandbox policy stays with each caller; mechanism goes to shared.** `shared.sandbox` holds the
  builders, the availability probes, the resolution order and the argv wrap. Orchestrator keeps its
  process-global preference, the fail-closed refusal and its escalation dedup. fused-memory keeps its
  containment check and `RemediationSandboxUnavailable`. The resolution order fused-memory "mirrors" by
  hand becomes the call `resolve_available_backend('auto')`.
- **D12 — invoke threads one runner.** `invoke_agent` gains `runner: CliRunner = run_cli_subprocess`,
  passed to every backend. The claude sandbox path passes `wrap_argv`; codex, gemini, pi, zcode
  (task 5399) and mimo (task 7993; both measured at `565d15a06f`) pass `transcript=None`. The patch sites (13 of
  `_run_subprocess` and 19 of `_run_subprocess_local`, §Freshness) move onto that parameter, or onto a
  fake binary through
  `shared.testing_fake_claude` where the test is about the subprocess. The public runner lands first,
  in `shared.cli_invoke`, ahead of the CLI-invocation plan's runner and result moves (S20 θ and β) and
  its orchestrator repoint (S20 ο), which carry the public names into `shared/claude_runner.py` and
  `shared/cli_result.py` (S19/Q-C2 (a), ruled by the lead).
- **D13 — The dashboard keeps today's two eviction rules.** `mcp_tool_call` evicts on hang-shaped
  exits (`CancelledError`, `TimeoutError`, `httpx.TimeoutException`), as today. A fan-out that gives up
  on a url evicts it by calling `mcp_client.invalidate_session`, imported at module level, which cuts
  the cycle. The dashboard-readers plan's γ1 then leaves `first_success` as the only fan-out eviction
  site (S22 C2a). `reset_sessions` resets only the client's sessions, and the test fixture that relied
  on it resetting fan-out streaks calls `mcp_fanout.reset_failure_streaks` itself.
- **D14 — Prose stance.** Docstrings a leaf rewrites follow the Comments stance in `docs/code-quality.md`:
  - the circularity essay on `_wmctrl_list` is corrected and then shortened;
  - the "copy-don't-import" paragraphs go with the copies;
  - the "vestigial underscore" paragraphs go with the underscores;
  - the "OPERATIONAL PRECONDITION — venv install contract" block goes with the import.

  Measurements and rejected alternatives go to commit messages.

## Contracts

S19 owns these. A dependent stream holds a leaf on `S19/C<n>`.

**C1 — `shared.proc_liveness.pid_alive(value: object) -> bool`** (S19/OQ-2 (a) and S19/Q-C1 (a)).
Landed by task 6615 (done, merged 2026-11-11, `f750373949`); S19 owns the seam per the G4 table, whose
row the lead rewords from `shared.config_dir.pid_alive`. It returns False for a non-int, a bool or
`pid <= 0` without calling `os.kill`. Otherwise it calls `os.kill(pid, 0)`: success → True,
`ProcessLookupError` → False, `PermissionError` → True, any other `OSError` or an `OverflowError` →
False. It never raises. A zombie reads alive until it is reaped. Leo ruled zombie detection
negligible "until evidence to the contrary" (2026-11-13), on the measurement in
`digests/discussion/S19.md` §S19-3: no lease, claim or folder held by a zombie in 60 days, and every
caller keeps something rather than destroying it. Every liveness check calls it (the rule and census
below), and it is the only public pid-liveness predicate: `service_restart`'s lease liveness already does
(6615), and the fleet-watchdog plan's lease leaf (S24 γ) replaces the watchdog's `_pid_alive` copy and
deletes 6615's parity test `tests/scripts/test_watchdog_pid_alive_parity.py`. Until that leaf lands, the
parity test pins the watchdog copy. `shared/src/shared/proc_group.py::pid_running` (reads `/proc`, a
zombie is not running, a malformed stat raises `MalformedStatLine`) stays the process-group snapshot's
own fact and its tests' helper, with no production caller, not a second public predicate. What would reopen
zombie detection is evidence, not a new caller: a recorded case (escalation, incident or commit) of a
zombie read as alive that cost something, such as a lease, claim, folder or record kept past its owner's
death with an observable effect. The change would then be to C1 alone (read `/proc` state where it
exists), which S19 owns. No task is filed now.

**The liveness rule** (Leo, 2026-11-13: "use the portable liveness detection throughout"). A liveness
check answers, for one pid, whether that process still exists. In production code (each package's
`src/` tree, `scripts/` and `skills/`) every liveness check calls C1; a shell script, which cannot
import it, uses the same signal-0 test (`kill -0`). The rule does not cover a test's termination probe
on a process the test or its system under test killed (a zombie is the expected state there, and the
probe waits it out), nor process-table reads for identity, ancestry, command line, descriptors, cgroups
or locks, which answer other questions.

**Census at `565d15a06f`** (`git grep` for `os.kill(…, 0)`, `kill -0`, `/proc/<pid>/stat` state reads,
`Z`/`X` state sets, `pid_running` and `psutil` status calls): 5 liveness call sites outside C1, in 3
places.

| Site | Call sites | Disposition |
|---|---|---|
| `scripts/orchestrator-watchdog.py::_pid_alive` (a signal-0 copy) | 2 | converted by S24's lease leaf (γ), which imports C1 and deletes the copy and its parity test |
| `shared/process_identity.py::_running_stat`'s `Z`/`X` arm (`_EXITED_STATES`), reached from `capture` and `liveness` | 2 | converted by ν |
| `orchestrator/scripts/warm-lane/warm-lane-gc-sweep.sh` (`kill -0 "$pid"`) | 1 | complies: the same portable signal-0 test, in bash |

Named as not liveness checks: `shared/proc_group.py::pid_running` (no production caller; 6 call sites in
3 test files, termination probes after a kill, task 3955), the test-harness probes
`df_pytest_isolation.py::wait_pids_exited` and `orchestrator/tests/_verify_boundary_harness.py::wait_pids_exited`,
23 `os.kill(pid, 0)` sites in 16 test files and 3 test-side `kill -0` lines (task 6919 owns three
test copies); and the process-table readers `spawn_t3_lane.py::read_proc_stat`,
`session_ownership_probes.py`, `session_registry.py`'s cmdline read, `verify_cancel.py::read_ppid_map`,
`relocated_scopes.py`, `rebase_recovery.py::scan_lock_holders`, `spawn_temp_files.py::_read_live_cmdlines`,
`sampler/src/sampler/metrics.py::_default_fd9_exists`, `scripts/cgroup-stall-ratio.py` and
`scripts/verify_scope_peaks.py`. No `psutil` status or `pid_exists` call exists outside tests.

**C2 — `fused_memory.recon_state`** (created by α). Its purpose: the reconciliation stores,
vocabularies and write-path policies that modules below reconciliation read. Membership rule: a module
belongs in recon_state if a unit below `reconciliation` in C3 imports it, or a recon_state member
imports it, and it is not an engine module. An engine module is one that imports a unit above
`recon_state`; such a module stays in `reconciliation/`, and its lower importers take a Protocol or a
moved helper (D4). A member imports only units at or below `recon_state`'s layer, plus `shared` and
the stdlib. `__init__.py` is a one-line docstring with no imports. S12 and S13
add modules under this rule. This is a packaging contract with no runtime error semantics; C3 enforces
it.

**C3 — `fused_memory/layers.py::LAYERS` and the DAG rule** (ε). The layers, bottom first (S19/OQ-1 (a)):

```python
LAYERS: tuple[frozenset[str], ...] = (
    frozenset({'utils', 'topic_slug', 'canonical_marker', 'layers'}),
    frozenset({'models', 'config', 'escalators'}),
    frozenset({'backends'}),
    frozenset({'memory_metadata', 'routing', 'mcp_tools'}),
    frozenset({'recon_state'}),
    frozenset({'services'}),
    frozenset({'middleware'}),
    frozenset({'reconciliation'}),
    frozenset({'maintenance'}),
    frozenset({'server'}),
    frozenset({'eval', 'arm_harness'}),
)
```

`layers` is the declaration module itself (D2 makes it a unit), and it imports nothing first-party.

Rule, over every import statement in `fused-memory/src` (module-level, function-local and
`TYPE_CHECKING`, as `scripts/source_measures.py::import_statements_in_tree` classifies them), with
edges as `source_measures.import_targets` resolves them: the snapshot's own resolution, which ε
promotes out of `scripts/quality_metrics_snapshot.py` so the guard and the snapshot share one edge
builder and one SCC (R17's "over the snapshot's typed edge list"):

- if the importer's unit differs from the target's unit, the target's layer is at or below the
  importer's;
- the unit graph has no strongly connected component larger than one unit;
- every unit in the tree appears in exactly one layer, and every listed unit exists.

A violation names the importer module, the target module, the kind, the line and both layers (INV-2).
The guard derives units from the tree and keeps no exemption list (INV-12 has nothing to own). The SCC
is `source_measures.strongly_connected_components(edges)`, the snapshot's Kosaraju
(`quality_metrics_snapshot.py::_cycles`) made public in ε; the snapshot calls the same two functions,
so the guard and the snapshot cannot disagree on an edge (heuristic 11).

**C4 — `shared.sandbox`** (ζ), stdlib-only:

```python
# shared/src/shared/sandbox/backends.py
Backend = Literal['auto', 'bwrap', 'landlock', 'none']
ResolvedBackend = Literal['bwrap', 'landlock', 'none']
def resolve_available_backend(preferred: Backend) -> ResolvedBackend: ...
    # none -> none; auto -> landlock if available, else bwrap if available, else none;
    # landlock|bwrap -> itself if available, else none with a WARNING naming the preferred backend.
def build_wrapped_command(backend: Literal['bwrap', 'landlock'], cmd: list[str], cwd: Path,
                          writable_modules: Sequence[str], *, writable_extras: Sequence[str] = ()) -> list[str]: ...
# shared/src/shared/sandbox/landlock.py: is_landlock_available(), build_landlock_command(...)  (moved unchanged)
# shared/src/shared/sandbox/landlock_exec.py: the exec helper, run as `sys.executable <path>` (moved unchanged)
# shared/src/shared/sandbox/bwrap.py: is_bwrap_available(), build_bwrap_command(...)  (moved from agents/sandbox.py)
```

Each probe is computed once per process, as today. Resolution never raises. Orchestrator's
`sandbox_dispatch` (`set_backend`, `resolve_backend_or_refuse`, `SandboxUnavailable` and its dedup,
`wrap_command`) and fused-memory's `sandbox_guard` (containment check, `RemediationSandboxUnavailable`)
import it at module level.

**C5 — `shared.session`** (θ1), stdlib plus stdlib-only shared leaves. Modules:

- `fleet_files` (from 6179);
- `records`: schema, `SessionRecord`, slugs, pid/claude/T3 pointers, read/write/update, `encode_cwd`,
  `transcript_path_for_cwd`;
- `identity`: `SessionIdentity`, `resolve_session_identity`, `resolve_session_pid`,
  `is_shared_session_host`, `lease_liveness_pid`;
- `decisions` (6179's `decision_registry`);
- `leases`;
- `reaper`, which also holds the window sweep (`mark_windowless_wm_sessions_exited`, `_wm_window_id_of`,
  `_names_window`): the sweep calls the reaper's `_mark_exited_if_still_non_terminal`, so the edge is
  reaper → windows and `windows` stays a leaf;
- `windows` (C6);
- `liveness`: today's stdlib leaf `shared/src/shared/session_liveness.py` (`capture_for_record`,
  `judge`), moved in so the session package is the one home of session-record facts (heuristic 6);
  its 7 importers (measured at `1b106418da`) are in θ1's files.

`__init__.py` has no imports. The command-line contract is unchanged: `python3
<checkout>/orchestrator/src/orchestrator/session_registry.py <verb>` runs with no `PYTHONPATH`. The
decision verbs live behind 6179's `decision_cli.py`, which `session_registry.py` dispatches. Exceptions
(`CorruptSessionRecord`, `DecisionCloseRefused`, `DecisionReopenRefused`, `DecisionAuditRefused`) move
with their modules, unchanged. No name is re-exported from `orchestrator.session_registry`.

**C6 — `shared.session.windows`** (η2 creates it in place; θ1 moves it):

```python
WMCTRL_TIMEOUT_SECS: int
def run_wmctrl(argv: list[str]) -> subprocess.CompletedProcess[str]: ...   # never raises; rc 127 = wmctrl missing
@dataclass(frozen=True)
class WmWindow: window_id: str; desktop: str; host: str; title: str     # window_id canonical
def parse_wmctrl_list(stdout: str) -> tuple[WmWindow, ...]: ...          # lines with < 4 columns skipped
def canonical_window_id(raw: str | int | None) -> str | None: ...        # '0x%08x'; None if blank/unparseable/negative
def window_id_int(raw: str | int | None) -> int | None: ...
def live_window_ids(run=run_wmctrl) -> frozenset[str] | None: ...        # None = cannot prove; callers keep, never reap
def resolve_window_id_by_title(title: str, *, run=run_wmctrl, sleep=time.sleep, attempts: int = ...) -> str | None: ...
@dataclass(frozen=True)
class WindowProbe: live_window_ids: Callable[[], frozenset[str] | None]; resolve_by_title: Callable[[str], str | None]
```

**C7 — `shared.cli_invoke` public runner** (ι):

```python
@dataclass(frozen=True)
class SubprocessResult: ...            # today's _SubprocessResult fields, public
ArgvWrapper = Callable[[list[str]], list[str]]
@dataclass(frozen=True)
class TranscriptWatch: session_id: str | None; config_dir: Path | None; startup_grace_secs: float
                       working_idle_secs: float | None; absolute_cap_secs: float | None
async def run_cli_subprocess(cmd: list[str], cwd: Path, env: dict[str, str], *, backend: str, model: str,
                             timeout_seconds: float | None = None, stdin_data: bytes | None = None,
                             wrap_argv: ArgvWrapper | None = None,
                             transcript: TranscriptWatch | None = None) -> SubprocessResult: ...
def parse_claude_output(result: SubprocessResult, ...) -> AgentResult: ...   # today's private, public
CliRunner = Callable[..., Awaitable[SubprocessResult]]
```

There is one spawn path. Stdin is materialized before spawn and the process starts in its own
session. A timeout or a cancellation terminates the process group. `transcript=None` keeps the flat
timeout of today's `_run_subprocess_local`. `wrap_argv` is applied once, before spawn. Only this module
constructs a `SubprocessResult`.

Order against the CLI-invocation plan (S20), S19/Q-C2 (a): ι lands first. S20 θ (runner leaf
`shared/claude_runner.py`) and S20 β (result leaf `shared/cli_result.py`) depend on ι and carry the
public names `run_cli_subprocess`, `SubprocessResult`, `parse_claude_output` and `CliRunner` into those
modules; S20 ο's repoint of `orchestrator/src/orchestrator/agents/invoke.py` imports public names only.
This breaks the cycle ι → S20 ο → 9232 → ι that S19's and S20's drafts closed together, and keeps ι
off the task-5040 chain that 9232 and 9233 reach.

**C8 — `dashboard.data.mcp_client`** (κ1), the dashboard's only MCP egress:

```python
async def mcp_tool_call(client: httpx.AsyncClient, base_url: str, tool_name: str, arguments: dict,
                        timeout: float = 10) -> dict: ...
def invalidate_session(url: str) -> None: ...
def reset_sessions() -> None: ...
def enclosing_deadline(per_request_timeout: float) -> float: ...
```

The signature is today's `dashboard/src/dashboard/data/memory.py::mcp_tool_call` signature exactly:
the caller owns the httpx pool and passes `client` (8 production callers and the test modules pass it
today, measured at `97be1d71fa`). Its per-url session cache sits over `shared.mcp_session.McpSession` (S14 κ). Eviction follows D13.
Neither `mcp_fanout` nor `memory` defines or imports an egress of its own.

**C9 — `shared.escalation_layout`** (θ0 creates it; θ2 adds `locate_record` and repoints the readers; S19/OQ-3 (a)):

```python
ARCHIVE_SUBDIR = 'archive'; DATE_FORMAT = '%Y-%m-%d'
def archive_dir_for_date(queue_dir: Path, resolved_at_iso: str) -> Path: ...   # moved from escalation.archive
def locate_record(queue_dir: Path, escalation_id: str) -> Path | None: ...
    # <queue_dir>/<id>.json if present; else the newest archive/<date>/<id>.json, scanning date dirs
    # newest first and stopping at the first hit; None when absent. Never raises on a missing dir.
```

S19 owns the whole module (G4). θ0 creates it with the layout constants and the terminal status names
and repoints nothing; θ2 adds `locate_record`. `escalation.archive` imports these names in place of its
own definitions, and so does `scripts/escalation_ladder.py` (its `ARCHIVE_SUBDIR` "Mirrors
escalation/src/escalation/archive.py" copy goes; it already imports `shared` through
`scripts/_checkout.py`); `scripts/check_l0_settle_outcomes.py` does too (its `ARCHIVE_SUBDIR` and
`ARCHIVE_DAY_FORMAT` copy, "Mirrors escalation/src/escalation/archive.py", goes; it gains the
`_checkout` import the ladder already has). Every file that reads the layout through
`escalation.archive` is repointed to `shared.escalation_layout` in θ2, so `escalation.archive` is not a
re-export (D6). Measured at `1b106418da`: `escalation/src/escalation/queue.py`
(`archive.ARCHIVE_SUBDIR` ×4, `archive.archive_dir_for_date`), `escalation/src/escalation/sweep.py` (×3)
and 9 test files (`escalation/tests/test_archive.py`, `test_note_to_file.py`, `test_sweep.py`,
`orchestrator/tests/test_steward.py`, `dashboard/tests/_boundary_payloads.py`,
`test_escalation_corpus.py`, `test_escalation_lifecycle_gate.py`, `test_memory_evals_data.py`, and
`scripts/tests/test_check_l0_settle_outcomes.py`, new since the draft).
`shared.session.decisions` reads a record's status through `locate_record`.

`locate_record` is the stateless locator for a process that holds no queue: the decision reaper and
`scripts/backfill_decision_queue_stamp.py`. The escalation package keeps its own index-backed locator,
`escalation/src/escalation/queue.py::EscalationQueue._locate_path` (memoised archive listing, negative
cache, a WARNING on a multi-date duplicate), which S3 θ (task 9177) extends with a per-task landing
index; `queue.py::iter_all_escalation_paths` is an enumerator, not a locator. Both locators read the
layout from this one home. θ2 changes no locator logic in `queue.py`; it repoints its layout reads.
ω's acceptance reason names `_locate_path` as the queue-side locator.

Under the lead's S19/Q-C8 (a) the module also holds the stdlib names of the terminal escalation statuses,
`RESOLVED = 'resolved'`, `DISMISSED = 'dismissed'` and `TERMINAL_STATUS_NAMES`.
`shared.session.decisions.DECISION_CLOSE_MAP` keys from them, so no module outside `escalation/`
re-spells a status (the typed-escalation-facts plan's invariant `escalation-vocabulary-imported`), and
that plan's typed vocabulary (S2 α, `EscalationStatus`) takes its member values from them. θ0
creates these names with the layout, as a small leaf with no prerequisites, and θ2 → θ0. The seam
returned to the typed-escalation-facts plan is `S19/C9` (θ0): S2 α → S19 θ0; S2 α builds
`EscalationStatus` from the names; S2's invariant rule and guard name `shared.escalation_layout` as the
vocabulary's stdlib home.

**C10 — `fused_memory.escalators`** (γ; S19/OQ-4 (b)) holds the write-path escalation filers that
import no other fused_memory unit:

- `folded_escalation.file_folded_escalation`;
- `entity_mint_storm_escalator`;
- `mem0_update_storm_escalator`;
- `referent_repair_storm_escalator`;
- `candidate_key_escalation`.

They import the stdlib, `shared`, the escalation package (guarded, as today) and each other, and no
other fused_memory module. `dead_letter_escalator` is not a member: since task 9143 it imports the
durable queue's types, so γ moves it to `services/dead_letter_escalator.py` (D5). Behaviour is
unchanged: same escalation shapes, folding and storm thresholds. The one-escalation-filing-entry-point
plan (S1), which rewrites how fused-memory files escalations, edits these modules at their new paths:
its folded-filer leaf θ1a depends on β2 (and through it γ), its θ1b and middleware-escalator leaf θ2
depend on γ, its C9 path
`middleware/_folded_escalation.py` becomes `escalators/folded_escalation.py`, and its dead-letter path
becomes `services/dead_letter_escalator.py`.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| B1 | guard | seeded fixture packages under `tmp_path`: (a) module-level upward import, (b) `TYPE_CHECKING`-only upward import, (c) function-local upward import, (d) a unit missing from `LAYERS`, (e) a same-layer two-unit cycle, (f) a clean tree | fixture `LAYERS` | (a)–(c) fail naming importer, target, kind, line and both layers; (d) fails naming the unit; (e) fails naming the SCC; (f) passes |
| B2 | fm | the real tree after α–δ | main | the guard passes; `LAYERS` covers exactly the tree's units |
| B3 | recon_state | a fresh interpreter imports every `fused_memory.recon_state` module | — | no `fused_memory.{services,middleware,reconciliation,maintenance,server}` module in `sys.modules` |
| B4 | escalators/backends | a fresh interpreter imports `fused_memory.backends.sqlite_task_backend` | — | no `fused_memory.middleware.*` in `sys.modules` |
| B5 | server | a fresh interpreter imports `fused_memory.reconciliation.consolidation_auto_executor` and `fused_memory.services.consolidation_ops` | — | no `fused_memory.server.*` loaded (the recon-report guidance hand-off is S12 ι1's and its own boundary row, S19/Q-C4 (a)) |
| B6 | sandbox | `resolve_available_backend` over a 4×4 table of (preferred, availability) with fake probes | — | equals today's `sandbox_dispatch.resolve_active_backend` table row for row; `resolve_recon_sandbox_wrap` with landlock off and bwrap on returns the bwrap wrapper; with neither, raises `RemediationSandboxUnavailable`; a fresh interpreter importing `sandbox_guard` loads no `orchestrator` module |
| B7 | session CLI | `python3 orchestrator/src/orchestrator/session_registry.py lease-show --name x` run from a bare shell with no venv and no `PYTHONPATH`, as `skills/spawn/spawn-claude.sh` and the watcher skills run it and as the `_BARE_SHELL_ENTRYPOINTS` tier-1 row does (no `-I`: it would drop the script's own directory from `sys.path` and break 6179's dual-mode sibling import of `decision_cli`), plus the `_BARE_SHELL_ENTRYPOINTS` rows | after η1, η2 and θ1 | exit 0; every tier-1/tier-2 row passes; `shared/tests/test_pure_stdlib_leaves.py` lists the `shared.session` modules |
| B8 | cockpit | the cockpit suite in an environment where `import orchestrator` raises | after θ2 | the suite passes; `uv tree --package cockpit` names no orchestrator |
| B9 | windows | `canonical_window_id('52428807') == canonical_window_id('0x3200007') == '0x03200007'`; a record captured with a decimal id whose window is live | fake `WindowProbe` | the reaper keeps it; with the window absent the reaper marks it exited; cockpit's `WmBackend.is_alive` agrees through the same function |
| B10 | escalation layout | a record archived by a real `EscalationQueue` in a temp queue dir; a pending record; an unknown id | real queue | `locate_record` finds the archived record in its dated dir and the pending one at the root, and returns None for the unknown id; `escalation.archive.archive_dir_for_date` and `locate_record` agree on the path |
| B11 | cli runner | `run_cli_subprocess` against a fake binary that records its argv: (a) `wrap_argv` set; (b) timeout; (c) cancellation; (d) each of codex, gemini, pi, zcode and mimo through `invoke_agent(runner=…)` | fake binary | (a) the wrapped argv runs; (b) `timed_out=True` and the process group is gone; (c) the process group is gone and `CancelledError` propagates; (d) results equal today's `_run_subprocess_local` results field for field |
| B12 | dashboard client | `mcp_tool_call` on a clean result, a `ValueError`, an HTTP 500, a timeout and a cancellation; `first_success` over two urls whose first fails with each `FANOUT_FAILURE_EXCEPTIONS` member | stub server | `mcp_tool_call` keeps the session on a clean result, a `ValueError` and a 500, and evicts on a timeout or a cancellation. `first_success` evicts the failing url for every `FANOUT_FAILURE_EXCEPTIONS` member, which a spy on `mcp_client.invalidate_session` observes. A fresh interpreter importing `dashboard.data.mcp_fanout` does not import `dashboard.data.memory` |
| B13 | dashboard names | an AST walk of `dashboard/src` | after κ2 | no `ImportFrom` of a `dashboard.*` module brings an underscore name; the word "vestigial" is absent from `dashboard/src` |

The rows face both sides of each seam. Producer rows are B1, B3, B6, B9, B10, B11 and B12. Consumer
rows, which drive a real importer of the seam, are B2, B4, B5, B7, B8 and B13. Gate λ has no boundary
row: it changes no code, and its signal is the escalation it files.

## Decision partition

Leo's principle of 2026-11-12, design invariant `one-decision-partition-per-area` (INV-23, minted last
of the wave-3 invariants by the scheduler plan's task; cited by its slug, not restated), applied to the
decision areas this plan touches:

| Decision area | Decided by | Where | LLM rows |
|---|---|---|---|
| Which sandbox backend a confined process gets | code | C4 `resolve_available_backend` (one table); fail-closed refusal stays with each caller (D11) | none |
| Whether a pid names a live process | code | C1 `pid_alive`, the one check every caller uses (ν converts `process_identity`; Leo 2026-11-13) | none |
| Whether the dashboard drops an MCP session | code | D13's two rules in `mcp_client` and the fan-out | none |
| Whether a cockpit decision record closes | code | `DECISION_CLOSE_MAP` over the record's escalation status, read through C9 `locate_record`, run by the `reap-decisions` verb; a watcher only runs the verb | none |
| Whether decisions should close at the source, and who owns escalation identity across queues | human | gate λ's session with Leo (a ruling, then a `/prd` if B or D) | none |

No row hands a decidable case to an LLM, so the plan complies; it adds no recovery, triage or alarm
table.

## Design invariant (ε performs the four-site lockstep edit)

Minted here: **`fused-memory-subpackage-dag`**. The slug is the stable id and every check anchors on it,
never on the number. INV-14 is the highest on main; INV-15, 16 and 17 are reserved by wave-2 leaves 9140,
9242 and 9154. The six wave-3 minting tasks (S1 ω `escalation-filing-one-door`, S1's eval-hermeticity
invariant task, S2 ι, S16 ι `config-default-one-home`, this ε, and the scheduler plan's
`one-decision-partition-per-area` task) take INV-18 to INV-23 in landing order, the decision-partition
invariant last as INV-23 (lead ruling after the wave-3 fold-verifiers), so ε takes one of INV-18 to
INV-22. The lead wires the invariant-file chain at decompose and corrects the number then if the order
changes; the FILED row records the number that landed. This PRD asks for no edge onto 9154, which waits
on 8528 and 5040.

The sites, as `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` lists them:

- `docs/legibility/design-invariants.md`: the section, the header dating paragraph and the `## Fixtures`
  date list;
- `docs/legibility/design-invariants-fixtures.md`: the `-PRD` and `-CODE` fixtures and a dated addendum
  walk, with its rehearsal rows and cumulative count;
- `skills/prd/references/gates.md`: the family-inventory row and the `inv-trigger-shapes` span;
- `CONTRIBUTING.md`: no change (it is pinned as an absence);
- `orchestrator/src/orchestrator/agents/code_quality.md` (`docs/code-quality.md` is a symlink to it):
  no change required. It is a partial INV-to-heuristic mapping, never pinned for completeness; if ε
  names the new invariant under heuristics 9 and 13 there, the pair-wise pin applies.

Draft normative text:

> **Rule**: fused-memory's second-level units import only units in the same or a lower layer of
> `fused_memory/layers.py::LAYERS`. The rule counts every import statement: module-level,
> function-local and `TYPE_CHECKING`. The unit graph has no cycle, and every unit appears in exactly
> one layer.
>
> **Checkable design question(s)**: Does this feature add an import between fused-memory units, a new
> unit, or a module that a lower unit will import? Is the target's layer at or below the importer's? Is
> a new unit listed in `LAYERS`? Is a function-local import or a `TYPE_CHECKING` import being used to
> hide an upward edge?
>
> **Evidence**: at `6551b75aba`, six unit pairs imported each other in both directions, and an 11-module
> cycle spanned middleware, reconciliation and server once `TYPE_CHECKING` edges were counted
> (`fk-4fb8ffbb0141`, run `review-all-dark_factory-20261008`). At `1f240e6da2` and `1b106418da`, 48 upward import
> statements.
>
> **House pattern**: `fused_memory/layers.py`; `fused_memory.recon_state` for write-path state that lower
> units read; the guard `fused-memory/tests/test_package_layering.py`.
>
> **Family boundary**: INV-9 `one-fact-one-home` governs facts in records; this invariant governs import
> direction between fused-memory's units. It is the checkable form of heuristics 9 and 13 for one
> package.

The trigger shape for gates.md: *an import added between fused-memory units, or a new fused-memory unit,
that is not at or below its importer in `fused_memory/layers.py::LAYERS` (`fused-memory-subpackage-dag`)?*

The guard runs the C3 rule and B1's fixtures (INV-10: a guard is exercised against fixture inputs). It
classifies imports with `scripts/source_measures.py::import_statements_in_tree`, resolves their targets
with `source_measures.import_targets` and finds cycles with `source_measures.strongly_connected_components`
(both promoted by ε from `scripts/quality_metrics_snapshot.py::_targets` and `::_cycles`, which the
snapshot then calls), put on `sys.path` the way `orchestrator/tests/test_merge_lane_facade_names.py`
does. It derives its scope from the tree and keeps no hand-listed exemption.

## Module placement (fused-memory; measured at `1b106418da`, the same statements as at `1f240e6da2`)

Every upward statement of F1 and the leaf that removes it:

| Upward edges | Count | Cut | Leaf |
|---|---|---|---|
| middleware guards → `reconciliation.recon_self_model` (`execution_class_guard`, `gitignored_deliverable_guard`, `operational_suggestion_guard`, `premise_lint_guard`, `routing_intent_guard`) | 5 | target moves to recon_state | β1 |
| `task_interceptor` → `audit_trail_rotation`, `event_buffer`, `event_queue` (typecheck) | 3 | targets move | α |
| `task_interceptor` → `backlog_policy`, `bulk_reset_guard` (typecheck) | 2 | targets move | β1 |
| `task_interceptor` → `consolidation_gate`, `stale_gate_citation_guard` | 2 | targets move | β2 |
| `dependency_direction_check` → `task_filter` | 1 | both move to recon_state | γ (guard), β2 (target) |
| `task_curator` → `cli_stage_runner.recon_archive_root`, `context_assembler.estimate_tokens` | 2 | helpers move down: `recon_archive_root` to a recon_state module, `estimate_tokens` to `utils/token_estimate.py` | γ |
| `task_interceptor` → `reconciliation.targeted` (typecheck) | 1 | Protocol in middleware naming the methods the interceptor calls | γ |
| services alarms (`classification_fallback_alarm`, `completion_claim_gate`, `journal_growth_alarm`, `mem0_retry_backlog_alarm`, `memory_metadata_census`) → `middleware._folded_escalation` | 5 | target moves to `escalators/folded_escalation.py` | γ |
| `memory_service` → `dead_letter_escalator`, `entity_mint_storm_escalator`, `mem0_update_storm_escalator`, `referent_repair_storm_escalator` | 4 | three targets move to escalators; `dead_letter_escalator` moves to `services` (D5) | γ |
| `memory_service` → `dependency_direction_check`; `completion_claim_gate` → `recon_claim_verification_guard` | 2 | targets move to recon_state | γ |
| `memory_service` → `event_buffer`, `journal`, `recon_ledger` (typecheck) | 3 | targets move | α |
| `memory_service` → `flag_record_contract`, `standing_decision_writer` | 2 | targets move | β2 |
| `memory_service` → `recon_pool_map` ×3, `standing_decision_constants` | 4 | targets move | β1 |
| `completion_claim_gate` → `task_filter` | 1 | target moves | β2 |
| `sqlite_task_backend` → `middleware.candidate_key`, `candidate_key_escalation` | 2 | `utils/candidate_key.py`, `escalators/candidate_key_escalation.py` | γ |
| `child_memory_link`, `prompts/stage1` → `server.grouped_read`; `maintenance.link_heal`, `link_heal_store` → `server.grouped_read` | 4 | repointed to `memory_metadata` | task 6720 (S14 ν) |
| `consolidation_auto_executor` → `server.consolidation`, `server.mem0_update_authz`; `services.consolidation_ops` → the same | 4 | targets move to services | δ |
| `reconciliation.prompts` → `server.recon_report` (deferred) | 1 | cut by S12 ι1: `server/main.py::run_server` renders the guidance and the harness hands it to stages through `StageCollaborators.report_tool_guidance` (S19/Q-C4 (a)) | S12 ι1 (ε → S12 ι1) |

Edges inside a moved set are cut in the move leaf: `journal → services.write_journal` and
`audit_trail_rotation → context_assembler.HINT_QUERIES_EXECUTED` (the constant moves down; α),
and `backlog_policy → services.orchestrator_detector` (the detector moves to `utils`; β1).
`dead_letter_escalator`'s module-level imports of `services.durable_queue` and `services.queued_writes`
(task 9143) become same-unit edges when γ moves it to `services/`; `POST_EXECUTE_DEAD_PREFIX` stays in
`durable_queue`.

Between γ and β2, `recon_state.dependency_direction_check` imports `reconciliation.task_filter`
upward. That is tolerated until β2, and ε's guard lands only after both.

## Moves

Paths each leaf moves or deletes. The clerk's moves scan reads this table, and so does the repath of
live tasks' `metadata.files` (D8).

| Leaf | From | To |
|---|---|---|
| α | `fused-memory/src/fused_memory/reconciliation/{event_buffer,event_queue,coalescing,throughput,dead_letter_writer,event_journal,journal,recon_ledger,audit_trail_rotation}.py` | `fused-memory/src/fused_memory/recon_state/<same>.py` |
| β1 | `reconciliation/{recon_self_model,graphiti_degradation_probe,standing_decision_constants,recon_pool_map,bulk_reset_guard,backlog_policy}.py`; `services/orchestrator_detector.py` | `recon_state/<same>.py`; `utils/orchestrator_detector.py` |
| β2 | `reconciliation/{task_filter,consolidation_gate,flag_record_contract,stale_gate_citation_guard,standing_decision_writer}.py`; `reconciliation/recon_prose_grammar.py` (created by S13 δ1; S19/Q-C5 (a)) | `recon_state/<same>.py` |
| γ | `middleware/{_folded_escalation,entity_mint_storm_escalator,mem0_update_storm_escalator,referent_repair_storm_escalator,candidate_key_escalation}.py`; `middleware/dead_letter_escalator.py`; `middleware/candidate_key.py`; `middleware/{dependency_direction_check,recon_claim_verification_guard}.py` | `escalators/{folded_escalation,…}.py` (S19/OQ-4 (b)); `services/dead_letter_escalator.py`; `utils/candidate_key.py`; `recon_state/<same>.py` |
| δ | `server/{grouped_read,consolidation,mem0_update_authz}.py` | `services/<same>.py` (the `server/storm_counter.py` shim is already deleted, task 6547) |
| ζ | `orchestrator/src/orchestrator/agents/{landlock,landlock_exec}.py`; `agents/sandbox.py` | `shared/src/shared/sandbox/{landlock,landlock_exec}.py`; `shared/src/shared/sandbox/bwrap.py` |
| ζ | `orchestrator/tests/test_landlock.py`; `orchestrator/tests/test_sandbox.py` | `shared/tests/test_landlock.py`; `shared/tests/test_bwrap.py` |
| η1 | `session_registry.py` sections | `orchestrator/src/orchestrator/{session_records,session_identity}.py` |
| η2 | `session_registry.py` sections; `session_hooks.py::_wmctrl_list`, `_canonical_window_id`, `_resolve_wm_window_id` | `orchestrator/src/orchestrator/{session_leases,session_reaper,session_windows}.py` |
| θ1 | `orchestrator/src/orchestrator/{fleet_files,decision_registry,session_records,session_identity,session_leases,session_reaper,session_windows}.py` | `shared/src/shared/session/{fleet_files,decisions,records,identity,leases,reaper,windows}.py` |
| θ1 | `shared/src/shared/session_liveness.py` | `shared/src/shared/session/liveness.py` |
| θ2 | `escalation/src/escalation/archive.py::ARCHIVE_SUBDIR`, `DATE_FORMAT`, `archive_dir_for_date` | `shared/src/shared/escalation_layout.py` |
| θ2 | `scripts/escalation_ladder.py::ARCHIVE_SUBDIR`; `scripts/check_l0_settle_outcomes.py::ARCHIVE_SUBDIR`, `ARCHIVE_DAY_FORMAT`; `shared/src/shared/session/decisions.py::ESCALATION_ARCHIVE_SUBDIR` (today `session_registry.py`'s) | deleted; each imports `shared.escalation_layout.ARCHIVE_SUBDIR` (and `DATE_FORMAT`) |
| κ1 | `dashboard/src/dashboard/data/memory.py::mcp_tool_call`, `invalidate_session`, `reset_sessions`, `enclosing_deadline`, the session cache | `dashboard/src/dashboard/data/mcp_client.py` |

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y, so Y lands first.

| Other PRD / stream / task | Direction | Seam mechanism | Owner | Edge or status |
|---|---|---|---|---|
| S14 ν, task 6720 (pending) | consumes | `memory_metadata.child_parent_id`, `PARENT_ID_KEY`, the kind constants; the four lower importers of `grouped_read` repointed | S14 | δ → 6720 |
| S14 κ, task 6722 (pending) | consumes | dashboard `memory.py` over `shared.mcp_session`; `loops.py`, `project_dbs.py`, `config.py` adopt `project_layout` | S14 | κ1 → 6722; κ2 → 6722 |
| S14 α, task 6710 (done) | consumes | `shared.project_id.fold_project_token` (already imported by `session_registry.py`) | S14 | none (landed) |
| S14 μ, task 6721 (pending, high) | same bodies | `flag_record_contract`, `stale_gate_citation_guard`, `standing_decision_writer`, `server/mem0_update_authz.py` | S14 | β2 → 6721; δ → 6721 (S19/OQ-5 (a)) |
| S25 ε, task 6760 (pending) | consumes | `scripts/sitting` and `scripts/legibility` bound through `scripts/_checkout.py`, so they import `shared` | S25 | θ1 → 6760 |
| S10 η 9240, λ 9254, ο 9285 (pending; 9240 high) | same bodies | `stale_gate_citation_guard`, `consolidation_gate`, `task_filter` | S10 | β2 → 9240, 9254 (S19/OQ-5 (a); neither waits on 5040 or 8528); 9285 → β2, because 9285 waits on 8528 through 9282/9283 (D8 exception; lead action), so 9285 edits `task_filter` at its recon_state path |
| S10 δ 9204 (pending) | same body | `middleware/task_curator.py` | S10 | γ → 9204 (S19/OQ-5 (a)) |
| S26 μ 9203, β 9147, ρ 9243 (pending); α 9143 (done, `3983fa2265`) | same bodies | `standing_decision_writer`; `dead_letter_escalator`; `grouped_read` | S26 | β2 → 9203; γ → 9147; δ → 9243 (S19/OQ-5 (a)). γ's edge onto 9143 is satisfied; 9143 made `dead_letter_escalator` import `services.durable_queue` and `services.queued_writes` at module level, so γ moves it to `services/` (D5) |
| S3 θ 9177 (pending) | same body | `escalation/archive.py` | S3 | θ2 → 9177 |
| S8 κ1 9232, κ2 9233 (pending, behind 5040: 301 and 303 live prerequisites at `97be1d71fa`) | same bodies | `agents/sandbox_dispatch.py`, `agents/invoke.py` | S8 | 9232 → ζ, ι; 9233 → ι (lead actions; D8 exception: they wait on 5040). Kept under S19/Q-C2 (a), as ruled; with S20 first they would have closed the cycle ι → S20 ο → 9232 → ι |
| S6 α 9140 (INV-15), S10 θ 9242 (INV-16), S7 ζ 9154 (INV-17), and the wave-3 minting tasks S1 ω, S1's eval-hermeticity task, S2 ι, S16 ι and the scheduler plan's `one-decision-partition-per-area` task | same docs | the three invariant files (`docs/legibility/design-invariants.md`, `design-invariants-fixtures.md`, `skills/prd/references/gates.md`) | lead | none proposed by S19; the six wave-3 minting tasks take INV-18 to INV-23 in landing order, `one-decision-partition-per-area` last as INV-23, and the lead wires the invariant-file chain at decompose (R51, R54; lead ruling after the wave-3 fold-verifiers) |
| S12 run lifecycle (wave 3, authoring) | populates C2 | its leaves that edit `journal`, `event_buffer` or other moved modules work at the recon_state path | S19 creates, S12 populates | S12 α, ζ, κ1 → α; S12 δ → β1 (S12 δ edits `recon_self_model.py`; S12 ι1 reaches β1 through S12 δ), as the lead ruled S12/OQ-12 (a) and S12's ruling fold states (xref) |
| S12 ι1, stage capabilities (wave 3, authoring; priority high) | produces (to) | ι1 removes `prompts/__init__.py`'s function-local `server.recon_report` import: `server/main.py::run_server` renders the guidance and passes it to the harness, which hands it to stages through `StageCollaborators.report_tool_guidance` | S12 (the upward-import half of `fk-20cb93d17ff3`) | ε → S12 ι1 (held on S12/ι1; S19/Q-C4 (a)); δ does not cut that import. ι1's `server/grouped_read.py` re-bind of `CHILD_KINDS` is task 6720's work (S14 ν removes the second copy); if S12 keeps the file, δ → S12 ι1, and δ carries ι1's re-bind. If ι1 instead depends on task 6720, which already moves `CHILD_KINDS` and `PARENT_ID_KEY` into `memory_metadata`, ι1 drops `server/grouped_read.py` and no edge is needed (xref) |
| S13 sweeps and flag identity (wave 3, authoring) | populates C2 | S13 δ1 splits `task_filter` into `recon_prose_grammar` and the tree renderer inside `reconciliation/`; middleware and services import `recon_prose_grammar`, so C2 puts it in `recon_state`, and β2 moves it with `task_filter` | S19 creates, S13 populates | S19/Q-C5 (a), as S13/OQ-4 (a) (lead): S13 β, μ → α; S13 κ1 → β1; S13 κ3 → β2 (S13 κ3 edits `flag_record_contract.py` and `consolidation_gate.py`, which β2 moves); β2 → S13 δ2 (held on S13/δ2). β2 never depends on S13 κ3: the reverse edge would be a two-task cycle (labels as in S13's draft of 2026-11-12) |
| S1 filing entry point (wave 3, authoring) | edits C10 modules | fused-memory escalators re-filed through `escalation.filing`; S1's C9 `file_folded_escalation` / `afile_folded_escalation` live at `escalators/folded_escalation.py` after γ | S1 | as S1's draft of 2026-11-12 states: S1 θ1b and θ2 (middleware escalators: `dead_letter_escalator`, `entity_mint_storm_escalator`, `mem0_update_storm_escalator`) → γ; S1 θ1a (the folded filer), ι1 and ι2 → β2, the last S19 move on their files, and through β2 → γ; S1's C9 path is `fused-memory/src/fused_memory/escalators/folded_escalation.py` (S19/OQ-4 (b)), and its `dead_letter_escalator` path is `fused-memory/src/fused_memory/services/dead_letter_escalator.py` (D5, freshness at `1b106418da`), as S1's draft now writes it (the lead corrected S1's earlier `escalators/` path at S1's ruling fold) |
| S2 typed escalation facts (wave 3, authoring) | same bodies; produces (to) | S2 η and κ edit `reconciliation/backlog_policy.py` and `standing_decision_constants.py` (β1 moves both); S2 κ edits the escalators γ moves (`dead_letter_escalator` lands at `services/dead_letter_escalator.py`); under the lead's S19/Q-C8 (a) S2 α builds `EscalationStatus` from the stdlib terminal-status names that θ0 creates in `shared/escalation_layout.py` (seam `S19/C9`) | S19 owns C9 | S2 η, κ → β1 (their `backlog_policy`/`standing_decision_constants` edits) and S2 κ → γ (escalators), all at the new paths (S2 κ already waits on 8528, so it must not precede a move: D8 exception); S2 α → θ0 (S19/Q-C8 (a)), and S2's invariant rule and guard name `shared.escalation_layout` as the vocabulary's stdlib home; R9's invariant text gains that home (lead) |
| S16 config shape and routing (wave 3, authoring; its fold of 2026-11-11 already orders β2, ε2b → S19 δ and α → S19 κ2) | same files | S16 γ1 edits `agents/invoke.py`; S16 ε2b edits `middleware/task_curator.py`, and its ε2a `reconciliation/cli_stage_runner.py`; S16 β2 deletes `fused_memory/config/reload.py`, whose comments δ repoints first, and edits `config/schema.py`; S16 α edits `dashboard/data/orchestrator.py` | S16 | S16 β2, ε2b → δ; S16 α → κ2 (as S16's draft states); S16 γ1 → ι (reached through 9233); S16 ε2a → γ (γ moves `recon_archive_root` out of `cli_stage_runner.py`; ε2b reaches γ through δ) (xref) |
| S15 server boundary (wave 3, ruling-folded 2026-11-12) | same files; consumes C2 | S15 κ (the dashboard read model, `get_status(panels=...)`) declares `recon_state/journal.py` and `recon_state/event_buffer.py`, which α moves; S15 ε5 edits files δ repoints; `server/tools.py` imports (δ repoints `grouped_read`/`consolidation`/`mem0_update_authz` to services); δ no longer edits `server/main.py` | S15 | S15 κ → α (lead S15/OQ-8 and the cross-stream alignment "S15 κ lands right after 9280 and S19 α"; every other S15 task, and the dashboard-readers plan's two consumer tasks, reach α through S15 κ); S15 ε5 → δ (as S15's draft states); the rest rebase-only, lead orders (xref) |
| S20 CLI invocation (wave 3, authoring) | consumes C7 | S20 β (`shared/cli_result.py`) and θ (`shared/claude_runner.py`) carry ι's public names; S20 ο's `invoke.py` repoint imports public names only | S19 owns C7 | S19/Q-C2 (a): S20 β, θ → ι; S20 ο reaches ι through them. Reverses S20's draft ("S19's runner API depends on θ and ο") (xref). Test files take no edge (lead ruling after the wave-3 fold-verifiers): S20 withdrew δ → ζ (`shared/tests/test_landlock.py`) and its requested θ0, θ1, θ2 → S20 γ (`shared/tests/test_pure_stdlib_leaves.py`); neither is wired |
| S22 dashboard readers (wave 3, authoring; its fold of 2026-11-11 already consumes C8 under its OQ-6, lead-ruled with S19/Q-C3) | builds on C8 | S22's author draft had γ1 create `data/mcp_sessions.py` and cut the `memory ↔ mcp_fanout` cycle (κ1's work), and ν make `_parse_window`, `_cost_dbs` and the loops public (κ2's work); S22's fold now takes both from S19 | S19 owns C8 and the two keys (R17) | S19/Q-C3 (a): S22 γ1, γ2, δ1 → κ1; S22 δ1, ν, λ → κ2 (as S22's fold states). `mcp_fanout._LOCK_ACQUIRE_TIMEOUT_SECONDS`: 6879 makes the name public; S22 δ1 moves it to `ttl_cache.py` (κ2 → 6879) (xref) |
| S24 fleet watchdog (wave 3, authoring) | consumes C1 | S19/Q-C1 (a), the cross-stream "one public liveness check": S24's lease leaf (γ) replaces the watchdog's `_pid_alive` copy with C1 and deletes 6615's parity test; `service_restart`'s lease liveness already imports C1 (6615); S24 does not adopt `shared.proc_group.pid_running` as a second public pid predicate | S24 | S24 γ → 6615 is satisfied (6615 done, merged 2026-11-11, `f750373949`). S24's draft also recommended "S19 η2 → S24 γ" for `service_restart.py`; no edge is needed: at `1b106418da` `service_restart.py` has no `session_registry` reference, so η2 and θ1 do not declare the file (xref) |
| task 7902 (pending, low, outside the program) | same file | session_hooks' window-id resolver family (`_resolve_wm_window_id` 9 sites, `_wmctrl_list` 2, `_WMCTRL_TIMEOUT_SECS` 1 in `test_session_hooks_display.py`) is η2's `WindowProbe` work | S19 (η2) | 7902 → η2; 7902 is amended to keep only its install and single-read families (lead action) |
| tasks 5040 / 8528 | none | no S19 leaf declares `merge_lane/worker.py` or a 5040 file; under S19/Q-C6 (a) no S19 leaf chains behind them through a hot file | — | none |
| gate task 6698, "Run /review-all on dark_factory" (pending; the review-all capstone, a delayed human gate that waits on this program's high and critical tasks) | consumed by λ | λ's delayed milestone starts when 6698 and θ2 are both done | lead (ruled_by: lead, on Leo's S19-2 addition) | λ → 6698, θ2. No S19 leaf is high or critical, so 6698 does not wait on S19 and λ needs its own edge onto θ2 |

Out-of-program pending tasks on moved files are rebase-only, and the clerk repaths them when the move
lands (D8). The largest groups:

- 49 tasks declare a recon_state module path (at `1f240e6da2`);
- 16 declare `session_registry.py` (at `1b106418da`);
- 25 declare `cli_invoke.py` (at `1b106418da`);
- 24 declare `task_curator.py` (at `1f240e6da2`).

## Decomposition plan

Every leaf is filed with `planning_mode=True` and carries:

- `metadata.x_finding_key` (a list);
- `metadata.x_finding_run = "review-all-dark_factory-20261008"`;
- `metadata.source = "review-all"`;
- `prd_path`, `prd_task_label` and file-level `metadata.files`;
- `priority` equal to the highest severity of its keys.

Gate λ is the exception to those fields: it is a human gate with no finding key and no files (its
fields are under λ). No leaf is `high` or `critical`, so no leaf adds an edge onto gate 6698; λ
depends on 6698 instead (the lead's decision on Leo's addition). Same-file leaves inside this plan are
serialized by real edges (below). The codemod leaves are R57 class (D7, S19/Q-C6 (a)). File counts are
measured at `1b106418da`. Held leaves (filed with the hold; the lead wires the sibling's task id when
it is filed): ε on S12 ι1 (S19/Q-C4 (a)), β2 on S13 δ2 (S19/Q-C5 (a)), and conditionally δ on S12 ι1
(only if ι1 keeps its `server/grouped_read.py` edit).

**α — Create `fused_memory.recon_state` and move the reconciliation event and journal stores into it**
(`task_kind=normal`, priority medium, est. ~900 changed LOC excluding renames)

- What: create the package (C2). `git mv` the 9 modules in §Moves. Repoint every importer (D6). Cut
  `journal`'s `TYPE_CHECKING` import of `services.write_journal` with a Protocol in `recon_state` that
  names the write-journal methods `ReconciliationJournal` calls. Move `HINT_QUERIES_EXECUTED` from
  `context_assembler` into `recon_state.audit_trail_rotation`; `context_assembler` imports it. The
  `python -m fused_memory.recon_state.throughput` CLI path replaces the old one in the config comment
  and in `tests/test_config_schema.py`.
- Keys: `fk-4fb8ffbb0141` (shared with β1, β2, γ, ε).
- Files: (158 files) fused-memory/src/fused_memory/recon_state/__init__.py, fused-memory/src/fused_memory/reconciliation/context_assembler.py, fused-memory/src/fused_memory/reconciliation/event_buffer.py, fused-memory/src/fused_memory/recon_state/event_buffer.py, fused-memory/src/fused_memory/reconciliation/event_queue.py, fused-memory/src/fused_memory/recon_state/event_queue.py, fused-memory/src/fused_memory/reconciliation/coalescing.py, fused-memory/src/fused_memory/recon_state/coalescing.py, fused-memory/src/fused_memory/reconciliation/throughput.py, fused-memory/src/fused_memory/recon_state/throughput.py, fused-memory/src/fused_memory/reconciliation/dead_letter_writer.py, fused-memory/src/fused_memory/recon_state/dead_letter_writer.py, fused-memory/src/fused_memory/reconciliation/event_journal.py, fused-memory/src/fused_memory/recon_state/event_journal.py, fused-memory/src/fused_memory/reconciliation/journal.py, fused-memory/src/fused_memory/recon_state/journal.py, fused-memory/src/fused_memory/reconciliation/recon_ledger.py, fused-memory/src/fused_memory/recon_state/recon_ledger.py, fused-memory/src/fused_memory/reconciliation/audit_trail_rotation.py, fused-memory/src/fused_memory/recon_state/audit_trail_rotation.py, dashboard/tests/test_tab_recon.py, docs/task-authoring.md, fused-memory/config/config.yaml, fused-memory/scripts/backfill_entity_standing_decision.py, fused-memory/scripts/probe_schema_max_turns.py, fused-memory/scripts/repair_recon_citation.py, fused-memory/src/fused_memory/middleware/task_interceptor.py, fused-memory/src/fused_memory/reconciliation/backlog_policy.py, fused-memory/src/fused_memory/reconciliation/citation_repair.py, fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py, fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py, fused-memory/src/fused_memory/reconciliation/consolidation_proposal.py, fused-memory/src/fused_memory/reconciliation/cross_run_citation_scan.py, fused-memory/src/fused_memory/reconciliation/drained_event_restore.py, fused-memory/src/fused_memory/reconciliation/flag_dedup.py, fused-memory/src/fused_memory/reconciliation/harness.py, fused-memory/src/fused_memory/reconciliation/judge.py, fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py, fused-memory/src/fused_memory/reconciliation/prompts/stage2.py, fused-memory/src/fused_memory/reconciliation/sqlite_watchdog.py, fused-memory/src/fused_memory/reconciliation/stages/base.py, fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py, fused-memory/src/fused_memory/reconciliation/standing_decision_storm_escape.py, fused-memory/src/fused_memory/reconciliation/standing_decision_writer.py, fused-memory/src/fused_memory/reconciliation/summary_pool.py, fused-memory/src/fused_memory/reconciliation/targeted.py, fused-memory/src/fused_memory/server/main.py, fused-memory/src/fused_memory/server/recon_report.py, fused-memory/src/fused_memory/server/tools.py, fused-memory/src/fused_memory/services/memory_service.py, fused-memory/tests/_dead_letter_helpers.py, fused-memory/tests/_fm_helpers.py, fused-memory/tests/_hermetic_task_stack.py, fused-memory/tests/_submit_stack.py, fused-memory/tests/reconciliation/harness_fixtures.py, fused-memory/tests/reconciliation/test_backlog_policy_offload.py, fused-memory/tests/reconciliation/test_coalescing.py, fused-memory/tests/reconciliation/test_drained_event_restore.py, fused-memory/tests/reconciliation/test_event_buffer_coalescing.py, fused-memory/tests/reconciliation/test_flag_dedup_project_qualified_marker.py, fused-memory/tests/reconciliation/test_integrity_check_dangling_citation_scan.py, fused-memory/tests/reconciliation/test_recon_cross_cycle_dedup_citations.py, fused-memory/tests/reconciliation/test_stage1.py, fused-memory/tests/reconciliation/test_stage1_auto_consolidation_wiring.py, fused-memory/tests/reconciliation/test_stage2_flag_carry_forward.py, fused-memory/tests/reconciliation/test_stage_agent_outcome.py, fused-memory/tests/reconciliation/test_standing_decision_e2e.py, fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py, fused-memory/tests/reconciliation/test_standing_decision_writer.py, fused-memory/tests/reconciliation/test_throughput.py, fused-memory/tests/server/test_get_cycle_summary_presence_tool.py, fused-memory/tests/server/test_recon_report_hook_b.py, fused-memory/tests/server/test_recon_report_standing_decision.py, fused-memory/tests/test_append_rejection_wire_shape.py, fused-memory/tests/test_audit_trail_archive_wiring.py, fused-memory/tests/test_audit_trail_rotation.py, fused-memory/tests/test_audit_trail_rotation_boundary.py, fused-memory/tests/test_auto_consolidation_e2e.py, fused-memory/tests/test_backfill_entity_standing_decision.py, fused-memory/tests/test_backlog_policy.py, fused-memory/tests/test_candidate_key_boundary.py, fused-memory/tests/test_check_consolidation_closure_cli.py, fused-memory/tests/test_config_schema.py, fused-memory/tests/test_consolidation_auto_executor.py, fused-memory/tests/test_consolidation_closure_seam.py, fused-memory/tests/test_consolidation_proposal_outcome.py, fused-memory/tests/test_consolidation_proposal_tool.py, fused-memory/tests/test_curation_deferral_acceptance.py, fused-memory/tests/test_curation_pipeline.py, fused-memory/tests/test_curator_corpus_streams.py, fused-memory/tests/test_daemon_connect_consolidation.py, fused-memory/tests/test_data_dir_hermeticity.py, fused-memory/tests/test_dead_letter_writer.py, fused-memory/tests/test_dead_letters_tool.py, fused-memory/tests/test_declared_scope_stamp.py, fused-memory/tests/test_defect_creation_floor.py, fused-memory/tests/test_defect_records_update_floor.py, fused-memory/tests/test_defect_write_tools.py, fused-memory/tests/test_deferral_gate.py, fused-memory/tests/test_delivered_checks_e2e.py, fused-memory/tests/test_done_provenance_chokepoint.py, fused-memory/tests/test_event_buffer.py, fused-memory/tests/test_event_journal.py, fused-memory/tests/test_event_queue.py, fused-memory/tests/test_filer_first_attribution.py, fused-memory/tests/test_find_tasks_by_metadata_tool.py, fused-memory/tests/test_flag_dedup.py, fused-memory/tests/test_flag_record_write_seam.py, fused-memory/tests/test_get_tasks_projection.py, fused-memory/tests/test_harness.py, fused-memory/tests/test_interceptor_amendment_advisory.py, fused-memory/tests/test_interceptor_terminal_claimant.py, fused-memory/tests/test_journal.py, fused-memory/tests/test_journal_migration.py, fused-memory/tests/test_live_task_write_guard.py, fused-memory/tests/test_main_ticket_janitor_wiring.py, fused-memory/tests/test_mem0_tombstone.py, fused-memory/tests/test_memory_service.py, fused-memory/tests/test_merge_standing_decision_invalidation.py, fused-memory/tests/test_operational_routing_boundary_matrix.py, fused-memory/tests/test_pending_since_anchor.py, fused-memory/tests/test_pragma_triad_consolidation.py, fused-memory/tests/test_queue_stats_tool.py, fused-memory/tests/test_r4_idempotency.py, fused-memory/tests/test_recon_db_atomicity.py, fused-memory/tests/test_recon_ledger.py, fused-memory/tests/test_recon_reliability_integration.py, fused-memory/tests/test_recon_self_model.py, fused-memory/tests/test_recon_write_policy.py, fused-memory/tests/test_recurrence_mint.py, fused-memory/tests/test_recurring_gate_boundary.py, fused-memory/tests/test_repair_recon_citation.py, fused-memory/tests/test_resolve_ticket_vocabulary.py, fused-memory/tests/test_review_suggestion_curator_intake_e2e.py, fused-memory/tests/test_sqlite_watchdog.py, fused-memory/tests/test_stage3_ledger_presence.py, fused-memory/tests/test_stages.py, fused-memory/tests/test_stale_gate_citation_guard.py, fused-memory/tests/test_standing_decision_constants.py, fused-memory/tests/test_standing_decision_storm_escape.py, fused-memory/tests/test_status_authority_gate.py, fused-memory/tests/test_summary_pool.py, fused-memory/tests/test_targeted.py, fused-memory/tests/test_task_creation_journal.py, fused-memory/tests/test_task_interceptor.py, fused-memory/tests/test_task_text_markup_internal_writers.py, fused-memory/tests/test_task_tools.py, fused-memory/tests/test_task_write_agent_id.py, fused-memory/tests/test_ticket_parking.py, fused-memory/tests/test_ticket_worker.py, orchestrator/tests/test_deterministic_provenance_wire_e2e.py, orchestrator/tests/test_reopen_sticks_e2e.py, scripts/tests/test_check_run_completion.py, scripts/tests/test_fm_wedge_forensics.py, shared/src/shared/phantom_verdict.py, shared/tests/loop_blocking_allowlist.py, tests/scripts/test_atomic_write_regrowth.py, orchestrator/tests/_trigger_chain_rig.py
- Deps: none; α opens the plan (lead ruling on test-file edges, after the wave-3 fold-verifiers). It
  shares only test files with ζ (`fused-memory/tests/test_stages.py`) and θ1
  (`tests/scripts/test_atomic_write_regrowth.py`), which are rebase-only (R54); no edge.
- Downstream (wave-3 siblings, as their drafts state): S12 α, ζ, κ1; S13 β, μ; S15 κ, and through it
  every other S15 task and the dashboard-readers plan's two consumer tasks. α carries the 24 h hand-carry
  trigger (D7) and the lead watches it for file-lock starvation like tasks 5040 and 4867 (lead adoption
  note of 2026-11-11). It declares 158 files at `565d15a06f`: 147 importer files plus the 10 new paths
  and `context_assembler.py`; every other hit is a dated snapshot, a data fixture or prose (D6). It was
  157 at `7848173a7d`, as the lead's note records; `orchestrator/tests/_trigger_chain_rig.py` (imports
  `EventBuffer`) landed since. The sitting's earlier "158" predates the fold that dropped
  `shared/src/shared/task_metadata.py` (D8) and is a different list.
- Signal: after merge, the systemd `fused-memory.service` restarts on the tree, and its next
  reconciliation cycle writes a journal run, as `mcp__fused-memory__get_status` shows.
  `python -m fused_memory.recon_state.throughput --help` exits 0. B3 passes for the α modules except
  `recon_ledger`, which still loads `reconciliation` (it imports `standing_decision_constants` at
  module level) until β1 moves that module.
  `git grep -nE 'fused_memory\.reconciliation\.(event_buffer|event_queue|journal|recon_ledger|audit_trail_rotation|coalescing|throughput|dead_letter_writer|event_journal)\b' fused-memory/src` is empty.
- DELETES: nothing but the old paths (pure move); the `TYPE_CHECKING` upward import.
- G7: INV-12: none of α's modules is keyed in `docs/legibility/exception-ratifications.yaml`
  (measured); γ repoints the one moved module that is (`recon_claim_verification_guard`). INV-9: no
  second copy is created.

**β1 — Move the recon self-model, pool map and backlog/bulk-reset policies into recon_state; orchestrator_detector into utils**
(`task_kind=normal`, priority medium, est. ~600 changed LOC excluding renames)

- What: `git mv` the 6 reconciliation modules and `services/orchestrator_detector.py` (§Moves).
  Repoint importers, `recon_state/recon_ledger.py` among them (α moved it; it imports
  `standing_decision_constants`). `orchestrator_detector` carries 6615's `shared.proc_liveness` import
  as landed. β1 declares neither `orchestrator/src/orchestrator/harness.py` nor
  `orchestrator/tests/test_pid_alive_contract.py`: 6615 deleted `harness._pid_alive`, whose docstring
  was the only hit, and the test (measured at `1b106418da`).
- Keys: `fk-4fb8ffbb0141`.
- Files: (91 files) fused-memory/src/fused_memory/reconciliation/recon_self_model.py, fused-memory/src/fused_memory/recon_state/recon_self_model.py, fused-memory/src/fused_memory/reconciliation/graphiti_degradation_probe.py, fused-memory/src/fused_memory/recon_state/graphiti_degradation_probe.py, fused-memory/src/fused_memory/reconciliation/standing_decision_constants.py, fused-memory/src/fused_memory/recon_state/standing_decision_constants.py, fused-memory/src/fused_memory/reconciliation/recon_pool_map.py, fused-memory/src/fused_memory/recon_state/recon_pool_map.py, fused-memory/src/fused_memory/reconciliation/bulk_reset_guard.py, fused-memory/src/fused_memory/recon_state/bulk_reset_guard.py, fused-memory/src/fused_memory/reconciliation/backlog_policy.py, fused-memory/src/fused_memory/recon_state/backlog_policy.py, fused-memory/src/fused_memory/services/orchestrator_detector.py, fused-memory/src/fused_memory/utils/orchestrator_detector.py, escalation/tests/test_canonical.py, fused-memory/scripts/backfill_entity_standing_decision.py, fused-memory/scripts/cleanup_test_collections.py, fused-memory/src/fused_memory/middleware/curator_escalator.py, fused-memory/src/fused_memory/middleware/execution_class_guard.py, fused-memory/src/fused_memory/middleware/gitignored_deliverable_guard.py, fused-memory/src/fused_memory/middleware/operational_ask_registry.py, fused-memory/src/fused_memory/middleware/operational_suggestion_guard.py, fused-memory/src/fused_memory/middleware/premise_lint_guard.py, fused-memory/src/fused_memory/middleware/recon_write_policy.py, fused-memory/src/fused_memory/middleware/routing_intent_guard.py, fused-memory/src/fused_memory/middleware/task_interceptor.py, fused-memory/src/fused_memory/middleware/ticket_janitor.py, fused-memory/src/fused_memory/reconciliation/consolidation_gate.py, fused-memory/src/fused_memory/reconciliation/flag_dedup.py, fused-memory/src/fused_memory/reconciliation/harness.py, fused-memory/src/fused_memory/reconciliation/live_workflow_section.py, fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py, fused-memory/src/fused_memory/reconciliation/predicate_contradiction.py, fused-memory/src/fused_memory/reconciliation/preservation_specimen_guard.py, fused-memory/src/fused_memory/reconciliation/prompts/__init__.py, fused-memory/src/fused_memory/reconciliation/prompts/stage1.py, fused-memory/src/fused_memory/reconciliation/prompts/stage2.py, fused-memory/src/fused_memory/reconciliation/prompts/stage3.py, fused-memory/src/fused_memory/recon_state/recon_ledger.py, fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py, fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py, fused-memory/src/fused_memory/reconciliation/standing_decision_storm_escape.py, fused-memory/src/fused_memory/reconciliation/standing_decision_writer.py, fused-memory/src/fused_memory/reconciliation/summary_pool.py, fused-memory/src/fused_memory/reconciliation/sweep_deletion_guard.py, fused-memory/src/fused_memory/reconciliation/targeted.py, fused-memory/src/fused_memory/server/main.py, fused-memory/src/fused_memory/server/tools.py, fused-memory/src/fused_memory/services/live_workflow_detector.py, fused-memory/src/fused_memory/services/memory_service.py, fused-memory/tests/reconciliation/test_backlog_policy_offload.py, fused-memory/tests/reconciliation/test_stage1.py, fused-memory/tests/reconciliation/test_standing_decision_e2e.py, fused-memory/tests/reconciliation/test_standing_decision_growth_sweep.py, fused-memory/tests/reconciliation/test_standing_decision_writer.py, fused-memory/tests/server/test_get_cycle_summary_presence_tool.py, fused-memory/tests/server/test_recon_report_hook_b.py, fused-memory/tests/server/test_recon_report_standing_decision.py, fused-memory/tests/test_backfill_entity_standing_decision.py, fused-memory/tests/test_backlog_policy.py, fused-memory/tests/test_bulk_reset_guard.py, fused-memory/tests/test_execution_class_guard.py, fused-memory/tests/test_flag_dedup.py, fused-memory/tests/test_graphiti_degradation_probe.py, fused-memory/tests/test_halt_visibility.py, fused-memory/tests/test_harness.py, fused-memory/tests/test_mem0_tombstone.py, fused-memory/tests/test_memory_service.py, fused-memory/tests/test_merge_standing_decision_invalidation.py, fused-memory/tests/test_metadata_vocabulary_prompt_pinning.py, fused-memory/tests/test_operational_routing_boundary_matrix.py, fused-memory/tests/test_orchestrator_detector.py, fused-memory/tests/test_predicate_contradiction.py, fused-memory/tests/test_premise_lint_guard.py, fused-memory/tests/test_queue_stats_tool.py, fused-memory/tests/test_recon_gate_subject_guidance.py, fused-memory/tests/test_recon_ledger.py, fused-memory/tests/test_recon_report_stage_gated_tool_advertisement.py, fused-memory/tests/test_recon_self_model.py, fused-memory/tests/test_resolve_ticket_vocabulary.py, fused-memory/tests/test_stage2_narrative_reconstruction_retired.py, fused-memory/tests/test_stage2_tasks_created_accounting.py, fused-memory/tests/test_stage3_ledger_presence.py, fused-memory/tests/test_stages.py, fused-memory/tests/test_standing_decision_constants.py, fused-memory/tests/test_standing_decision_prompt_drift.py, fused-memory/tests/test_standing_decision_storm_escape.py, fused-memory/tests/test_stats_verifier.py, fused-memory/tests/test_summary_pool.py, fused-memory/tests/test_targeted.py, fused-memory/tests/test_task_interceptor.py
- Deps: α (shared importer files: `task_interceptor.py`, `memory_service.py`, `server/main.py`,
  `server/tools.py`, `reconciliation/harness.py`). Its edge onto task 6615 (`orchestrator_detector.py`
  body) is satisfied: 6615 is done (`f750373949`).
- Signal: as α. In addition, a `submit_task` through the fused-memory MCP whose shape
  `execution_class_guard` rejected before the move is rejected with the same error type after it (an
  existing guard suite, run against the moved module). B3 passes for the β1 modules.
- DELETES: the `services → orchestrator_detector` placement (one upward edge).

**β2 — Move task_filter, consolidation_gate and the flag/standing-decision contracts into recon_state**
(`task_kind=normal`, priority medium, est. ~600 changed LOC excluding renames)

- What: `git mv` the 5 modules (§Moves) and repoint importers, after the program tasks that edit their
  bodies have landed and this move has carried their code. Under S19/Q-C5 (a), also `git mv`
  `reconciliation/recon_prose_grammar.py`, which S13 δ1 creates by splitting `task_filter.py` and which
  middleware and services import (C2 puts it in recon_state). The files S13 δ1 and δ2 create or edit
  that import either module (`reconciliation/flag_filters/count_snapshot.py`,
  `flag_filters/task_state_refutations.py`, `tests/test_recon_prose_grammar.py`,
  `docs/recon-prose-grammar.md`) are declared at the paths S13's draft names; they do not exist at
  `1b106418da`, so the implementer re-greps the importers of both modules at dispatch. Task 6621 (done)
  left two test importers of `consolidation_gate`, now declared.
  `dependency_direction_check` is edited at `recon_state/dependency_direction_check.py`, where γ moved it.
- Keys: `fk-4fb8ffbb0141`.
- Files: (96 files) fused-memory/src/fused_memory/reconciliation/task_filter.py, fused-memory/src/fused_memory/recon_state/task_filter.py, fused-memory/src/fused_memory/reconciliation/consolidation_gate.py, fused-memory/src/fused_memory/recon_state/consolidation_gate.py, fused-memory/src/fused_memory/reconciliation/flag_record_contract.py, fused-memory/src/fused_memory/recon_state/flag_record_contract.py, fused-memory/src/fused_memory/reconciliation/stale_gate_citation_guard.py, fused-memory/src/fused_memory/recon_state/stale_gate_citation_guard.py, fused-memory/src/fused_memory/reconciliation/standing_decision_writer.py, fused-memory/src/fused_memory/recon_state/standing_decision_writer.py, docs/flag-marker-sweep-recurring.md, fused-memory/scripts/audit_duplicate_memories.py, fused-memory/scripts/audit_ruling_overreach.py, fused-memory/scripts/backfill_entity_standing_decision.py, fused-memory/scripts/cleanup_count_snapshots.py, fused-memory/scripts/measure_stale_status_snapshot_sweep.py, fused-memory/scripts/memory_eval_staleness_sweep.py, fused-memory/scripts/normalize_topic_slugs.py, fused-memory/src/fused_memory/recon_state/dependency_direction_check.py, fused-memory/src/fused_memory/middleware/task_interceptor.py, fused-memory/src/fused_memory/reconciliation/blocked_gate_audit_section.py, fused-memory/src/fused_memory/reconciliation/capped_task_list.py, fused-memory/src/fused_memory/reconciliation/citation_verifier.py, fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py, fused-memory/src/fused_memory/reconciliation/degenerate_task_node_sweep.py, fused-memory/src/fused_memory/reconciliation/flag_dedup.py, fused-memory/src/fused_memory/reconciliation/harness.py, fused-memory/src/fused_memory/reconciliation/live_workflow_section.py, fused-memory/src/fused_memory/reconciliation/prompts/__init__.py, fused-memory/src/fused_memory/reconciliation/prompts/stage1.py, fused-memory/src/fused_memory/reconciliation/prompts/stage2.py, fused-memory/src/fused_memory/reconciliation/stage1_payload_format.py, fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py, fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py, fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py, fused-memory/src/fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py, fused-memory/src/fused_memory/reconciliation/status_snapshot_extraction.py, fused-memory/src/fused_memory/reconciliation/targeted.py, fused-memory/src/fused_memory/server/main.py, fused-memory/src/fused_memory/server/recon_report.py, fused-memory/src/fused_memory/server/tools.py, fused-memory/src/fused_memory/services/completion_claim_gate.py, fused-memory/src/fused_memory/services/memory_service.py, fused-memory/tests/reconciliation/test_curator_gate_resolution_sweep.py, fused-memory/tests/reconciliation/test_degenerate_task_node_sweep.py, fused-memory/tests/reconciliation/test_enumeration_guard.py, fused-memory/tests/reconciliation/test_flag_record_contract.py, fused-memory/tests/reconciliation/test_gate_owned_finding_phrasing.py, fused-memory/tests/reconciliation/test_live_workflow_section.py, fused-memory/tests/reconciliation/test_recon_window_anchor.py, fused-memory/tests/reconciliation/test_remediation_tail_offload.py, fused-memory/tests/reconciliation/test_stage1.py, fused-memory/tests/reconciliation/test_stage1_auto_consolidation_wiring.py, fused-memory/tests/reconciliation/test_stage1_payload_section_parity.py, fused-memory/tests/reconciliation/test_stage1_run_failure_isolation.py, fused-memory/tests/reconciliation/test_stale_status_snapshot_edge_sweep.py, fused-memory/tests/reconciliation/test_standing_decision_writer.py, fused-memory/tests/reconciliation/test_task_count_snapshot_cadence.py, fused-memory/tests/reconciliation/test_task_metadata_selection.py, fused-memory/tests/server/test_add_memory_flag_record_schema_error.py, fused-memory/tests/test_audit_ruling_overreach.py, fused-memory/tests/test_auto_consolidation_e2e.py, fused-memory/tests/test_backfill_entity_standing_decision.py, fused-memory/tests/test_check_consolidation_closure_cli.py, fused-memory/tests/test_consolidation_auto.py, fused-memory/tests/test_consolidation_auto_executor.py, fused-memory/tests/test_consolidation_closure_seam.py, fused-memory/tests/test_consolidation_gate.py, fused-memory/tests/test_flag_record_write_seam.py, fused-memory/tests/test_harness.py, fused-memory/tests/test_live_workflow_authority.py, fused-memory/tests/test_memory_eval_staleness_sweep.py, fused-memory/tests/test_recon_self_model.py, fused-memory/tests/test_recon_suppression_prompt_retirement.py, fused-memory/tests/test_stage1_consolidation_guidance.py, fused-memory/tests/test_stages.py, fused-memory/tests/test_stale_gate_citation_guard.py, fused-memory/tests/test_status_vocab_consolidation.py, fused-memory/tests/test_task_filter.py, fused-memory/tests/test_task_interceptor.py, orchestrator/src/orchestrator/config_census_ignore.py, orchestrator/tests/test_scheduler.py, scripts/check_consolidation_closure.py, scripts/migrate_open_consolidation_gates.py, scripts/tests/test_migrate_open_consolidation_gates.py, shared/src/shared/task_statuses.py, shared/tests/test_task_statuses.py, fused-memory/src/fused_memory/reconciliation/recon_prose_grammar.py, fused-memory/src/fused_memory/recon_state/recon_prose_grammar.py, fused-memory/src/fused_memory/reconciliation/flag_filters/count_snapshot.py, fused-memory/src/fused_memory/reconciliation/flag_filters/task_state_refutations.py, fused-memory/tests/test_recon_prose_grammar.py, fused-memory/tests/test_audit_duplicate_memories.py, docs/recon-prose-grammar.md, fused-memory/tests/test_operational_routing_boundary_matrix.py, fused-memory/tests/test_operational_routing_guard.py
- Deps: β1; γ (`dependency_direction_check` lives in recon_state by then); tasks 6599, 6721, 9203,
  9240, 9254 (S19/OQ-5 (a); the edge onto 6621 is satisfied, done `9c3f608c88`); S13 δ2, the
  reconciliation-sweeps plan's status-lexicon leaf (held on S13/δ2; its task id is wired when S13
  files; S19/Q-C5 (a)). Task 9285 depends on β2 instead (it waits on
  8528; D8). β2 never depends on S13 κ3: S13 κ3 → β2 (it edits `flag_record_contract.py` and
  `consolidation_gate.py` at their recon_state paths), so the reverse edge would be a two-task cycle.
- Signal: as α. B3 passes for every recon_state module. Middleware imports no reconciliation module:
  `git grep -nE '^\s*(from|import) fused_memory\.reconciliation' fused-memory/src/fused_memory/middleware` is empty.
- DELETES: the last services/middleware → reconciliation edges (F1: middleware→reconciliation 16 and
  services→reconciliation 10, together with α, β1 and γ).

**γ — A low `fused_memory.escalators` package for the write-path escalation filers; cut the remaining middleware and backends upward edges**
(`task_kind=normal`, priority medium, est. ~700 changed LOC excluding renames)

- What:
  - C10: move the five filers to `escalators/` (S19/OQ-4 (b)), with `_folded_escalation` becoming
    `folded_escalation.file_folded_escalation`;
  - move `dead_letter_escalator` to `services/dead_letter_escalator.py` (D5: since task 9143 it imports
    `services.durable_queue` and `services.queued_writes` at module level, measured at `1b106418da`);
    `POST_EXECUTE_DEAD_PREFIX` stays in `durable_queue`;
  - move `candidate_key` to `utils/candidate_key.py`;
  - move `dependency_direction_check` and `recon_claim_verification_guard` to `recon_state`;
  - move `recon_archive_root` to a recon_state module and `estimate_tokens` to `utils/token_estimate.py`,
    with no shim (D6): their importers `reconciliation/harness.py`, `reconciliation/stages/base.py`,
    `tests/test_harness.py`, `tests/reconciliation/test_cli_stage_runner_cutover.py`,
    `tests/test_config_reload.py`, `tests/test_task_curator_config_dir_archival.py`,
    `tests/test_context_assembler.py`, and the bake-off script and its test (function-local imports of
    `estimate_tokens`) are repointed (measured at `1b106418da`; the other mentions are prose, D6);
  - replace `task_interceptor`'s `TYPE_CHECKING` `TargetedReconciler` with a Protocol in middleware;
  - repoint all importers, `docs/legibility/exception-ratifications.yaml` and
    `skills/curate-fused-memories/SKILL.md` included.
- Keys: `fk-4fb8ffbb0141`.
- Files: (69 files) fused-memory/src/fused_memory/escalators/__init__.py, fused-memory/src/fused_memory/escalators/folded_escalation.py, fused-memory/src/fused_memory/services/dead_letter_escalator.py, fused-memory/src/fused_memory/escalators/entity_mint_storm_escalator.py, fused-memory/src/fused_memory/escalators/mem0_update_storm_escalator.py, fused-memory/src/fused_memory/escalators/referent_repair_storm_escalator.py, fused-memory/src/fused_memory/escalators/candidate_key_escalation.py, fused-memory/src/fused_memory/utils/candidate_key.py, fused-memory/src/fused_memory/recon_state/dependency_direction_check.py, fused-memory/src/fused_memory/recon_state/recon_claim_verification_guard.py, fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py, fused-memory/src/fused_memory/reconciliation/context_assembler.py, fused-memory/src/fused_memory/utils/token_estimate.py, docs/legibility/exception-ratifications.yaml, fused-memory/src/fused_memory/backends/sqlite_task_backend.py, fused-memory/src/fused_memory/maintenance/link_heal_executor.py, fused-memory/src/fused_memory/middleware/_folded_escalation.py, fused-memory/src/fused_memory/middleware/candidate_key_escalation.py, fused-memory/src/fused_memory/middleware/dead_letter_escalator.py, fused-memory/src/fused_memory/middleware/entity_mint_storm_escalator.py, fused-memory/src/fused_memory/middleware/mem0_update_storm_escalator.py, fused-memory/src/fused_memory/middleware/recon_claim_verification_guard.py, fused-memory/src/fused_memory/middleware/referent_repair_storm_escalator.py, fused-memory/src/fused_memory/middleware/task_curator.py, fused-memory/src/fused_memory/middleware/task_interceptor.py, fused-memory/src/fused_memory/middleware/ticket_janitor.py, fused-memory/src/fused_memory/server/markup_tripwire.py, fused-memory/src/fused_memory/server/write_triage.py, fused-memory/src/fused_memory/services/classification_fallback_alarm.py, fused-memory/src/fused_memory/services/completion_claim_gate.py, fused-memory/src/fused_memory/services/journal_growth_alarm.py, fused-memory/src/fused_memory/services/mem0_retry_backlog_alarm.py, fused-memory/src/fused_memory/services/memory_metadata_census.py, fused-memory/src/fused_memory/services/memory_service.py, fused-memory/tests/middleware/test_dead_letter_escalator.py, fused-memory/tests/middleware/test_entity_mint_storm_escalator.py, fused-memory/tests/middleware/test_mem0_update_storm_escalator.py, fused-memory/tests/test_add_memory_classification_fallback.py, fused-memory/tests/test_candidate_key.py, fused-memory/tests/test_candidate_key_escalation.py, fused-memory/tests/test_classification_fallback_alarm.py, fused-memory/tests/test_consolidation_gate.py, fused-memory/tests/test_curation_deferral_acceptance.py, fused-memory/tests/test_dependency_direction_check.py, fused-memory/tests/test_e2e_durable_queue.py, fused-memory/tests/test_folded_escalation.py, fused-memory/tests/test_fused_memory_governed_lists.py, fused-memory/tests/test_journal_growth_alarm.py, fused-memory/tests/test_link_heal_apply.py, fused-memory/tests/test_memory_service.py, fused-memory/tests/test_recon_claim_verification_guard.py, fused-memory/tests/test_recon_claim_verification_wiring.py, fused-memory/tests/test_referent_repair_storm_escalator.py, fused-memory/tests/test_sqlite_task_backend.py, fused-memory/tests/test_task_curator.py, fused-memory/tests/test_ticket_parking.py, orchestrator/src/orchestrator/agents/triage.py, scripts/tests/test_exception_owner_sweep.py, skills/curate-fused-memories/SKILL.md, fused-memory/src/fused_memory/reconciliation/harness.py, fused-memory/src/fused_memory/reconciliation/stages/base.py, fused-memory/tests/test_harness.py, fused-memory/tests/reconciliation/test_cli_stage_runner_cutover.py, fused-memory/tests/test_config_reload.py, fused-memory/tests/test_task_curator_config_dir_archival.py, fused-memory/tests/test_context_assembler.py, fused-memory/scripts/bake_off_storage_shape.py, fused-memory/tests/test_bake_off_storage_shape.py, fused-memory/tests/test_task_created_at.py
- Deps: β1; tasks 6622, 6633, 9204 (`task_curator.py` body) and 9147 (`dead_letter_escalator.py` body)
  (S19/OQ-5 (a)). The edge onto 9143 is satisfied (done, `3983fa2265`).
- Signal: B4. A fresh interpreter importing `fused_memory.services.memory_service` loads no
  `fused_memory.middleware.*` module. The live server restarts and its next storm-escalator or
  dead-letter filing lands in the escalation queue unchanged (existing escalator suites, through their
  public entry points).
- DELETES: 13 services/backends → middleware edges (services 11, backends 2, F1); the false underscore
  on `_folded_escalation`. (The deferred import in `dead_letter_escalator` the draft listed here was
  hoisted by task 9143, `204de188c4`.)

**δ — Demote server/: grouped-read assembly, consolidation and mem0-update authz into services**
(`task_kind=normal`, priority medium, est. ~650 changed LOC excluding renames)

- What:
  - move `server/grouped_read.py`, `server/consolidation.py` and `server/mem0_update_authz.py` to
    `services/`;
  - make `memory_service._MEM0_CONTENT_KEYS` public at its owner (the private reach in `grouped_read`
    goes);
  - repoint the prose citations of the three moved modules in `config/reload.py` and `config/schema.py`
    (2 each, measured at `1b106418da`), which S16 β2 relies on before it deletes `reload.py`.

  The `server/storm_counter.py` shim the draft had δ delete is gone: task 6547 deleted it
  (`0f86a2cdcc`) and repointed its prose to `shared.storm_counter`.

  δ does not cut `reconciliation/prompts/__init__.py`'s deferred `server.recon_report` import: the
  reconciliation-lifecycle plan's stage-capabilities leaf (S12 ι1) owns that cut, and ε waits on it
  (S19/Q-C4 (a)). δ therefore does not edit `server/main.py`, `server/recon_report.py` or
  `reconciliation/prompts/__init__.py`, none of which references a module δ moves (measured at
  `1b106418da`). This subsumes task 5983.
- Keys: `fk-420bc8f9a425`.
- Files: (66 files) fused-memory/src/fused_memory/services/grouped_read.py, fused-memory/src/fused_memory/services/consolidation.py, fused-memory/src/fused_memory/services/mem0_update_authz.py, fused-memory/src/fused_memory/services/memory_service.py, fused-memory/src/fused_memory/config/reload.py, fused-memory/src/fused_memory/config/schema.py, fused-memory/scripts/attribute_write_triage_soak_misfiles.py, fused-memory/scripts/eval_link_adjudicator.py, fused-memory/scripts/memory_eval_write_after_miss.py, fused-memory/scripts/write_triage_reader_rubric.py, fused-memory/src/fused_memory/maintenance/link_heal.py, fused-memory/src/fused_memory/maintenance/link_heal_store.py, fused-memory/src/fused_memory/reconciliation/child_memory_link.py, fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py, fused-memory/src/fused_memory/reconciliation/prompts/stage1.py, fused-memory/src/fused_memory/server/consolidation.py, fused-memory/src/fused_memory/server/entity_mint_authz.py, fused-memory/src/fused_memory/server/tools.py, fused-memory/src/fused_memory/server/write_triage.py, fused-memory/src/fused_memory/server/write_triage_judge.py, fused-memory/src/fused_memory/services/consolidation_ops.py, fused-memory/tests/_link_heal_harness.py, fused-memory/tests/reconciliation/test_stage1_child_memories.py, fused-memory/tests/server/test_add_memory_write_triage_gate.py, fused-memory/tests/server/test_amendment_bodies_search_reply.py, fused-memory/tests/server/test_contesting_child_search_reply.py, fused-memory/tests/server/test_grouped_read.py, fused-memory/tests/server/test_grouped_read_amendment_text.py, fused-memory/tests/server/test_grouped_read_supersession.py, fused-memory/tests/server/test_update_memory_authz_gate.py, fused-memory/tests/server/test_write_triage.py, fused-memory/tests/server/test_write_triage_attach_guard.py, fused-memory/tests/server/test_write_triage_flip_gate_invariants.py, fused-memory/tests/server/test_write_triage_judge.py, fused-memory/tests/server/test_write_triage_judged_attach.py, fused-memory/tests/server/test_write_triage_supersession.py, fused-memory/tests/test_build_write_triage_v2_soak_baseline.py, fused-memory/tests/test_calibrate_write_triage_recall.py, fused-memory/tests/test_config_reload.py, fused-memory/tests/test_consolidate_memories_tool.py, fused-memory/tests/test_consolidate_memories_validation.py, fused-memory/tests/test_eval_link_adjudicator.py, fused-memory/tests/test_eval_write_triage_reranker.py, fused-memory/tests/test_eval_write_triage_retrieval.py, fused-memory/tests/test_link_heal_adjudicated.py, fused-memory/tests/test_link_heal_apply.py, fused-memory/tests/test_link_heal_cli.py, fused-memory/tests/test_link_heal_completion_records.py, fused-memory/tests/test_link_heal_config.py, fused-memory/tests/test_link_heal_ledger.py, fused-memory/tests/test_link_heal_plan.py, fused-memory/tests/test_link_heal_table.py, fused-memory/tests/test_link_heal_undo.py, fused-memory/tests/test_topic_anchor.py, fused-memory/tests/test_write_triage_rating_rounds.py, fused-memory/tests/test_write_triage_reader_rubric.py, orchestrator/src/orchestrator/agents/memory_recall.py, orchestrator/src/orchestrator/agents/roles.py, orchestrator/tests/_briefing_helpers.py, orchestrator/tests/test_memory_recall.py, orchestrator/tests/test_roles_memory_amend_routing.py, scripts/exception_register.py, scripts/migrate_open_consolidation_gates.py, scripts/tests/test_design_invariants_consistency.py, skills/curate-fused-memories/SKILL.md, fused-memory/tests/server/test_write_triage_completion_records.py
- Deps: β2 (`server/tools.py`, `services/memory_service.py`, `reconciliation/prompts/stage1.py`,
  `consolidation_auto_executor.py`); task 6720 (S14 ν); tasks 9243 and 6721 (S19/OQ-5 (a)); δ → S12 ι1 (conditional, held on S12/ι1): only if
  ι1 keeps its `server/grouped_read.py` re-bind, which δ then carries.
- Signal: B5. The `consolidate_memories` and grouped `search` MCP tools answer as before against the
  live server. No module imports the three moved server modules at their server paths:
  `git grep -nE '^\s*(from|import) fused_memory\.server\.(grouped_read|consolidation|mem0_update_authz)\b' fused-memory/src`
  is empty (13 statements in 9 files at `1b106418da`). (The last lower import of `server`, the deferred `recon_report` one, goes with S12 ι1;
  ε's guard covers the whole rule.)
- DELETES: `grouped_read`'s private import; the three server-path modules (moved).

**ε — Invariant `fused-memory-subpackage-dag`: declared layers, the guard, and the four-site lockstep edit (integration gate)**
(`task_kind=normal`, priority medium, est. ~550 LOC)

- What: create `fused_memory/layers.py` (C3), with a one-line pointer in `fused_memory/__init__.py`'s
  docstring. Promote the snapshot's target resolution and SCC to public functions in
  `scripts/source_measures.py`: `import_targets(statement, importer, is_package, known)` (from
  `quality_metrics_snapshot.py::_targets`, `_from_targets`, `_from_module`, `_from_base`) and
  `strongly_connected_components(edges)` (from `::_cycles`); `quality_metrics_snapshot.py` calls them
  and `tests/scripts/test_source_measures.py` pins them through the public names. Add the guard
  `fused-memory/tests/test_package_layering.py` (C3 rule over those two functions; B1 fixtures; B2).
  Make the four-site lockstep edit (§Design invariant). Out-of-program pending tasks that declare
  `scripts/source_measures.py` or `scripts/quality_metrics_snapshot.py` (6819, 6941, 6982, 6986, 7286,
  8090, 8324, 8402, 8684; all pending at `7848173a7d`) are rebase-only (D8).
- Keys: `fk-4fb8ffbb0141`.
- Files: (9 files) fused-memory/src/fused_memory/layers.py, fused-memory/src/fused_memory/__init__.py, fused-memory/tests/test_package_layering.py, docs/legibility/design-invariants.md, docs/legibility/design-invariants-fixtures.md, skills/prd/references/gates.md, scripts/source_measures.py, scripts/quality_metrics_snapshot.py, tests/scripts/test_source_measures.py
- Deps: δ (and through it α, β1, γ, β2); S12 ι1, the reconciliation-lifecycle plan's
  stage-capabilities leaf, which removes the last upward `reconciliation → server` import (held on
  S12/ι1; its task id is wired when S12 files; S19/Q-C4 (a)). For the three invariant files, the lead
  wires the invariant-file chain of the six wave-3 minting tasks at decompose (INV-18 to INV-23 in
  landing order, `one-decision-partition-per-area` last as INV-23); no edge onto 9154 is proposed.
- Signal: `uv run --project fused-memory pytest fused-memory/tests/test_package_layering.py` passes on
  main. Each B1 fixture fails with the edge named. `quality_metrics_snapshot.py` and the guard report
  the same fused-memory edge set (both call `source_measures.import_targets`).
  `uv run --project shared pytest scripts/tests/test_design_invariants_consistency.py` passes with the
  new section present at every pinned site.
- DELETES: the snapshot's private `_targets`/`_cycles` pair as a second edge builder (it becomes the
  public one). Cross-reference for S13 (task 5996 owns `flag_dedup.py`): after β2,
  `flag_dedup`'s two local copies of `task_filter` logic (`COUNT_SNAPSHOT_RE` and an extraction
  routine) can import `recon_state.task_filter`, because `task_filter` then sits below reconciliation.
- G7: INV-10 holds: the guard runs against fixtures. INV-12 holds: there is no exemption list. INV-2
  holds: the message names the values.

**ζ — `shared.sandbox`: landlock, bwrap and backend resolution in one stdlib-only package**
(`task_kind=normal`, priority medium, est. ~650 LOC)

- What: C4. Move `landlock.py` and `landlock_exec.py`; move `agents/sandbox.py` to `bwrap.py`. Add
  `backends.py` holding the resolution and the argv wrap that today live in
  `sandbox_dispatch._resolve_auto`, `resolve_active_backend` and the per-backend branch of
  `wrap_command`.
  - `sandbox_dispatch` keeps its process-global preference, `resolve_backend_or_refuse` with its dedup,
    and `wrap_command`, delegating to shared.
  - `sandbox_guard` imports shared at module level and keeps the containment check and
    `RemediationSandboxUnavailable` for `'none'`.
  - The two type-ignored imports, the "mirrors sandbox_dispatch" sentence and the venv-precondition
    block are deleted.
  - Register the new modules in `shared/tests/test_pure_stdlib_leaves.py`.
  - Move `orchestrator/tests/test_landlock.py` to `shared/tests/test_landlock.py` and
    `orchestrator/tests/test_sandbox.py` to `shared/tests/test_bwrap.py` (§Moves rows, which the
    clerk's repath of task 9232's file list reads).
- Keys: `fk-64e06774ee0e`.
- Files: (32 files) shared/src/shared/sandbox/__init__.py, shared/src/shared/sandbox/backends.py, shared/src/shared/sandbox/landlock.py, shared/src/shared/sandbox/landlock_exec.py, shared/src/shared/sandbox/bwrap.py, orchestrator/src/orchestrator/agents/sandbox.py, shared/tests/test_sandbox_backends.py, shared/tests/test_landlock.py, shared/tests/test_bwrap.py, shared/tests/test_pure_stdlib_leaves.py, conftest.py, orchestrator/tests/conftest.py, docs/sandbox-fleet-status.md, fused-memory/src/fused_memory/reconciliation/sandbox_guard.py, fused-memory/tests/reconciliation/test_cli_stage_runner_cutover.py, fused-memory/tests/reconciliation/test_cli_stage_runner_sandbox.py, fused-memory/tests/reconciliation/test_recon_sandbox_guard.py, fused-memory/tests/test_stages.py, orchestrator/src/orchestrator/agents/landlock.py, orchestrator/src/orchestrator/agents/landlock_exec.py, orchestrator/src/orchestrator/agents/sandbox_dispatch.py, orchestrator/tests/test_dry_run_unblock.py, orchestrator/tests/test_git_ops.py, orchestrator/tests/test_invoke.py, orchestrator/tests/test_sandbox_dispatch.py, orchestrator/tests/test_sandbox_enforcement_matrix.py, orchestrator/tests/test_shadow_invocation_sandbox.py, orchestrator/tests/test_workflow_sandbox_refusal.py, scripts/tests/test_sitting_nightly_prepare.py, orchestrator/tests/test_landlock.py, orchestrator/tests/test_sandbox.py, orchestrator/tests/_landlock_required.py
- Deps: none. α, θ1 and ι share only test files with ζ (`fused-memory/tests/test_stages.py`,
  `shared/tests/test_pure_stdlib_leaves.py`, `orchestrator/tests/test_invoke.py`,
  `orchestrator/tests/test_shadow_invocation_sandbox.py`) and take no edge on it (lead ruling on
  test-file edges; R54). Tasks 5345, 4444 and 9232 depend on ζ (lead actions). After ζ, 4444 and 5345 fix one probe, `shared.sandbox.bwrap.is_bwrap_available`, reached
  from two event loops, so 5345 is amended to fold into 4444 (lead action). The other pending tasks
  that edit the moved sandbox bodies are rebase-only and the clerk repaths them (D8): 4627 and 4840
  (medium), 6852 and 7607 (low), and 3419 (deferred), measured by `find_tasks_by_metadata` on
  2026-11-11.
- Signal: B6. The next live reconciliation cycle logs its stage agents' sandbox as `landlock`. The
  orchestrator's next sandboxed dispatch still wraps with Landlock (its existing sandbox event).
  `git grep -nE '^\s*from orchestrator\.' fused-memory/src` is empty.
- DELETES: `orchestrator/src/orchestrator/agents/{landlock,landlock_exec,sandbox}.py` (moved); 2
  type-ignored imports; the hand-mirrored resolution order; the venv-install precondition prose.
- G7: INV-11 holds. Fail-closed stays fail-closed on both sides, and B6 pins it.

**η1 — Split session_registry.py in place, step 1: session records and identity modules**
(`task_kind=normal`, priority medium, est. ~1,300 changed LOC, mostly moved lines)

- What: extract the records partition into `orchestrator/src/orchestrator/session_records.py`: schema
  constants, `ResultOutcome`, `ExitRequest`, `Status`, `Display`, `Question`, `SessionRecord`, slugs, the
  pid/claude/T3 pointers, read/write/update/refresh, `encode_cwd` and `transcript_path_for_cwd`. Extract
  the identity cluster into `session_identity.py`. Use 6179's dual-mode sibling import (D9).
  `session_hooks` imports `shared.proc_liveness.pid_alive` directly, which subsumes task 9100.
  `TestStdlibOnlySelfContainment` gains rows for the new modules. Every consumer of a moved name is
  repointed (no shim). Its one hot file, `orchestrator/src/orchestrator/workflow.py`, takes only the
  `build_session_slug` import line (rebase-only, S19/Q-C6 (a)).
- Keys: `fk-cfcf00669c46`.
- Files: (59 files) orchestrator/src/orchestrator/session_registry.py, orchestrator/src/orchestrator/session_records.py, orchestrator/src/orchestrator/session_identity.py, cockpit/src/cockpit/app.py, cockpit/src/cockpit/backends/base.py, cockpit/src/cockpit/panes/decision_queue.py, cockpit/src/cockpit/panes/detail_pane.py, cockpit/src/cockpit/panes/session_table.py, cockpit/src/cockpit/panes/spawn_tree.py, cockpit/src/cockpit/panes/weight_editor.py, cockpit/src/cockpit/registry_reader.py, cockpit/src/cockpit/ui_config.py, cockpit/tests/test_app.py, cockpit/tests/test_backends_base.py, cockpit/tests/test_decision_queue.py, cockpit/tests/test_detail_pane.py, cockpit/tests/test_project_token_coherence.py, cockpit/tests/test_registry_reader.py, cockpit/tests/test_session_table.py, cockpit/tests/test_spawn_tree.py, cockpit/tests/test_ui_config.py, cockpit/tests/test_weight_editor.py, orchestrator/src/orchestrator/session_hooks.py, orchestrator/src/orchestrator/session_self_exit.py, orchestrator/src/orchestrator/spawn_t3_lane.py, orchestrator/src/orchestrator/t3_mcp_client.py, orchestrator/src/orchestrator/workflow.py, orchestrator/tests/_session_hook_fixtures.py, orchestrator/tests/test_session_hooks.py, orchestrator/tests/test_session_hooks_bash.py, orchestrator/tests/test_session_hooks_display.py, orchestrator/tests/test_session_hooks_launch_window.py, orchestrator/tests/test_session_hooks_ownership.py, orchestrator/tests/test_session_hooks_single_read.py, orchestrator/tests/test_session_launch_session_id.py, orchestrator/tests/test_session_lease_shared_host.py, orchestrator/tests/test_session_project_from_checkout.py, orchestrator/tests/test_session_record_linkage.py, orchestrator/tests/test_session_registry.py, orchestrator/tests/test_session_registry_spawn_contract.py, orchestrator/tests/test_session_self_exit.py, orchestrator/tests/test_spawn_t3_lane.py, orchestrator/tests/test_workflow_e2e.py, scripts/legibility/check_transcript_persistence.py, scripts/legibility/digest.py, scripts/legibility/inventory.py, scripts/sitting/apply.py, scripts/sitting/ownership.py, scripts/sitting/prepare_sitting.py, scripts/sitting/return_brief.py, scripts/tests/test_legibility_inventory.py, scripts/tests/test_legibility_nightly.py, scripts/tests/test_legibility_transcript_persistence.py, scripts/tests/test_prepare_sitting_cli.py, scripts/tests/test_sitting_ownership.py, skills/review-all/scripts/program/gen_brief.py, skills/review-all/scripts/program/watch_sessions.py, tests/scripts/test_review_all_program_state.py, tests/scripts/test_spawn_claude.py
- Deps: task 6179 (decisions out first, same file). The edge onto task 6615 (`session_registry.py` pid
  predicate) is satisfied: 6615 is done (`f750373949`).
- Signal: `skills/spawn/spawn-claude.sh` launches a session from a bare shell and its `record.json`
  appears (`tests/scripts/test_spawn_claude.py` plus a manual launch). The cockpit session table renders
  live records. B7's tier-1 rows pass.
- DELETES: `session_registry.pid_alive`'s pass-through (task 9100's finding).

**η2 — Split session_registry.py in place, step 2: leases, reaper and one window-helper module; the hook window probe as a seam**
(`task_kind=normal`, priority medium, est. ~1,400 changed LOC, mostly moved lines)

- What: extract `session_leases.py`, `session_reaper.py` and `session_windows.py` (C6). The window
  sweep (`mark_windowless_wm_sessions_exited`, `_wm_window_id_of`, `_names_window`) goes to
  `session_reaper.py`, because it calls the reaper's `_mark_exited_if_still_non_terminal` (reaper →
  windows; `session_windows` stays a leaf, C5). `session_hooks` imports `session_windows` for the wmctrl
  runner, the parse, the canonical id and the resolve-by-title. Its entry points take
  `windows: WindowProbe | None` beside `process_tree`. The circularity docstring is corrected.
  `session_registry.py` keeps only the CLI runners and parser. Lease verbs stay reachable at the same
  path. `orchestrator/src/orchestrator/service_restart.py` is not declared: since task 6615 it has no
  `session_registry` reference (measured at `1b106418da`).
- Keys: `fk-cfcf00669c46`, `fk-57599c7fd907`, `fk-ea4463b19d86`.
- Files: (19 files) orchestrator/src/orchestrator/session_registry.py, orchestrator/src/orchestrator/session_leases.py, orchestrator/src/orchestrator/session_reaper.py, orchestrator/src/orchestrator/session_windows.py, orchestrator/src/orchestrator/session_hooks.py, orchestrator/tests/test_session_hooks_display.py, orchestrator/tests/test_session_hooks_ownership.py, docs/session-hook-ownership.md, orchestrator/tests/test_lease_holder_body_types.py, orchestrator/tests/test_session_lease_shared_host.py, orchestrator/tests/test_session_record_linkage.py, orchestrator/tests/test_session_registry.py, orchestrator/src/orchestrator/session_self_exit.py, orchestrator/src/orchestrator/spawn_t3_lane.py, orchestrator/tests/_reap_report_wait.py, orchestrator/tests/test_spawn_t3_lane.py, skills/review-all/scripts/program/watch_sessions.py, tests/scripts/test_spawn_claude.py, orchestrator/tests/test_session_hooks_launch_window.py
- Deps: η1. Task 7902 depends on η2 (lead action: 7902 keeps its install and single-read families).
- Signal: `python3 …/session_registry.py lease-claim` / `lease-show` / `lease-reap` run from a bare
  shell, as the watcher skills invoke them. B9's orchestrator half passes. The 8
  `_resolve_wm_window_id` patches are gone:
  `git grep -nE "_resolve_wm_window_id|_wmctrl_list|_normalize_wm_window_id" orchestrator` is empty.
- DELETES: one `_wmctrl_list` copy, one `_WMCTRL_TIMEOUT_SECS`, `_canonical_window_id` and
  `_normalize_wm_window_id` (one form remains, D10); the second `wmctrl -l` line parser.

**θ1 — Move the stdlib-only session modules to `shared.session`; repoint every consumer**
(`task_kind=normal`, priority medium, est. ~800 changed LOC excluding renames)

- What: C5. `git mv` the seven modules (§Moves). The dual-mode imports become `from shared.session
  import …` through `session_registry.py`'s existing bootstrap. Repoint cockpit, scripts, skills,
  orchestrator and tests. Register the modules in `shared/tests/test_pure_stdlib_leaves.py`, and point
  `TestStdlibOnlySelfContainment`'s module rows at the shared package. The command lines in
  `spawn-claude.sh`, the skills and `OPERATIONS.md` are unchanged. Also `git mv`
  `shared/src/shared/session_liveness.py` to `shared/session/liveness.py` (C5) and repoint its 7
  importers. `tests/scripts/test_atomic_write_regrowth.py` keys the atomic writer by path; 6179's plan
  puts that key at `orchestrator/src/orchestrator/fleet_files.py`, which θ1 moves, so θ1 rewrites the
  key (the file is shared with α and is rebase-only, R54). `workflow.py` takes only the
  `build_session_slug` import line (rebase-only, S19/Q-C6 (a)). `orchestrator/src/orchestrator/service_restart.py`
  is not declared (6615 removed its `session_registry` import).
- Keys: `fk-d91317f88f21`.
- Files: (105 files) orchestrator/src/orchestrator/fleet_files.py, orchestrator/src/orchestrator/decision_registry.py, orchestrator/src/orchestrator/session_records.py, orchestrator/src/orchestrator/session_identity.py, orchestrator/src/orchestrator/session_leases.py, orchestrator/src/orchestrator/session_reaper.py, orchestrator/src/orchestrator/session_windows.py, shared/src/shared/session/__init__.py, shared/src/shared/session/fleet_files.py, shared/src/shared/session/decisions.py, shared/src/shared/session/records.py, shared/src/shared/session/identity.py, shared/src/shared/session/leases.py, shared/src/shared/session/reaper.py, shared/src/shared/session/windows.py, orchestrator/src/orchestrator/session_registry.py, orchestrator/src/orchestrator/decision_cli.py, orchestrator/src/orchestrator/session_hooks.py, shared/tests/test_pure_stdlib_leaves.py, orchestrator/tests/test_fleet_files.py, orchestrator/tests/test_decision_registry.py, orchestrator/tests/test_decision_cli_write.py, orchestrator/tests/test_decision_cli_lifecycle.py, orchestrator/tests/_decision_fixtures.py, orchestrator/tests/test_decision_audit_outcome.py, cockpit/src/cockpit/app.py, cockpit/src/cockpit/backends/base.py, cockpit/src/cockpit/panes/decision_queue.py, cockpit/src/cockpit/panes/detail_pane.py, cockpit/src/cockpit/panes/session_table.py, cockpit/src/cockpit/panes/spawn_tree.py, cockpit/src/cockpit/panes/weight_editor.py, cockpit/src/cockpit/registry_reader.py, cockpit/src/cockpit/ui_config.py, cockpit/tests/test_app.py, cockpit/tests/test_backends_base.py, cockpit/tests/test_decision_queue.py, cockpit/tests/test_detail_pane.py, cockpit/tests/test_project_token_coherence.py, cockpit/tests/test_registry_reader.py, cockpit/tests/test_session_table.py, cockpit/tests/test_spawn_tree.py, cockpit/tests/test_ui_config.py, cockpit/tests/test_weight_editor.py, orchestrator/src/orchestrator/session_self_exit.py, orchestrator/src/orchestrator/spawn_t3_lane.py, orchestrator/src/orchestrator/t3_mcp_client.py, orchestrator/src/orchestrator/workflow.py, orchestrator/tests/_session_hook_fixtures.py, orchestrator/tests/test_session_hooks.py, orchestrator/tests/test_session_hooks_bash.py, orchestrator/tests/test_session_hooks_display.py, orchestrator/tests/test_session_hooks_launch_window.py, orchestrator/tests/test_session_hooks_ownership.py, orchestrator/tests/test_session_hooks_single_read.py, orchestrator/tests/test_session_launch_session_id.py, orchestrator/tests/test_session_lease_shared_host.py, orchestrator/tests/test_session_project_from_checkout.py, orchestrator/tests/test_session_record_linkage.py, orchestrator/tests/test_session_registry.py, orchestrator/tests/test_session_registry_spawn_contract.py, orchestrator/tests/test_session_self_exit.py, orchestrator/tests/test_spawn_t3_lane.py, orchestrator/tests/test_workflow_e2e.py, scripts/legibility/check_transcript_persistence.py, scripts/legibility/digest.py, scripts/legibility/inventory.py, scripts/sitting/apply.py, scripts/sitting/ownership.py, scripts/sitting/prepare_sitting.py, scripts/sitting/return_brief.py, scripts/tests/test_legibility_inventory.py, scripts/tests/test_legibility_nightly.py, scripts/tests/test_legibility_transcript_persistence.py, scripts/tests/test_prepare_sitting_cli.py, scripts/tests/test_sitting_ownership.py, skills/review-all/scripts/program/gen_brief.py, skills/review-all/scripts/program/watch_sessions.py, tests/scripts/test_review_all_program_state.py, tests/scripts/test_spawn_claude.py, orchestrator/tests/test_session_registry_audit_outcome.py, scripts/backfill_decision_queue_stamp.py, scripts/sitting/audits.py, scripts/sitting/fleet_state.py, scripts/sitting/inventory.py, scripts/sitting/payloads.py, scripts/sitting/ports.py, scripts/tests/test_scripts_layout_parity.py, scripts/tests/test_sitting_apply.py, scripts/tests/test_sitting_audits.py, scripts/tests/test_sitting_fleet_state.py, scripts/tests/test_sitting_inventory.py, scripts/tests/test_sitting_payloads.py, scripts/tests/test_sitting_ports.py, scripts/tests/test_sitting_return_brief.py, tests/scripts/test_backfill_decision_queue_stamp.py, orchestrator/tests/test_lease_holder_body_types.py, orchestrator/tests/_reap_report_wait.py, orchestrator/tests/test_eval_spawn_identity.py, tests/scripts/test_atomic_write_regrowth.py, shared/src/shared/session_liveness.py, shared/src/shared/session/liveness.py, shared/tests/test_session_liveness.py, orchestrator/tests/_process_identity_variants.py, docs/session-hook-ownership.md
- Deps: η2; task 6760 (S25 ε: scripts import `shared`
  through `_checkout`).
- Signal: B7. The cockpit renders sessions and decisions. `skills/spawn/spawn-claude.sh` launches a
  session that registers. The escalation-watcher's `write-decision` and `reap-decisions` verbs run from
  a bare shell. No cockpit file imports any orchestrator module:
  `git grep -nE '^[[:space:]]*(from|import) orchestrator\b' cockpit` is empty (19 statements in 17 files
  at `1b106418da`; after 6179, η1 and η2 they name the new orchestrator session modules, so a pattern
  naming only `session_registry` would pass before θ1).
- DELETES: the dual-mode sibling-import blocks; the orchestrator-side copies of the moved modules.

**θ0 — Create shared.escalation_layout: the archive layout constants and the terminal status names**
(`task_kind=normal`, `complexity='simple'`, priority low, est. ~120 LOC)

- What: C9's `ARCHIVE_SUBDIR`, `DATE_FORMAT`, `archive_dir_for_date`, `RESOLVED`, `DISMISSED` and
  `TERMINAL_STATUS_NAMES`, stdlib-only, registered in `shared/tests/test_pure_stdlib_leaves.py`. No
  importer is repointed here (θ2 repoints them). S19 owns the module (G4); the typed-escalation-facts
  plan's vocabulary leaf (S2 α) depends on θ0 and builds `EscalationStatus` from the names (S19/Q-C8 (a)).
- Keys: `fk-e8c835c3f412` (contributes; θ2 closes it).
- Files: (3 files) shared/src/shared/escalation_layout.py, shared/tests/test_escalation_layout.py, shared/tests/test_pure_stdlib_leaves.py
- Deps: none. θ2 and S2 α depend on θ0. θ0 takes no edge onto the CLI-invocation plan's transcript
  leaf (S20 γ): they share only a test file, and the lead ruled such edges are not wired.
- Signal: `uv run --project shared pytest shared/tests/test_escalation_layout.py
  shared/tests/test_pure_stdlib_leaves.py` passes; the module imports only the stdlib.
- DELETES: nothing (it adds the one home; θ2 deletes the copies).

**θ2 — cockpit depends on shared only; one encode_cwd, one window-id parse and one escalation archive layout (integration gate of the session cluster)**
(`task_kind=normal`, priority medium, est. ~450 LOC)

- What:
  - `cockpit/pyproject.toml` drops the orchestrator dependency, its `[tool.uv.sources]` entry and the
    pyright `extraPaths` entry; `uv.lock` is regenerated;
  - `cockpit/backends/wm.py` imports `shared.session.windows.window_id_int`;
  - `scripts/legibility/inventory.py` and `digest.py` import `shared.session.records.encode_cwd`,
    their copies go, and `TestEncoderLockstep` becomes a single-home test (this subsumes task 6603);
  - S19/OQ-3 (a): add `locate_record` to `shared/escalation_layout.py`, which θ0 created (C9); θ2
    creates nothing else in that module. `escalation/archive.py`, `scripts/escalation_ladder.py` and
    `scripts/check_l0_settle_outcomes.py` import `ARCHIVE_SUBDIR` (and `DATE_FORMAT`) from it (the two
    scripts' "Mirrors" copies go; the settle check gains the `_checkout` import); the 11 files that read
    the layout through `escalation.archive` (`queue.py`, `sweep.py` and 9 test files, C9) are repointed to
    `shared.escalation_layout`, so `escalation.archive` is not a re-export; `shared.session.decisions`
    reads status through `locate_record`; `scripts/backfill_decision_queue_stamp.py` uses it;
    `ESCALATION_ARCHIVE_SUBDIR` is deleted. θ2 changes no locator logic in `queue.py`; it repoints its
    layout reads (θ2 → 9177 orders it against S3 θ);
  - S19/Q-C8 (a): `DECISION_CLOSE_MAP` keys from the module's `RESOLVED` and `DISMISSED` names, which
    θ0 created.
- Keys: `fk-d91317f88f21`, `fk-e8c835c3f412` (its one disposition; ω names the residual arm),
  `fk-57599c7fd907` (cockpit half).
- Files: (31 files) cockpit/pyproject.toml, uv.lock, cockpit/src/cockpit/backends/wm.py, cockpit/tests/test_backends_wm.py, cockpit/tests/conftest.py, scripts/legibility/inventory.py, scripts/legibility/digest.py, scripts/legibility/check_transcript_persistence.py, scripts/tests/test_legibility_inventory.py, scripts/tests/test_legibility_digest.py, shared/src/shared/escalation_layout.py, shared/tests/test_escalation_layout.py, escalation/src/escalation/archive.py, shared/src/shared/session/decisions.py, orchestrator/tests/test_decision_registry.py, scripts/backfill_decision_queue_stamp.py, tests/scripts/test_backfill_decision_queue_stamp.py, shared/tests/test_pure_stdlib_leaves.py, scripts/escalation_ladder.py, escalation/src/escalation/queue.py, escalation/src/escalation/sweep.py, escalation/tests/test_archive.py, escalation/tests/test_note_to_file.py, escalation/tests/test_sweep.py, orchestrator/tests/test_steward.py, dashboard/tests/_boundary_payloads.py, dashboard/tests/test_escalation_corpus.py, dashboard/tests/test_escalation_lifecycle_gate.py, dashboard/tests/test_memory_evals_data.py, scripts/check_l0_settle_outcomes.py, scripts/tests/test_check_l0_settle_outcomes.py
- Deps: θ1; θ0; task 9177 (S3 θ, `escalation/archive.py`, `queue.py`, `sweep.py`).
- Signal: B8 (`uv tree --package cockpit` without orchestrator; the `cockpit` script starts). B9's
  cockpit half and B10 pass. The next `reap-decisions` run by a watcher closes a decision whose
  escalation was archived, as today.
- DELETES: cockpit's whole-orchestrator dependency; 2 `encode_cwd` copies; `_window_id_int`'s copy of
  the parse; the `ESCALATION_ARCHIVE_SUBDIR` mirror and its by-hand-sync docstring; the ladder's and
  the settle check's `ARCHIVE_SUBDIR` mirrors (and the settle check's `ARCHIVE_DAY_FORMAT`); the
  unbounded `rglob` in `read_escalation_status`; the decision registry's spelled escalation statuses.

**ω — Acceptance record for the residual decision-record join** (filed and cancelled at decompose, no
code, no sidecar row; S19/OQ-3 (a)).

- Files: none.
- Deps: θ2.
- It carries no `x_finding_key`: θ2 carries `fk-e8c835c3f412`, its one disposition
  (`docs/quality-findings-contract.md` §7), as S6's acceptance task 9211 does for its keys' arms. It is
  cancelled with this `x_acceptance_reason`: "the polled-join arm of fk-e8c835c3f412: after θ2,
  DecisionRecord's state is still synced by the reap-decisions poll and joined by (project,
  escalations_dir, id); the queue-side locator stays `escalation/src/escalation/queue.py::EscalationQueue._locate_path`;
  the queue's resolve callback (`queue.py::EscalationQueue.set_resolve_callback`, wired by
  `orchestrator/src/orchestrator/harness.py::_on_escalation_resolved`) could close decisions at the
  source, but that needs a harness edit on the task-5040 file chain and wiring in every queue host;
  operator gate <λ's task id> explores that option (D) and the full move into the escalation package
  (B) with Leo after the review-all capstone, task 6698 (Leo, 2026-11-12)". The decompose seat writes
  λ's task id into the reason, since both are filed in the same planning batch.

**λ — Operator gate: explore, assess and discuss moving cockpit decision closing into the escalation package, after the review-all capstone**
(`task_kind='deterministic'`, `metadata.always_escalates=true`, no `before_done`,
`metadata.milestone = {"mode": "delayed", "after_secs": 604800}`, priority medium; Leo's addition to
S19-2, ruled 2026-11-12)

- Shape: a pure human gate (`docs/task-authoring.md` §5 and §6). Seven days after both dependencies are
  done (the lead may shorten the delay), it files a born-at-L2 escalation and blocks; the escalation
  watcher surfaces it to Leo, and nothing runs unattended. Precedents: tasks 5394 and 3676. Metadata:
  `source: "review-all"`, `x_finding_run: "review-all-dark_factory-20261008"`, `prd_path`,
  `prd_task_label: "λ"`; no `x_finding_key` (θ2 carries `fk-e8c835c3f412` as its one disposition) and no
  `metadata.files`.
- Deps: task 6698 (gate "Run /review-all on dark_factory", the review-all capstone; ruled_by: lead) and
  θ2. Both edges are needed: 6698 waits only on this program's high and critical tasks, and no S19 leaf
  is high, so 6698 could be done before the layout work lands.
- Description (the session brief, written in plain language at decompose):
  - What option B is: the escalation package issues a stable queue id and a typed escalation reference
    and closes cockpit decisions itself; the cockpit's close map (`DECISION_CLOSE_MAP`), status reader,
    `reap-decisions` verb and project-token normalisers go.
  - Option D, the cheaper alternative: the queue's resolve callback closes matching open decisions at
    the source, with the poll kept as a backstop.
  - The facts, measured 2026-11-12 (`digests/discussion/S19.md` §S19-2): 26 open decisions, 25 pointing
    at escalations still pending and 1 with no escalation id; the callback exists
    (`escalation/src/escalation/queue.py::EscalationQueue.set_resolve_callback`) and only the
    orchestrator wires it (`orchestrator/src/orchestrator/harness.py::_on_escalation_resolved`), not
    fused-memory's reconciliation queue; no escalation-package owner and no queue-identity design exist;
    after θ2 the layout has one home and the poll uses the bounded `locate_record`, and ω records the
    remaining polled join.
  - The three questions the session answers: who owns escalation identity across queues; whether
    decisions should close at the source; whether the cockpit's copy of escalation state should go
    entirely.
  - The timing caveat: the gate fires about a week after 6698 is done, and 6698 starts the next
    whole-project review, so the session may run while that review does; Leo may fold it into that
    review's deliberation instead and resolve the gate with that note.
  - Output: a ruling recorded on the escalation, and a `/prd` if Leo picks B or D (its path in the
    resolution). Resolving the escalation with the resume path marks λ done.
- Signal: the born-at-L2 escalation appears on the dark-factory queue after the delay, and is resolved
  with a ruling.
- DELETES: nothing (a gate).

**ν — process_identity asks the portable liveness check whether a pid exists; its zombie arm goes**
(`task_kind=normal`, `complexity='simple'`, priority medium, est. ~80 LOC)

- What: under Leo's ruling of 2026-11-13, `shared/src/shared/process_identity.py` stops reading a
  `/proc` state of `Z` or `X` as death. `liveness` and `capture` ask C1 (`pid_alive`) whether the pid
  exists. The `/proc/<pid>/stat` read stays only for what C1 cannot answer: the start ticks that make
  pid reuse read as death, with the boot id for a reboot. That is process identity, not liveness.
  `_EXITED_STATES` and the state test in `_running_stat` go. `shared/tests/test_process_identity.py`'s
  `test_unreaped_zombie_is_dead` becomes a test that an unreaped zombie reads `ALIVE` and is captured
  until it is reaped, then `DEAD`. `ProcessIdentityUnreadable` and the contract of
  `plans/deferral-flip-condition-prd.md` §5.2 are unchanged. Readers through it
  (`shared/src/shared/session_liveness.py::judge`, `dashboard/src/dashboard/data/deferral_summary.py`,
  `dashboard/src/dashboard/data/task_snapshot.py`, `fused-memory/src/fused_memory/middleware/deferral_gate.py`)
  change no code. Below the ~100 LOC floor, hence `complexity='simple'`; it does not fold into θ1, which
  moves `session_liveness.py` but does not edit `process_identity.py`.
- Keys: `fk-e22732500172` (contributes; its disposition is `filed:6615` + ν).
- Files: (2 files) shared/src/shared/process_identity.py, shared/tests/test_process_identity.py
- Deps: none. No other S19 leaf declares either file. Out-of-program pending tasks on the same file
  (D8): 6918 (single-source the stat parse and the exited-state set with `shared.proc_group`) is
  amended to drop its exited-state half, which ν deletes (lead action); 6918's parse half and 7283 are
  rebase-only.
- Signal: `uv run --project shared pytest shared/tests/test_process_identity.py
  shared/tests/test_session_liveness.py` passes; `git grep -n _EXITED_STATES shared/src` is empty; the
  dashboard's deferral panel and the cockpit keep showing holder liveness (their existing suites).
- DELETES: `_EXITED_STATES` and the zombie arm, the last zombie-aware liveness test in production code.
- G7: INV-9 (one home for the liveness fact).

**ι — A public CLI runner in shared.cli_invoke; invoke.py drops its four private imports and its second runner**
(`task_kind=normal`, priority low, est. ~700 LOC)

- What: C7 and D12. `_SubprocessResult`, `_run_subprocess` and `_parse_claude_output` become public as
  `SubprocessResult`, `run_cli_subprocess` and `parse_claude_output`. `_materialize_stdin` stays private
  inside the runner. `invoke.py` imports public names only, threads `runner` from `invoke_agent`, and
  deletes `_run_subprocess_local` and the `noqa: F401` block. The 32 patch sites move onto the
  `runner` parameter or onto `shared.testing_fake_claude`. The fifth and sixth backend arms, ZCode (task 5399)
  and MiMo (task 7993, landed 2026-11-12), take the runner like codex, gemini and pi, and
  `agents/zcode_backend.py` and `agents/mimo_backend.py` take the public `SubprocessResult`. The renamed
  names' code importers are all declared (18 files, measured at `565d15a06f`; 16 at `1b106418da`):
  besides the author's 11 files, `agents/zcode_backend.py`, `orchestrator/tests/test_zcode_backend.py`,
  `agents/mimo_backend.py` and `orchestrator/tests/test_mimo_backend.py`, the shared tests
  `test_api_refusal.py`, `test_cli_input_rejected.py`, `test_cli_invoke_background.py` and
  `test_cli_invoke_transcript_offloop.py`, and `orchestrator/tests/test_liveness_boundary_gate.py`
  (`from shared.cli_invoke import _run_subprocess`). Docstring and comment mentions elsewhere
  (`shared/tests/startup_completion_probe.py`, `fused-memory/tests/test_judge.py`,
  `fused-memory/tests/test_task_curator_timeout_evidence.py`, `steward.py`, `proc_group.py` and others)
  are prose, left to the next editor (D6). ι lands before the CLI-invocation plan's runner and result
  moves (S20 θ, β), which carry its public names (S19/Q-C2 (a)).
- Keys: `fk-ad5323d27142`.
- Files: (20 files) shared/src/shared/cli_invoke.py, orchestrator/src/orchestrator/agents/invoke.py, shared/src/shared/testing_stdin.py, orchestrator/tests/test_invoke_argv_parity.py, orchestrator/tests/test_invoke_parsers.py, orchestrator/tests/test_invoke.py, orchestrator/tests/test_shadow_invocation_sandbox.py, orchestrator/tests/test_shared_output_schema_intact.py, orchestrator/tests/test_verdict_servers_integration_gate.py, shared/tests/test_cli_invoke.py, shared/tests/test_cli_invoke_sandbox_wrap.py, shared/tests/test_api_refusal.py, shared/tests/test_cli_input_rejected.py, shared/tests/test_cli_invoke_background.py, shared/tests/test_cli_invoke_transcript_offloop.py, orchestrator/tests/test_liveness_boundary_gate.py, orchestrator/src/orchestrator/agents/zcode_backend.py, orchestrator/tests/test_zcode_backend.py, orchestrator/src/orchestrator/agents/mimo_backend.py, orchestrator/tests/test_mimo_backend.py
- Deps: none (it shares only test files with ζ: `orchestrator/tests/test_invoke.py`,
  `test_shadow_invocation_sandbox.py`; lead ruling on test-file edges). Tasks 9232 and
  9233 depend on ι (lead actions, under S19/Q-C2 (a)); S20 β and θ depend on ι, and S16 γ1
  reaches it through 9233. Task 6605: cancelled by the CLI-invocation plan's state-machine leaf (S20 η)
  per its draft; rebase-only if not.
- Signal: B11. The orchestrator's next codex-, gemini-, pi- or zcode-routed invocation, if any is routed in the
  window, and every claude invocation produce `AgentResult`s through the one runner (existing
  `test_invoke` e2e plus the live dispatch).
  `git grep -nE '_run_subprocess_local|from shared\.cli_invoke import .*\b_[a-zA-Z]' orchestrator/src`
  is empty.
- DELETES: `invoke._run_subprocess_local` (~100 lines); 4 cross-package private imports; 2 duplicated
  `_SubprocessResult` constructions.

**κ1 — Dashboard MCP egress module below memory and mcp_fanout; the import cycle cut**
(`task_kind=normal`, priority medium, est. ~500 LOC)

- What: C8 and D13. Create `data/mcp_client.py` with today's `mcp_tool_call` parameters (C8).
  `memory.py` keeps memory-endpoint shaping. `mcp_fanout` imports nothing from `memory`. The seven
  importers repoint, `metrics.py`'s multi-line import included, and `app.py`'s
  `memory_data.mcp_tool_call` and `memory_data.enclosing_deadline` calls go through `mcp_client`
  (measured at `1b106418da`). `reset_sessions` stops resetting fan-out streaks.
- Keys: `fk-99f44f47a6cc`.
- Files: (58 files) dashboard/src/dashboard/data/mcp_client.py, dashboard/src/dashboard/data/memory.py, dashboard/src/dashboard/data/mcp_fanout.py, dashboard/src/dashboard/data/escalations.py, dashboard/src/dashboard/data/merge_halt.py, dashboard/src/dashboard/data/merge_queue.py, dashboard/src/dashboard/data/scheduler.py, dashboard/src/dashboard/data/task_runtime.py, dashboard/src/dashboard/data/tasks.py, dashboard/tests/_boundary_payloads.py, dashboard/tests/_canned_mcp.py, dashboard/tests/_dashboard_helpers.py, dashboard/tests/test_active_tasks.py, dashboard/tests/test_api_curator.py, dashboard/tests/test_api_curator_cancel.py, dashboard/tests/test_app.py, dashboard/tests/test_app_http_limits.py, dashboard/tests/test_burndown_data.py, dashboard/tests/test_dashboard_endpoints_survive_hung_mcp.py, dashboard/tests/test_deferral_summary.py, dashboard/tests/test_escalation_lifecycle_gate.py, dashboard/tests/test_escalations_data.py, dashboard/tests/test_fetch_statuses.py, dashboard/tests/test_fetch_tasks_whole_operation_budget.py, dashboard/tests/test_healthz_data_plane.py, dashboard/tests/test_mcp_fanout.py, dashboard/tests/test_mcp_write_proxy.py, dashboard/tests/test_memory.py, dashboard/tests/test_memory_evals_data.py, dashboard/tests/test_memory_exchange_isolation.py, dashboard/tests/test_merge_halt.py, dashboard/tests/test_merge_queue_data.py, dashboard/tests/test_metrics_curator.py, dashboard/tests/test_metrics_curator_cap_pairing.py, dashboard/tests/test_metrics_data.py, dashboard/tests/test_orchestrator.py, dashboard/tests/test_project_roots.py, dashboard/tests/test_same_basename_roots.py, dashboard/tests/test_saturated_pool_is_diagnosable.py, dashboard/tests/test_scheduler_holder_live.py, dashboard/tests/test_scheduler_page.py, dashboard/tests/test_scheduler_state_access.py, dashboard/tests/test_tab_tasks_deferral.py, dashboard/tests/test_task_lineage.py, dashboard/tests/test_task_lookup.py, dashboard/tests/test_task_prose.py, dashboard/tests/test_task_runtime.py, dashboard/tests/test_task_snapshot.py, dashboard/tests/test_tasks.py, dashboard/tests/test_tasks_budget.py, dashboard/tests/test_tasks_census_projection.py, dashboard/tests/test_tasks_collector_seams.py, dashboard/tests/test_tasks_fanout.py, dashboard/tests/test_tasks_pagination.py, dashboard/tests/test_tasks_read_record.py, dashboard/tests/test_tasks_terminal_answer.py, dashboard/src/dashboard/data/metrics.py, dashboard/src/dashboard/app.py
- Deps: task 6722 (S14 κ, `data/memory.py`).
- Signal: B12. Against the live fused-memory, the dashboard's Memory, Tasks, Escalations, Merge-queue
  and Scheduler panels render as before. With an endpoint stopped, they show the outage state. After
  the endpoint returns, the next refresh succeeds on a fresh session.
- DELETES: the function-local import; the reset-resets-streaks coupling (the two `memory.py` loop
  sites go in S22 γ1).

**κ2 — Dashboard cross-module underscore names become public; the "vestigial underscore" paragraphs go**
(`task_kind=normal`, priority low, est. ~350 LOC)

- What: make each cross-module underscore name public at its owner:
  - `project_dbs`: 5 names;
  - `loops`: 6 names;
  - `api/window.parse_window`;
  - `data/orchestrator.resolve_project_root` and `read_project_root_from_config`;
  - `config.CANONICAL_CONFIG_NAME` and `LEGACY_CONFIG_NAMES`, whose deferred import in
    `data/orchestrator.read_max_concurrent_tasks` is hoisted to the module top.

  `mcp_fanout._LOCK_ACQUIRE_TIMEOUT_SECONDS` is not in this list: made public by task 6879, which κ2
  depends on. Delete the three paragraphs. Tests patch the public names. The dashboard-readers plan's
  app.py partition (S22 ν) and its fan-out leaves build on these names and depend on κ2 (S19/Q-C3 (a)).
- Keys: `fk-3ddc73250c64`.
- Files: (42 files) dashboard/src/dashboard/api/burndown.py, dashboard/src/dashboard/api/merge_queue.py, dashboard/src/dashboard/api/prds.py, dashboard/src/dashboard/api/window.py, dashboard/src/dashboard/app.py, dashboard/src/dashboard/config.py, dashboard/src/dashboard/data/burndown.py, dashboard/src/dashboard/data/db.py, dashboard/src/dashboard/data/mcp_fanout.py, dashboard/src/dashboard/data/merge_queue.py, dashboard/src/dashboard/data/metrics.py, dashboard/src/dashboard/data/orchestrator.py, dashboard/src/dashboard/http_pool.py, dashboard/src/dashboard/loops.py, dashboard/src/dashboard/project_dbs.py, dashboard/tests/_dashboard_helpers.py, dashboard/tests/test_api_curator.py, dashboard/tests/test_api_curator_cancel.py, dashboard/tests/test_api_prds.py, dashboard/tests/test_app.py, dashboard/tests/test_app_lifespan_reap.py, dashboard/tests/test_app_module_boundaries.py, dashboard/tests/test_burndown_data.py, dashboard/tests/test_burndown_parity_alarm.py, dashboard/tests/test_config.py, dashboard/tests/test_costs_data.py, dashboard/tests/test_dashboard_governed_lists.py, dashboard/tests/test_db.py, dashboard/tests/test_durability.py, dashboard/tests/test_fixture_isolation.py, dashboard/tests/test_http_pool.py, dashboard/tests/test_lifespan_resource_binding.py, dashboard/tests/test_lifespan_samplers.py, dashboard/tests/test_mcp_fanout.py, dashboard/tests/test_merge_queue_data.py, dashboard/tests/test_metrics_curator.py, dashboard/tests/test_metrics_loop_harness.py, dashboard/tests/test_orchestrator.py, dashboard/tests/test_project_roots.py, dashboard/tests/test_quality_sampler.py, dashboard/tests/test_quality_store.py, dashboard/tests/test_tasks_budget.py
- Deps: κ1 (`mcp_fanout.py`); task 6722 (`loops.py`, `project_dbs.py`, `config.py`, `app.py`); task 6879
  (renames `_LOCK_ACQUIRE_TIMEOUT_SECONDS` and `_TASKS_PER_PROJECT_BUDGET`; lands first).
- Signal: B13. The dashboard serves `/api/burndown`, `/api/merge_queue` and `/api/prds` and the cost
  panels as before.
- DELETES: 12 cross-module underscore imports (21 names; task 6879 removes the 13th) and 3 "vestigial" paragraphs.

**Intra-plan edges** ("X → Y": X depends on Y): β1 → α; γ → β1; β2 → β1, γ; δ → β2;
ε → δ; η2 → η1; θ1 → η2; θ2 → θ1, θ0; κ2 → κ1; ω → θ2; λ → θ2. α, ζ, η1, θ0, ι, κ1 and ν have no
intra-plan prerequisite. η1 depends on task 6179
and λ on gate task 6698 (out of batch). Held edges onto wave-3 siblings: ε → S12 ι1, β2 → S13 δ2, and conditionally δ → S12 ι1. The
fused-memory chain is serial because its leaves share importer files; every pair of leaves that shares
a non-test file is joined by a path of edges (measured over the folded file lists). The pairs sharing
only test files take no edge (R54, restated by the lead after the wave-3 fold-verifiers for intra-plan
edges too): ζ with α, β1, β2, γ (through `test_stages.py` and `test_cli_stage_runner_cutover.py`), θ1, θ0,
θ2 and ι; α and θ1 on `tests/scripts/test_atomic_write_regrowth.py`; θ0 with θ1 on
`shared/tests/test_pure_stdlib_leaves.py`; θ2 and κ1 on three dashboard test files.

**Signal leaves:** ε (fused-memory cluster), θ2 (session cluster), ι (CLI runner) and κ2 (dashboard,
reached through κ1).

### Deletable accounting (synthesis `deletable`)

| Deletable | Where |
|---|---|
| 6 `_pid_alive` copies | task 6615 (5; done, `f750373949`); the watchdog copy: S24's lease leaf (S19/Q-C1 (a)) |
| 2 `_wmctrl_list` copies and 3 window-id canonicalisers | η2 (orchestrator), θ2 (cockpit) |
| the fm→orchestrator type-ignored imports and venv prose | ζ |
| the second `CHILD_KINDS` | task 6720 (S14 ν) |
| the storm_counter shim | task 6547 (`0f86a2cdcc`; fixed before this plan) |
| cockpit's whole-orchestrator dependency | θ2 |
| `invoke._run_subprocess_local` | ι |
| 14 underscore imports (13 at main) and 2 "vestigial underscore" paragraphs (3 at main) | κ2 |
| the memory↔mcp_fanout deferred cycle | κ1 |

## Existing tasks (contract §8, re-run 2026-11-11; statuses re-read 2026-11-12)

- **6615** (done, merged 2026-11-11, `f750373949`; medium, fk-5d02d7903533): delivered C1. β1's and
  η1's edges onto it are satisfied. Stamp `fk-e22732500172` on it (lead), so the key's `filed:6615`
  disposition names the task that did the work.
- **6179** (pending, low, planned 2026-11-10): the decisions extraction, step 1 of D9. η1 depends on it.
  Amend: its modules move to `shared.session` in θ1. Stamp `fk-cfcf00669c46`.
- **6955** (pending, low): subsumed by η1 and η2 (the ruled partition). Cancel, naming η1/η2.
- **9100** (pending, low): subsumed by η1. Cancel.
- **7444**, **5768** (pending, low): subsumed by 6615, which is done. Cancel now, naming 6615.
- **5983** (pending, low): subsumed by δ. Cancel.
- **6603** (pending, low, Phase-5, fk-34c546efccc9): subsumed by θ2. Cancel.
- **6879** (pending, low, `complexity: simple`, outside the program): promotes the dashboard timing
  constants `_LOCK_ACQUIRE_TIMEOUT_SECONDS` and `_TASKS_PER_PROJECT_BUDGET` to public names. κ2 depends
  on it (lead action); it lands first and renames both constants, so κ2 does not rename
  `_LOCK_ACQUIRE_TIMEOUT_SECONDS`.
- **6635** (pending, low, Phase-5): amend to drop its `dead_letter_escalator` half, which task 9143
  already did (`204de188c4`: the escalator imports the services layer at module scope). Its
  `services/__init__.py` half stands.
- **6621** (done, `9c3f608c88`), **9143** (done, `3983fa2265`): their edges from β2 and γ are satisfied.
- **6547** (done, merged `d614ee408b`): deleted the `server/storm_counter.py` shim; δ no longer does.
- **6698** (pending, medium, gate "Run /review-all on dark_factory"): λ depends on it.
- **5345**, **4444** (pending, low): they edit the moved sandbox probes. Each depends on ζ. After ζ
  they fix one probe (`shared.sandbox.bwrap.is_bwrap_available`) reached from two event loops: amend
  5345 to fold into 4444 once ζ lands.
- **4627**, **4840** (pending, medium), **6852**, **7607** (pending, low), **3419** (deferred): they
  declare a moved sandbox file; rebase-only, the clerk repaths them from §Moves (D8).
- **7902** (pending, low, outside the program; `search_tasks` 0.659): public seams for session_hooks'
  private test families. Its window-id resolver family is η2's `WindowProbe` work. Amend it to drop that
  family and keep the install and single-read families; 7902 → η2.
- **9232**, **9233** (S8, behind 5040): each depends on ι; 9232 also on ζ (D8 exception), under
  S19/Q-C2 (a).
- **9285** (S10 ο, pending): depends on β2, not the other way, because it waits on 8528 through 9282/9283 (D8 exception).
- **6720**, **6722** (S14): dependencies, as in the cross-PRD table.

## Out of scope

- Splitting `reconciliation/` engine modules further (S12, S13) and `server/tools.py` / `server/main.py`
  (S15).
- `flag_dedup.py`'s partition (task 5996, S13).
- The watchdog's stdlib constraint (S24).
- Moving escalation-lifecycle ownership of cockpit decisions to the escalation package (options B and
  D of S19-2): gate λ opens that session after the capstone; any code is a later `/prd` if Leo picks
  one.
- Zombie detection (S19/Q-C1, Leo 2026-11-13: negligible until evidence to the contrary). A recorded
  case of a zombie read as alive that cost something reopens it, as a change to C1 alone; no task now.
- The other session_hooks private patches (`__file__`, `_run_install`, `_env_slug_is_owned`): out of
  the key's claim.
- `orchestrator.session_registry`'s CLI layout and verb set: unchanged.
- Any change to the dashboard's fan-out cache (S22).

## Open questions (tactical)

1. **Where `recon_archive_root` lands inside recon_state.** A module of its own, or with the journal?
   **Suggested:** its own small module named for the run archive. Decide in γ.
2. **The Protocol names** for `task_interceptor`'s targeted reference and `journal`'s write journal.
   Decide in γ and α.
3. **Whether `migrate_decision_project_tokens` and its verb are still needed.** It is a one-shot repair
   (task 3807). If the live corpus has no non-canonical token, θ1 may delete it with a commit-message
   measurement. Decide in θ1.
4. **`throughput`'s CLI entry.** It keeps `python -m fused_memory.recon_state.throughput`. A console
   script is out of scope.

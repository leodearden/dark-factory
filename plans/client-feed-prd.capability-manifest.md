# Capability manifest — client feed: a per-project, client-safe evidence feed

Binds each task's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `client-feed-prd.capability-manifest.yaml`.
PRD: `plans/client-feed-prd.md` (committed `ea7ab27c21`, 2026-10-14).
As of main `b7688370cf` (2026-10-14); anchors re-verified by a sonnet seat before filing.

**Verdict summary: 25 bindings, all PASS; 21 mechanical checks (all grep `present`) and 4 manual; zero polarity findings at `main`** (`shared.delivered_check_polarity.lint_delivered_checks` run by hand at decompose: every `present` pattern names a file or string absent on main today; the two `absent` checks first drafted for ε and ζ were rejected as `vacuous_absent`, since their target directories do not exist yet, and are recorded as manual with the leaf's own test named instead).

The DAG has no inversion:

- α → β1, β2 → γ → δ, ε → ζ
- θ → β1
- γ → η

## Filed batch (2026-10-14, planning mode; committed pending in one `commit_planning`)

| Label | Task | Depends on | Mechanical checks |
|---|---|---|---|
| α | 6966 | — | 3 |
| θ | 6967 | — | 2 |
| β1 | 6968 | α, θ | 3 |
| β2 | 6969 | α | 3 |
| γ | 6970 | β1, β2 | 3 |
| δ | 6971 | γ | 2 |
| ε | 6972 | γ | 2 |
| ζ | 6973 | δ, ε | 1 |
| η | 6974 | γ | 2 |

## Substrate findings that shaped the bindings (G3)

| Capability | Status at decompose | Consequence |
|---|---|---|
| Task store shape | **CONFIRMED** via `scripts/tasks_db_schema.py`: `tasks(tag, id, title, description, details, test_strategy, status, priority, metadata, updated_at, claimant_run_id, heartbeat_at, candidate_key)`, `dependencies(tag, task_id, depends_on)`; one tag (`master`) on the live store | β1 pins `tag` and projects by allowlist |
| Event types | **CONFIRMED** members of `orchestrator/src/orchestrator/event_store.py::EventType` with live rows on 2026-10-14: merge_finalized 4,283; train_merged 58; merge_verify 3,297; workflow_verify 5,936; phase_enter 53,703; phase_exit 53,487; scheduler_paused 15; scheduler_resumed 15; escalation_created 2,675; escalation_resolved 755 | β1's readers have producers that have written in production (INV-13) |
| `escalation_created.data.level` | **PARTIAL**: present on 452 of 2,675 rows | `waiting_on: leo` only when a recorded level is 2; otherwise `factory` (PRD §4.10 amended; β1's details amended) |
| `review_completed` | **ABSENT** on main (the planned θ) | β1 depends on θ; the feed says `not_recorded` until a row exists |
| Verify archive | **CONFIRMED**: 2,294 `junit-*.xml.gz` under 1,885 task dirs; task legs archive via `TaskWorkflow._run_scoped_verification_with_infra_retry` (PRD §2 anchor corrected from `_execute_verify_review_loop`); pruning in `verify.py::_prune_archive` at 30 days / 500 MB | β2 reads the archive; copies rows into the feed |
| junit pytest-only | **CONFIRMED** `orchestrator/src/orchestrator/verify_cmd.py::with_junitxml` is a no-op unless the tool is pytest and the command is structured (a raw-retained pytest chain also gets no junit) | the feed and docs say pytest projects only |
| pytest `record_property` | **CONFIRMED** warns under the default `junit_family=xunit2` (`_pytest/junitxml.py::_warn_incompatibility_with_xunit2`) | ε's plugin writes a map file and must not use `record_property` (ε's details amended; an `absent` check pins it) |
| `ReviewAggregation` | **CONFIRMED** `orchestrator/src/orchestrator/artifacts.py::ReviewAggregation` with the five cited fields; `EventStore.emit` exists | θ's emit site is real |
| Lockstep lists | **CONFIRMED** `pyproject.toml` members, `hooks/project-checks` `PYRIGHT_PACKAGES` and `MAGICMOCK_TEST_DIRS`, `dark-factory-orchestrator.yaml` `test_command`/`lint_command`/`type_check_command`, and the four drift tests | α edits all of them in one leaf (INV-5) |
| `shared.project_layout` | **CONFIRMED** `tasks_db`, `runs_db`, `default_project_root` | β1/β2 use the first two; ζ bans the third |
| Dashboard reuse | **CONFIRMED** none: `fetch_tasks` fans out over roots through fused-memory; `project_label` is a basename | nothing from `dashboard/` is imported |

## Per-leaf bindings

| Leaf | Capability | Evidence | Check |
|---|---|---|---|
| α | workspace member registered | producer α; absent on main | grep `"client-feed"` in `pyproject.toml` |
| α | contract models | producer α; `client-feed/` absent | grep `FEED_VERSION` under `client-feed/src/client_feed/` |
| α | fold is the single reducer | α's tests (state == fold(feed)) | manual |
| α | verify command runs the package | producer α; `test_command` lacks it | grep `cd ../client-feed && uv run pytest` in the yaml |
| θ | event type | producer θ; absent on main | grep `review_completed` in `event_store.py` |
| θ | emitted from the review loop | producer θ | grep `EventType.review_completed` in `workflow.py` |
| θ | counts-only payload | θ's tests | manual |
| β1 | tag-pinned projection | producer β1 | grep tag pin in `sources/tasks.py` |
| β1 | events cursor | producer β1 | grep `merge_finalized` in `sources/events.py` |
| β1 | review counts from θ else not_recorded | producer θ upstream (wired) | grep `review_completed` in `sources/events.py` |
| β2 | policy with required visible globs | producer β2 | grep `visible_globs` in `policy.py` |
| β2 | content via cat-file only | producer β2 | grep `cat-file` in `sources/git.py` |
| β2 | junit → test_results | producer β2; archive confirmed | grep `junit` in `sources/archive.py` |
| γ | append-only writer, no symlink following | producer γ | grep `O_NOFOLLOW` in `feed_io.py` |
| γ | snapshot subcommand | producer γ | grep `snapshot` in `cli.py` |
| γ | degraded heartbeat | producer γ; B19 | grep `degraded` in `publisher.py` |
| δ | typed refusal reasons | δ's tests B11/B13/B15 | grep `not_in_tree` in `budget.py` |
| δ | reveal floor | producer δ | grep `reveal_floor_lines` in `budget.py` |
| ε | criteria map reader (amended 2026-10-16) | producer ε; map producer defect-tests β, upstream | grep `tests\.json` in `criteria.py` |
| ε | no record_property | now defect-tests β's plugin; β's tests | manual (client-feed ships no pytest plugin) |
| ε | criteria-results CLI | producer ε | grep `criteria-results` in `cli.py` |
| ζ | audit-hook confinement test | producer ζ; B1 | grep `addaudithook` in `test_isolation.py` |
| ζ | default_project_root banned | rejection-style; ζ's ban test | manual (same reason) |
| η | unit template rendered by setup-host | producer η; dashboard template precedent | grep `client-feed.service.template` in `setup-host.sh` |
| η | OPERATIONS section | producer η | grep `### Client feed` in `OPERATIONS.md` |

## G6 premises

No leaf signal asserts a number the fixture does not carry. B12/B13's figures (1,000 lines, 0.25,
floor 50, 200/100/10-line files) are fixture-defined. B21's `≥ 1` is producible by θ alone after one
real review on this repo. B22's "prds non-empty" holds: the live store carries `prd_path` on hundreds of
tasks (the dashboard PRD-lanes PRD measured 255 direct holders on 2026-10-09).

## G7 walk (design invariants, per task)

| Task | Invariant questions answered | Hit / waiver |
|---|---|---|
| α | INV-1: the contract is a pydantic model with committed fixtures validated on both sides. INV-5: α edits every lockstep list in one leaf and the existing drift tests pin them. INV-9: `fold.py` is the one reducer; `state.json` is rendered from it | none |
| θ | INV-2: the event carries structured counts at the failure point (blocking). INV-5: one emit idiom, two call sites in one method. INV-13: the reader (β1) renders `not_recorded` until a row exists | none |
| β1 | INV-13: every reader's producer has written rows in production (counts above). INV-11: a store that cannot be opened is reported to γ's degraded list, never an empty result | none |
| β2 | INV-11: an absent junit is `not_recorded`, never pass. INV-9: the counting rule has one home used by δ | none |
| γ | INV-6: heartbeat age makes liveness visible; INV-11: degraded list; INV-12: refuses to start on a bad config or a corrupt feed rather than overwriting | none |
| δ | INV-10: refusals are exercised by hostile inputs, not asserted in prose. INV-9: `engagement.json` is a projection echoing its version | none |
| ε | INV-13: `criteria_results {verify}` appears only when a map and junit exist; otherwise absent, not "all passing". INV-5: one join, in `criteria.py` | none |
| ζ | INV-10: the confinement guard exercises behaviour (audit hook over a real run), not a grep of prose. INV-3: three independent instruments corroborate the invariant | none |
| η | INV-5: the unit is rendered from one template, like the dashboard's; INV-9: install steps live in OPERATIONS, pointers elsewhere | none |

## Decompose decisions

- Nine leaves, no out-of-batch dependencies in this repo; the portal PRD (other repo) will depend on α
  (`dark_factory:6966`) and ζ (`dark_factory:6973`).
- Task details amended after the anchor verification: β1 (`waiting_on` only on a recorded level 2),
  ε (no `record_property`).
- The orchestrator does not read `user_observable_signal` / `consumer_ref` / `grammar_confirmed`; they
  are substrate for a future tracking-infra session.

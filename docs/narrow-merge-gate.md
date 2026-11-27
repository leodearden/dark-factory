# The narrow merge gate: docs-only and comments-only merges

Ruled by Leo as R4 on 2026-09-11, with no shadow since 2026-11-19. Implemented by
task 5411. This file is the rationale home that the `narrow_gate_*` modules and
the `narrow_merge_gate` block of `dark-factory-orchestrator.yaml` point at.

## The ruling

Before this gate, a merge whose diff held no source file reached the INV-1
sites in `orchestrator/src/orchestrator/verify.py::_run_scoped_verification_body`.
Those sites emit `trivial_pass_escalated(resolution='full_gate')` and run the
whole gate, about 35 to 40 minutes on dark-factory. A comment-only `.py` diff
ran the whole gate as well. Because a comment fix cost a full gate, policy left
stale pins and drifted docstrings alone until a code change touched their
file.

- **R4, 2026-09-11.** The merge gate may take a narrow command set for two diff
  classes, `docs_gate` and `comments_gate`, provided that:
  - the classifier is positive;
  - the command set is non-empty;
  - evidence absence stays on the full gate;
  - the task 2823 red-main precheck and the task 2838 config-diff escalation
    are kept.
- **Amended 2026-10-09.** The docs allowlist is `plans/*.md`, not `plans/**`,
  because `shared/tests/test_capability_manifest.py` reads the `plans/*.yaml`
  manifests. This amendment also kept FIFO order; 2026-11-20 replaced that
  part.
- **Amended 2026-11-02.** The allowlist admits the stamped sidecar
  `plans/*.capability-manifest.yaml` and `skills/**/*.md`. A PRD family (PRD
  `.md`, capability manifest and sidecar) is the unit this gate exists for.
- **No shadow, 2026-11-19.** Dark-factory adopts the gate when this task lands.
  There is no narrow-vs-full comparison run. A red main caused by a docs or
  comments merge is debugged directly from the main-tip sweep's finding and
  the merge's `merge_gate_resolution` event.
- **Amended 2026-11-20.** Three changes:
  - Queue order belongs to the docs lane, task 10996. This gate adds no order
    bypass.
  - The gate-delta re-verify, task 6661, reuses this gate's command set on a
    rebased tree.
  - Direct-to-main docs commits continue until migration task 8260 lands.

## The two classes

The classifier reads the real tree delta `base..merge_sha` from git:
`orchestrator/src/orchestrator/narrow_gate_delta.py::classify_tree_delta`
runs `git diff --raw --no-renames` and reads blobs with `git cat-file`. It never
reads `MergeRequest.task_files`, because that list is the plan's declared files,
not the change. A task that declares only docs but changes code must not be
narrowed.

The rules are in `orchestrator/src/orchestrator/narrow_gate_class.py`:

- **`docs_gate`**: every changed path matches a `narrow_merge_gate.docs_paths`
  glob (`::is_docs_path`). The match is a full match on the repo-relative path
  (`::matches_path_glob`). `*` and `?` stay within one path segment, and a `**`
  segment spans any depth. So `*.md` matches only root-level Markdown and never
  `fused-memory/.../prompts/stage1.md`. `::classify_docs_change` applies this
  class on its own.
- **`comments_gate`**: every changed path is either a docs path or an existing
  `.py` file that `::python_comments_only` accepts. That means both sides parse,
  their `ast.dump` matches once the leading string statement of every module,
  class and function body is dropped, and the shebang and coding cookie are
  unchanged. Comments never reach the AST, so they are stripped by parsing.
  `::classify_change` decides both classes.
- **`full_gate`** is everything else. The first miss is recorded as a
  structured `::MissReason`:
  - a deletion, or a rename (which `--no-renames` splits into a deletion and an
    addition);
  - a mode change, or a path that is not a regular file;
  - an unnormalised path;
  - a non-docs path whose suffix has no comment predicate. Only `.py` has one,
    so `.rs`, `.yaml`, `.toml`, `.json` and `.sh` all miss;
  - a new `.py` file, or a `.py` file that is not UTF-8 on either side;
  - an empty or unreadable delta.

YAML, TOML, JSON, shell scripts and prompt files are not docs, wherever they
sit. They take the full gate because tests read them. `docs/` holds
machine-read yaml and json too (`docs/legibility/legibility.yaml`,
`docs/legibility/census-state.json`), so dark-factory's allowlist admits only
`docs/**/*.md` there, and `tests/scripts/test_narrow_merge_gate_policy.py`
pins that boundary.

## What the narrow gate runs

`orchestrator/src/orchestrator/narrow_gate_plan.py::narrow_gate_plan` builds
the command set. For each registered module it runs one FILE_SCOPED pytest run,
over two sets of tests:

- the module's whole-tree guard family. Its single home is
  `shared/src/shared/whole_tree_guard_family.py`. The census in
  `shared/src/shared/testing_repo_wide_tests.py` fails any test module that
  reads the repository beyond its imports but is not a member. The census is a
  floor, not a proof of completeness: it sees only the reads a module spells
  itself (see Accepted residual risk);
- the consumer tests of every touched behavioural artifact, from
  `orchestrator/src/orchestrator/verify_artifact_consumers.py::BEHAVIOURAL_ARTIFACTS`.
  An artifact whose consumer is a whole suite gives that module its full suite.

The family already holds the document-reading guards the ruling named, such as
the design-invariants consistency suite, the capability-manifest suite, the
codebook guards, cited-test-class drift, the line-pin policy and the source
measures. No suite list is kept in yaml, where it would drift.

`comments_gate` also runs the module's `lint_command` (ruff) and
`type_check_command` (pyright), scoped to that module's changed `.py` files.
Both tools read comments (`# noqa`, `# type: ignore`).

The command set is never empty. If a changed `.py` belongs to no module, or
nothing would run, `narrow_gate_plan` returns None and the merge takes the full
gate with `EMPTY_COMMAND_SET`.

`docs_gate` runs no ruff. ruff 0.15.9 given a `.md` path warns "No Python files
found under the given path(s)" and exits 0, so running it would only make the
set look non-empty.

## Where it sits in the merge lane

- **Decision.** `orchestrator/src/orchestrator/verify.py::narrow_merge_gate_decision`
  decides on the dispatcher, once per in-flight verify
  (`orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::_PostMergeVerify._merge_gate_decision`).
  Only the lane's in-flight verify offers it a base:
  `orchestrator/src/orchestrator/merge_lane/inflight_verify.py::InflightVerifyRun._launch`
  passes `narrow_gate_base=item.base_sha`. A history-leg re-verify, and any
  caller whose tree is not one merge onto a base, keeps the full gate.
- **Execution.** A narrow decision always runs on the local host. The wire spec
  carries no narrow plan, so a remote could never be the sole narrow verdict,
  which was the incident 966f23a6 shape. It runs through
  `verify.py::run_scoped_verification(narrow_gate=...)`, the same
  plan-to-execution bridge, verify-result cache and archive as any merge
  verify. `orchestrator/src/orchestrator/verify_runner.py::LocalRunner` skips its
  unscoped typecheck on a narrow verdict.
- **The screen.** The merge pre-verify screen (task 7919,
  `orchestrator/src/orchestrator/merge_lane/screen.py`, manifest
  `scripts/merge_screen.yaml`) still runs at the merger site before the
  in-flight verify, for narrow items too. The narrow gate replaces only the
  in-flight verify's command set, never the screen.
- **The event.** Every decision taken while the gate is on whose verify
  reaches a final verdict emits `EventType.merge_gate_resolution`. A verify
  that raises instead (a remote's `RunnerUnavailable`, a cancellation) lands
  nothing and emits none. The event carries:
  - `resolution`, the gate the verdict actually ran;
  - `decided`, the gate the decision chose. The two differ when the narrow
    plan never reached execution, for example on a project with
    `merge_verify_workspace: true`, whose workspace-forced verify runs the full
    gate;
  - the miss reason and path;
  - the changed paths;
  - `commands`, the narrow verdict's commands
    (`narrow_gate_plan.py::planned_commands`), or null on the full gate. The
    verify-result cache may replay a unit rather than run its command;
  - whether it passed;
  - `base_sha` and `merge_sha`.

## What stays on the full gate by construction

- **Evidence absence.** Both incidents behind task 2883 are on the full gate
  by construction:
  - 966f23a6: a remote trivial pass was the sole verdict. Narrow decisions run
    only on the local host.
  - 83336a32: a 0 ms local pass after a worktree was clobbered (ENOENT).
    `narrow_merge_gate_decision` requires the worktree's HEAD to be
    `merge_sha`, and every present changed path and every registered module's
    test file to be on disk. Otherwise it gives `EVIDENCE_ABSENT`.
- **The 2838 globs and reify's guard.**
  `verify.py::_merge_config_only_diff_forces_full_gate` and
  `verify.py::_verify_pipeline_guard_requires_full_gate` still force the full
  gate, with `FORCED_FULL_GATE`.
- **Any classifier miss**, and **an empty command set**.
- **Any fault.** `narrow_merge_gate_decision` never raises. An unexpected
  error degrades to the full gate with `DECISION_FAULT`.

## A narrow pass is not whole-tree evidence

`verify.py::VerifyResult.is_whole_tree_evidence` is false for a trivial pass and
for a narrow one. Every pass-tail consumer reads that one predicate:

- **The 2823 red-main withholding**
  (`post_merge_verify.py::_PostMergeVerify._withheld_partial_pass`). A narrow
  pass over a known-red main is withheld. Its reason keeps
  `TRIVIAL_PASS_MAIN_RED_REASON_PREFIX` and names the resolution, from
  `verify.py::VerifyResult.partial_evidence_label`.
- **The remote cross-check** (`::_PostMergeVerify._cross_check_remote`). There
  is no full suite to cross-check.
- **Baseline seeding** (`::_PostMergeVerify._seed_main_baseline`). A partial run
  never records `merge_sha` as green.
- **The shadow's warm map** (`on_result`). A narrow land stays map-less, so it
  remains in the population whose cold leg
  (`orchestrator/src/orchestrator/merge_lane/shadow.py`) runs the complete suite.

## Trains and queue order

Docs and comments items never join a train, which would make them pay the full
gate they were routed away from. `narrow_gate_delta.py::train_exclusions` is
the rule's one home, and it reads no git while the gate is off. Both train
formers take their ids from it:

- **The workflow's train former.** Its anchor is the forming task itself, so it
  passes the ids to
  `orchestrator/src/orchestrator/merge_lane/train_selection.py::select_train_members`
  as `excluded=`. An excluded anchor forms no train, and an excluded candidate
  is skipped.
- **The worker's coalesce pass.** Its anchor is only the head of its buffer.
  The pass drops narrow items in its audited exclusion split, with reason
  `merge_lane/worker.py::COALESCE_EXCLUDED_NARROW_MERGE_GATE`, before it picks
  an anchor. A docs-only head therefore still lets the code singles behind it
  form a train.

Queue order is unchanged: the docs lane is task 10996. That lane classifies a
request before its verify with `narrow_gate_delta.py::classify_branch_change`
and `narrow_gate_class.py::classify_docs_change`. Task 6661 takes the command
set from `narrow_gate_plan.py::narrow_gate_plan`.

## Accepted residual risk

Docstrings count as comments, although `__doc__` is visible at runtime: MCP tool
descriptions and argparse help are built from it. A comments-only merge can
therefore change runtime text without any runtime test in the narrow set. The
backstops are:

- the task-role verify the branch already passed;
- the whole-tree guard family in the narrow set;
- the main-tip sweep;
- the shadow's cold leg over map-less lands.

The guard family itself is only as complete as its census. A repo-reading test
that reaches the tree through a helper imported from elsewhere is invisible to
`shared/src/shared/testing_repo_wide_tests.py`, so it can stay out of the
family and out of the narrow set. The same backstops apply.

## Kill switch

`narrow_merge_gate` is a green-tier, hot-reloadable section
(`orchestrator/src/orchestrator/narrow_gate_config.py::NarrowMergeGateConfig`).
To roll back, set `narrow_merge_gate.enabled: false` in
`dark-factory-orchestrator.yaml` and apply it with
`mcp__escalation__reload_config`. No restart is needed. The code default is off.

## Not yet: Rust and Reify

Python is the only comment language. Task 11763 carries the rest:

- a Rust comment-only predicate, added as the `.rs` entry of
  `narrow_gate_class.py`'s suffix-to-predicate map;
- the global-command command set Reify needs, since Reify has no
  `module_configs` (clippy and `cargo test --doc` on touched crates, the PTODO
  check, `hooks/project-checks`);
- Reify's yaml.

Reify adopts the gate only after task 7423 has observed a main-tip sweep
complete, because that sweep is this class's post-merge backstop.

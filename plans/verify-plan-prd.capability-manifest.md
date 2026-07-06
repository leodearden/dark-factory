# Capability manifest — verify-plan PRD (W7)

Mechanizes G3 + G6 per leaf: every capability each task's user-observable signal
asserts, bound to evidence. Verified 2026-07-06 against `main`. Any binding
resolving to a FAIL value (`declared-only`/`test-only`/`producer-absent`/
`producer-downstream`/`producer-extent-short`/`rejection-absent`) blocks the
batch. **Result: no hard FAIL — one soft/deferred binding (θ·E2, M1-owned) is
recorded, not asserted as a hard leaf signal.**

Evidence vocabulary: `grep:<file>:<line>` = wired/present on main today ·
`producer:<label> upstream` = delivered by an upstream task in this batch's
dependency closure · `substrate:<x>` = language/tool substrate verified ·
`self` = a property/rejection the task's own code produces (bound by the
task's own RED test) · `soft:M1` = M1-owned, consuming-direction registration.

Tool substrate (shared by δ; verified via `--help`): `pyright --outputjson`
(1.1.408), `ruff --output-format json` (0.15.9, `possible values: … json`),
`cargo --message-format json` (present). `shlex`, `StrEnum` = py3.13 stdlib.

---

## α — FailureCategory enum + one policy table

Signal: a synthetic category with no policy row raises at import (exhaustiveness);
all on-the-wire category strings stay byte-identical.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The 5+ scattered registries exist to be derived from one table | `grep:verify.py:577` (_ARCHIVE_DENY_LIST), `:584` (_CATEGORY_PRIORITY), `:606` (PREEXISTING_BREAK_SKIP), `:640` (endswith heuristic); `grep:verify_runner.py:109` (UNSCOPED_TYPECHECK_*); `grep:merge_queue.py:721`,`:924`; `grep:workflow.py:4894`,`:4939` | PASS |
| `StrEnum` keeps JSON byte-identical (verify_runner Invariant 1) | `substrate:StrEnum` (members are `str`; `json.dumps` unchanged) | PASS |
| Missing-policy-row is rejected at import | `self` — F1 exhaustiveness assert; RED test adds a member sans row, observes ImportError-time assert fire | PASS (self-produced rejection) |

## β — VerifyCmd structured command model

Signal: `render(parse(x))` round-trips a well-formed command; an unparseable
command → OPAQUE and is never scoped.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The six string-rewrite helpers + bash-wrap exist to delete | `grep:verify.py:85` (_scope_command), `:204` (_strip_directory_flag), `:222` (_strip_leading_cd), `:254` (_reproject_bare_uv_run), `:666` (_force_serial_pytest), `:1246` (_scope_cargo_workspace), `:2399` (_maybe_govern_merge_cmd) | PASS |
| shlex parsing | `substrate:shlex` (stdlib) | PASS |
| OPAQUE is never scoped (rejection of the scope mutators on OPAQUE) | `self` — Invariant P1; RED test feeds the historical broken layout, asserts `scope_to` is a no-op | PASS (self-produced rejection) |
| render round-trip argv-equivalence | `self` — Invariant P2 | PASS (self-produced) |

## γ — derive_verify_plan() + FileKind

Signal: plan goldens — root conftest→FULL_SUITE, data-module→SKIPPED-with-reason,
structural→unscoped pyright in both paths.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The two twice-fixed functions exist to unify | `grep:verify.py:1437` (scope_module_config), `:1548` (_build_fallback_config) | PASS |
| has_conftest→full-suite convention | `grep:verify.py:1478`,`:1525` | PASS |
| Historical incident diffs are reconstructable (goldens, G6) | `producer:git-history` — commits exist: `d7504d432d`+`cb7277926d` (task 1077), `4fbed6c4fb`+`7c9b316260` (task 1852), all `git cat-file`-confirmed | PASS |
| `VerifyCmd` carried in each `PlannedRun` | `producer:β upstream` (β is γ's prereq; extent = the full VerifyCmd model) | PASS |

## δ — tool-dispatched classify_failure

Signal: a cargo token can no longer swallow a pytest/rustc line; expected
categories derived from historical fix commits.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| pyright/ruff/cargo structured-output flags | `substrate` — all three `--help`-verified on pinned versions | PASS |
| The tool-blind classifier sites exist | `grep:verify.py:522` (_classify_failure), `:359` (_extract_cause_hint), `:703` (_summarize_checks) | PASS |
| Cargo re-grounding goldens (G6, expected categories from real commits) | `producer:git-history` — `1703f86f95`,`18f57fe922`,`1aed67cd56`,`264d5b5e8a`,`b40a3e0a7f` (tasks 1103/1109/1116) all `git cat-file`-confirmed | PASS |
| `FailureCategory` return type | `producer:α upstream` | PASS |
| `ToolKind` at the call site | `producer:β upstream` | PASS |

## ε — CheckRun / VerifyAttempt dataclasses

Signal: an env-recovery run hitting the wall clock cannot flip category to
infra_timeout while leaving `timed_out=False` (the 2735-2744 drift).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The 15 parallel locals + two formula copies exist | `grep:verify.py:704` (test_rc/test_out/test_timed_out signature region), survey-cited `:2631-2665`,`:2672-2678`,`:2747-2753`; drift documented `:2735-2744` | PASS |
| `pure_timeout_failure`/`any_timed_out` single-source computation | `self` — VerifyAttempt derived properties; RED test drives both branches | PASS (self-produced) |
| classifier produces the category consumed in the attempt | `producer:δ upstream` | PASS |

## ζ — typed BlockRecord + b3_gate branches on block_class

Signal: POST_MERGE_RED_MAIN still hard-aborts (task 1680); legacy proposal
(no block_class) routes identically; MERGE_VERIFY_RED is gateable.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| b3_gate's prose-prefix + status-sniff + proposal-read exist to replace | `grep:b3_gate.py:42` (POST_MERGE_RED_MAIN_REASON_PREFIX), `:347` (startswith), `:359` (status sniff), `:398` (_read_latest_proposal) | PASS |
| Producer sites (workflow + entry builder) | `grep:workflow.py:8200` (_spawn_dry_run_unblock), `grep:dry_run_unblock.py:467` (_build_entry) | PASS |
| `unblock_types.py` is a clean new module | `grep`-absent (confirmed not on main) → `producer:ζ` (this task) | PASS |
| POST_MERGE_RED_MAIN hard-abort preserved (rejection, task-1680) | `self` — Invariant B2; RED test asserts ABORT before risk/git check | PASS (self-produced rejection) |
| Legacy-proposal bridge (no block_class) | `self` — Invariant B3; RED test feeds a block_class-less dict | PASS (self-produced) |

## η — merge_queue block path spawns the dry-run investigation

Signal: a merge-verify RED now yields a `metadata.dry_run_proposals[]` entry
(today: none); b3_gate returns non-ABORT for a low-risk one.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The block site holds worktree + failing VerifyResult + scoped diff (end-to-end, G6 branch-3) | `grep:merge_queue.py:602` (merge_wt param), `:692` (verify: VerifyResult), `:829` (_derive_task_files_from_git(merge_wt,…)) — all in the dependent's own extent | PASS |
| `run_dry_run_unblock` keyword API exists | `grep:dry_run_unblock.py:244` (task_id, worktree, reason, detail, scheduler, mcp, config, …) | PASS |
| `BlockRecord(MERGE_VERIFY_RED)` | `producer:ζ upstream` | PASS |
| scheduler/mcp/config handles threaded into the merge worker | tactical (Open Q4) — capability exists (workflow calls it); wiring is impl, not a substrate gap | PASS (capability present; wiring tactical) |

## θ — git_ops.ephemeral_worktree() extraction

Signal (hard leaf): both verify probes run through the CM and it never invokes
`git worktree prune` (scoped remove only).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Both verify probes exist to consume the CM | `grep:verify.py:3260` (verify_failure_is_preexisting_on_main), `:3431` (run_main_tip_sweep) | PASS |
| git worktree add/remove primitives | `grep:git_ops.py` (worktree add/remove used throughout; the probes call them today) | PASS |
| E1: the CM never broad-prunes (DD5 as code — the incident-prevention value) | `self` — RED test asserts `git worktree prune` is never invoked | PASS (self-produced, the load-bearing signal) |
| E2: kind prefix registered into PROTECTED_PREFIXES so reapers skip it | `soft:M1` — M1-owned registry, not yet filed; **consuming-direction** registration. **Not asserted as θ's hard leaf signal.** At decompose: wire dep on M1's chokepoint task if filed → `producer:M1 upstream`; else θ registers if the symbol exists at dispatch and the E2 assertion is deferred (Open Q1) | DEFERRED (not a hard leaf signal; recorded, not faked PASS) |

## ι — B+H integration gate (the leaf)

Signal: the boundary-test module (rows 1-12 of the PRD sketch) passes.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| VerifyCmd render + OPAQUE (rows 1-2) | `producer:β upstream` | PASS |
| Plan goldens (rows 3-5) | `producer:γ upstream` + `producer:git-history` (diffs) | PASS |
| Classifier isolation (row 6) | `producer:δ upstream` | PASS |
| Category exhaustiveness (row 7) | `producer:α upstream` | PASS |
| CheckRun timeout consistency (row 8) | `producer:ε upstream` | PASS |
| Block-path end-to-end + POST_MERGE_RED_MAIN + legacy bridge (rows 9-11) | `producer:ζ,η upstream` | PASS |
| ephemeral_worktree no-prune (row 12) | `producer:θ upstream` (E1 only; E2 per θ note) | PASS |

All producers are **upstream** of ι in the DAG (anti-inversion satisfied):
ι depends on θ (verify spine tip) and η (block spine tip); transitive closure =
{α,β,γ,δ,ε,ζ,η,θ}.

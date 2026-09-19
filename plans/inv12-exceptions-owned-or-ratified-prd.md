# INV-12 `exceptions-owned-or-ratified` — exception register, inline-suppression ratchet, owner-liveness sweep

**Status:** active — authored 2026-09-18 from Leo's ruling of the same day (brief
`~/.claude/spawn-briefs/2026-09-18-inv12-exceptions-owned-or-ratified-prd.md`).
**Type:** greenfield mechanism + a new design invariant. **B+H** (contract +
boundary tests): five packages touched, nine mechanisms, and the merge gate is a
load-bearing seam.
**Code anchors** verified against main `890be05658` (2026-09-18). Main moves
fast — everything below is cited by symbol; re-locate at implementation time.
**Measurements** are dated snapshots taken 2026-09-18 on that commit; they are
provenance, not live counts. CLI forms are written
`uv run --project shared python scripts/<x>.py …` throughout: a bare `python3`
under `orchestrator/src/orchestrator/verify.py::_target_subprocess_env` cannot
import `shared` modules.

## Goal

Make this true, and keep it true:

> For allow-lists that track exceptions to a rule intended to be uniform, every
> entry is either a **debt** exception carrying the task that owns removing it —
> and that task is live (not `done`, `cancelled` or `deferred`) — or a **policy**
> exception explicitly ratified by the operator and marked with the ratification
> reference. No entry may be neither. Inline suppressions are in scope, under
> their own register and ratchet.

What an operator or agent observes when the batch lands:

1. A branch that adds an inline suppression with no disposition, or adds an entry
   to a governed list with no disposition, goes RED at the merge gate with a
   message naming the site and the accepted forms. A branch that adds nothing is
   green: this is a ratchet, not a day-one gate.
2. When a debt owner reaches `done`/`cancelled`/`deferred`, or does not resolve,
   the operator hears within a day, once, not once per night: a task for inline
   suppressions, an escalation for a named-list entry.
3. `uv run --project shared python scripts/exception_register.py --report`
   prints the whole register: governed lists with their dispositions; inline
   suppressions per kind / code / file; a headline `grandfathered (unowned): N`
   that is never folded into another bucket; baseline slack; the age of the last
   sweep.
4. `/prd` decompose (G7) and `/review` phase 2 cite `exceptions-owned-or-ratified`
   by slug, with fixtures.

## Background

**The motivating incident (already owned — do not duplicate).**
`orchestrator/src/orchestrator/verify.py::_KNOWN_LOAD_FLAKE_NODEID_RES` gained
`test_cli.py::test_verify_merge_cancel_end_to_end` on 2026-07-16 (`1ef27fe25a`),
between de-flake tasks 2350 (done 07-08) and 2733 (done 07-18). The entry
outlived the fix, carried no owner, and the test recurred at task 4545's gate on
2026-09-06 (esc-4545-6). Task 5578 (`merge-deferred`, 2026-09-18) now carries
the retirement of that list. A missing-id check would not have caught the
incident; only a **liveness** check does — 2733 existed, and was done.

**The general defect is an unrecorded link, not an absent owner.** Every task
the governed lists *cite* is `done` (5147, 5333, 4246, 4016, 4793, 4796, 2698).
Several lists nevertheless have a live owner that the list does not name: 5149
(pending) migrates the 61 timeout markers, 5034 (pending) retires the serial
merge worker and the reach-back patch guard, 4354 (pending) burns down the Rule B
budget, 4920 and 5215 (pending) the Rule C budget. Nothing connects those owners
to the entries, so nothing notices when one completes and entries remain — the
2733 shape — and other lists (the silent-fallthrough allowlist, the legacy key
and filename lists) have no live owner at all.

**Named lists.** A four-pass census found 52 named tables; an adversarial pass
added `shared/tests/loop_blocking_allowlist.py::AUDITED_SITES` (62 rows, its own
`DISPOSITIONS = {accepted, filed, to_file}` vocabulary) and
`shared/tests/config_dir_archival_allowlist.py`. Two readers classified the same
rows and disagreed on ten, which is why D1 is an operational test and the
borderline rows go to the operator. One brief census row is a false positive:
`dashboard/tests/test_memory_evals_data.py::_FP_GRANDFATHERED` is a fixture
fingerprint string.

**Inline suppressions — stock.** Token-accurate census (`tokenize` COMMENT tokens
over `git ls-files '*.py'`, 1,918 files, 0 tokenize failures):

| kind | count | in tests | consumer at the gate |
|---|---|---|---|
| `type: ignore` | 3,996 | 93% (`orchestrator/tests` alone 2,963) | pyright; the bracketed code is NOT validated — a made-up code suppresses like a bare ignore |
| `noqa` | 1,818 | 68% | ruff for selected rules only (`select = E,F,UP,B,SIM,I` in all 8 `pyproject.toml`); 1,499 are flagged unused by `uv run ruff check --extend-select RUF100 --statistics --exclude graphiti --exclude mem0 --exclude .worktrees .` (952 are `PLC0415`) |
| `pragma: no cover` | 116 | 43% | none — no coverage config or `--cov` anywhere |
| `pyright: ignore` | 72 | 90% | pyright; the code IS validated |
| `nosec` | 0 | — | none (the brief's "5" were `grep -F` hits on "nanosecond") |

697 files carry a marker; the top file carries 391; top `type: ignore` codes are
`attr-defined` 1,278, `method-assign` 954, `arg-type` 506. Zero of the 413
markers with a trailing reason cite a task, escalation or ticket.

**Inline suppressions — flow.** `git log --since=30d -p -- '*.py'`: 1,052 marker
lines added, 336 removed, **865 content-net-new ≈ 29 per day**, 75% in tests.
Top added: `noqa: PLC0415` 223 (a rule ruff does not run), `type:
ignore[method-assign]` 95, `[attr-defined]` 92, `[arg-type]` 77, bare `noqa` 53,
`noqa: E402` 47, `[possibly-unbound]` 46, `noqa: BLE001` 43. What an agent does
at a red gate is determined by this table, not the stock one.

**Ratchet substrate.** The repo has two ratchet idioms. *Scalar*:
`scripts/merge_lane_metrics.py` (committed JSON, `now > was`, derived totals,
ceilings for items absent from the baseline, INV-11 polarity, exit 0/1/2) and the
in-source per-file budgets in `fused-memory/scripts/check_bare_magicmock_config.py`.
*Multiset*: `shared/tests/silent_fallthrough_scan.py::reconcile_against_allowlist`
and the loop-blocking ledger (content-hashed keys without line numbers, `Counter`
subtraction, enforced stale-entry removal). The scalar idiom's
`--write-baseline` absorbs a task's own increase; task 5406 (in-progress) owns
that hole.

## Sketch of approach

Nine mechanisms, one vocabulary.

1. **The invariant text** — INV-12 in the pinned sites.
2. **Disposition vocabulary** (`shared/src/shared/governed_exceptions.py`) — the
   typed union everything else speaks, and the one parser for the inline marker
   grammar.
3. **Ratification table** (`docs/legibility/exception-ratifications.yaml`) — the
   closed world of operator rulings. A `Policy` names a row; nothing else is a
   valid ratification.
4. **Governed-list declarations** — `governed_exceptions(...)` calls in the
   owning package's test tree, importing the container they govern.
5. **Multiset ratchet kernel** (`shared/src/shared/ratchet.py`).
6. **Inline scanner and ratchet** (`scripts/inline_suppressions.py`,
   `scripts/inline_suppression_baseline.json`) — scan, consumer model, class
   table, `--check / --seed / --tighten`.
7. **Exception register** (`scripts/exception_register.py`) — static discovery
   of every declaration, closed-world check of every `Policy`, `--report /
   --json`.
8. **One guard-test module per half** under `scripts/tests/` — the complete,
   tree-pure enforcement point, run once per merge under `merge_verify_breadth:
   "full"`.
9. **Owner-liveness sweep** (`scripts/exception_owner_sweep.py` + a timer pair)
   — the only non-hermetic part.

Hermetic and non-hermetic halves never mix. Everything the gate runs is a pure
function of the tree. Anything that reads the task store or the ticket store
lives in the sweep, which runs nightly from the main checkout — `.taskmaster/`
exists nowhere else, and tests whose inputs are outside the tree are the class
that broke verify caching on 09-14..16.

## Resolved design decisions

**D1 — What is governed: suppressors in, grants out.** A list is governed iff
its entries **silence a detector** — a lint, guard test, census warning, alarm,
sweep, failure classifier or finding pipeline would have reported the entry's
subject, and the list makes it not. A list whose entries **grant a capability**
(who may promote, what auto-heals, which fields hot-reload) is design, reviewed
as design. Appendix A part 1 is the unambiguous core, declared without waiting
for anyone. Part 2 is the borderline set; the operator rules each row in δ, and
the rulings seed the register's reasoned `NOT_EXCEPTION_LISTS` classification
(D14).

**D2 — Three legal states, and the third is not an entry-level state.**
`Debt(owner)` with `owner` a `TaskRef(id)` or `TicketRef('tkt_…')`.
`Policy(ratified)` with `ratified` the id of a row in the ratification table.
*Grandfathered* is legal only as membership in a machine-generated, shrink-only
baseline seeded when its rule was adopted; each baseline is itself ONE governed
entry carrying a disposition, and the report counts grandfathered entries under
their own headline. The invariant's Rule says this plainly: true per entry for
anything added or touched after the seed; N legacy entries are neither, and N
may only fall. `TicketRef` exists because no agent role holds `resolve_ticket`
(`orchestrator/src/orchestrator/agents/roles.py`: `submit_task` is held by
architect, implementer, steward and deep reviewer; none can learn the task id);
tickets persist (`fused-memory/src/fused_memory/middleware/ticket_store.py` has
no delete path). A ticket that resolved `failed` or `refused`, or to a task that
is not live, is a dead owner.

**D3 — Policy is a closed world.** `docs/legibility/exception-ratifications.yaml`
holds one row per operator ruling: `id`, `date`, `source` (an escalation id, a
sitting ruling, or a document path — provenance, never liveness-checked, because
`escalation/src/escalation/archive.py::prune_archive` deletes resolved
escalations after 30 days), and `covers`. A `Policy` whose id is not a row is a
violation, checked hermetically. Minting policy is therefore one reviewable edit
to one file; an agent that invents a reference cannot make it resolve.

**D4 — Declarations live in the owning package's test tree.**
`governed_exceptions(list_id, rule, keys, *, default=None, default_covers=None,
dispositions={})` is called from a test module that imports the container, one
module per package. The container keeps its shape and its home. Production
modules and stdlib-only scripts gain no import and no import-time raise:
`check_bare_magicmock_config.py` runs under bare `python3` in every member
`lint_command` and in `hooks/project-checks`, `check_dashboard_unit_parity.py`
under bare `python3` in `scripts/setup-host.sh`, and every project's orchestrator
imports `verify.py`. A mis-edited disposition fails a test, never an import.
`default` covers a **count**, not a list: `default_covers=61` is a literal,
checked equal to the number of keys without an override, so a new entry needs an
explicit override or a visible increment beside the disposition it borrows.
Rejected: a central sidecar register (needs the same key enumeration and hides
the disposition from both files); rewriting containers as typed mappings (churn
in hot files for no added guarantee).

**D5 — Division of knowledge.** The static reader (AST over `git ls-files
'*.py'`, imports nothing) knows `list_id`, `rule`, `default`, `default_covers`
and the overrides — all literals. The runtime call alone knows the key set, and
checks keys ↔ dispositions when the test module is collected. The report prints
defaults with their counts, and overrides. A non-literal argument other than
`keys`, a duplicate `list_id`, or an unparseable file is an instrument failure
(exit 2), never a skip (INV-11). An override restates its entry's key; drift
between the two is loud by construction.

**D6 — Inline dispositions are inline.** Grammar, in the same COMMENT token,
after the suppression:

```
# type: ignore[attr-defined]  # debt: task 5601
# noqa: E402  # debt: ticket tkt_0RTCC80EM92A7WD08D6RF6ZZPY
# pyright: ignore[reportArgumentType]  # ratified: inv12-day-one-test-doubles
```

One parser in `shared.governed_exceptions` returns D2's values — parse at the
boundary into typed values (heuristic 12). A marker that does not parse, or that
sits on a line with no suppression, is a violation. Verified with positive
controls 2026-09-18: ruff and pyright honour the suppression through the trailing
text; no `Invalid # noqa directive` warning; `RUF100` still sees through it.
Rejected: a sidecar keyed by an anchor — the pin ruling forbids line anchors, a
content-hash anchor re-blesses on every edit, the agent hand-computes a hash in a
second file, and the reader at the site cannot tell debt from policy.

**D7 — The inline baseline is a content-keyed multiset.** Key = `(kind, sorted
codes, sha256(stripped physical line)[:12])`, no path. The check is
`unowned(current) − baseline = ∅`; equality and shrinkage are green. This departs
from the brief's per-file count:

- *A touch is visible.* Editing the line that bears the marker changes the key.
  Editing other lines of the same statement does not; that is the limit of
  "touched", stated rather than hidden.
- *Splits and renames of files are free.* Under per-file counts, splitting the
  391-marker file makes every moved marker new and regenerating the baseline the
  only exit.
- *Ordinary tasks never edit the baseline.* No hot file, no regenerate verb:
  `--seed` refuses when a baseline exists, `--tighten` writes `baseline ∩
  current`.

Its costs, accepted knowingly: 67% of sites share a key with another site, so an
un-tightened baseline lets an identical line in where one was removed (slack);
two branches can each be green and red together when both spend the same slack,
or when one tightens what the other re-adds — rare, and the red names the line.
The sweep files one `tighten` task whenever slack is non-zero (D11). The
baseline is unreviewable by content — 12-hex keys — and acceptable only because
its legal diff is deletions; `--report` carries an example line per key. A mechanical
rename rewrites every marker-bearing line it touches; under the ruling's words
that is a touch, and the exit is one follow-up ticket cited on all of them. δ
asks the operator to confirm or overrule that reading.

**D8 — A new marker nothing consumes is rejected, not registered.** The scanner
models consumers: `type: ignore` and `pyright: ignore` → pyright; `noqa` codes →
ruff when the nearest `pyproject.toml` selects the rule, or a first-party checker
when the code is one it defines (`bare-magicmock`, `bare-dataclass-double`);
`pragma: no cover` and `nosec` → none today. A marker absent from the baseline
with no consumer fails with "delete this marker — no tool reads it" and accepts
no disposition. That removes the largest single inflow (`PLC0415`) honestly.
Grandfathered dead markers stay counted.

**D9 — Ratified classes are the pressure valve, and the operator holds it.**
`scripts/inline_suppressions.py::RATIFIED_SUPPRESSION_CLASSES`: key
`kind[code]@scope`, scope ∈ `src | tests | any`, each row a `Policy`. A matching
site is policy by reference, outside the unowned multiset, and counted per class
in the report so blanket policy stays visible. The table ships empty; δ carries
the inflow table so the operator sizes the valve against the filing rate each
row prevents. Rows added after the seed turn their sites' baseline keys into
slack, which the next tighten removes; removing a row makes its sites new.

**D10 — No burn-down is scheduled; that is the operator's to confirm.**
Grandfathered suppressions convert when their line is touched. The baseline
unit's disposition is `Debt(TaskRef(δ))` until the operator rules — honest and
live: owned by the pending decision. Recorded opportunity, not queued: `RUF100`
autofix would delete the 1,499 dead `noqa` markers.

**D11 — The sweep: venue, outputs, idempotence.** Nightly systemd timer,
`WorkingDirectory=` the main checkout, because owner liveness changes with task
status, not with the tree (the offline lane's merge-landed clock and its
per-failing-test-set red path are both the wrong shape). It reads only the
register's `--json`; resolves the store with
`scripts/_task_db_scan.py::tasks_db_path` / `::connect_ro`; the dead set is
`shared.task_statuses.TERMINAL | {TaskStatus.DEFERRED}`; before calling an owner
dead it follows `x_coalesced_into` and curator combine markers to a live
successor, and re-reads status immediately before acting (INV-3).

- *Key.* One finding per `(scope, dead owner)`; scope is a `list_id`, or
  `inline`. The key rides in `metadata.x_inv12_finding` (Tier-C).
- *Idempotence is the sweep's own.* It skips when a live task or pending
  escalation carries the key. The curator is not the mechanism: its `drop` pool
  excludes only `cancelled`, so it can fold a finding onto the very `done` owner
  it reports (`fused_memory/middleware/task_curator.py`), and the store's
  `ux_tasks_candidate_key` index keys on title **and** files.
- *A terminal finding with the condition unchanged escalates; it is never
  re-filed.* That ends the loop where a finding task re-owns to itself.
- *Output by scope.* Inline → a task through
  `scripts/legibility/census_trigger.py::post_mcp_tool_call` (`submit_task`,
  `planning_mode` omitted). Named list → an escalation: rare, consequential, and
  a decision rather than an edit.
- *Notification.* On first sight of a live owner, the sweep merges an
  `x_inv12_owns` note into that task's metadata, so the record an operator or a
  later agent reads says what cites it.
- *Storm bound.* At most 10 outputs per run; the overflow is one roll-up with a
  constant key whose body says "run `--report`" (INV-4).
- *Slack.* Non-zero baseline slack files one constant-key `tighten` task.
- *Fail loud.* Store or MCP unreachable → zero outputs and an `escalate_info`;
  the unit wrapper itself exits 0, as
  `scripts/memory-metadata-coverage-census.sh` does, because a failed oneshot
  stays failed and silently stops the timer. `--dry-run` corroborates and prints,
  files nothing, writes no stamp.
- *Its own liveness.* The run stamp feeds `--report`, which exits non-zero when
  the stamp is older than two timer periods. Activation is an explicit operator
  leaf (κ2), and θ's signal proves it on the real report. A deterministic
  predicate leaf was declined: `before_done.script` must exist and be executable
  when the task is filed, and a second copy of
  `scripts/legibility/check_trickle_liveness.sh` for another unit is lockstep
  duplication. A timer that dies later is heard only by whoever reads the
  report; that residual is named in `OPERATIONS.md`, not hidden.
- *Where it runs from.* `--project-root` defaults to the main checkout (first
  entry of `git worktree list --porcelain`), so `--dry-run` works from a task
  worktree; `tasks_db_path(project_root)` is a plain join and `.taskmaster/`
  never exists in a worktree.

What the sweep guarantees is detection when a nominal owner dies. It does not
make a live owner diligent; nothing mechanical can, and the ruling does not ask
for it.

**D12 — Enforcement is two guard tests, and the seed is an operator step.**
`scripts/tests/` runs once per merge under breadth `full`, under
`uv run --project shared`, with no edit to nine `lint_command` strings and their
drift mirrors. A root `lint_command` entry would never run (the yaml's own
commentary: with `module_configs` non-empty it is not invoked), a module-scoped
check is blind to the 67% of sites that pool across modules, and an unscoped
check in every leg costs the full scan nine times on a serial gate. Scan budget:
≤10 s, from a byte prefilter — tokenize only files containing a marker substring
(697 of 1,918; the measured full tokenize pass is 11.9 s), AST only files
containing `governed_exceptions(`. `--check PATH…` remains for early feedback and
labels its green `partial`. The inline guard enforces iff the baseline file
exists; until then it reports and skips loudly, naming κ1. Seeding in a worktree
would go red on the merged tree at ~29 new markers a day, so κ1 seeds on main
with the merge queue halted and commits the baseline alone.

**D13 — Two ratchet idioms; SPOT per idiom; this PRD owns the multiset kernel.**
`shared.ratchet` is new. The brief's "reuse the merge-lane module" is declined:
that module is 1,634 lines and 60% cluster-bound, its gate test reaches its
private names, tasks 5414 and 5406 are both about to edit it, and its contract
has clauses with no multiset analogue (ceilings, derived totals). Porting it onto
the kernel would share one line of `Counter` arithmetic at real regression risk
to someone else's gate. The natural future adopters are the multiset reconcilers
in `shared/tests/`; they are named, not scheduled.

**D14 — Unregistered-list detection is report-only in v1.** `--report` lists
module-level collection assignments whose names match the census pattern and
carry no declaration, minus `NOT_EXCEPTION_LISTS` — rows written `path::symbol`
with a reason. A classification says the rule does not apply; it is not a
tolerated violation, so it needs a reason, not a ratification.

**D15 — Agents and reviewers are told before the gate tells them.** The
implementer prompt gains the accepted forms, "fix the type before suppressing
it, and never trade an ignore for `Any` or `cast`", "one follow-up ticket per
branch, cited on every marker it adds", and "never cite your own task". The
reviewer prompt gains six blockers no tree-pure check can see: self-citation;
citing a task or ticket unrelated to the debt; an `Any`/`cast` contortion in
place of a scoped ignore; any added line in the inline baseline; an incremented
`default_covers`; any edit to the ratification table the task's brief did not
order.

## Pre-conditions for activating

- None external for α, β, γ1, γ2, ε1, ε2, η, λ.
- δ needs the operator's rulings, and dispatches only after γ2 has created the
  ratification table it adds rows to.
- κ1 and κ2 need an operator at the host (halt the queue and seed; install the
  timer).
- Task 5578 may retire `_KNOWN_LOAD_FLAKE_NODEID_RES` before ε1 lands; ε1
  declares what exists at that time.

## Cross-PRD relationship

| Other PRD / work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| tasks 5149, 5034, 4354, 4920, 5215 (pending) | consumes them as owners | `Debt(TaskRef(n))` in ε1/ε2's declarations | those tasks own the burn-downs; this PRD only names them | live |
| task 5578 (merge-deferred) and the spawned flake-entry session | consumes the outcome | `verify.py::_KNOWN_LOAD_FLAKE_NODEID_RES` | 5578 | running |
| `plans/merge-lane-quality-prd.md`; tasks 5406 (in-progress), 5414 (pending) | none in code | `scripts/merge_lane_metrics.py` is NOT edited by this batch; its baseline unit is declared from `orchestrator/tests/` | merge-lane PRD; 5406 owns the absorb hole | no seam |
| tasks 5086, 5096 (pending) | adjacent | `reportUnnecessaryTypeIgnoreComment` measurement | those tasks; D8's consumer model is where a result would plug in | no seam |
| tasks 5094, 4600 (pending) | adjacent | consolidation of whole-tree AST scans in guard tests | those tasks; D12's prefilter keeps this batch's scan out of their cost problem | no seam |
| task 4941 (pending) | same file | `scripts/tests/test_design_invariants_consistency.py` | 4941; no leaf of this batch edits the test module. δ's rulings file and θ's `OPERATIONS.md` section name no slug but INV-12's own, so they stay under `_ENUMERATION_THRESHOLD` and need no `PINNED_SITES` row; a leaf that finds otherwise serialises behind 4941 | no seam |
| `plans/flake-ledger-prd.md` (3790 pending) | sibling invariant | `flake_debt.owner_task_id` re-corroborated in `open_debt()` | ledger PRD owns runtime rows in `runs.db`; this PRD owns tree-resident lists | no code seam |
| INV-8 ledger `shared/tests/loop_blocking_allowlist.py` | vocabulary seam | its `DISPOSITIONS` (`accepted / filed / to_file`) stay; ε2's declaration maps `filed → Debt(TaskRef)`, `to_file → Debt(TicketRef)` or the row's owner, `accepted → Policy` via δ | INV-8's gate owns the ledger; this PRD owns the mapping | queued |
| `plans/confusion-reduction-prd.md` | produces a slug | `invariant_violated` reads `design-invariants.md` at run time | that PRD; nothing to wire | wired on α |
| `plans/design-invariants-gate-prd.md` (3802) | extends the family | the five sites `test_design_invariants_consistency.py::PINNED_SITES` already pins; the family is derived from the doc's headings | this PRD (α) | queued |

No reciprocal-ownership ambiguity.

## Contract (B+H)

**Vocabulary** (`shared.governed_exceptions`), frozen dataclasses:
`TaskRef(id: int)`, `TicketRef(id: str)`, `Debt(owner: TaskRef | TicketRef)`,
`Policy(ratified: str)`, `Disposition = Debt | Policy`. Constructors validate
shape only; whether a `Policy` id is a table row, and whether a path exists, is
the register's check, not the dataclass's.

**Declaration.** `governed_exceptions(list_id: str, rule: str, keys:
Iterable[str], *, default: Disposition | None = None, default_covers: int | None
= None, dispositions: Mapping[str, Disposition] = {}) -> GovernedList`. Raises
`UndisposedException` naming `list_id`, the key and the accepted forms; raises on
an override for an absent key, a duplicate key, `default` without
`default_covers` or a count mismatch. `list_id` is dotted and globally unique. A
baseline unit is a declaration with exactly one key: the baseline's
repo-relative path.

**Inline grammar.** `parse_disposition_marker(comment: str) -> Disposition |
None`; `None` when neither keyword is present; raises `MalformedDisposition`
otherwise. One disposition covers every suppression in that comment.

**Ratification table.** YAML list of `{id, date, source, covers}`; ids unique,
kebab-case; loaded by the register only.

**Kernel** (`shared.ratchet`). `excess(current, baseline) -> Counter`;
`tighten(current, baseline) -> Counter` (pointwise minimum); `slack(current,
baseline) -> Counter`; `load / dump` of a one-entry-per-line, stable-order JSON
with a `params` block whose mismatch refuses comparison; an enumeration flagged
incomplete refuses comparison. No function can add a key to an existing baseline.

**CLIs.** Exit 0 clean (a scoped run says `partial`); 1 violations, one per
line: site, kind, codes, the reason, the accepted forms; 2 instrument failure —
the entry point is wrapped so an `ImportError` is 2, never 1.
`exception_register.py --json` is the sweep's and the report's only data source.

**Sweep.** Exit 0 always from the unit wrapper; the result object `{checked,
findings, filed, escalated, skipped_existing, overflow, slack, last_run, ok}` is
written to the state path the report reads.

**Ordering invariants.** λ lands before κ1, so the forms are known before
anything is red. The ratification table exists (γ2) before δ dispatches. ζb runs
after δ, ε1, ε2 and κ1. κ2 follows ζb, so the sweep never runs while a
declaration still cites the closed δ. The pure-gate leaves (δ, κ1, κ2) go `done` on `resume`
— `orchestrator/src/orchestrator/deterministic_runner.py` does not consult
`delivered_checks` — so what holds their dependents is the producer's
`delivered_checks` in the scheduler's delivered-check cache: ζb does not
dispatch until the ruled rows and the baseline are on main, whoever closed the
gate.

## Boundary-test sketch

Scenarios 1–11 are γ1's; 12 is β's; 13–15 are γ2's (the closed-world check of
inline `ratified:` ids is γ2's too, reading γ1's `--json`); 16–24 are η's. θ
does not re-run them.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | New bare `type: ignore[arg-type]` | fixture tree + seeded baseline | exit 1; names file, kind, code, both forms |
| 2 | Same with `# debt: task N` | same | exit 0; site under `debt` in `--json` |
| 3 | Grandfathered marker, code on its line edited | same | exit 1 |
| 4 | File with 40 grandfathered markers renamed and split | same | exit 0 |
| 5 | Marker removed | same | exit 0; slack 1; `--tighten` drops it; second `--tighten` is a no-op |
| 6 | `--seed` with a baseline present | same | exit 2, baseline untouched |
| 7 | `# debt: soon`; and `x = 1  # debt: task 5` | same | exit 1 naming each comment |
| 8 | Untokenizable tracked file | same | exit 2 naming the file |
| 9 | New `# noqa: PLC0415`; new `# pragma: no cover` | rule unselected; no coverage config | exit 1 "delete this marker"; a disposition does not rescue it |
| 10 | Site matches a ratified class | class row + table row | exit 0; counted under the class |
| 11 | Baseline absent | same | guard test skips with a reason naming κ1; `--check` exit 0 `advisory` |
| 12 | Governed list gains a key, no override, `default_covers` unchanged | declaration in a test module | collection raises `UndisposedException` naming the key |
| 13 | Declaration with a computed `list_id`; two declarations sharing one | same | exit 2 |
| 14 | `Policy('no-such-row')`, in a declaration and inline | table without the row | exit 1 naming the id and the table path |
| 15 | Static report of a defaulted list | same | prints default, `default_covers`, N overrides — no key set |
| 16 | Owner task marked `done` | temp tasks.db + stub MCP | inline scope: one `submit_task`; list scope: one escalation; second run outputs nothing |
| 17 | Owner id absent; `TicketRef` resolving `failed`; `TicketRef` → cancelled task | same | one finding each |
| 18 | Owner `done` with `x_coalesced_into` → a live task | same | no finding; report line "via successor" |
| 19 | Finding task `done`, condition unchanged | same | one escalation, no task |
| 20 | 14 dead owners in one run | same | 10 outputs + 1 roll-up with the constant key |
| 21 | Store unreadable | same | zero outputs, one `escalate_info`, stamp not advanced, `ok: false` |
| 22 | `pending` at scan → `done` at re-read; and `done` at scan → `pending` at re-read | injected | first filed, second not |
| 23 | `--dry-run` | any | prints would-file findings; zero writes; no stamp |
| 24 | Slack non-zero on two consecutive runs | same | one `tighten` task in total |

## Decomposition plan

Greek labels; ids at decompose. Every leaf is reviewed against
`docs/code-quality.md` — especially *structured data instead of meaningful
strings*, *files make internal sense in isolation*, *no file too large*, and the
stance that tests reaching a module's internals are an interface smell
(declarations import private containers the guard tests already import; they add
no new reach).

- **α — INV-12 in the pinned sites.** `docs/legibility/design-invariants.md`
  (Rule per D2 including the grandfathered sentence; Checkable question per D1;
  Evidence: the incident and the unrecorded-owner table; House pattern; a
  Family-boundary note against INV-7; header and Fixtures date lists),
  `docs/legibility/design-invariants-fixtures.md` (`INV-12-PRD`, `INV-12-CODE`, a
  dated addendum walk), `skills/prd/references/gates.md` (family row +
  `inv-trigger-shapes` span). `CONTRIBUTING.md` is pinned as an absence and
  `docs/code-quality.md` as a partial pairing: add nothing to either.
  `complexity='simple'`. Signal: `uv run --project shared pytest
  scripts/tests/test_design_invariants_consistency.py` green with INV-12 in the
  derived family; a G7 walk of `INV-12-PRD` returns a hit. Unlocks θ, which
  edits the same section.
- **β — `shared.governed_exceptions` + `shared.ratchet`.** The Contract's
  vocabulary, declaration, grammar and kernel; scenarios 12 and 15's runtime
  half. Modules: `shared/src`, `shared/tests`. Unlocks γ1, γ2, ε1, ε2, η, λ.
- **γ1 — Inline scanner and ratchet.** `scripts/inline_suppressions.py`
  (prefiltered `tokenize` scan, D8 consumer model read from the workspace
  `pyproject.toml` files via `tomllib`, the empty class table, `--check / --seed
  / --tighten / --json`), `scripts/tests/test_inline_suppression_ratchet.py`
  (fixture scenarios 1–11; the live-tree assertion that enforces iff the baseline
  exists). Prereq β. Signal: `--json` on main reports non-zero totals for four
  kinds, zero for `nosec`, and completes inside the 10 s budget. Unlocks γ2, η,
  κ1, λ.
- **γ2 — Exception register.** `scripts/exception_register.py` (static
  discovery, ratification-table load, closed-world check, D14 listing, `--report
  / --json`), `docs/legibility/exception-ratifications.yaml` (empty list with its
  schema comment), `scripts/tests/test_exception_register.py` (scenarios 13–15
  and the live-tree assertion). Prereqs β, γ1. Signal: `--report` on main prints
  the grandfathered headline (reading `not seeded` until κ1), the
  per-`(kind, code)` table and zero declarations.
  Unlocks ε1, ε2, ζb, η.
- **δ — Day-one rulings (operator decision).** `metadata.execution_class:
  "decision"`. The escalation carries Appendix A parts 1–3. Deliverable: rows in
  the ratification table and `docs/legibility/inv12-day-one-rulings.md` — a
  fixed-column table `list_id | key-or-default | disposition literal` plus the
  part 2 and part 3 answers. `delivered_checks`: `kind: grep`, `present`, a
  ratification-row id pattern in the YAML path — it holds ζb, not δ's own close
  (Ordering invariants). Prereq γ2. Decompose writes δ's real task id into the descriptions of ε1, ε2 and ζb
  before the batch is committed — they cite it as `TaskRef(δ)`. Unlocks ζb.
- **ε1 — Declare the governed lists of `orchestrator/` and `scripts/`.** One
  declarations test module each. Appendix A part 1 rows for those packages: live
  owners cited as `Debt(TaskRef(n))` after re-checking they are still live;
  policy-proposed rows as `Debt(TaskRef(δ))`. Prereqs β, γ2. Unlocks ζb.
- **ε2 — Declare the governed lists of `shared/`, `fused-memory/`,
  `dashboard/`.** Same, including the loop-blocking ledger mapping and the
  config-suppression declaration (keys from `pyproject.toml` via `tomllib`).
  `ALLOWLIST_ENTRIES` has 14 elements with repeated sites; the key function must
  keep duplicates distinct. Prereqs β, γ2. Unlocks ζb.
- **λ — Tell the agents.** D15's blocks in
  `orchestrator/src/orchestrator/agents/roles.py`; a `docs/task-authoring.md`
  entry for `x_inv12_finding`, `x_inv12_owns` and what a sweep-filed task asks
  for; the guard named in `CONTRIBUTING.md` §4 without the slug list.
  `complexity='simple'`. Prereqs β, γ1. Unlocks κ1.
- **κ1 — Seed cutover (operator action).** `metadata.execution_class:
  "operational"`. Halt the merge queue; on main HEAD run `--seed`; `git commit
  --only scripts/inline_suppression_baseline.json`; unhalt. `delivered_checks`:
  `grep present` on the baseline's `params` key, holding ζb. In-flight branches carrying a
  new marker go red once; the message tells them what to do. Prereqs γ1, λ.
  Unlocks ζb.
- **ζb — Apply the rulings.** Replace every `TaskRef(δ)` with its ruled
  disposition; add the ruled class rows; declare the part 2 lists ruled
  governed; seed `NOT_EXCEPTION_LISTS` from those ruled design; `--tighten`.
  Prereqs δ, ε1, ε2, κ1, γ2. Signal: `--report` shows no disposition citing δ.
  Unlocks θ.
- **η — Owner-liveness sweep.** `scripts/exception_owner_sweep.py`, the unit
  pair and always-exit-0 wrapper, the installer and its test, scenarios 16–24 against a temp store and stub MCP —
  exercise the mechanism, do not configure it (INV-10). Prereqs β, γ1, γ2. Signal:
  from the task worktree, `--dry-run` resolves the main checkout's store, prints
  a corroborated finding list and writes nothing. Unlocks κ2.
- **κ2 — Sweep activation (operator action).** `metadata.execution_class:
  "operational"`. Run η's installer on the host, wait for one timer run, confirm
  `--report` shows a sweep age. Prereqs η, ζb — ζb because δ goes `done` on
  `resume` while ε1/ε2's declarations still cite `TaskRef(δ)` until ζb lands,
  and a sweep running in that window would read δ as a dead owner on every list
  citing it (added at decompose, 2026-09-18). Unlocks θ.
- **θ — Integration gate.** On the real tree: `--report` shows every part 1 list
  with a disposition, a grandfathered headline, slack and a sweep age under one
  period; a planted branch with one bare suppression fails
  `test_inline_suppression_ratchet.py` with the scenario-1 message; the sweep's
  `--dry-run` (from the worktree, against the main checkout's store) agrees with
  the report's dead-owner list. INV-12's
  House pattern re-pointed at the landed paths; an `OPERATIONS.md` section
  (reading the report, the cutover, the timer, the residual in D11, what a
  sweep-filed task asks for). Prereqs α, ζb, κ2. **The batch's only leaf.**

Same-file serialisation: `scripts/inline_suppressions.py` γ1 → ζb;
`scripts/exception_register.py` γ2 → ζb; declarations modules ε1/ε2 → ζb;
`design-invariants.md` α → θ; the ratification table γ2 → δ → ζb.

## Out of scope

- The 428 bare `module.py:1234` pins — tolerated drift, not debt (esc-3815-7).
- Any formatting sweep (task 3441), and any burn-down of grandfathered
  suppressions, including the `RUF100` autofix.
- Gating an owner's `done` on its citations (`metadata.delivered_checks`,
  `expect: absent`). It would make ownership binding at the transition, but the
  same descriptors feed the scheduler's delivered-capability cache, and a stamp
  that reads "absent" before the work is finished could close an owner early.
  Worth its own design pass.
- `pytest.mark.skip` / `xfail` (114), `shellcheck disable` (15),
  `config_key_census.ignore` (its own audit,
  `orchestrator/src/orchestrator/config_census_ignore.py::audit_census_ignore_entries`).
  The scanner's kind table is the extension point.
- Porting any existing ratchet onto `shared.ratchet`; runtime debt rows
  (`flake_debt`); other factory-operated projects.

## Open questions (tactical)

1. **Working tree or `HEAD` blobs in the sweep?** The main checkout can be
   mid-merge. Suggested: working tree, since findings are re-corroborated before
   output. Decide in η.
2. **Escalation dedupe field.** Whether `escalate_info`'s fingerprint can carry
   the finding key, or the sweep keeps its own ledger of escalated keys in its
   state file. Decide in η.
3. **Promote D14's detector to a gate?** Measure the candidate count in γ2 (a
   naive one-pass AST scan gives 162); decide after θ.
4. **Labelling self-citation.** Whether the sweep can cheaply tell that a dead
   owner's own landed diff introduced the marker (`git log -S`), so the finding
   says "fix, do not re-cite". Decide in η.
5. **Early feedback.** Whether λ's implementer block should tell agents to run
   `--check` on their changed files before finishing, or whether the merge-time
   red is cheap enough. Measure bounce cost after κ1.

## Appendix A — day-one docket

### Part 1 — governed under D1 without argument (ε1/ε2 declare; δ rules the rows marked ◆)

| Governed list | Entries | Live owner found 2026-09-18 | Declared as |
|---|---|---|---|
| `orchestrator/src/orchestrator/verify.py::_KNOWN_LOAD_FLAKE_NODEID_RES` | 1 | 5578 retires the list | `Debt(TaskRef(5578))` if the list still exists |
| `orchestrator/tests/test_timeout_marker_inversion_guard.py::_GRANDFATHERED` | 61 | 5149 | default `Debt(TaskRef(5149))`, covers 61 |
| `orchestrator/tests/test_serial_merge_worker_import_guard.py::ALLOWLIST` | 9 | 5034 | default `Debt(TaskRef(5034))` |
| `orchestrator/tests/test_merge_queue_reachback_patch_guard.py::ALLOWLIST` | 34 | 5034 | same |
| baseline unit `fused-memory/scripts/check_bare_magicmock_config.py::_DATACLASS_DOUBLE_DEBT` (95 sites) | 1 | 4354 | `Debt(TaskRef(4354))` |
| baseline unit `…::_WALL_CLOCK_DEADLINE_DEBT` (588 sites) | 1 | 4920, 5215 | `Debt(TaskRef(4920))` |
| baseline unit `scripts/inline_suppression_baseline.json` | 1 | δ | `Debt(TaskRef(δ))` ◆ — confirm "no scheduled burn-down" (D10) |
| baseline unit `orchestrator/tests/merge_lane_ratchet_baseline.json` | 1 | none names the burn-down | ◆ |
| `shared/tests/silent_fallthrough_allowlist.py::ALLOWLIST_ENTRIES` | 14 | none | ◆ |
| `shared/tests/loop_blocking_allowlist.py::AUDITED_SITES` | 62 | per row (`filed` rows) | mapped per row; `accepted` rows ◆ |
| `shared/tests/config_dir_archival_allowlist.py` | — | 3970–3972 adjacent | ◆ |
| `scripts/merge_lane_metrics.py::SIZE_CEILING_EXEMPT` | 1 | none ("its split is a follow-up PRD") | ◆ |
| `scripts/check_dashboard_unit_parity.py::DIVERGENCE_ALLOWLIST` | 2 | — deliberate | ◆ propose `Policy` |
| `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`, `scripts/tests/test_merge_state_vocabulary_consistency.py::PINNED_SITES` | 5 + 5 | — INV-5's text tolerates them | ◆ propose `Policy` |
| `shared/src/shared/task_metadata.py::_BLESSED_METADATA_KEYS` | 47 | — ~20 cite a ruling | ◆ propose `Policy` |
| `shared/src/shared/task_metadata.py::_LEGACY_RETRY_LEDGER_COUNTER_KEYS` | 2 | none | ◆ |
| `fused-memory/src/fused_memory/memory_metadata.py::BLESSED_METADATA_KEYS` | 8 | — | ◆ propose `Policy` |
| `dashboard/src/dashboard/config.py::_LEGACY_CONFIG_NAMES` | 3 | none | ◆ |
| config: ruff `lint.ignore = ["E501"]` (8 files, one class); `orchestrator/pyproject.toml` per-file-ignore `tests/**/*.py: F811` | 2 | — | ◆ propose `Policy` |

### Part 2 — borderline under D1; the operator rules "govern?" per row

Suppressors of a runtime detector (propose: govern):
`orchestrator/src/orchestrator/harness.py::_BY_DESIGN_SESSION_RESUME_REASONS`;
`scripts/sweep_toolcall_markup.py::NEVER_TOUCH`;
`fused-memory/config/cancelled_premise_blocklist.yaml` and
`fused-memory/config/recon_code_fix_premise_registry.yaml` (YAML — declared from
a fused-memory test that loads the keys);
`fused-memory/src/fused_memory/reconciliation/flag_dedup.py::_SUPPRESSED_SNAPSHOT_CATEGORIES`
and `::_SUPPRESSED_CEILING_CATEGORIES`;
`fused-memory/src/fused_memory/reconciliation/internal_writers.py::INTERNAL_WRITER_AGENT_ID_PREFIXES`;
`fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py::PROTECTED_AUDIT_KINDS`;
`fused-memory/src/fused_memory/middleware/operational_suggestion_guard.py::_WEAK_MARKER_LABELS`;
`fused-memory/src/fused_memory/middleware/recon_claim_verification_guard.py::_EXCLUDED_PATHSPECS`;
the `_EXEMPT_EXECUTION_CLASSES` pair in `routing_intent_guard.py` and
`operational_suggestion_guard.py`.

Grants (propose: design, not governed):
`escalation/src/escalation/authority.py::ROLE_LEVEL_ALLOWLIST`,
`::L2_AUTO_CLOSE_ALLOWLIST`, `::PROMOTE_ALLOWED`;
`escalation/src/escalation/models.py::BORN_AT_L2_SEVERITIES`;
`orchestrator/src/orchestrator/merge_queue.py::AUTO_HEAL_MECHANICAL_CATEGORIES`;
`orchestrator/src/orchestrator/harness.py::MERGE_REMEDIABLE_ESC_CATEGORIES`;
`orchestrator/src/orchestrator/verify_categories.py`'s derived sets;
`fused-memory/src/fused_memory/config/reload.py::RELOADABLE_FIELDS` and
`orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS`; the entity-mint and
mem0-update prefix configs;
`fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py::_BRIEFING_REFRESH_PROJECT_ALLOWLIST`;
`scripts/render_dashboard_unit.py::HOST_LOCAL_ENVIRONMENT`.

### Part 3 — questions

1. **Class table.** Which `kind[code]@scope` rows are policy? 30-day inflow a row
   would absorb: `type: ignore[method-assign]@tests` 95, `[attr-defined]@tests`
   92, `[arg-type]@tests` 77, `[possibly-unbound]` 46. An empty table is a valid
   ruling; it means each of those is a ticket.
2. **Is a mechanical rename a touch?** (D7.) Renaming one widely-mocked method
   rewrites ≥144 marker-bearing lines.
3. **Burn-down.** Confirm none is scheduled (D10), or name the owner.
4. **Dead markers.** Run the `RUF100` autofix, or leave 1,499 grandfathered?

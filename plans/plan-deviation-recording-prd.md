# PRD: Plan-deviation recording — make a frozen plan's wrongness cheap to record and countable in aggregate

**Status:** active — authored 2026-09-21 from Leo's rulings of the same day on the
`plan-text-unchecked-assertions-2026-09-19` cockpit decision record. Executes the ruling; it
does not reopen it. Reviewed by a fresh adversarial pass before commit (twenty findings, all
applied or recorded below).

**Code anchors** verified against main `2c78539443` (2026-09-21). Cite-by-symbol throughout;
no `path:line` anchors are introduced by this document.

**Governing frame (Leo, 2026-09-21):** imperfect plans are interesting data that can drive
process and quality improvements; they are not a serious problem. This is an optimisation,
not an urgent fix. Nothing below may block or slow plan authoring, add an authoring-time
verification pass, or route a per-instance record to any queue a human reads.

---

## 1. Goal

A frozen `plan.json` step whose text turns out to be wrong at execution time is corrected by
the implementer under a ratified doctrine, the correction is recorded **once**, in a home
that requests nothing of anyone, and a deterministic nightly census counts those records and
escalates to a human **only** on an actionable pattern or a significant rate rise.

**Consumer (G1).** The **plan-deviation census** (leaf δ) is the named consumer of both record
surfaces this PRD introduces — the per-step disposition record (leaf β) and the note-to-file
escalation class (leaf α). **Leo**, via the census's escalation, is the consumer of the census.
No other reader is claimed: the reviewer prompt takes no plan
(`orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_reviewer_prompt`)
and the completion judge is idle in practice (ruling 7) — neither is a design argument here.
(`plans/info-l0-disposition-router-prd.md` leaf β will show the reviewer *ordinary* info
notes; a note-to-file record is by construction not one of those — see D2.)

**User-observable surface.** (1) A dispatched implementer's system prompt carries the
deviation doctrine and names the two record tools. (2) `escalate_info(..., note_to_file=True)`
returns `status: resolved` with `resolution_class: note-to-file` and the record is in
`data/escalations/archive/<date>/` without ever being pending. (3) `record_step_deviation`
appends a typed record to the step in `plan.json`. (4) The nightly journal carries one
`plan-deviation census: ...` line per run; on a pattern or rate rise a `blocking` escalation
on a synthetic task id reaches the L2 watcher with structured counts.

---

## 2. Background — a dated snapshot (measured 2026-09-21)

Everything in this section was measured on 2026-09-21 by a seven-seat investigation plus an
adversarial critic, over the window 2026-09-18..20 (W). It is provenance, not current state;
do not re-anchor it. Full brief: the investigation scratchpad (session
`d18be7a4-0132-4e27-972e-f6779d57a8f4`), fused-memory `decisions_and_rationale` record filed
2026-09-21, local memory `project_plan_text_corrections_study_2026_09_21`.

- **The eleven are the promoted tail of a larger stream.** 46 implementer-filed info
  `design_concern` L0s in W; 43 mention the plan across 31 tasks; 21–28 records / 17–22 tasks
  assert the plan text was wrong under a strict rule. Whether one reached a human was decided
  by disposal timing, not content: the orphan reaper promoted 12, a restart swept 11, a
  sibling L1 dismissed 15, a steward closed 11.
- **Promotion is deterministic — under the reaper as it stood in W.** 12/12 reaper promotions
  fired 34–87 s after `task_completed`; the liveness arm defers an info L0 for the whole task
  lifetime. `orphan_l0_timeout_secs` is irrelevant to this class. This mechanism is being
  replaced by `plans/info-l0-disposition-router-prd.md` (tasks 5221–5227, pending; 5221
  pinned critical by Leo on 2026-09-21), whose reaper info branch routes an orphaned info L0
  instead of promoting it verbatim — see §3 ruling 5 and §8.
- **Sub-shape (i) "false claim about the code" dissolves** into 12 distinct wrong assertions
  across 8 tasks: 4 never-checked facts · 3 wrong inferences from correctly gathered facts ·
  3 plan-internal contradictions · 1 inherited from task text · 1 design narrowing refuted by
  measurement. **0/12 were stale since freeze** — revalidation and shorter plan lifetime buy
  nothing.
- **The implementer catches well.** 5/5 falsified predictions and 4/4 contradictions handled
  correctly; two resolutions strictly stronger than literal compliance; one correctly needed
  no change. Deviation cost ≈ 0 (no extra iterations, blocked work or merge delays).
- **Residual risk is prose-only claims.** A false claim destined for a docstring / CHANGELOG
  / SKILL.md is not refutable by running the code: ≥3 shipped into the artefact and were
  caught at review; one (task 5582) shipped to main uncorrected and spawned a steward ticket.
- **Nobody human paid per instance.** All eleven L2s were resolved by the
  `escalation-watcher` agent (~3.5 k chars of verification each); 4 cap-parked 4–36 h; two
  stewards' lifetime budgets ran out on the filing call.
- **Prediction split (3-plan census, a selected sample):** Tier-1 `RED`/`GREEN` step headers
  1/46 falsified; Tier-2 substantive predictions (exact counts, error types, "must stay
  green") 4/39 ≈ 10 %. Nothing downstream consumes Tier-2 predictions.
- **Citations are pre-rebase SHAs.** All 7 SHAs cited in the eleven's `detail` fields have
  patch-id-identical twins on main (re-verified 2026-09-21: each cited object exists and is
  not an ancestor of `main`; each twin is); the watcher re-proved each by hand and wrote the
  landed SHA only into resolution prose (verification table: investigation seat E).

### 2.1 Corrections to beliefs that circulate in the records

- **Task 3890 does not own the precedent ladder.** The ladder is the fused-memory
  `procedural_knowledge` topic `amendment-pass-precedent-ladder`; its canonical (memory
  `663c84bc`) calls itself "a POINTER, not a roster", records that task-number citations
  drift, and instructs readers to query the topic rather than trust any list. Leaf γ's
  implementer queries the topic for rung 2's wording — it does not trust a task number.
- **The ladder already reaches implementers** (they hold `_MEMORY_TOOLS`; 9 of 11 cited it).
  What is unreachable to them is the consumer-side six-condition standing rule in
  `skills/escalation-watcher/SKILL.md`. Leaf γ ratifies and stabilises a doctrine already in
  operation; it does not make it reachable.
- **`citation_sha`** on an escalation record is a dedup key scoped to task 4499's
  landing-evidence filings, not a general "which commit" field. `None` on a
  `design_concern` is correct.
- **`plan.json._finalized_at` is overwritten by every `confirm_plan`** (including amendment
  re-confirms) and `_created_at` (stamped by `artifacts.py::TaskArtifacts.stamp_plan_provenance`)
  is a post-hoc stamp that is always later. Neither dates the freeze. Leaf β documents this
  at the source.
- **`.worktrees/.task-meta/<id>/plan.json` is not reaped.** The plan for this PRD assumed the
  directory shrinks over time. Measured 2026-09-21: 2,154 directories, 2,022 with a
  `plan.json`, 230 MB, oldest 2026-07-07; every deletion site in `git_ops.py` is
  lane-reacquisition-keyed (`_clear_foreign_meta_root` / `_relocate_foreign_meta_root`) and
  `orchestrator/scripts/warm-lane/lib_lane_state.sh::LANE_PROTECT_GLOB_FALLBACK` lists
  `.task-meta` as *protected from* GC. The escalation archive is the corpus that shrinks
  (`escalation/src/escalation/archive.py::prune_archive`, `DEFAULT_RETENTION_DAYS = 30`).
  D9 is built on the measured shapes, including the cost of an append-only 230 MB corpus.
- **The architect prompt does not solicit Tier-2 forecasts.** `roles.py::ARCHITECT` asks for
  "write a failing test, then implement to make it pass" and nothing about counts or error
  types; the forecasts are emergent (64/66 plans carry them). Leaf γ therefore *instructs
  against* them rather than removing a solicitation that does not exist.
- **Task 5637's step-2 defect is not a step-graph shape.** Its plan's `type` sequence is
  `test, impl, test, impl, test, impl, impl`; the one non-alternation (steps 6–7) is
  unrelated to the defect, and steps 1–4, where the defect lives, alternate. "GREEN for
  step-1" when only step-4 greens it is a semantic dependency that only execution reveals.
  The originally proposed TDD-alternation check at `_confirm_plan` therefore rested on a
  false premise; Leo dropped it on 2026-09-21 (§3, ruling 9).

---

## 3. Rulings this PRD executes (Leo, 2026-09-21 — settled inputs, not open questions)

1. Optimisation, not urgent: prefer cheap, general mechanisms.
2. Doctrine home = inline. Ratify a frozen snapshot of the ladder's rung 2 inline in the
   IMPLEMENTER role prompt, with a parity drift guard.
3. The stream stays countable, addressed by a deterministically run periodic census that
   escalates only on an actionable pattern or a significant rate increase — not by
   per-instance routing.
4. Plan-size dependence is not significant; the factory is moving to fewer, larger tasks.
   No step-count limits.
5. Reopen task 2725's exclusion of `found_on_main` so the orphan reaper's terminal-task arm
   can dismiss an info L0 on a done task, paired with the note-to-file class. **Discharged
   by an existing queued leaf rather than a new one (D7):** the router PRD's leaf δ (task
   5223, item 5) replaces the reaper's verbatim promotion of every `severity == 'info'` L0
   with `escalation.disposition.route_info_l0(..., exit_kind='orphan')`, whose
   terminal-subject branch resolves a non-work-shaped note on a `done` task
   `observation-consumed` **without consulting `done_provenance`** — so an aged info L0 on a
   `found_on_main` task is dismissed with a countable class and never promoted, which is what
   the ruling asked for, while `_MERGED_DONE_PROVENANCE_KINDS` and task 2725's hollow-done
   protection stay untouched. This PRD adds no reaper leaf.
6. Clock: rolling window or EWA preferred over any day boundary; UTC if a boundary is
   unavoidable.
7. The completion judge is idle; the census is the consumer of the disposition record.
8. One PRD; the census retires the watcher's ">3 qualifying in one day" count but keeps the
   per-task "a second one goes to a human" limit; decompose and queue in this session.
9. (Same day, while authoring.) Drop the confirm-time TDD-alternation check; its purpose
   moves to the disposition record. The census lives in a sibling module hooked into the
   nightly trickle, not inside `scripts/legibility/census.py`.

### 3.1 Rejected, with reasons — do not reopen

- **Architects verify load-bearing claims before freezing** as the primary remedy — reaches
  only sub-shape (i), ≤7/12 of those; architects already verify and the failure is mis-aimed
  verification (task 5542 proved a log line parseable; nobody asked what its ids meant).
  Survives in reduced form as the interface-shape duty (γ).
- **Stop filing these** — conflicts with rung 2, which mandates the record, and discards the
  signal. Replaced by ratifying the doctrine (γ) and giving the record a non-routed home
  (α, β).
- **Change the 3/day cap** — hop 7 of an 8-hop routing chain; superseded by the census (δ).
- **Run per-step green claims at authoring time** — undeliverable by the architect role
  (`Edit`/`Write` disallowed, $5 / 50 turns, the tree does not exist). Purpose relocated to
  the disposition record (β).
- **Another architect pass** — falsified on this corpus (task 3816's contradictory step-19
  came out of a second pass; task 5586, the only plan without one, produced the most
  text-obvious contradiction).
- **Hoisting `_is_terminal_merged` out of the done-step-commit branch** — that predicate
  excludes `found_on_main` and 9 of 11 tasks are `found_on_main`; reaches 0/11.
- **A new reaper arm keyed on `severity == 'info'` + terminal subject** (this PRD's own first
  draft) — it would have intercepted task 2725's own class: the done-step-commit tripwire is
  filed at `severity='info'` (`workflow.py::TaskWorkflow._escalate_unreconciled_done_step`),
  so an arm placed before that branch is a strict superset of it and turns the 2725 tests
  red. Superseded by 5223 (ruling 5).
- **A typed `expected_suite_state` field** — indistinguishable from the disposition record
  unless the orchestrator compares and acts, and acting is wrong (task 5637 was right to
  proceed red). `step.type` is the natural experiment: a typed field agents restate in prose
  because nothing consumes it.
- **A prose linter over frozen plan text** — an ad-hoc parser of internal values
  (*structured data instead of meaningful strings*, violated in the name of enforcing it).
- **Smaller plans / step-count limits** — ruling 4.
- **A new severity below `info`** — `escalation/src/escalation/models.py::SEVERITY_RANK` is
  total over `KNOWN_SEVERITIES` by contract, rank 0 is already `info`, every severity fold
  takes the max, and `orchestrator/src/orchestrator/session_registry.py::_DECISION_SEVERITY_RANK`
  is a second, deliberately unshared map. A record flag touches routing only (D2).
- **A TDD-alternation check at `_confirm_plan`** — ruling 9; premise refuted (§2.1).
- **Reusing `resolution_class='moot-terminal-subject'`** for anything here — that class
  pre-exists with two producers (`harness.py::_ESCALATION_REVALIDATION_RESOLUTION_CLASS`,
  `fused-memory/scripts/derive_orphaned_recon_escalations.py::RESOLUTION_CLASS`); counting it
  would poison the census corpus.

---

## 4. Resolved design decisions

**D1 — Two record surfaces, disjoint by subject, one consumer.** A deviation attributable to
one plan step is recorded on that step (`record_step_deviation`, leaf β). A deviation not
attributable to a step — a design decision whose rationale proved wrong, an amendment-pass
rung-2 resolution against a reviewer suggestion, a plan-level premise — is recorded as a
note-to-file escalation (leaf α). Never both for one fact (INV-9 `one-fact-one-home`). The
census (δ) reads the **union** of both corpora; there is no join.

**D2 — Note-to-file is a filer-declared flag on `escalate_info`, evaluated inside the
chokepoint, not a severity.** `escalate_info(..., note_to_file=True)` is accepted only with
`severity='info'` (any other severity → structured error `code: note_to_file_requires_info`,
nothing filed). The flag is evaluated inside `server.py::_chokepoint_or_submit` **after** the
task-3550 `filing_claimant_run_id` stamp and before gate 1, so the identity stamp is kept and
gates 2–4 and dedupe are short-circuited (every note is its own record). The server writes the
record directly in resolved state via `queue.py::EscalationQueue.submit_resolved` with
`resolution_class='note-to-file'` **and `resolution_action='close_only'`** — `submit_resolved`
fires `_resolve_callback` once, and `harness.py::Harness._on_escalation_resolved` maps a
resolved record with no `resolution_action` to the legacy `resume` effect
(`escalation/src/escalation/action_effects.py`: `('resume', ANY, ANY) → TaskEffect('pending',
WORKFLOW_RESUME)`), which would re-pend the filer's own task; `close_only` maps to
`TaskEffect(None, WORKFLOW_NONE)`. `'note-to-file'` joins `models.py::RESOLUTION_CLASSES`;
that is a **two-site lockstep edit** because
`escalation/tests/test_models.py::test_resolution_classes_contains_exactly_the_legal_values`
asserts set equality — and it lands *after* the router PRD's task 5221 adds its four classes
to the same constant (§8). The record never enters pending, so the orphan reaper,
`has_open_l1`, pins, the router PRD's reviewer notes block and the auto-watcher never see it;
it is archived under `archive/<resolved-date>/` and countable by `category`, `severity`,
`agent_role`, `resolution_class`, `timestamp`. `note_to_file` is also persisted as a boolean
field on the record (zero-migration via `from_dict`'s `__dataclass_fields__` filter).
Category stays unvalidated prose — this PRD adds no category, so `models.py`'s "next category
addition promotes this to an enum" trigger does not fire.

**D3 — One `escalate_info` contract, stated once.** Today three prompts state three
contracts (`roles.py::ESCALATION_LADDER_CORE` "non-blocking observation";
`roles.py::DEEP_REVIEWER` "findings that need human judgment"; `roles.py::_FOLLOWUP_FILING_INSTRUCTIONS`
"a pure observation with no work item"). The single contract: *`escalate_info` files a
non-blocking observation. With `note_to_file=True` it is a record that requests nothing and
is archived unrouted — no reviewer, steward or watcher will read it. Without it, it is an
observation the reviewer dispositions in its verdict, or the router closes at workflow exit
(`plans/info-l0-disposition-router-prd.md`). A work item is `submit_task`; a decision you need
before continuing is `escalate_blocker`.* The contract lives **once**, as
`roles.py::ESCALATE_INFO_CONTRACT`, rendered into `ESCALATION_LADDER_CORE` (which every
escalating role already carries); `DEEP_REVIEWER`'s table row and
`_FOLLOWUP_FILING_INSTRUCTIONS` become one-line pointers to "## Escalation", not restatements
(INV-5 `no-lockstep-duplication`). `_FOLLOWUP_FILING_INSTRUCTIONS` is appended to ARCHITECT
and IMPLEMENTER after construction, so a restatement there would render the contract twice
into the same prompt. This also renders Leo's 2026-09-08 norm (memory `3b011d26`): an info
record is an annotation, never a hold.

**D4 — The per-step disposition record is a typed list on the step.** Each entry:
`{kind, observed, expected, action_taken, prose_only, note, recorded_at, recorded_by}`.
`kind` is a closed vocabulary validated at the tool boundary (INV-1
`contracts-machine-checked`), derived from the 2026-09-21 taxonomy: `never_checked_fact |
wrong_inference | plan_internal_contradiction | inherited_from_task_text |
narrowing_refuted_by_measurement | prediction_falsified | other`. The vocabulary lives in
**`shared/src/shared/plan_deviation.py::DEVIATION_KINDS`** — three packages consume it (the
plan-tools server validates against it, `roles.py` renders the doctrine from it, the census
reads it), and `shared` is the one home every consumer can import (*deep modules with
coherent narrow interfaces*; `orchestrator.mcp.plan_tools` exports only `create_server` and
`main` today and should stay that way). `prose_only=True` marks the residual-risk class (a
claim destined for a docstring / CHANGELOG / SKILL.md). `recorded_by` is the plan-tools
session identity that is actually in hand — `TaskArtifacts.read_agent_session().session_id`
when present, else `plan['_session_id']`, else `None` — not the `filing_claimant_run_id`
shape, which the plan-tools server cannot compose (no run id or pid reaches it). The record
changes neither `status`, `commit` nor `description`; it is written by a dedicated
`artifacts.py::TaskArtifacts.append_step_deviation` beside `update_step_status` (whose
contract stays "status and commit only" — *well-defined purpose for each entity*). It is
allowed on any step status, including `done`. Known limit, accepted: the record's prose
fields sit two levels deep (`steps[i].deviations[j].observed`) and
`plan_tools.py::_REPAIRABLE_PLAN_FIELDS` / `_walk_repairable` address exactly one level, so
they get inbound `MarkupGuardMiddleware` coverage but no stored-damage repair; neither
`artifacts.py::_normalize_plan` nor `plan_tools.py::_read_plan_repaired` drops unknown step
keys, so the list survives both read-write-back paths. **Seam with
`plans/task-amendment-delivery-prd.md`:** its C4 `_descoped_steps` record (disposition
`orphan|revert|ride-along`, `authority`, `note`) is **not on main** — its task 4032 is
pending, and that PRD's open question 2 ("`_descoped_steps` vs per-step tombstone") is still
open. β does not extend a field that does not exist; it uses the same `note`/`recorded_by`
vocabulary on a step-level list, and 4032 should decide its open question 2 in light of
β's shape (a dropped step vs an executed-with-deviation step are one family). That PRD
records task 2971 as the incident whose plan was repaired **by hand** through four follow-up
escalations because no tool existed; β's record goes through a tool and never edits plan
text, which is the same lesson applied one layer earlier.

**D5 — Doctrine home and INV-9.** The ratified rung-2 snapshot becomes a prompt constant,
`roles.py::IMPLEMENTER_DEVIATION_DOCTRINE`, spliced into IMPLEMENTER in place of "note it and
stop. Do NOT modify the plan." **The constant is the home** (the `CODE_QUALITY_GUIDANCE` /
`docs/code-quality.md` precedent from task 5225: inlined because a dispatched agent cannot
follow a cross-reference). The fused-memory canonical of topic
`amendment-pass-precedent-ladder` carries a **dated pointer** to it (an amendment record
written as γ's last step: "rung 2 ratified 2026-09-21; home =
`roles.py::IMPLEMENTER_DEVIATION_DOCTRINE` at commit <sha>"). Reconciliation direction:
**code wins**. The ladder keeps growing in memory for every other shape; a future rung that
*contradicts* the frozen rung 2 is not a memory write but a proposal to amend the constant —
it is filed as a task against `roles.py`, and until that task lands the constant is what
dispatched implementers follow. The doctrine covers the IMPLEMENTER role only (the role that
executes frozen text); the DEBUGGER's "note it and stop rather than applying band-aids"
concerns design issues, not plan text, and is left alone.

**D6 — The parity drift guard mirrors named constants; it pins no prose and scrapes none.**
The doctrine is *rendered from* the vocabulary it cites (INV-5's render-from-source): its
list of kinds is produced from `shared.plan_deviation.DEVIATION_KINDS`, and the tool it names
is the bare name of the entry in `roles.py::_PLAN_DEVIATION_TOOLS`.
`orchestrator/tests/test_roles_deviation_doctrine.py` (layout of
`test_code_quality_guidance_parity.py`) asserts: (a) IMPLEMENTER carries
`IMPLEMENTER_DEVIATION_DOCTRINE` exactly once and the retired sentence is absent —
containment against named constants, the `test_roles_escalation_ladder.py` convention; (b)
every member of `DEVIATION_KINDS` appears in the rendered IMPLEMENTER prompt and the bare
name of every `_PLAN_DEVIATION_TOOLS` entry appears in the doctrine — containment against
the live constants; (c) `ESCALATE_INFO_CONTRACT` appears exactly once in every composed
escalating-role prompt. No regex over prose: a first draft proposed scraping backticked tool
names, which is the untested-matcher shape INV-10 `guards-exercise-behaviour` names (a silent
green when the extraction stops matching), and `plan_tools.py::_params_of` cannot introspect
the registered FastMCP closures anyway (it requires an `artifacts` first parameter and there
is no tool registry). Fully rewording the doctrine leaves the guard green; dropping a kind or
renaming the tool turns it red.

**D7 — No reaper leaf.** Ruling 5's purpose is delivered by task 5223 (§3). What this PRD
adds instead is the *declared* channel: an implementer that knows its note requests nothing
says so at filing (`note_to_file=True`), so the router never has to infer "work-shaped vs
observation" from prose for that class — the router's inference stays for records whose
filer did not declare.

**D8 — The census is a sibling module hooked into the nightly trickle's one recording
point.** `scripts/legibility/plan_deviation_census.py`, called from
`scripts/legibility/nightly.py::run_nightly`'s `finally` block beside
`nightly.py::_record_trickle_progress` — the one site every return path and an unexpected
raise pass through (`run_nightly` has four early `return result` sites before
`evaluate_census_step`, so hooking after that call would deliver a census only on clean
nights). It guards itself (`evaluate_census_step`'s discipline is per function, not
inherited): never raises, one journal line per run in the `plan-deviation census: FIRE|NO-FIRE
-- ...` grammar, an evaluation failure rendered in that same grammar (INV-11
`no-silent-fail-soft`). Reuses: the installed `legibility-trickle@.timer` (a `systemd --user`
unit that can reach `127.0.0.1:8102`); `census_trigger.post_mcp_tool_call` for the
escalation; the `trickle_state.trickle_state_path` directory for its own state file
(`plan-deviation-census-state.json`, not git-tracked); a `plan_deviation_census:` block in
`docs/legibility/legibility.yaml` modelled on `config.py::Census` — noting that every model in
`config.py` sets `extra='allow'`, so the block's validation is δ's own `from_mapping`
(validated-never-coerced, one WARNING per block, per `census_trigger.CensusConfig.from_mapping`),
not something pydantic gives for free. It does not touch `census.py`,
`census_trigger.evaluate` or the codebook (*well-defined purpose*, *no file too large*).

**D9 — Corpus, cost and metric.** Two corpora, read into one flat list: (i) `plan.json`
deviation records under every configured meta root (default
`<project_root>/.worktrees/.task-meta`; `scripts/scan_plan_decision_pairing.py` is the
bulk-reader precedent; task id from the directory name, which disagrees with the document's
`task_id` on 4 of that script's 1,299-plan 2026-08-16 sample). The corpus is append-only and
230 MB today, so the reader **stats before it reads**: a `plan.json` is rewritten on every
write, so a file whose mtime is older than the 28-day horizon cannot hold an in-window record
and is skipped without being opened. (ii) Archived escalations from
`escalation/src/escalation/queue.py::iter_all_escalation_paths`, filtered to
`resolution_class ∈ {'note-to-file', 'observation-consumed'}` — the second is the router
PRD's class for a pure observation — **or** the pre-router fallback (`severity == 'info'`,
`category == 'design_concern'`, `agent_role` in the implementer/amendment set, resolved by
any tier), which keeps counting records filed before the flag and the router exist. The
archive is a rolling ~30 days; the metric needs 28. Metric: **records per day**, the unit
the 2026-09-21 study used (4.2 → 9.0/day). Trigger A (rate rise): the trailing 7-day mean
exceeds `rate_rise_factor` (default 2.0) × the mean of the preceding 21 days **and** the
7-day count ≥ `min_records` (default 10). Trigger B (pattern): one `kind`'s share of the
7-day records ≥ `pattern_share` (default 0.5) with count ≥ `min_records`, or `prose_only`
records ≥ `prose_only_min` (default 3) in 7 days. Windows are measured backwards from `now`
(UTC-aware); no day boundary. No denominator: a per-plan rate would need runs.db, which the
nightly cannot reach; the human the escalation reaches has it. `Verdict(fire, reasons,
evidence)` is *modelled on* `census_trigger.Decision(fire, reasons)` — the `evidence` list
is new, and is what carries the structured counts (INV-2 `structured-facts-at-failure`).

**D10 — The census escalates as `blocking` on a synthetic task id, through the unchanged
blocking path.** On a trigger it files `escalate_info(task_id='plan-deviation-census-<project_id>',
agent_role='plan-deviation-census', category='design_concern', severity='blocking',
evidence=[…])`. `info` cannot be used: the router PRD's D4/D8 close all-info clusters at L1
with a digest and route an unknown-role info L0 to the curator leg — the census's own filing
would become a task, not a human page. `blocking` L0s take the reaper's unchanged verbatim
promotion (router PRD D6) after `orphan_l0_timeout_secs` (600 s default — the "~10 min" the
legibility trickle's comment records is that timeout, not a property of any category), the
auto-watcher promotes a `design_concern` L1 to L2 with an explicit severity (never
auto-closable), and the L2 watcher reads the structured `evidence` (window, 7-day and 21-day
means, per-kind counts, prose-only count, top task ids). The synthetic task id pins nothing
real — the `__recovery_veto_streak__` sentinel precedent in the auto-watcher skill. It files
at most once per `cooldown_days` (default 7) and never while a record with that `task_id` is
pending (`get_pending_escalations`). `escalate_info` accepts any `KNOWN_SEVERITIES` value;
if a future change restricts it, `escalate_blocker` with the returned `action` ignored is the
fallback (open question 2).

**D11 — The watcher's per-day count retires; the per-task limit stays.**
`skills/escalation-watcher/SKILL.md` "Standing rule: accept verified info-level design
deviations" → **Limits** becomes: a second such escalation on the same task still goes to
the human as a pattern; the ">3 qualifying in one day" clause is deleted with a pointer to
the census as the owner of the aggregate question (ruling 8).

**D12 — SHA citation legibility, no sweep, no signature change to the sync read tools.**
`ESCALATION_LADDER_CORE` instructs: cite `task/<id>` and the commit subject beside any SHA,
because a task-branch SHA is rewritten by rebase (`git_ops.py::GitOps.rebase_onto_main`). A
**new** async escalation tool, `resolve_escalation_shas(escalation_id)`, returns `{escalation_id,
resolved_shas: [{cited, landed, subject, method}]}` and never rewrites the record; the
existing `get_escalation` / `get_task_escalations` are sync and stay untouched. Resolution is
a new `git_ops.py::GitOps.resolve_rebased_citation(sha) -> {landed, subject, method} | None`:
base = `git merge-base <sha> main` (the cited object exists, so a base is computable); tier 1
patch-id equality over `base..main` (the same `git log -p | git patch-id --stable` shape as
`find_equivalent_commit`, which cannot be reused directly — it takes a worktree and a
`base_sha` an archived record does not carry); tier 2 exact-subject match within the same
range. `find_task_citation_commit` is **not** the fallback — it matches a task id in a
subject, so for task 5264's three cited SHAs it would return one commit three times. The
server reaches git through the wired `harness.git_ops` exactly as `claim_warm_worktree` does;
a standalone server (`create_server(harness=None)`) returns `{'error': 'no git authority
wired'}`. Fixture: a synthetic repo with a rebased branch as the executable test; the seven
verified pairs (5264: a7ceed6597→570dc88ad6, 05b8f66ad7→3236eb44d7, 98122b1acf→9dfea6419d;
5542: ce6c0a8a2b→c12a5ec749; 5262: 1b627a06a5→e796bab6b9; 3816: da2e8c021b→8fdecc5125; 5586:
8b2e641917→c9968710c3) are a `manual` check. **Seam:** the `plan.json` per-step `commit`
field has the same defect one layer down and is owned by tasks 3651/3665 (pending); α is the
escalation-layer twin and adds no dependency on them.

**D13 — Architect-facing duties widen, they do not add a pass.** Rule 2's scope moves from
*provenance* ("any file or symbol the task description claims already exists") to
*load-bearingness*: for every existing symbol a step routes through, calls or extends, read
its signature and confirm the step's other requirements are expressible through it (catches
task 3816's step-19 exactly). And a new rule: step text states what changes and which test it
greens; it does not forecast suite outcomes (exact counts, error types, "must stay green")
— those carried 4 of 5 falsifications and have no consumer. "The plan structure is IMMUTABLE
after creation" is replaced by what is true: only an architect-role pass may change a plan's
structure, and only a non-`done` step (`plan_tools.py::_remove_plan_step` /
`_replace_plan_step`); the implementer's immutability is its tool allowlist.

**D14 — Riders land where the fact lives.** `plan_tools.py::_add_plan_step` and
`_replace_plan_step` reject `step_type ∉ STEP_TYPES` (a public `frozenset({'test', 'impl'})`
in `plan_tools.py`) with a structured error naming both values (INV-2 shape; task 5520's
symptom). The `add_plan_step` docstring documents that the stored key is `type`.
`workflow.py::TaskWorkflow._replan`'s inline prompt stops telling the architect to "Write the
updated plan to `.task/plan.json`" — it points at the plan-tools the system prompt already
lists, and does **not** paste the recipe (task 5532 owns collapsing the two existing copies;
a third would widen its problem). `_confirm_plan`'s comment and the `confirm_plan` docstring
state that `_finalized_at` dates the *last* confirm. `record_step_deviation` is exposed
through a **new** list `roles.py::_PLAN_DEVIATION_TOOLS`, added to IMPLEMENTER and DEBUGGER
only — not through `_PLAN_STATUS_TOOLS`, which `SIMPLE_TASK` also holds and which would hand
the tool to a role with no frozen plan and no doctrine.

---

## 5. Sketch of approach

```
 implementer (frozen plan)                  escalation server               nightly trickle (finally)
 ─────────────────────────                  ─────────────────               ─────────────────────────
 step text wrong, rationale clear           escalate_info(note_to_file)     plan_deviation_census.py
   → apply rationale in full                  → chokepoint: stamp id,       reads .task-meta/*/plan.json
   → self-verify                                submit_resolved             (mtime-pruned) ∪ archive
   → record_step_deviation(kind,…)  ──►  plan.json step.deviations[]        → records/day, 7d vs 21d
   → (not step-attributable)        ──►  archive/<date>/esc-*.json          → per-kind shares, prose_only
                                           resolution_class=note-to-file    → blocking escalate_info on
 ordinary info notes: router PRD chain                                        a synthetic task id, ≤1 per
   (reviewer → steward → router) → observation-consumed ──► archive           cooldown
```

Nothing new runs at plan-authoring time. The only new agent-side calls are two record
writes the doctrine tells the implementer to make at the moment it deviates.

---

## 6. Contract (G5: three packages, one load-bearing seam — the escalation record)

### 6.1 `escalate_info` — note-to-file (α)

```
escalate_info(task_id, agent_role, category, summary, severity='info', detail='',
              suggested_action='', worktree=None, workflow_state=None, evidence=None,
              terminal_state_is_the_bug=False, note_to_file=False) -> dict
```
- `note_to_file=True` and `severity != 'info'` → `{'error': ..., 'code': 'note_to_file_requires_info'}`,
  nothing written.
- `note_to_file=True` → inside `_chokepoint_or_submit`, after the `filing_claimant_run_id`
  stamp: `esc.resolution_action = 'close_only'`; `submit_resolved(esc, resolution='note to file
  — archived unrouted at filing', resolved_by='escalation-server',
  resolution_class='note-to-file')`; response `{id, status: 'resolved', resolution_class:
  'note-to-file'}`. Gates 1–4 and dedupe are not evaluated.
- `Escalation.note_to_file: bool = False` persisted; `RESOLUTION_CLASSES` gains
  `'note-to-file'` (with the set-equality test in `escalation/tests/test_models.py`).
- Invariant: a note-to-file record is never `pending` at any instant on disk, and filing one
  never changes the subject task's status or wakes a workflow.

### 6.2 `record_step_deviation` (β)

```
record_step_deviation(step_id, kind, observed, expected, action_taken,
                      prose_only=False, note='') -> dict
```
- `kind ∉ shared.plan_deviation.DEVIATION_KINDS` → `{'status': 'error', 'message': <names the vocabulary>}`.
- Unknown `step_id` → error naming the id. Any step status accepted.
- Appends `{kind, observed, expected, action_taken, prose_only, note, recorded_at (UTC ISO),
  recorded_by}` to `step['deviations']` (created on first write); `status`, `commit`,
  `description` untouched; returns `{'status': 'ok', 'step_id', 'deviations': <count>}`.
- Registered on the plan-tools server by β with the same `@accepts_markup_override` guard as
  `add_plan_step`; the allowlist `roles.py::_PLAN_DEVIATION_TOOLS =
  ['mcp__plan-tools__record_step_deviation']` is added to IMPLEMENTER and DEBUGGER by γ, the
  one leaf that edits `roles.py` — until γ lands the tool exists but no dispatched role holds
  it. `STEP_TYPES` is a public constant in `plan_tools.py`.

### 6.3 Census (δ)

`plan_deviation_census.evaluate(records: list[DeviationRecord], *, now, config) -> Verdict`
is pure (no I/O): `Verdict(fire: bool, reasons: list[str], evidence: list[dict])`. Readers
(`read_plan_deviations(meta_roots, *, horizon)`, `read_archived_notes(escalations_dir)`)
normalise both corpora to `DeviationRecord(task_id, recorded_at, kind | None, prose_only,
source)` and return one flat list (a union). `run(cfg, *, now, filer, state_path)` wires
readers → evaluate → cooldown / pending check → `escalate_info(severity='blocking')` → state
write, never raises, returns the one journal line.

### 6.4 Boundary-test sketch (the integration gate is δ's test module)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Note-to-file never pending | server with chokepoint enabled and a resolve callback wired | `escalate_info(note_to_file=True)` → file under `archive/<today>/`, `status='resolved'`, `resolution_class='note-to-file'`, `resolution_action='close_only'`; `get_pending()` unchanged; the callback observes no task-status effect |
| Note-to-file severity guard | `severity='blocking'` | structured error, no file written |
| Note-to-file keeps identity | `task_claimant_lookup` wired | archived record carries `filing_claimant_run_id` |
| Step record survives status change | `record_step_deviation` on pending step, then `mark_step_done` | `deviations` retained, `commit` set |
| Step record rejects unknown kind | `kind='typo'` | structured error naming `DEVIATION_KINDS`; plan unchanged |
| Census positive control | synthetic corpus: 3/day for 21 d then 8/day for 7 d | `fire=True`, evidence carries both means; exactly one `escalate_info` with `severity='blocking'` and the synthetic task id |
| Census negative control | flat 3/day for 28 d | `fire=False`, no filing, journal line present |
| Census cooldown | fired 2 days ago (state file) | no second filing; line says why |
| Census reads the union | one plan.json record + one archived note, same task | two records; old-mtime plan.json never opened |
| Census runs on a red night | `run_nightly` returns early on a coder storm | the census line is still produced from the `finally` |
| SHA resolver | fixture repo with a rebased branch; record citing the pre-rebase SHA | `resolved_shas[0].landed` is the replayed SHA; record bytes unchanged; standalone server returns the structured error |

---

## 7. Pre-conditions (G3 — assumed substrate, verified 2026-09-21 on `2c78539443`)

| Capability | Evidence | Verdict |
|---|---|---|
| `escalate_info` tool with `severity`/`evidence` params; any `KNOWN_SEVERITIES` value accepted | `escalation/src/escalation/server.py::escalate_info` | PASS |
| Chokepoint with identity stamp above all gates | `server.py::_chokepoint_or_submit` (task-3550 block) | PASS |
| Atomic submit-as-resolved with `resolution_class`; fires `_resolve_callback` once | `queue.py::EscalationQueue.submit_resolved` | PASS |
| `resolution_action='close_only'` has no task effect | `escalation/src/escalation/action_effects.py::ACTION_EFFECTS`; `models.py::Escalation.resolution_action` | PASS |
| `RESOLUTION_CLASSES` extensible, set-equality-tested | `models.py::RESOLUTION_CLASSES`; `escalation/tests/test_models.py` | PASS (two-site edit) |
| Zero-migration record fields | `Escalation.from_dict` `__dataclass_fields__` filter | PASS |
| Plan write path + step-scoped mutation | `artifacts.py::TaskArtifacts.write_plan`, `::update_step_status`, `::read_agent_session`; `plan_tools.py::_mark_step_done` | PASS |
| Unknown step keys survive read/normalise | `artifacts.py::_normalize_plan`, `plan_tools.py::_read_plan_repaired` | PASS |
| Nightly one-recording-point hook | `nightly.py::run_nightly` `finally` → `_record_trickle_progress` | PASS |
| Per-project state dir; timer can reach the port | `trickle_state.py::trickle_state_path`; `legibility-trickle@dark_factory.timer` (systemd --user, no network sandbox) | PASS |
| Config block precedent (validation is δ's own) | `config.py::Census` (`extra='allow'`); `census_trigger.CensusConfig.from_mapping` | PASS |
| Escalation envelope from a script | `census_trigger.post_mcp_tool_call`; `nightly.py::post_escalation` | PASS |
| Bulk archive reader | `queue.py::iter_all_escalation_paths` | PASS |
| Bulk plan reader precedent | `scripts/scan_plan_decision_pairing.py` | PASS |
| Patch-id shape to imitate | `git_ops.py::GitOps.find_equivalent_commit` | PASS (imitated, not reused) |
| Escalation server can reach git when a harness is wired | `server.py::claim_warm_worktree` → `harness.git_ops` | PASS (standalone path degrades explicitly) |
| `shared` importable from the nightly's env and from `orchestrator`/`escalation` | `uv run --project shared`; workspace layout | PASS |
| C4 `_descoped_steps` record | **absent** — task 4032 pending | N/A by D4 (not extended) |
| `.task-meta` reaper | **absent** (measured) | N/A by D9 (mtime pre-filter) |
| Router PRD vocabulary (`observation-consumed`) in the archive | lands with task 5221/5223 | δ reads it when present; the fallback clause covers its absence |

No novel substrate is queued; every leaf builds on capabilities present on main or on the
already-queued router batch (α's edges).

---

## 8. Cross-PRD relationship and seam table (G4)

| Other PRD / task | Status 2026-09-21 | Direction | Seam mechanism | Disposition |
|---|---|---|---|---|
| **`plans/info-l0-disposition-router-prd.md`** (5221–5227; 5221 pinned critical) | pending | **overlapping** | `models.py::RESOLUTION_CLASSES` (5221), `server.py` + auto-watcher SKILL.md (5222), reaper info branch (5223), `roles.py`/`artifacts.py` (5224) | **α depends on 5221 and 5222** (real edges: same constant, same server file, same skill section); ruling 5 discharged by 5223 (D7) — no reaper leaf; 5224 and γ/β edit the same files with disjoint symbols — whichever lands second rebases; that PRD's "no change to `escalate_info`'s filing contract" out-of-scope line is what α does, after it |
| `plans/task-amendment-delivery-prd.md` (4031–4037) | 4031/4032 pending | adjacent | C4 `_descoped_steps`; both declare `artifacts.py` + `workflow.py`, which β edits | **distinct**; same vocabulary; no edge; same-file collision noted — whichever lands second rebases (D4) |
| **3865** plan-tools: supersede a design decision | pending | adjacent | design-decision corrections | **distinct**; design-decision deviations use α's note-to-file until 3865 gives decisions ids; 3865 should adopt D4's vocabulary; no edge |
| **4560** architect refutes premise, no write-back | pending | adjacent | task-text correction path | **distinct**; γ's load-bearing duty is about the architect's *own* prose, not task text; named in γ's brief |
| **4799** architect premise check needs a main-ref diff | pending | adjacent | `roles.py::ARCHITECT` rule 2 | **distinct but same paragraph**; γ widens rule 2's *scope*, 4799 adds a *step*; whichever lands second rebases |
| **5520** implementer guesses `step_type` key | pending | overlapping | `plan_tools.py::_add_plan_step` | **superseded by β** (validation + documented `type` key); annotated at decompose, recommend cancel once β lands |
| **5532** architect recipe spelled twice | pending | adjacent | `roles.py::ARCHITECT` / `briefing.py` | **distinct**; β's `_replan` fix must not add a third copy (D14); γ edits the same file — rebase on landing order |
| **5225** reviewer heuristics inline + parity test | done | precedent | `test_code_quality_guidance_parity.py` | γ's guard imitates its layout |
| **3651 / 3665** `_reconcile_done_step_commits` | pending | twin | plan.json step `commit` SHA | **distinct**; α is the escalation-layer twin (D12); no edge |
| **2725** reaper merged-family skip | done | untouched | `_is_terminal_merged` | unchanged; its exclusion stays (D7) |
| **4764** rulings never reach re-dispatches/reviewers | pending | adjacent | briefing injection | **distinct**; this PRD claims no reviewer consumer (G1) |
| **5721** late-minted gating L2 | pending | none | cites `ESCALATION_LADDER_CORE` in prose only | **distinct** |
| Leo's 2026-09-08 norm (memory `3b011d26`) | ratified | consumed | info = annotation; all-info clusters close at L1 | rendered by D3; enforced by the router PRD (5222) |

No reciprocal-ownership statements found.

---

## 9. Decomposition plan

Sizing per the overlay bands (target 300–1,500 changed LOC, ≤10–12 files; sibling froth
coalesced; same-file leaves serialised by real edges). `roles.py` is edited by exactly one
leaf (γ); `plan_tools.py` by exactly one (β). All leaves `task_kind="normal"`.

| Label | Leaf | Modules | Signal (G2) | Depends on | G7 notes |
|---|---|---|---|---|---|
| **α** | Note-to-file escalation class + `resolve_escalation_shas` tool + auto-watcher routing row (D2, D3's server half, D12) | `escalation/` (server, models, queue, tests); `orchestrator/git_ops.py` (`resolve_rebased_citation`); `skills/escalation-watcher-auto/SKILL.md` | `escalate_info(note_to_file=True)` returns `status: resolved`, `resolution_class: note-to-file`, the record is in `archive/<date>/` with nothing pending and no task effect; `resolve_escalation_shas` returns `resolved_shas` for a rebased citation | **5221, 5222** (out-of-batch) | INV-1 (flag declared at the tool boundary, checked); INV-2 (structured refusal); INV-11 (a note's response is distinguishable from a routed filing's); INV-4: note volume is unbounded by design and the census (δ) is its rate watch — stated, not waived |
| **β** | Per-step disposition record + `DEVIATION_KINDS` in `shared` + `step_type` validation + `_replan` prompt fix + `_finalized_at` / immutability docs at source (D4, D14) | `shared/src/shared/plan_deviation.py` (new), `orchestrator/` (`mcp/plan_tools.py`, `artifacts.py`, `workflow.py`, tests) | `record_step_deviation` appends a typed entry visible in `plan.json` without changing `status`/`commit`; `add_plan_step(step_type='tdd')` is refused naming `test`/`impl` | — | INV-1 (`kind` enum checked); INV-9 (step-attributable facts live only in plan.json); INV-2; no INV-12 list added |
| **γ** | Doctrine inline + parity guard + one `escalate_info` contract + SHA citation instruction + architect duties + `_PLAN_DEVIATION_TOOLS` + memory pointer (D3's prompt half, D5, D6, D12's prompt half, D13, D14's allowlist) | `orchestrator/agents/roles.py`, new `orchestrator/tests/test_roles_deviation_doctrine.py`, existing role tests | The rendered IMPLEMENTER prompt names `record_step_deviation` and `escalate_info(note_to_file=True)` and no longer says "note it and stop"; a dispatched implementer's note lands as an archived note-to-file record | α, β | INV-9 (constant is the home; memory carries a dated pointer; code wins); INV-5 (three contracts → one constant, rendered once per prompt); INV-10 satisfied by containment against live constants, not a prose matcher |
| **δ** | Plan-deviation census: sibling module + nightly `finally` hook + config block with its own validation + state + watcher count retirement (D8–D11) — **the consumer / integration gate** | `scripts/legibility/plan_deviation_census.py` (new), `nightly.py`, `config.py`, `docs/legibility/legibility.yaml`, `scripts/tests/test_legibility_plan_deviation_census.py` (new), `skills/escalation-watcher/SKILL.md` | The nightly journal carries one `plan-deviation census: FIRE\|NO-FIRE -- …` line on every night including red ones; the positive-control fixture files exactly one `blocking` `escalate_info` on the synthetic task id with structured `evidence`; the negative control files none; SKILL.md's ">3 qualifying in one day" clause is gone and the per-task limit remains | α, β | INV-4 (this *is* the storm escape for the record stream); INV-2 (evidence is structured counts); INV-11 (evaluation failure → a journal line in the same grammar); INV-9 (state has one home); INV-7 (its L0 is bounded by the ladder) |

Order of landing: β immediately; α after 5221/5222; then γ and δ.

---

## 10. Out of scope

- Any retroactive sweep rewriting archived escalation records' SHAs (D12 is read-time only).
- Bare line-pin cleanup and formatting cleanup (CLAUDE.md / CONTRIBUTING.md §2–3).
- Step-count limits, plan-size caps, or any authoring-time verification pass (rulings 1, 4).
- Changes to the completion judge or to reviewer briefings (ruling 7; task 4764 and the
  router PRD's leaf β own the reviewer-briefing question).
- Any reaper change (ruling 5 is discharged by task 5223; D7).
- A `plan.json` reaper or retention policy for `.task-meta` (measured append-only; the
  census stats before it reads).
- Extending `plan_tools.py::_REPAIRABLE_PLAN_FIELDS` to nested prose (D4 records the limit).
- Giving design decisions ids or a supersede path (task 3865).
- Category validation for escalations (no category is added; `models.py`'s trigger stands).
- Collapsing the architect recipe's two copies (task 5532).

---

## 11. Open questions (tactical — decided at implementation time)

1. **Default thresholds for δ** (`rate_rise_factor` 2.0, `min_records` 10, `pattern_share`
   0.5, `prose_only_min` 3, `cooldown_days` 7, `horizon_days` 28). Seeded from the study's
   4.2→9.0/day doubling; tune after the first month of journal lines. Decide in δ.
2. **Filing primitive for the census** if `escalate_info(severity='blocking')` is ever
   restricted: `escalate_blocker` with the returned `action` ignored. Decide in δ; default
   `escalate_info`.
3. **Which amendment-pass roles count in D9's fallback clause** (`agent_role` set). Read the
   roles that carry `_ESCALATION_INSTRUCTIONS` and are invoked by `workflow.py::_amend`.
   Decide in δ.
4. **Whether `resolve_escalation_shas` should also accept a bare SHA list** (for the L2
   watcher resolving a citation outside any record). Decide in α; default record-id only.

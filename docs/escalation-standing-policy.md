# Standing policy for L2 adjudication

## Status — what is and is not in force

**Every candidate class in this document is SHADOW MODE ONLY. None is adopted.**
The L2 session's actions are unchanged by anything below: it stamps what it
*would* have ruled, then handles the record exactly as
`skills/escalation-watcher/SKILL.md` already says. Nothing here grants any
authority, and nothing in the task that wrote it changed
`escalation/src/escalation/authority.py`.

This is *not* the same as saying no standing rule exists. One does, it is live
today, and it is the next section.

## The one standing rule already in force

`skills/escalation-watcher/SKILL.md` — "Standing rule: accept verified
info-level design deviations (Leo, 2026-09-17)", task 5361 — lets the
interactive watcher itself `close_only` an info-level `design_concern` that
ratifies an already-made, evidence-checked deviation, under six stated
conditions.

**That skill owns the rule. Its six conditions are not restated here** — a
second copy would drift, and this document is not their authority. Read them
there.

What it is evidence *for*: the skeleton below generalises the shape that rule
already demonstrates in production — a reversible action, the deciding evidence
quoted, a narrow class, and the human retaining the audit.

## Provenance

Ruling D8 (Leo, 2026-09-10); agent-capacity study 2026-09-09 §4/§5/§10,
appendix P3-2 and P3-11; task 5374. Cited as a named ruling and study rather
than as a repo path — that study is not tracked in this repository, and a
dangling path in the document a later session reads as authority is worse than
a citation it has to go and find.

The measurement that motivated the ruling:

- 585 L2 records in 30 days.
- 69% of them closed `close_only`.
- 58% of archived L2s were routed to the human by skill *text* rather than by
  difficulty.
- The 2026-07-15 audit put rubber-stamping at ~45%. Its remedy — task 2630, the
  auto-close carve-out in `escalation/src/escalation/authority.py` — has not
  been re-measured since.

## The ratified skeleton

An L2 in a listed class may be ruled by the L2 session without waiting for Leo
when **all** of the following hold:

1. The action is reversible.
2. The DecisionRecord quotes the deciding evidence verbatim.
3. The class's shadow agreement was 95% or better over at least 10 items.
4. The record is not a `milestone_gate`, a deterministic-runner filing, a spend
   or model-admission question, or a physical operator action.

Every autonomous ruling appears in the return brief. Leo audits at least one in
five. A class with any audited divergence below n=20 — or above 5% at n=20 or
more — reverts to human routing until it has been re-shadowed.

## Reversible actions

<!-- Each bullet below OPENS with a backticked slug, and those slugs are held
     equal to escalation.shadow_ruling.REVERSIBLE_ACTIONS by
     tests/scripts/test_shadow_ruling_doc_contract.py. The prose is free; the
     leading slug of each bullet is not. -->

- `close_only` — close the escalation, leave the task untouched. A C1
  `resolution_action` value.
- `resume` — re-pend the subject task. A C1 `resolution_action` value.
- `add_dependency` — wire the subject behind the work it is actually waiting on.
- `update_task_amendment` — amend the task record rather than escalate.
- `file_task` — file the follow-up the record is really asking for.

Only the first two are C1 `resolution_action` values, so only those two leave a
trace the weekly count can compare against. The other three are task-side and
are reported `not_comparable` — see "How the shadow measurement is read" below.

## Human-forever gates

These stay with the human forever, whatever any shadow measurement shows. No
agreement rate promotes one of them.

<!-- Same extraction contract as the reversible-action list above: the leading
     backticked slug of each bullet is held equal to
     escalation.shadow_ruling.HUMAN_FOREVER_GATES. -->

- `milestone_gate` — a milestone gate of any kind. **Mechanically detected** by
  `escalation/src/escalation/shadow_ruling.py::mechanically_gated`.
- `deterministic_runner_filing` — anything filed by the deterministic runner's
  sentinel role. **Mechanically detected** by the same function, by that role
  and also by a runner-filed category that is not itself a milestone
  (`curator_adjudication_missing`): the slug names the gate a record actually
  trips, so such a record is not reported as a milestone it has nothing to do
  with.
- `model_admission` — admitting a model to the routing allowlist, or changing
  which model a role gets.
- `physical_operator_action` — anything needing hands on the machine.
- `irreversible_deletion` — deleting work, history or data that cannot be
  restored.
- `spend_or_eval_launch` — committing spend, or launching an evaluation that
  commits it.
- `post_breaker_resume_scheduler` — resuming the scheduler after a breaker has
  tripped.

**Two of the seven are detected; five are not.** `mechanically_gated` covers
exactly `milestone_gate` and `deterministic_runner_filing`, because those are
the only two with a signal on the record itself. The remaining five are semantic
judgements with nothing to match on, so **this list — not the detector — is the
authority.** A `None` from that function means "no mechanical gate detected",
never "not gated".

**The detector is not `authority.py`'s denylist, and `design_concern` is the one
cell where they differ** (esc-5374-1). `design_concern` is not on the gate list
above, so it is not gated here: it is a first-tranche class the same ratified
text says to shadow, and gating it would make that class unmeasurable. The
detector therefore derives its categories from the denylist **minus that one
member**, so a new runner-filed category added to `authority.py` still gates
here for free. Nothing about the auto-watcher's authority changes: it still may
not auto-close a `design_concern`.

The argument for why those two tables answer different questions is not
repeated here — it lives beside the subtraction it justifies, in
`escalation/src/escalation/shadow_ruling.py::_UNGATED_DENIED_CATEGORY`.

## Time-boxing a human gate

A human gate that has gone unanswered for N days gets the adjudicator's
recommendation **attached in shadow**. It is never applied. The gate stays the
human's; the time box only means the recommendation is waiting when they get to
it, rather than having to be re-derived then.

## First-tranche candidate classes

**To be shadowed, not adopted.** Each of these is a recurring L2 shape the
measurement exists to judge. None of them may be ruled today.

<!-- Same extraction contract as the two lists above: the leading backticked
     slug of each bullet is held equal to
     escalation.shadow_ruling.FIRST_TRANCHE_CLASSES. -->

- `risk_identified_branch_behind_main` — a `risk_identified` reporting a branch
  N commits behind main, or a save-WIP tip. Proposal: hand it to the
  branch-rescue path.
- `risk_identified_recovery_veto_streak` — a `risk_identified` filed by
  `orchestrator-recovery-veto-streak`. Proposal: close it under task 4541's fix.
- `design_concern_semantic_collision` — a `design_concern` reporting a semantic
  collision with landed work. Proposal: draft the reconciled design as a task
  amendment, and escalate only when the two are genuinely incompatible.
- `infra_issue_transient_self_cleared` — an `infra_issue` of a transient class
  whose probe has since self-cleared. Proposal: close it.

**How `design_concern_semantic_collision` is scoped against the live rule
above.** It is the *semantic-collision* class, and a semantic collision is a
choice about what happens **next**. The 5361 standing rule covers only
after-the-fact ratification of a deviation an agent has **already made**. The
two do not overlap. A record already closable under 5361 must therefore **not**
be shadow-stamped: the session is the adjudicator there, so a stamp would be
measuring the session against itself. See "The integrity rule" below.

## Adoption preconditions

A class adopts only when **both** hold:

1. Task 3346 has landed.
2. The class has met the threshold in the skeleton above — 95% or better over at
   least 10 items.

**What `authority.py` does and does not require.** Its
`L2_AUTO_CLOSE_DENY_CATEGORIES` / `L2_AUTO_CLOSE_DENY_ROLES` constrain **only
identified callers** listed in `ROLE_LEVEL_ALLOWLIST`, whose sole member is the
auto-watcher identity `orchestrator-escalation-watcher-auto`. A header-less
interactive connection is never narrowed by that module — its own docstring says
so, and that is the esc-2087-2 human-channel guarantee.

So adopting a class for the **interactive** session needs no `authority.py`
change. Task 5361 is the existence proof: a live rule closing `design_concern`
from the interactive session, shipped with no `authority.py` edit at all.
Extending any class to the **auto-watcher** arm is the case that would need one.

Which arm you are extending decides whether `authority.py` is in your diff.

## The integrity rule

**Never shadow-stamp a record this session will itself rule.**

`escalation/src/escalation/classify.py::_HUMAN_RESOLVERS` contains
`escalation-watcher`, so the session's own close is reported in the same `human`
tier as a Leo ruling. A stamp plus a self-close is therefore the session
agreeing with itself — and without a defence it would inflate a class toward its
own adoption threshold, silently, using its own actions as the evidence.

The weekly count buckets such a record as `self_resolved` and excludes it from
every rate. That is a backstop that makes a violation *visible*, not a licence
to produce one: a violation costs the sample either way.

## How the shadow measurement is read

Each stamped proposal is compared against the record's observed outcome and
lands in exactly one bucket:

- `agreed` / `diverged` — the proposal was a C1 action *and* the record recorded
  one, so the two could be checked against each other.
- `not_comparable` — one of those two sides is missing: the proposal was one of
  the three task-side actions, which leave `resolution_action` unset, or the
  record itself recorded no `resolution_action` (the legacy shape D10 in
  `escalation/src/escalation/server.py::resolve_issue` describes). Counted
  separately and never folded into either side: a class whose proposals are
  mostly task-side is visibly not yet measurable rather than falsely green, and
  missing data never reads as disagreement.
- `gated_stamps`, `self_resolved`, `rejected_stamps` — records excluded from
  every rate, counted over the SAME window as `agreed`/`diverged`, so a small
  sample and a discarded one cannot look alike. `rejected_stamps` counts a
  record that carries a marker line the codec could not read: a sample this
  measurement lost, and one a pasted report must not be able to hide.
- `unresolved_lifetime` — stamps still pending, including a pending record that
  would also have been gated or whose marker is unreadable. The one number in
  the report that is not from the window: a pending record has no `resolved_at` to window on, so this is the
  standing backlog as of the sweep, and its name carries that.

The rate is computed over the comparable subset only, and the comparable
denominator is printed beside it, so "95% or better over at least 10 items" is
decidable from a pasted report alone.

## Pointers

- The shadow-mode arm the L2 session follows:
  `skills/escalation-watcher/SKILL.md`, "Shadow-mode standing-policy rulings
  (measurement only)".
- The weekly count:

  ```
  uv run --directory escalation python -m escalation.shadow_ruling \
      --queue-dir <project_root>/data/escalations
  ```

- The vocabularies, the detector and the report:
  `escalation/src/escalation/shadow_ruling.py`.

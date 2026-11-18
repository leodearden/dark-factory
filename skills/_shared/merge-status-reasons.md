# Acting on an Epistemic merge_status Answer — Canonical Reference

This document is the single source of truth for **what an agent does when `merge_status`
answers with an epistemic state**: what each `reason` means, and the action it permits. Four
skills consume it:

- [`skills/merge-queue/SKILL.md`](../merge-queue/SKILL.md) — interactive submit-and-poll; the only consumer that resubmits on its own
- [`skills/unblock/SKILL.md`](../unblock/SKILL.md) — the step-7 poll loop and the `merge_cancel` cleanup
- [`skills/unblock-low-risk/SKILL.md`](../unblock-low-risk/SKILL.md) — unattended; an answer that leaves the merge possibly in flight (a wait-state reason at its deadline, or a task_id re-poll still live or waiting) is handed off, and every other answer except a verified landing ends in ABORT
- [`skills/escalation-watcher/SKILL.md`](../escalation-watcher/SKILL.md) — its merge-status fallback note

Each consumer keeps its own **dispositions** inline in its own file — resubmit, report, abort,
follow a train. This document owns only the meaning of each reason and the action it permits,
the same split as [Deriving the landed sha](deriving-landed-sha.md).

Sources of truth, named once: the reason vocabulary is
`shared/src/shared/merge_state.py::MergeReason` (partition `MERGE_REASONS`); the state partition is
`shared/src/shared/merge_state.py::EPISTEMIC_STATES`; the response shape (`state`, `reason`,
`hint`, `source`, `last_outcome`) and the hint table are
`escalation/src/escalation/merge_status_answer.py`.

<!-- merge-state-vocab:begin partition=EPISTEMIC_STATES
     Mirrors shared/src/shared/merge_state.py::EPISTEMIC_STATES. Pinned by
     scripts/tests/test_merge_state_vocabulary_consistency.py — extend the
     enum and this list goes red until it matches. -->
`no_record`, `stale_record`, `journaled` and `unknown` describe the server's records rather than the merge, and every answer in one of them carries a `reason`.
<!-- merge-state-vocab:end -->

---

<a id="before-you-switch"></a>
## Before you switch

1. **Switch on `reason`.** Never on `hint`, which is prose rendered from `reason` for a human
   reader, and never on a git ancestry exit code: the server's git tier has already run the
   landing probe, and its verdict arrives as a field. Where a hint and the action column
   below disagree, the action column governs. The `never_submitted` and
   `prior_run_terminal` hints both invite a resubmit, but the server writes a hint without
   knowing which handle you polled or whether anyone is watching. A resubmit is licensed
   only on the terms stated in the [switch](#the-switch).
2. **`last_outcome` is history unless it says otherwise.** It accompanies any state whenever a
   durable finalize row exists for the key, labelled with `run_id`, `finished_at` and
   `is_current_run`. `is_current_run: false` means another orchestrator run wrote it — never
   read it as this poll's outcome.
3. <a id="request-id-polls"></a>**Re-poll a request_id answer by task_id.** Only a task_id- or
   branch-keyed poll reaches the git tier (`escalation/src/escalation/server.py::_PollKey.landing_key`).
   Before acting on `stale_record` or `no_record` from a request_id poll, re-poll once with
   `merge_status(task_id="<TASK_ID>")` and switch on that answer instead. The exception is
   `coalesce_train_id`: it names a train, not your request, so act on its row directly. A
   landing shows up in the re-poll as an ordinary `done` with `kind: "found_on_main"` and a
   `merge_sha`, and only that shape counts as one. A re-poll `done` without that `kind` may be
   an earlier round's current-run row, so it is unconfirmed, and any `merge_sha` it carries is
   the record's, not a git proof. If the request_id answer was `prior_run_terminal` and the
   re-poll is also `stale_record`, report the request_id answer's `last_outcome`: it names
   your request.

<a id="the-switch"></a>
## The switch

<!-- merge-state-vocab:begin partition=MERGE_REASONS
     Mirrors shared/src/shared/merge_state.py::MERGE_REASONS. Pinned by
     scripts/tests/test_merge_state_vocabulary_consistency.py — extend the
     enum and this list goes red until it matches. The guard checks this
     line, not the table below: add or rename the matching table row in the
     same edit. -->
The reasons are `probe_failed`, `journaled_not_live`, `prior_run_terminal`, `never_submitted`, `coalesce_train_id` and `landing_unattributable`, one row each below.
<!-- merge-state-vocab:end -->

| `reason` | rides on | meaning | action |
|---|---|---|---|
| `probe_failed` | `unknown`; `source` names the failing tier | a probe raised — the answer is not negative | **WAIT**: retry on the normal backoff, under the [wait bound](#wait-bound) |
| `journaled_not_live` | `journaled` | the durable journal holds the request but the live worker does not show it; startup recovery re-drives it | **WAIT**, under the [wait bound](#wait-bound). Never resubmit: the journal still owns it |
| `prior_run_terminal` | `stale_record` | only another orchestrator run's finalize row exists | **REPORT** `last_outcome` (`state`, `run_id`, `finished_at`, `reason`) as history, and stop. Never resubmit on it and never stamp `done_provenance` from it: resubmitting after a prior-run record is a human's decision, and an unattended caller aborts |
| `never_submitted` | `no_record` | no tier holds a record (and, on a task_id or branch poll, git shows no landing): never enqueued (expected for the own `request_id` of an `attached` submit whose `poll_by` is `task_id` or `branch`), or its record was lost | **NOT IN FLIGHT HERE**: the consumer's disposition decides. A resubmit is licensed only by a task_id- or branch-keyed `never_submitted` with `get_merge_queue()` showing no entry for the branch |
| `coalesce_train_id` | `stale_record` | the id names a coalesce train with no outcome row of its own; `last_outcome` is a member's absorption record, never your outcome | **FOLLOW THE MEMBERS**: for your own branch, take merge-queue's [Follow the superseded successor](../merge-queue/SKILL.md#follow-the-superseded-successor) Escape |
| `landing_unattributable` | any epistemic state but `unknown`, replacing that state's own reason | git shows the branch's work on main but could not confirm attribution or effect | **VERIFY** at once, never resubmit. It is not transient, so it exits every poll loop, including an arm that otherwise treats epistemic answers as its live state. Run [the ladder](deriving-landed-sha.md#the-ladder) in full: a stampable sha is a landing (`done_provenance` kind `found_on_main`, with `commit` and the ladder's `note`); no stampable sha means stop and report |
| any reason not listed here | — | the vocabulary grew past this table | **FAIL CLOSED**: stop and report `state`, `reason` and `hint` verbatim |

<a id="wait-bound"></a>
## The wait bound

An unbroken run of wait-state answers (the **WAIT** rows of [the switch](#the-switch)) is bounded at
**20 minutes from the first of them**. At expiry, stop and report `state`, `reason`, `source`
and `hint` — never resubmit, never direct-merge. The bound exists because the merge journal is
not complete across restarts (`plans/merge-status-durable-non-landed-prd.md`, Out of scope:
task 3047), so recovery may never re-drive an entry. A caller with its own tighter deadline
uses that instead.

<a id="safety-rules"></a>
## Rules no reason relaxes

- **Never fall back to a direct merge in response to an epistemic answer.** Direct merge is a
  response to the orchestrator being down, never to what its records say.
- **Never resubmit while the merge may still be in flight.** Only the `never_submitted` row can
  license a resubmit, and only on the terms it states.

<a id="merge-cancel"></a>
## merge_cancel answers

`merge_cancel`'s no-waiter path (`escalation/src/escalation/server.py::merge_cancel`) answers
the same epistemic states, but its `reason` field is the hint prose, not the code. To switch,
poll `merge_status(task_id="<TASK_ID>")` and switch on that answer's `reason`.

# Task 4835: DecisionRecord plumbing decisions

Task 4835 coalesced five workstreams on the fleet-global DecisionRecord store
(`orchestrator/src/orchestrator/session_registry.py`). Each section below states
its outcome first, then why. The facts were measured read-only on 2026-10-01
against base `8be021797f` and the live fleet store: 1017 records, 40 open, 882
with bare `esc-…` ids, plus hand-prefixed legacy ids (`df-esc` 64, `reify-esc`
7, `recon-esc` 3, `scp-esc` 2, `dark_factory-esc` 2).

## #3873: multi-queue decision. Outcome: WON'T DO

`DecisionRecord.escalations_dir` stays a scalar, and the first filer's queue
wins (`merge_decision_enrichment`, with the existing WARNING). This is a
decided limit, not a deferral. The MODE-2 reap gap pinned by
`test_main_reap_decisions_mode2_collapsed_decision_is_reapable_only_by_its_stamped_queue`
is accepted.

- **A list of queues is the wrong shape.** Escalation ids are minted per queue,
  so a MODE-2 twin can carry a different id in the other queue. A faithful
  generalisation would be a list of (queue, escalation_id) pairs. That
  contradicts `_merge_queue_and_escalation_id`'s one-pair invariant, which
  exists so a pair is never synthesized across queues.
- **Any-of closure fails closed.** "Close when either queue's escalation is
  terminal" drops the shared row as soon as one queue dismisses its copy as a
  duplicate of the other, while the gate is still live in the other queue.
  All-of closure is no more reapable than first-writer-wins.
- **The current gap fails open.** The row stays visible, and `close-decision`
  (task 5376) and `reopen-decision` (below) give a human or agent a remedy. The
  module's asymmetry-of-harm rule (`_run_reap_decisions` docstring) ranks that
  the cheaper mistake.
- **The 3640 evidence was about another population.** The "16 ambiguous"
  figure in `plans/3640-decision-queue-backfill-report.md` concerned the legacy
  UNSTAMPED population, not MODE-2 collapses. On 2026-10-01 that population
  measured 0 open (`scripts/backfill_decision_queue_stamp.py --verify` with the
  report's six `--queue` args, rc 0).

## #3905: project-scoped decision ids. Outcome: DONE

`write-decision` treats `--id` as project-LOCAL. It files and prints
`qualify_decision_id(project, local_id)`, which is
`f'{normalize_project_token(project)}-{local_id}'`, or `local_id` unchanged when
the project folds to `''`. The same escalation id in two projects lands on two
rows. Both queues of one project share the prefix, so the MODE-2 collapse still
gives one row; the queue stays out of the id on purpose.

**Why `-` is the separator:**

- Canonical tokens never contain `-` (the fold maps it to `_`), so the join is
  injective.
- `-` survives `_DECISION_ID_SANITIZE_RE`, so the file stem equals the id.
- It is byte-identical to the `<project>[-<queue tag>]-<esc>` ids the sitting
  (`scripts/sitting/payloads.py::sitting_decision_id`) already filed, and to the
  live `dark_factory-esc-*` ids. The sitting now delegates the join to
  `qualify_decision_id`, files the local id, and closes the qualified one.

The join is pure. It never asks "is this already qualified?", because that
would be an ad-hoc string parser (heuristic 12). A watcher that passes a
printed id back as `--id` gets a visible duplicate row, which fails open.

**Migration is lazy, existing-key-wins** (`_locked_filing_target`):

- If the qualified key has no file, but the bare legacy key holds a record of
  the SAME project, the filing continues that record in place. It keeps its
  custody, including the task 3872 held-closed state, so the first post-deploy
  restart of a watcher undoes no operator dismissal.
- A legacy record of another project is ignored.
- Locks go qualified-then-legacy, and the legacy lock is taken only when its
  file exists, so a fresh filing leaves no orphan sidecar. Only `write-decision`
  ever holds two locks, so the order cannot deadlock.

**There is no eager rename.** Renaming would break references in flight (the
cockpit's `set_manual_boost(old id)`, sitting close payloads) and could crash
between write and delete, leaving twin files. Closed legacy records are inert,
and `list_decisions` finds open ones whatever their key.
`migrate_decision_project_tokens` invariant (e), "ids are never rewritten",
still holds.

**There is no `SCHEMA_VERSION` or `SCHEMA_MINOR` bump.** The record's shape is
unchanged; only the value format of new ids changes. The cockpit
(`decision:{id}`), the sitting and the reaper all treat ids as opaque.

**The residual refusal arm stays.** `_run_write_decision` still refuses, with
an ERROR and rc 0, a filing whose qualified id is held OPEN by another project.
That is reachable when a legacy hand-prefixed id sits on another project's
future key, as `recon-esc-7459-1` (held by project reify) shows.

**Known limit.** A future `PROJECT_TOKEN_ALIASES` entry changes the qualifier
for NEW filings only. Records already filed under the old canonical token keep
their ids, so a re-file after such an alias opens a new row unless the old
qualified key is migrated by hand.

## #4268: residual unstamped decision. Outcome: NO CODE FIX NEEDED

- `reify-task-dispatch-stalled-20260815` was filed 2026-08-15. That is four days
  before Merge task/3559 (`23ce883356`, 2026-08-19) made `--escalations-dir`
  argparse-required and added the refusals for an empty or `<unknown>` stamp.
  Its id is ad hoc: no SKILL template contains it. It was closed 2026-10-01T19:53Z
  via `close-decision` with evidence (state `dropped`).
- **Sentinel parks are NOT exempt** from `--escalations-dir`. This covers a park
  with no `--escalation-id`, such as a lease orphan or a pipeline stall. The
  3872 same-queue custody arm keys on stamp equality. An unstamped sentinel
  later re-filed with a stamp falls to the different-queue arm and, if dropped,
  is fully overwritten, which re-opens it. Both watcher SKILLs say so.
- **No recurring check.** `_run_write_decision` is the only production code that
  creates a decision file; every other writer is a read-modify-write of an
  existing record. It refuses an unstamped filing at two layers, so nothing can
  regrow the population.
- `backfill_decision_queue_stamp.py --verify` with the six `--queue` args
  returned rc 0, "unstamped open records: 0", on 2026-10-01.

## #4460: re-open a held-closed row. Outcome: `reopen-decision` CLI verb

`session_registry.py reopen-decision --id <printed id> --project <p>
--escalations-dir <q> [--root]` calls `reopen_decision`.

- It shares close-decision's compare-and-swap (`_refuse_unless_named_record`),
  so another project's or queue's record at that id is refused.
- It exits 1 on a refusal or an unreadable record. Its caller is a person or
  agent who must see the failure, which is `_run_close_decision`'s rationale,
  not the rc-0 contract that protects `spawn-claude.sh`.
- It sets state `open` and clears `closing_evidence` and `closed_at`, because
  `close_decision_with_evidence` never overwrites evidence. Without that, a
  re-opened gate could never be closed with evidence again. `filed_at` and
  `manual_boost` are kept (custody).
- **Why reopen-only, not a generic `update-decision-state`.** A generic setter
  would allow closing without quoted evidence and moving between terminal
  states, bypassing task 5376's evidence discipline
  (`docs/escalation-standing-policy.md`). Closed-to-open is the one transition
  with no operator surface, so it is the one interface added (heuristic 9).
- **Why no cockpit (C5b) action.** The decision pane
  (`cockpit/src/cockpit/panes/decision_queue.py`) shows only `state == open`
  rows, so a re-open action needs a closed-rows view. That is a UI design task
  outside this one.

The held-closed WARNING in `write-decision` and both watcher SKILLs now name
this verb as the in-place remedy. A new id is only for a genuinely different
ask.

## #5159: zero-match reap warning. Outcome: OPT-IN `reap-decisions --expect-matches`

With the flag, `reap-decisions` warns once when the folded `--project` matches
ZERO registry records in ANY state. The warning names the folded tokens that do
exist, with their counts (`project_token_census`,
`unmatched_project_token_hint`).

- **Why not always on.** By count alone, a never-filed project (healthy, zero
  decisions ever) cannot be told apart from a mismatched token. An always-on
  warning would fire every Main Loop cycle, forever, for every project whose
  watcher never parks.
- **Why not won't-do.** The `collections.Counter` one-liners both SKILLs shipped
  counted RAW spellings (`df`, `dark-factory` and `dark_factory` as three
  buckets) while the reaper folds them, so the hand diagnostic misled. The
  verb-native census reuses the one fold (heuristic 11). The one-liners are
  retired.
- **One emission point.** `_reap_scope_hint` returns task 3813's
  `declined_project_token_hint` first, as the special case, and the opt-in
  census hint otherwise. The verb never warns twice, and there are not two
  independent warning paths over one verb.
- **Census cost.** The census is a separate `list_decisions()` scan, outside the
  `_status` closure, which only sees reap candidates. It is read only when the
  flag is set.

## File size

`session_registry.py` stays one file. Splitting out the decision machinery
would break its by-path execution (`spawn-claude.sh` and every watcher run it
with only the script's own directory on `sys.path`, which
`TestStdlibOnlySelfContainment` pins), and would make the new module reach back
into this one for `fleet_root`, `_atomic_write_text` and
`normalize_project_token`.

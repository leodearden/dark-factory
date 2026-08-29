#!/usr/bin/env python3
"""One-shot (idempotently re-runnable) corpus correction: version-scope the
stale "Claude CLI --resume is broken because sessions are per-project-directory"
claim in Mem0, and retire the hygiene warning's now-false STATUS clause
(task 4610, esc-3578-5).

Why a script exists at all
--------------------------
The same two writes are available as the ``update_memory`` MCP tool, and this
script calls the SAME function underneath —
``MemoryService.update_memory``. It owns no copy of the amendment logic; it
parses flags, corroborates each pre-image, and delegates. It exists because
NOBODY WHO CAN BE DISPATCHED AT THIS WORK MAY CALL THAT TOOL:

  * ``update_memory``'s authorization gate admits only ``recon-stage-*`` and
    ``curator-*`` agent ids. The task-3578 steward hit it on both the
    ``content_amend`` and ``metadata_patch`` arms and recorded the refusal in
    esc-3578-5 rather than working around it — which is why 6403e96b was still
    uncorrected months later.
  * An orchestrator-dispatched agent CANNOT clear that gate:
    ``orchestrator/src/orchestrator/agents/briefing.py`` hardcodes
    ``agent_id = f'claude-task-{task_id}-{role}'``, so every role this task
    could route to fails both arms exactly as the steward did.
  * And it MUST NOT try. ``skills/curate-fused-memories/SKILL.md`` names this
    precise situation: adopting a ``curator-`` id "because you hit
    ``Mem0UpdateNotAuthorized`` in the middle of unrelated work and want past
    it" is "self-authorization for a silent-rewrite primitive ... not a key you
    borrow". Widening the allowlist is closed too — the ``Mem0UpdateConfig``
    field descriptions in ``fused_memory.config.schema`` record Leo's
    esc-3524-1 ruling that the bar is deliberately narrow.

The gate is MCP-TOOL-LAYER ONLY; the service seam beneath it carries no prefix
check. So the sanctioned route — the one six sibling scripts in this directory
already take — is to land the change as REVIEWABLE CODE and have a session that
legitimately holds the capability run ``--apply``. That is the whole point: the
corpus mutation stays visible in git history and in review, instead of
happening once, invisibly, inside somebody's chat transcript.

What it does
------------
Exactly TWO in-place CONTENT amendments, in a fixed order, and NO deletes:

  **A — 6403e96b** ("Broken Claude CLI --resume due to sessions being
  per-project-directory"). Replaced with a version-scoped rewording modelled on
  the already-corrected Graphiti half of this same contradiction (edge
  fb96a8c0-da93-4e34-9e2c-6a1e7a3d1c08, reworded 2026-08-22). It PRESERVES the
  real 2026-04-10 incident and the commit that fixed it, labels the causal
  explanation as an inference that was never measured, states the contrary
  2026-08-19 measurement, and records what remains UNDETERMINED.

  **B — d007aa46** (the esc-3578-5 corpus-hygiene warning). Its final STATUS
  sentence claims 6403e96b "is STILL UNCORRECTED". Once A lands that sentence
  is false, so it is replaced by one recording the correction. Every measured
  score and the do-not-delete rationale are kept verbatim.

Why an amend and not a delete
-----------------------------
The April incident was REAL: a steward CWD switch to project_root did break
``--resume``, with instant "No conversation found" failures at 0 cost / 0 turns
/ 0 duration on all three retries, and removing the switch (commit e001dd3746)
fixed it. Only the CAUSE bolted onto that symptom was an unmeasured inference.
Deleting the record — or flatly invalidating it — would retire a true
historical observation, which is exactly the move the task-3578 steward
declined on the Graphiti side after provenance showed a measured symptom with
an inferred cause. There is no delete arm anywhere in this script, and the test
suite asserts its absence.

Safety properties
-----------------
* **Dry-run is the DEFAULT.** ``--apply`` is opt-in.
* **Corroborate before acting.** Each target's exact pre-image is pinned here
  and re-read live IMMEDIATELY BEFORE THAT TARGET'S OWN WRITE; a mismatch
  REFUSES rather than clobbers. An in-place amend is invisible to every
  downstream reader, so blindly overwriting a record that changed underneath us
  (a curator sitting, a recon Stage-1/2 consolidation) would destroy someone
  else's correction with no trace. The per-target re-read matters because the
  batch read happens before ANY write: without it, the second target's
  corroboration would predate the first target's write by a Qdrant round-trip
  plus a re-embed, and a race landing in that window would be clobbered by the
  very guard meant to catch it. ``MemoryService.update_memory`` offers no
  compare-and-swap, so the residual read-to-write window is narrowed to a
  single await rather than eliminated.
* **Idempotent.** Re-running after a successful apply finds the sentinel in
  both records and writes nothing. Safe to re-run after a partial failure.
* **Ordered.** B is written only if A actually succeeded (or was already
  amended). If A fails, B is short-circuited untouched — a corpus asserting
  "corrected" while the stale record still misleads is strictly worse than the
  status quo, and undetectable to anyone who trusts the warning.
* **Fail-closed capability preflight** before the first write, so a run that
  cannot write mem0's history dir refuses up front instead of half-writing.
* **Attributed writes** (``_source=WRITE_SOURCE``, and ``agent_id`` set to the
  same value) so the amendment storm alarm and the metadata-vocabulary census
  both read this sweep rather than a generic ``mcp_tool`` and a null agent.

Usage
-----
  # Dry run (default): corroborate both pre-images, print the report, write
  # nothing. Safe from anywhere, including a sandboxed task worktree.
  python scripts/amend_stale_resume_cwd_records.py

  # Commit both amendments.
  python scripts/amend_stale_resume_cwd_records.py --apply

Exit code is 0 for a clean run (dry-run, fully applied, or all-skipped) and 1
if ANY target was refused, so an operator can gate on it without parsing the
report.

Status of this correction: NOT YET APPLIED. Like
``scripts/repair_recon_citation.py`` before it, the authoring task agent could
not run ``--apply`` — a task agent holds neither the ``update_memory``
authorization prefix nor (from a sandboxed worktree) write access to mem0's
history directory. An operator or steward session must run it. Update this line
to ``DONE — APPLIED <date> by <session>`` once it has been.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Module-level (NOT deferred) so the test suite can monkeypatch the module
# attribute, and so an import-time failure of the guard is loud rather than
# discovered halfway through a write. The rest of the live stack --
# FusedMemoryConfig / MemoryService -- is imported lazily inside main(),
# mirroring tag_cgl_eta_rehome_scope, so importing this module for tests never
# touches config or a backend.
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

logger = logging.getLogger('amend_stale_resume_cwd_records')

# ---------------------------------------------------------------------------
# Write attribution
# ---------------------------------------------------------------------------

#: Written to every ``update_memory`` -- as ``_source`` AND as ``agent_id`` --
#: so the write journal attributes each amendment to this sweep rather than to
#: a generic ``mcp_tool``. The amendment storm alarm reads ``_source``; an
#: unattributed bulk rewrite of records this old is exactly the shape that
#: alarm exists to catch. The metadata-vocabulary census and the unknown-key
#: storm detector, which a ``metadata_patch`` triggers at the service seam,
#: read ``agent_id`` instead -- so the same value goes to both and neither view
#: reads this run as anonymous. Two writes is far under either threshold, but
#: attributing them correctly costs nothing and keeps both journals readable.
WRITE_SOURCE = 'amend_stale_resume_cwd_records'

#: Recorded on the write journal row beside the amendment.
WRITE_REASON = (
    'Version-scope the stale Claude-CLI --resume/cwd claim and retire the '
    'hygiene warning that says it is uncorrected (task 4610, esc-3578-5)'
)

#: Distinctive substring present in BOTH replacement texts. This is what
#: carries idempotency: a record already containing it has been amended, so a
#: re-run skips it rather than tripping the pre-image mismatch refusal.
#:
#: It lives in CONTENT rather than metadata deliberately — content is the field
#: already read for the pre-image check (no extra fetch), it survives a
#: metadata wipe or a ``metadata_mode='replace'`` from an unrelated sweep, and
#: it keeps idempotency independent of whether the optional metadata arm
#: survives vocabulary validation.
AMENDED_SENTINEL = 'amended in place by task 4610'

#: Outcomes meaning the corpus did not get what the plan intended. Named once,
#: here, so :func:`resolve_exit_code` and the report agree on what "clean"
#: means instead of each keeping its own list -- the drift that would let a new
#: refusal exit 0 and read to an automated caller as success.
ERROR_OUTCOMES: frozenset[str] = frozenset({
    'refuse:read_error',
    'refuse:not_found',
    'refuse:preimage_mismatch',
    'refuse:store_unavailable',
    'refuse:write_error',
    'refuse:write_failed',
    'refuse:precondition_failed',
})


# ---------------------------------------------------------------------------
# Amendment targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AmendTarget:
    """One record to amend, with the pre-image it must still match.

    ``expected_preimage`` is the record's exact content as read live from Mem0
    on 2026-08-28 while authoring this script. It is the corroboration key, not
    documentation: :func:`classify_amend_target` refuses to write unless the
    live content still equals it, so a record that moved underneath us produces
    a loud refusal instead of a silent clobber.

    ``metadata_patch`` is optional and deliberately NOT load-bearing — the
    content sentinel carries idempotency on its own, so dropping the metadata
    arm entirely would lose provenance and nothing else.
    """

    memory_id: str
    expected_preimage: str
    new_content: str
    metadata_patch: dict[str, Any] | None = field(default=None)


#: The stale record's pre-image: one bare sentence asserting an inferred cause
#: as fact, with no date visible to a reader and nothing marking it historical.
_STALE_PREIMAGE = (
    'Broken Claude CLI --resume due to sessions being per-project-directory'
)

#: Modelled on Graphiti edge fb96a8c0-da93-4e34-9e2c-6a1e7a3d1c08's current
#: text (reworded 2026-08-22 for the same contradiction), adapted to a Mem0
#: single-claim record. Keeping the two stores telling the same story with the
#: same caveats is the point — two independently-worded corrections drift.
_STALE_REPLACEMENT = (
    'SUPERSEDED AS A GENERAL CLAIM — historical record only; do NOT cite as '
    'evidence that a moved cwd blocks --resume today. '
    'WHAT WAS REALLY OBSERVED (2026-04-10, Claude CLI version unrecorded): a '
    'steward CWD switch to project_root broke --resume with instant "No '
    'conversation found" failures — 0 cost, 0 turns, 0 duration on all 3 '
    'retries. That incident was real and was fixed by commit e001dd3746, which '
    'removed the CWD switch. '
    'WHAT WAS ONLY INFERRED: the causal explanation recorded alongside it — '
    '"sessions are stored per-project-directory" — was an inference from that '
    'symptom, never a measurement, and it is FALSE for the current CLI. '
    'MEASURED CONTRARY FACT (task 3578, Claude Code CLI 2.1.236, 2026-08-19): '
    'the CLI is cwd-AGNOSTIC for --resume. It scans every projects/*/ subdir of '
    'CLAUDE_CONFIG_DIR by session id and ignores BOTH the encoded directory '
    'name AND the cwd recorded inside the JSONL records; a transcript under a '
    'completely unrelated projects/<enc>/ still resolves. Arms M1 (re-encoded '
    'under new cwd), M2 (original encoding, new cwd) and Y (unrelated path) all '
    'resolved and reached auth. Full method and results: '
    'plans/session-resume-eligibility-seam-prd.md §14; mem0 '
    '0ecc40cf-db46-4c91-9240-cf8991fe4dfc. '
    'UNDETERMINED: whether the CLI changed behaviour between 2026-04 and '
    '2026-08, or the original 2026-04 diagnosis mis-attributed a real failure '
    'to the wrong cause. Nobody has measured the April-era CLI, so this is not '
    'resolvable from the record. '
    f'(Content {AMENDED_SENTINEL} on the basis of esc-3578-5; the record is '
    'kept, not deleted, because the April symptom was measured.)'
)

#: The hygiene warning's pre-image, read live 2026-08-28. Everything up to the
#: STATUS clause is preserved verbatim in the replacement below.
_WARNING_PREIMAGE = (
    'CORPUS-HYGIENE WARNING (2026-08-22, esc-3578-5): the query "claude CLI '
    '--resume session cwd changed / sessions stored per-project-directory" '
    'returns STALE records ABOVE the correct ones, so rank is not a truth '
    'signal here. Measured on that exact query: mem0 '
    '6403e96b-f1af-403a-9513-59f007ed6d39 ("Broken Claude CLI --resume due to '
    'sessions being per-project-directory", 2026-04-10) returned at store_score '
    '0.786 — the TOP hit of the whole result set — while the correct entry '
    '0ecc40cf-db46-4c91-9240-cf8991fe4dfc returned at 0.692. Graphiti edge '
    'fb96a8c0-da93-4e34-9e2c-6a1e7a3d1c08 returned at graphiti store_rank 1 '
    'with temporal=null and created_at=null, i.e. NO visible date, so a reader '
    'cannot tell it is old. THE TRUTH: Claude Code CLI 2.1.236, measured '
    '2026-08-19 by task 3578, is cwd-AGNOSTIC for --resume — it scans every '
    'projects/*/ subdir of CLAUDE_CONFIG_DIR by session id and ignores both the '
    'encoded dir name and the cwd inside the JSONL. See '
    'plans/session-resume-eligibility-seam-prd.md §14. WHY THE OLD RECORDS '
    'EXIST AND ARE NOT SIMPLY WRONG: the 2026-04-10 incident was REAL — a '
    'steward CWD switch to project_root did break --resume (0 cost, 0 turns, '
    'all 3 retries), fixed by commit e001dd3746. Only the CAUSE attached to it '
    'was an inference. Whether the CLI changed between April and August or the '
    'April diagnosis mis-attributed the cause is UNDETERMINED; the April-era '
    'CLI was never measured. STATUS: edge fb96a8c0 was reworded in place on '
    '2026-08-22 and now self-dates and carries the correction. Mem0 6403e96b is '
    'STILL UNCORRECTED — update_memory rejected both its content_amend and '
    'metadata_patch arms for a steward agent_id (authorized prefixes are '
    'recon-stage-* and curator-* only), so it needs an authorized agent. Task '
    '3730 is the consumer at risk, since its acceptance criterion is to obtain '
    'this exact gate answer.'
)

#: The prefix of the warning that stays byte-identical: everything before its
#: STATUS sentence. Split out so the replacement is visibly an edit to ONE
#: clause rather than a rewrite of a record whose value is its measurements.
_WARNING_PRESERVED_PREFIX = _WARNING_PREIMAGE.split(' STATUS: ')[0]

_WARNING_REPLACEMENT = (
    _WARNING_PRESERVED_PREFIX
    + ' STATUS (updated 2026-08-28, task 4610): BOTH records are now '
    'corrected. Edge fb96a8c0 was reworded in place on 2026-08-22 and '
    'self-dates. Mem0 6403e96b was subsequently '
    f'{AMENDED_SENTINEL} — its bare causal claim is now version-scoped, marked '
    'SUPERSEDED AS A GENERAL CLAIM, and carries the 2026-08-19 contrary '
    'measurement. NEITHER was deleted: the April symptom was measured and real, '
    'only its inferred cause was wrong, so both are kept as dated historical '
    'records rather than retracted. The correction had to be routed through '
    'scripts/amend_stale_resume_cwd_records.py because update_memory admits '
    'only recon-stage-* and curator-* agent ids and no dispatchable role holds '
    'one. Task 3730 was the consumer at risk.'
)


#: The two amendments, in the order they MUST be applied. d007aa46's new text
#: asserts that 6403e96b was corrected, so it is written only after that is
#: true — see the ordering guard in ``run()``.
AMEND_TARGETS: tuple[AmendTarget, ...] = (
    AmendTarget(
        memory_id='6403e96b-f1af-403a-9513-59f007ed6d39',
        expected_preimage=_STALE_PREIMAGE,
        new_content=_STALE_REPLACEMENT,
        # 'correction_basis' — the spelling d007aa46's own metadata already
        # carries — is NOT in memory_metadata.BLESSED_METADATA_KEYS, so it
        # would census as an (non-fatal) unknown_key drift violation on every
        # write. The 'x_' experimental namespace is the sanctioned home for an
        # unblessed key, so the provenance is recorded there instead.
        metadata_patch={
            'x_correction_basis': 'esc-3578-5',
            'x_corrected_by_task': '4610',
        },
    ),
    AmendTarget(
        memory_id='d007aa46-5800-455c-af3c-32d8fd8445b2',
        expected_preimage=_WARNING_PREIMAGE,
        new_content=_WARNING_REPLACEMENT,
        metadata_patch={'x_corrected_by_task': '4610'},
    ),
)


# ---------------------------------------------------------------------------
# Pure core: classify_amend_target
# ---------------------------------------------------------------------------

def classify_amend_target(
    target: AmendTarget,
    fetched: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide what a live read of *target* means. Pure function -- no I/O.

    *fetched* is a ``get_memory_by_id``-shaped envelope: ``{'found': True,
    'content': ...}`` on a hit, ``{'found': False, ...}`` on a genuine miss, or
    ``{'error', 'error_type'}`` with ``found`` ABSENT on a backend failure.

    Returns ``{'id', 'action'}`` -- mirroring
    ``tag_cgl_eta_rehome_scope.classify_rehome_record``'s decision shape --
    where ``action`` is one of:

    - ``'refuse:read_error'`` -- the read itself failed. NOT the same claim as
      not-found: ``get_memory_by_id``'s no-silent-fail contract keeps "memory
      genuinely absent" and "backend timed out" distinguishable, and unknown is
      not absent. Collapsing them would let a transient Qdrant fault be
      reported as a vanished record.
    - ``'refuse:not_found'`` -- the record is genuinely gone (consolidated
      away, reaped). Nothing to amend, and inventing it back is not this
      script's job.
    - ``'skip:already_amended'`` -- content carries :data:`AMENDED_SENTINEL`.
    - ``'amend'`` -- content still equals the pinned pre-image exactly.
    - ``'refuse:preimage_mismatch'`` -- content exists but is neither. Somebody
      else changed this record; REFUSE rather than clobber.

    The check ORDER is load-bearing. The sentinel test precedes the pre-image
    comparison because after a successful apply the live content no longer
    equals the pre-image -- it equals the replacement. Pre-image-first would
    therefore report every re-run as ``refuse:preimage_mismatch``, turning the
    idempotency property into a false tamper alarm and leaving an operator
    unable to tell a safe re-run from a genuine race.

    And the mismatch branch REFUSES rather than overwrites because an in-place
    amend is invisible to every downstream reader: silently rewriting a record
    that moved underneath us (a curator sitting, a recon Stage-1/2
    consolidation) would destroy someone else's correction with no trace. A
    loud refusal converts that race into an operator decision.
    """
    result = {'id': target.memory_id}

    if not isinstance(fetched, dict) or 'found' not in fetched:
        return {**result, 'action': 'refuse:read_error'}

    if not fetched.get('found'):
        return {**result, 'action': 'refuse:not_found'}

    content = fetched.get('content') or ''

    if AMENDED_SENTINEL in content:
        return {**result, 'action': 'skip:already_amended'}

    if content == target.expected_preimage:
        return {**result, 'action': 'amend'}

    return {**result, 'action': 'refuse:preimage_mismatch'}


# ---------------------------------------------------------------------------
# Pure core: build_amend_report
# ---------------------------------------------------------------------------

def build_amend_report(
    decisions: list[dict[str, Any]],
    applied_ids: set[str],
    dry_run: bool,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the structured JSON-serialisable amendment report.

    Mirrors ``tag_cgl_eta_rehome_scope.build_tag_report``'s shape and
    determinism -- two runs over identical inputs serialise byte-identically,
    so an operator can diff two dry-runs and see only what actually moved.

    Two things differ from the sibling sweeps' reports, both deliberately:

    * ``changes`` preserves :data:`AMEND_TARGETS` ORDER rather than sorting by
      id. Order is semantic here (it encodes the precondition chain), and
      sorting would put d007aa46 first -- reading as though the dependent write
      were attempted before the one it depends on.
    * EVERY decision gets a ``changes`` entry, including skips and refusals.
      The sibling sweeps list only the records they touched because their pools
      run to thousands; here there are exactly two records and both are
      load-bearing, so an operator must be able to see what happened to each
      without inferring it from a count.

    Args:
        decisions: :func:`classify_amend_target` results, in AMEND_TARGETS
            order.
        applied_ids: memory ids actually written (empty on a dry run).
        dry_run: True when no writes were made.
        generated_at: ISO timestamp string.

    Returns:
        ``{'dry_run', 'generated_at', 'targets', 'totals', 'changes'}``, all
        JSON-serialisable primitives (no ``default=`` hook needed).
    """
    changes: list[dict[str, Any]] = []
    for decision in decisions:
        entry = {
            'id': decision['id'],
            'action': decision['action'],
            'applied': decision['id'] in applied_ids,
        }
        # Carry any diagnostic the write path attached (error text, the
        # precondition that failed) so a refusal explains itself in the
        # artifact rather than only in the logs.
        for key in ('error', 'error_type', 'detail'):
            if key in decision:
                entry[key] = decision[key]
        changes.append(entry)

    return {
        'dry_run': dry_run,
        'generated_at': generated_at,
        'targets': len(decisions),
        'totals': {
            # 'amended' counts what was actually WRITTEN, not what was
            # amendable -- on a dry run every entry is 'amend' and none is
            # applied, and the report must not read as though two writes
            # happened.
            'amended': sum(1 for c in changes if c['applied']),
            'skipped': sum(
                1 for c in changes if c['action'].startswith('skip:')
            ),
            'refused': sum(
                1 for c in changes if c['action'].startswith('refuse:')
            ),
        },
        'changes': changes,
    }


# ---------------------------------------------------------------------------
# Live shell: run
# ---------------------------------------------------------------------------

def _normalise_fetched(record: dict[str, Any] | None) -> dict[str, Any]:
    """Map ``MemoryService.get_memory_by_id``'s return to the tool-layer shape.

    The service returns ``dict | None`` and PROPAGATES a backend read failure
    as an exception; the MCP tool above it converts those into
    ``{'found': True/False, ...}`` and ``{'error', 'error_type'}`` respectively.
    :func:`classify_amend_target` is written against the tool-layer shape --
    the documented one, and the one that keeps "absent" and "unreadable"
    distinguishable -- so the raising/None seam is normalised here, at the I/O
    boundary, rather than teaching the pure classifier about two shapes.
    """
    if record is None:
        return {'found': False}
    return {'found': True, 'content': record.get('content') or ''}


async def run(
    memory_service: Any,
    *,
    project_id: str,
    apply: bool,
) -> dict[str, Any]:
    """Corroborate both targets and, with *apply*, amend the stale ones.

    Reads each target in :data:`AMEND_TARGETS` order, classifies it via
    :func:`classify_amend_target`, and assembles the report. See the module
    docstring for the two-phase (dry-run default / ``--apply``) model.

    This batch read decides WHAT to write; it is not the corroboration a write
    acts on. Under *apply*, :func:`_apply_amendments` re-reads and re-classifies
    each target immediately before that target's own write, because this pass
    completes before any write and so cannot see a concurrent edit landing
    mid-batch.

    A read that RAISES is caught and classified ``'refuse:read_error'`` rather
    than aborting the run: with two targets, one unreadable record must still
    leave the operator with a report about the other.

    Returns the report dict (see :func:`build_amend_report`).
    """
    generated_at = datetime.now(UTC).isoformat()

    decisions: list[dict[str, Any]] = []
    for target in AMEND_TARGETS:
        try:
            record = await memory_service.get_memory_by_id(
                project_id=project_id, memory_id=target.memory_id,
            )
        except Exception as exc:  # noqa: BLE001 -- unknown is not absent
            logger.warning(
                'amend_stale_resume_cwd_records: read FAILED for %s: %r',
                target.memory_id, exc,
            )
            decisions.append({
                'id': target.memory_id,
                'action': 'refuse:read_error',
                'error': f'{type(exc).__name__}: {exc}',
            })
            continue
        decisions.append(classify_amend_target(target, _normalise_fetched(record)))

    applied_ids: set[str] = set()

    if apply:
        decisions = await _apply_amendments(
            memory_service, decisions, project_id=project_id,
            applied_ids=applied_ids,
        )

    return build_amend_report(
        decisions=decisions,
        applied_ids=applied_ids,
        dry_run=not apply,
        generated_at=generated_at,
    )


def _precondition_satisfied(decision: dict[str, Any]) -> bool:
    """Does *decision* leave the corpus in a state later targets may assert?

    True for a landed amendment and for ``'skip:already_amended'``. The skip
    counts BECAUSE the claim a later record makes -- "6403e96b was corrected"
    -- is equally true whether this run corrected it or a previous run did.
    Treating a skip as a failure would make the script unable to finish what a
    partially-failed earlier run started, which is precisely the re-run an
    operator needs after a transient backend fault.

    False for every ``'refuse:*'`` outcome, including a pre-image mismatch: if
    somebody else edited the record underneath us we do not know what it now
    says, and asserting it carries our correction would be a guess.
    """
    return decision['action'] in ('amend', 'skip:already_amended')


async def _apply_amendments(
    memory_service: Any,
    decisions: list[dict[str, Any]],
    *,
    project_id: str,
    applied_ids: set[str],
) -> list[dict[str, Any]]:
    """Write every ``'amend'`` decision, in order, behind one capability probe.

    Returns the decision list with write outcomes folded in; mutates
    *applied_ids* to hold the ids actually written.

    The capability probe runs ONCE, before the first write, and only if there
    is something to write:

    * ``once`` rather than per-write because it is a real filesystem probe and
      one refusal is the whole answer;
    * ``before`` the first write because
      ``store_mutation_preflight`` exists to turn a half-written batch into an
      up-front refusal -- ``repair_recon_citation.py`` records this exact class
      of write silently failing from a task worktree on a read-only data dir;
    * and only ``if there is something to write`` so an all-skip re-run needs
      no capability at all (pinned in step-13).

    A refusal converts every pending target to ``'refuse:store_unavailable'``
    rather than raising, so a sandboxed operator still gets the report and can
    see why nothing happened.

    Each write is preceded by a re-read of ITS OWN record, re-classified
    through :func:`classify_amend_target`. The batch corroboration in
    :func:`run` happens before any write, so it cannot see a concurrent edit
    that lands while an earlier target is being written; without the re-read
    the pre-image guard would silently clobber exactly the race it exists to
    refuse. A record that moved in that window is reported with its NEW
    classification (a mismatch refuses and breaks the chain; a sentinel means
    somebody else already landed the correction, which keeps it intact).
    """
    amendable = [d for d in decisions if d['action'] == 'amend']
    if not amendable:
        return decisions

    try:
        assert_store_mutation_allowed(
            operation='amend_stale_resume_cwd_records --apply',
        )
    except StoreMutationUnavailable as exc:
        logger.error(
            'amend_stale_resume_cwd_records: --apply NOT started (fail-closed) '
            "-- this process cannot write mem0's history directory, so an "
            'amendment would rewrite a record and then fail to journal the '
            'change. Nothing was written. Re-run from the fused-memory MCP '
            "server's environment or an unsandboxed operator shell; to obtain "
            'the report safely from anywhere, re-run without --apply. (%r)',
            exc,
        )
        return [
            {**d, 'action': 'refuse:store_unavailable', 'error': str(exc)}
            if d['action'] == 'amend' else d
            for d in decisions
        ]

    targets_by_id = {t.memory_id: t for t in AMEND_TARGETS}
    updated: list[dict[str, Any]] = []
    # The precondition chain. Every target after the first asserts, in its own
    # text, that the earlier correction landed -- so once one link fails, no
    # later one may be written. See _precondition_satisfied for what counts.
    chain_intact = True
    for decision in decisions:
        if decision['action'] != 'amend':
            # A non-amend outcome still decides the chain: a skip keeps it
            # intact (the record already carries the correction), a refusal
            # breaks it.
            chain_intact = chain_intact and _precondition_satisfied(decision)
            updated.append(decision)
            continue

        if not chain_intact:
            # Deliberately BEFORE the await: an unsatisfied precondition means
            # the write must not be attempted at all, not attempted and undone.
            updated.append({
                **decision,
                'action': 'refuse:precondition_failed',
                'detail': (
                    'an earlier target in AMEND_TARGETS was not corrected, so '
                    "this record's claim that it was would be false"
                ),
            })
            continue

        target = targets_by_id[decision['id']]

        # RE-CORROBORATE, immediately before THIS record's own write. The
        # batch read in run() happens before ANY write, so every target after
        # the first was corroborated at least one Qdrant round-trip plus one
        # full re-embed of a ~2KB replacement ago. A curator sitting or a recon
        # Stage-1/2 consolidation landing inside that window is precisely the
        # race the pre-image guard exists to catch, and the batch read cannot
        # see it. Two records, so the extra reads cost nothing.
        #
        # This NARROWS the window to a single await; it does not close it.
        # ``MemoryService.update_memory`` offers no compare-and-swap, so there
        # is no seam at which read-then-write could be made atomic -- which is
        # why the safety property is stated as a narrowed race, not none.
        try:
            fresh = await memory_service.get_memory_by_id(
                project_id=project_id, memory_id=target.memory_id,
            )
        except Exception as exc:  # noqa: BLE001 -- unknown is not absent
            logger.warning(
                'amend_stale_resume_cwd_records: pre-write re-read FAILED '
                'for %s: %r -- NOT writing',
                target.memory_id, exc,
            )
            chain_intact = False
            updated.append({
                **decision,
                'action': 'refuse:read_error',
                'error': f'{type(exc).__name__}: {exc}',
                'detail': (
                    'the pre-write re-read failed, so the pre-image could not '
                    'be re-corroborated and the write was not attempted'
                ),
            })
            continue

        recheck = classify_amend_target(target, _normalise_fetched(fresh))
        if recheck['action'] != 'amend':
            logger.warning(
                'amend_stale_resume_cwd_records: %s MOVED between '
                'corroboration and write (now %s) -- NOT writing',
                target.memory_id, recheck['action'],
            )
            # A skip keeps the chain intact (the record already carries the
            # correction, whoever landed it); any refusal breaks it.
            chain_intact = _precondition_satisfied(recheck)
            updated.append({
                **decision,
                **recheck,
                'detail': (
                    'this record changed between the corroborating read and '
                    'its write; the pre-write re-read reclassified it, so '
                    'nothing was overwritten'
                ),
            })
            continue

        try:
            response = await memory_service.update_memory(
                memory_id=target.memory_id,
                project_id=project_id,
                content=target.new_content,
                metadata_patch=target.metadata_patch or None,
                reason=WRITE_REASON,
                # Attribution has TWO consumers and they read different
                # fields: the write journal / storm alarm read ``_source``,
                # while the metadata-vocabulary check this patch triggers
                # (emit_schema_warnings, UnknownKeyStormDetector.record,
                # file_unknown_key_storm_escalation) is keyed by ``agent_id``.
                # Omitting the latter leaves every census line and storm
                # bucket from this run attributed to a null agent, so both are
                # set to the same value and the two views agree.
                agent_id=WRITE_SOURCE,
                _source=WRITE_SOURCE,
            )
        except Exception as exc:  # noqa: BLE001 -- report, never propagate
            logger.warning(
                'amend_stale_resume_cwd_records: write RAISED for %s: %r',
                target.memory_id, exc,
            )
            chain_intact = False
            updated.append({
                **decision,
                'action': 'refuse:write_error',
                'error': f'{type(exc).__name__}: {exc}',
            })
            continue

        # `update_memory` reports a refusal by RETURNING a structured envelope
        # rather than raising, so a caller that only guarded against exceptions
        # would score a rejected write as an amendment.
        if isinstance(response, dict) and response.get('error_type'):
            logger.warning(
                'amend_stale_resume_cwd_records: write REFUSED for %s: %s',
                target.memory_id, response.get('error'),
            )
            chain_intact = False
            updated.append({
                **decision,
                'action': 'refuse:write_failed',
                'error': response.get('error'),
                'error_type': response.get('error_type'),
            })
            continue

        applied_ids.add(target.memory_id)
        updated.append(decision)

    return updated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Extracted from :func:`main` so tests can assert flag defaults without
    invoking the live entry point (mirrors
    ``tag_cgl_eta_rehome_scope.build_parser``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help='Commit both amendments (default: dry-run, report only)',
    )
    parser.add_argument(
        '--project-id', dest='project_id', default='dark_factory',
        help='Project whose Mem0 collection holds the two records '
             '(default: dark_factory -- both target ids are dark_factory records)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def resolve_exit_code(report: dict[str, Any]) -> int:
    """0 on a clean run, 1 when anything did not land as planned.

    Graded off :data:`ERROR_OUTCOMES` -- the SAME set the report counts as
    ``refused`` -- so the exit code and the artifact can never disagree about
    whether a run was clean.

    A dry run, a fully-applied run and an all-skipped run are all clean: none
    of them leaves the corpus in a state the plan did not intend. Only a
    refusal is non-zero, which is what an operator gates on.
    """
    return 1 if any(
        change['action'] in ERROR_OUTCOMES
        for change in report.get('changes', ())
    ) else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, build a live MemoryService, run, report."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    args = build_parser().parse_args(argv)

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> int:
        # Deferred so importing this module for tests never touches config or
        # a backend (mirrors tag_cgl_eta_rehome_scope.main).
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        memory = MemoryService(FusedMemoryConfig())
        await memory.initialize()
        try:
            report = await run(
                memory, project_id=args.project_id, apply=args.apply,
            )
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

        print(json.dumps(report, indent=2))
        print(
            f"amend_stale_resume_cwd_records "
            f"({'DRY RUN' if report['dry_run'] else 'APPLIED'}) at "
            f"{report['generated_at']}: targets={report['targets']} "
            f"amended={report['totals']['amended']} "
            f"skipped={report['totals']['skipped']} "
            f"refused={report['totals']['refused']}",
            file=sys.stderr,
        )
        return resolve_exit_code(report)

    return asyncio.run(_run_live())


if __name__ == '__main__':
    sys.exit(main())

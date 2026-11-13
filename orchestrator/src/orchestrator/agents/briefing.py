"""Prompt assembly — builds full prompts for each agent invocation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from escalation.models import Escalation
from shared.briefing_queries import BriefingScope
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, parse_item
from shared.task_text_fingerprint import TaskTextDrift

from orchestrator.agents.harness_guidance import WAIT_PATTERN_REMINDER
from orchestrator.agents.landed_sha_searches import LandedShaSearches, already_merged_step
from orchestrator.agents.memory_recall import MemoryRecall
from orchestrator.agents.review_briefing import (
    ReviewSubject,
    amendment_scope_section,
    render_context_complete_review,
    review_action_section,
)
from orchestrator.config import OrchestratorConfig
from orchestrator.implementer_brief_variant import ImplementerBriefVariant, is_one_step
from orchestrator.plan_descope import DESCOPE_DISPOSITIONS, DoneStepIntegrityVerdict
from orchestrator.scope_drift import ScopeDriftReport

logger = logging.getLogger(__name__)

IMPLEMENTER_ACTIONS: Mapping[ImplementerBriefVariant, str] = MappingProxyType({
    ImplementerBriefVariant.WHOLE_PLAN: (
        'Execute the next pending steps in TDD order. Commit after each step. '
        'Call `mark_step_done(step_id, commit_sha)` to record progress. '
        'Stop at a logical boundary.'
    ),
    ImplementerBriefVariant.ONE_STEP: (
        'One-step mode: complete exactly ONE plan step this session — the next '
        'pending item in `.task/plan.json`, prerequisites first, then steps. Do it '
        'in TDD order, commit, call `mark_step_done(step_id, commit_sha)`, then '
        'stop, even if more steps are pending; the orchestrator re-invokes you for '
        'the next step.'
    ),
})
"""The implementer brief's Action paragraph per ``config.implementer_brief_variant``."""


def _implementer_action(variant: ImplementerBriefVariant) -> str:
    """The Action paragraph *variant* selects, by the :func:`is_one_step` rule the cap and the bounds also use."""
    return IMPLEMENTER_ACTIONS[
        ImplementerBriefVariant.ONE_STEP if is_one_step(variant) else ImplementerBriefVariant.WHOLE_PLAN
    ]


DIFF_ONLY_DIFF_CAP_CHARS = 50_000
"""Where the incumbent reviewer briefing cuts its diff; ``diff_cap_chars=0`` lifts the cut."""


OPEN_INFO_NOTES_HEADING = '# Implementer notes (open info escalations)'
"""The heading of the reviewer's notes block; the reviewer contract in roles.py names it."""

OPEN_INFO_NOTES_LIMIT = 20
"""Max notes one reviewer briefing lists, the most recent kept; the rest stay pending for the exit router."""


def _open_info_notes_section(
    notes: Sequence[Escalation], *, reviewer_type: str, detail_chars: int,
) -> str:
    """The reviewer's notes block (plans/info-l0-disposition-router-prd.md D2).

    It closes on the verdict call with ``note_dispositions``, so the last call
    the reviewer is shown names the field.
    """
    by_age = sorted(notes, key=lambda note: note.timestamp or '')
    listed = by_age[-OPEN_INFO_NOTES_LIMIT:]
    entries = '\n'.join(_open_info_note_entry(note, detail_chars) for note in listed)
    unlisted = len(by_age) - len(listed)
    overflow = (
        f'{unlisted} older note(s) are not listed; they stay pending for the exit router.\n\n'
        if unlisted else ''
    )
    return f"""\
{OPEN_INFO_NOTES_HEADING}

These info notes were filed against this task while it was worked.

{entries}
{overflow}Disposition every id above as `addressed`, `no_action` or `work`, as your
contract defines them, in the `note_dispositions` of your verdict call:
`submit_review_verdict(reviewer="{reviewer_type}", verdict=..., issues=..., summary=..., note_dispositions=[...])`
"""


def _open_info_note_entry(note: Escalation, detail_chars: int) -> str:
    full_detail = note.detail or ''
    detail = full_detail[:detail_chars]
    quoted = '\n'.join(f'> {line}' for line in detail.splitlines()) or '> (none)'
    if len(full_detail) > detail_chars:
        quoted += f'\n[detail truncated at {detail_chars} characters]'
    return f"""\
## {note.id}

- filed by: {_one_line(note.agent_role)}
- filed at: {note.timestamp}
- summary: {_one_line(note.summary)}
- suggested action: {_one_line(note.suggested_action) or '(none)'}
- detail:

{quoted}
"""


def _one_line(text: str | None) -> str:
    """*text* with its line breaks collapsed, so a field cannot start a heading of its own; null reads as empty."""
    return ' '.join((text or '').split())

COMMIT_BULLET_LIMIT = 40
"""Max commit bullets rendered in one briefing section (task 3033 amendment).

Bounds the architect's already-committed-work section, whose source
(``TaskWorkflow._detect_committed_branch_work``) returns ALL of
``base_commit..HEAD`` — unbounded on a long-lived branch that accumulated WIP
safety-commits across many requeues/rebases plus normal step commits. Without a
cap the section grows with branch length and its ``git show <sha>`` protocol
drives one architect tool call per commit. Its sibling
``_detect_tip_wip_commits`` needs no cap: it is naturally bounded by the
contiguous WIP run at HEAD.
"""


def _format_commit_bullets(commits: list[dict], limit: int | None = None) -> str:
    """Render ``[{'sha': ..., 'subject': ...}]`` as HEAD-first markdown bullets.

    Single source of the 12-char abbreviation convention, shared by
    :meth:`BriefingAssembler.build_architect_prompt`'s committed-work section
    and :meth:`BriefingAssembler.build_implementer_prompt`'s WIP section — both
    consume the same shape from the two sibling detectors, so the format must
    not drift between them.

    ``limit`` caps the rendered bullets (HEAD-first, i.e. most recent) and
    appends an explicit "…and N more" line: a truncation must be VISIBLE in the
    prompt, never silent, so the agent knows the list is partial and can widen
    it itself. ``None`` (the default) renders everything.
    """
    shown = commits if limit is None else commits[:limit]
    lines = [f"- `{c['sha'][:12]}` — {c['subject']}" for c in shown]
    hidden = len(commits) - len(shown)
    if hidden > 0:
        lines.append(
            f'- …and {hidden} more commit(s) on this branch (not shown — run '
            f'`git log --oneline` to see the rest)'
        )
    return '\n'.join(lines)

DELIVERED_CHECK_BULLET_LIMIT = 20
"""Max delivered-check bullets rendered in one briefing section (task 5359).

Bounds the capability-gate section, whose source — a task's author-supplied
``metadata.delivered_checks`` — carries no length ceiling of its own, and whose
per-entry cost is unbounded too: a ``grep`` descriptor's ``pattern`` is an
arbitrary regex. Without a cap the section grows with the author's list in
EVERY architect/simple_task dispatch of that task. As with
``COMMIT_BULLET_LIMIT``, the truncation is rendered VISIBLY so the agent knows
the list is partial and can read the rest from the task record itself.
"""

_TEXT_DRIFT_NOTICES = {
    TaskTextDrift.CURRENT: '',
    TaskTextDrift.AMENDED: (
        '## The Task Text Has Changed\n\n'
        'The description or details under # Task have been amended since this '
        'plan was authored. Re-read them and change the plan to cover the '
        'amended scope, then call confirm_plan(); if it already covers it, '
        'confirm it unchanged.'
    ),
    TaskTextDrift.UNSTAMPED: (
        '## The Task Text May Have Changed\n\n'
        'This plan carries no record of the task text it was authored against, '
        'so the description or details under # Task may have been amended '
        'since. Re-read them and change the plan if its scope has moved, then '
        'call confirm_plan(); if it already covers it, confirm it unchanged.'
    ),
}



def _integrity_refusal_section(verdict: DoneStepIntegrityVerdict | None) -> str:
    """The refused done-step changes of a prior revalidation pass, or ''."""
    if verdict is None:
        return ''
    lines = [
        f'- `{step_id}` @ `{commit}` was dropped without a descope record'
        for step_id, commit in verdict.undisposed_drops
    ]
    lines += [
        f'- `{dup.commit}` is now the commit of done steps '
        + ', '.join(f'`{step_id}`' for step_id in dup.step_ids)
        for dup in verdict.introduced_duplicates
    ]
    lines += [f'- invalid descope record: {message}' for message in verdict.invalid_descope_records]
    return (
        '## Your Revised Plan Was Refused\n\n'
        + '\n'.join(lines)
        + '\n\nRestore each dropped step (`add_plan_step`, then `mark_step_committed` '
        'when its commit is on the branch) or record it with `record_descoped_step`; '
        'mark a step done only at its own commit. Then call `confirm_plan()`.\n\n'
    )


_DELIVERED_CHECK_FIELDS_BY_KIND = {
    'grep': ('pattern', 'expect', 'paths'),
    'script': ('script', 'args', 'timeout_secs'),
}
"""Descriptor fields to render per ``kind``, mirroring
``shared.capability_manifest.DeliveredCheckMeta``'s mutually exclusive
grep/script slices. Single source for BOTH the recognised-kind bullet and the
unrecognised-kind fallback (which renders whichever of the union is present),
so the two paths cannot drift. ``test_briefing.py`` pins the union against the
live model's ``model_fields``, so a field added to ``_CheckFieldsBase`` and not
added here fails there rather than vanishing silently from every prompt.
"""

_DELIVERED_CHECK_LIST_FIELDS = {'paths': 'whole tree', 'args': 'none'}
"""List-valued descriptor fields, mapped to what an EMPTY list means to the
runner — ``paths: []`` greps the whole tree, ``args: []`` passes none. Rendering
the meaning rather than a bare ``[]`` keeps the bullet readable for an agent.
"""


def _format_delivered_checks(checks: object) -> str:
    """Render a task's ``metadata.delivered_checks`` as an agent-readable block.

    Returns '' for a falsy input so ``BriefingAssembler._format_task`` can omit
    the whole section on the common path (most tasks declare no capability).

    One bullet per descriptor, field-labelled so the rendered text names every
    field of :class:`shared.capability_manifest.DeliveredCheckMeta` — a field
    added to that model must not vanish silently from the prompt, which
    ``test_briefing.py::TestFormatDeliveredChecks`` pins against the live
    schema. Output is capped at ``DELIVERED_CHECK_BULLET_LIMIT`` with a visible
    "…and N more" line, the never-silent-truncation convention
    :func:`_format_commit_bullets` documents.

    ``metadata`` is persisted, untyped data and ``_format_task`` runs on every
    dispatch, so this fails OPEN the way
    ``BriefingAssembler._format_prior_proposal`` does for ``files_referenced``:
    a bare dict is accepted as a one-element list (a shape ``parse_metadata``
    accepts, so a real task can carry one), ``paths``/``args`` elements are
    ``str()``-coerced, and an entry whose ``kind`` is missing or unrecognised
    degrades to a VISIBLE partial bullet rather than being dropped — dropping it
    would reproduce this section's own reason for existing one level down, since
    an unseen check still blocks mark-done. An element that is not a dict at all
    cannot be given a bullet, so it is COUNTED visibly instead and the section is
    still emitted: ``parse_metadata`` preserves a wrong-shaped slice with only a
    ``SchemaWarning``, ``run_delivered_check`` maps such an entry to ERRORED, and
    the gate turns that into a withheld mark-done — so returning '' would block a
    task from ever stamping done while showing its agent no gate at all. '' is
    reserved for a value that is neither a dict nor a list/tuple, i.e. genuinely
    nothing to show.

    The gate's own arming is CONDITIONAL (``delivered_checks.enabled``,
    green-tier hot-reloadable), and this renderer is pure with no view of
    config — so the block states the mechanical claim conditionally rather than
    asserting a gate that may be disarmed at dispatch time. The directive is
    unconditional either way: the descriptors are the acceptance contract the
    task was filed under, armed gate or not.

    The block carries its OWN reading directive rather than relying on the six
    consuming prompt bodies to explain it — the same co-location precedent
    ``_format_prior_proposal`` follows with its verify-before-reuse line. The
    directive is load-bearing: ``docs/task-authoring.md`` §3.3 records that a
    symbol-name grep "is satisfiable by prose — a comment, a docstring, or a
    variable named after the thing", so handing an agent the literal pattern is
    a teach-to-the-test hazard unless the same text tells it that matching the
    pattern without delivering the behaviour is a defect.
    """
    if not checks:
        return ''
    if isinstance(checks, dict):
        checks = [checks]
    if not isinstance(checks, (list, tuple)):
        return ''

    entries = [check for check in checks if isinstance(check, dict)]
    dropped = len(checks) - len(entries)

    shown = entries[:DELIVERED_CHECK_BULLET_LIMIT]
    lines = [_delivered_check_bullet(check) for check in shown]
    hidden = len(entries) - len(shown)
    if hidden > 0:
        lines.append(
            f'- …and {hidden} more declared check(s) (not shown — read the full '
            f'list from this task\'s `metadata.delivered_checks`)'
        )
    if dropped > 0:
        noun = 'entry' if dropped == 1 else 'entries'
        lines.append(
            f'- ⚠ {dropped} malformed {noun} on this task\'s '
            f'`metadata.delivered_checks` could not be rendered here — still '
            f'evaluated by the gate, so read the raw list from the task record.'
        )
    bullets = '\n'.join(lines)

    return f"""\
## Declared Capability Gate (metadata.delivered_checks)

When the capability gate is enabled (the `delivered_checks.enabled` config leaf,
which an operator can hot-reload), this task's OWN mark-done is gated on the
capability checks below — not only its dependents'
(`orchestrator/src/orchestrator/delivered_checks.py::gate_mark_done_on_delivered_checks`).

{bullets}

After this task lands, a `grep` check is re-run with `git grep -E` against the
COMMITTED `main` tree and a `script` check against the working checkout. Under
that gate a FAILED check blocks mark-done; on the dependency path it also fires
an escalation routed straight at a human. Treat the checks as binding whether or
not the gate is armed on this fleet: they are the acceptance contract this task
was filed under.

Deliver the BEHAVIOUR each capability names. Satisfying a pattern with a comment, a
docstring, or a variable named after the thing is a defect, not a pass. If a
descriptor cannot be satisfied by the work this task should do, escalate it
(`escalate_blocker(category='design_concern')`) rather than writing the string to
make the check match. The descriptor contract lives in `docs/task-authoring.md` §3.3
— read it there rather than inferring it from these bullets.
"""


def _delivered_check_bullet(check: dict) -> str:
    """Render one delivered-check descriptor as a single field-labelled bullet.

    An unrecognised or missing ``kind`` renders whichever descriptor fields are
    present, flagged UNRECOGNISED so the malformation is visible to whoever
    reads the prompt and the agent can ask rather than guess.
    """
    kind = check.get('kind')
    fields = [f'kind: `{kind}`']
    if kind in _DELIVERED_CHECK_FIELDS_BY_KIND:
        fields += [
            _delivered_check_field(check, name)
            for name in _DELIVERED_CHECK_FIELDS_BY_KIND[kind]
        ]
    else:
        fields.append('UNRECOGNISED descriptor, shown as-is')
        fields += [
            _delivered_check_field(check, name)
            for group in _DELIVERED_CHECK_FIELDS_BY_KIND.values()
            for name in group
            if name in check
        ]
    return f'- name: `{check.get("name")}` — ' + ', '.join(fields)


def _delivered_check_field(check: dict, name: str) -> str:
    """Render one descriptor field as ``name: value``, coercing defensively."""
    value = check.get(name)
    if name not in _DELIVERED_CHECK_LIST_FIELDS:
        return f'{name}: `{value}`'
    if value is None:
        items: list = []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return f'{name}: {", ".join(str(v) for v in items) or _DELIVERED_CHECK_LIST_FIELDS[name]}'


_FOLLOWUP_ITEMS_HEADING = f'## Follow-up Items (metadata.{FOLLOWUP_ITEMS_METADATA_KEY})'


def _format_followup_items(raw: object) -> str:
    """Render the task's follow-up items numbered (plans/follow-up-filing-shape-prd.md decision 1).

    Fails open like :func:`_format_delivered_checks`: '' when absent, malformed entries counted visibly.
    """
    if not raw:
        return ''
    if not isinstance(raw, list):
        return (
            f'{_FOLLOWUP_ITEMS_HEADING}\n\n'
            f'⚠ `metadata.{FOLLOWUP_ITEMS_METADATA_KEY}` is malformed (a {type(raw).__name__}, not a '
            'list), so its observations cannot be shown here; read them from the task record.'
        )
    items = [item for item in map(parse_item, raw) if item is not None]
    lines = [
        f'{number}. {item.summary} — files: {", ".join(item.files)} (observed in task {item.origin_task})'
        for number, item in enumerate(items, start=1)
    ]
    malformed = len(raw) - len(items)
    if malformed:
        noun = 'entry' if malformed == 1 else 'entries'
        lines.append(
            f'⚠ {malformed} malformed {noun} on `metadata.{FOLLOWUP_ITEMS_METADATA_KEY}` could not be '
            'rendered here; read the full list from the task record.'
        )
    numbered = '\n'.join(lines)
    return (
        f'{_FOLLOWUP_ITEMS_HEADING}\n\n'
        'Each numbered item is one observation this task was filed to address; the plan must '
        f'cover every one.\n\n{numbered}'
    )


def _caller_agent_id(task_id: str | None, role: str) -> str:
    """The agent-id a dispatch is known by, in its prompt and in the journal.

    One home for the format (INV-5): :meth:`BriefingAssembler._agent_identity`
    renders the prompt's ``## Agent Identity`` block from this, and every
    memory search declares the same string as ``caller_agent_id`` (D8). A
    second copy at the search call site would let the identity an agent is
    TOLD it has drift from the one the journal records it asking under.
    """
    return f'claude-task-{task_id}-{role}' if task_id else f'claude-{role}'


def _plan_scope(plan: dict, task_id: str | None) -> BriefingScope:
    """Scope a post-planning role from the plan it was handed.

    An explicitly-passed *task_id* wins over the plan's own copy, exactly as
    the ``task_id or plan.get('task_id')`` these call sites used to spell did:
    the workflow holds the authoritative id, and a plan written in an earlier
    session can carry a stale one.
    """
    scope = BriefingScope.from_plan(plan)
    return replace(scope, task_id=str(task_id)) if task_id else scope


@dataclass(frozen=True)
class PriorAttemptCause:
    """Why one earlier verify attempt failed: its failure category and cause text."""

    category: str
    cause: str


@dataclass(frozen=True)
class VerifyAttemptContext:
    """Where a debugger dispatch sits in its verify/debug loop.

    ``attempt`` is the 1-based number of the verify attempt that just failed.
    ``prior_causes`` holds one entry per earlier failed attempt in the same
    loop, oldest first.
    """

    attempt: int
    max_attempts: int
    prior_causes: tuple[PriorAttemptCause, ...] = ()


def _format_prior_cause(attempt: int, prior: PriorAttemptCause) -> str:
    cause = ' '.join(prior.cause.split())
    return f'- attempt {attempt}: {prior.category or "uncategorized"}: {cause}'


def _format_verify_attempt(ctx: VerifyAttemptContext) -> str:
    lines = [
        f'Verify attempt {ctx.attempt} of {ctx.max_attempts} failed; '
        'the failures below are from that run.',
    ]
    if not ctx.prior_causes:
        lines.append('This is the first failed attempt in this verify/debug loop.')
    else:
        lines.append(
            'Earlier attempts in this loop also failed, and a debugger pass '
            'followed each one:'
        )
        lines.extend(
            _format_prior_cause(i, prior) for i, prior in enumerate(ctx.prior_causes, start=1)
        )
        lines.append('')
        lines.append(
            'If the current failure matches one of these, the earlier fix did '
            'not work — find a different root cause instead of repeating it.'
        )
    return '# Verify Attempt\n\n' + '\n'.join(lines) + '\n\n'


def _format_scope_drift(report: ScopeDriftReport | None) -> str:
    """The briefing block reporting what the last writer boundary did with scope requests.

    Empty for None or a report with nothing acquired or contested. The single
    home of this briefing line; task 4371 reuses it.
    """
    if report is None or not (report.acquired or report.contested):
        return ''
    lines = [
        '## Scope Requests',
        '',
        'The orchestrator applied the files reported with `report_missing_plan_file`:',
        '',
    ]
    lines.extend(
        f'- `{path}`: granted. It is now in plan.json `files`, and this task holds its lock.'
        for path in report.acquired
    )
    lines.extend(
        f'- `{path}`: contested, held by task {holder}.' for path, holder in report.contested
    )
    if report.contested:
        lines.extend([
            '',
            'Do not edit a contested file. Revert any edit already made to it, or call '
            "`escalate_blocker(category='scope_violation')` naming the file, the "
            'holder and the design reason it must change.',
        ])
    return '\n'.join(lines) + '\n'


@dataclass
class CompletionJudgeVerdict:
    """Structured verdict returned by the completion judge agent.

    Distinct from ``evals.judge.JudgeVerdict`` (the Elo pairwise comparison
    judge) — this verdict exits the implementer loop early when the judge
    decides the substantive work is complete, regardless of plan.json
    bookkeeping state.
    """

    complete: bool
    reasoning: str
    uncovered_plan_steps: list[str]
    substantive_work: bool


COMPLETION_JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'complete': {'type': 'boolean'},
        'reasoning': {'type': 'string'},
        'uncovered_plan_steps': {
            'type': 'array', 'items': {'type': 'string'},
        },
        'substantive_work': {'type': 'boolean'},
    },
    'required': ['complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'],
    'additionalProperties': False,
}


class BriefingAssembler:
    """Builds prompts for agent invocations."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.project_id = config.fused_memory.project_id
        self._memory_recall = MemoryRecall(config.fused_memory.url, self.project_id)

    def _agent_identity(self, task_id: str | None, role: str) -> str:
        agent_id = _caller_agent_id(task_id, role)
        return (
            f'## Agent Identity\n\n'
            f'- **agent_id:** `{agent_id}`\n'
            f'- **project_id:** `{self.project_id}`\n'
        )

    async def _task_preamble(
        self,
        task: dict,
        role: str,
        context: str | None,
        *,
        include_files: bool = True,
    ) -> str:
        """The memory, identity and ``# Task`` header a task-scoped briefing opens with."""
        if context is None:
            context = await self._get_memory_context(BriefingScope.from_task(task), role)
        identity = self._agent_identity(task.get('id'), role)
        task_block = self._format_task(task, include_files=include_files)
        return f'{context}\n\n{identity}\n\n# Task\n\n{task_block}'

    async def build_architect_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
        *,
        include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
    ) -> str:
        """Build prompt for the architect agent.

        Args:
            include_prior_proposals: When True, surface the task's most
                recent ``dry_run_proposals`` entry (if any) as a prior
                block-time investigation. Defaults to False so a truly-fresh
                first dispatch stays proposal-free (C-A1 anti-anchoring) —
                only the re-plan path (an existing plan fell through to
                architect) should pass True.
            committed_work: HEAD-first ``[{'sha': ..., 'subject': ...}]`` for
                every commit this branch already carries beyond its base, as
                produced by ``TaskWorkflow._detect_committed_branch_work``
                (task 3033 / PRD §A1). When truthy, renders the
                already-committed-work section that teaches the architect to
                pre-satisfy already-implemented steps via
                ``mark_step_committed`` instead of leaving them pending. When
                None or empty — the truly-fresh first dispatch — the prompt is
                byte-identical to the pre-γ baseline, so nothing is added on
                the common path.
        """
        preamble = await self._task_preamble(
            task, 'architect', context, include_files=False,
        )

        prior_proposal_section = ''
        if include_prior_proposals:
            prior_proposal_section = self._format_prior_proposal(task)

        # Mirrors build_implementer_prompt's wip_section structurally (same
        # bullet format, same corroborate-then-attribute numbered protocol) so
        # both briefings teach one consistent habit; the substantive differences
        # are the tool named (mark_step_committed, an authoring-time authority,
        # vs mark_step_done) and the explicit VERIFY-is-the-gate warning.
        committed_section = ''
        if committed_work:
            commits_list = _format_commit_bullets(
                committed_work, limit=COMMIT_BULLET_LIMIT,
            )
            committed_section = f"""
## Already-Committed Work On This Branch — Pre-Satisfy, Don't Re-Plan-As-Pending

This branch already carries the commit(s) below beyond its base. A prior
dispatch may have implemented some (or all) of this task before the plan was
lost, so part of the work you are about to plan may already be done, committed,
and green on this very branch.

{commits_list}

Protocol:

1. Run `git show <sha>` for each commit above to see exactly what it contains.
2. Author the plan's steps as normal — do NOT drop or merge steps just because
   the work exists. The full TDD structure and its provenance must be preserved.
3. For every step the committed work ALREADY satisfies — confirmed first-hand by
   running that step's tests on this branch and seeing them pass — call
   `mark_step_committed(step_id, <sha>)` with the commit that carries it,
   instead of leaving the step pending. That marks the step done and tags its
   description `[COMMITTED <sha>]` so the provenance is durable.
4. Leave genuinely-unsatisfied steps pending, so the implementer does only the
   remaining work.
5. If EVERY step is pre-satisfied, the EXECUTE loop is skipped entirely and the
   branch flows PLAN → VERIFY → REVIEW → MERGE with zero implementer turns.

VERIFY is the semantic gate, not `mark_step_committed`. A falsely pre-satisfied
step surfaces as a VERIFY failure and BLOCKS the task — so never pre-satisfy a
step you have not actually seen pass on this branch. When in doubt, leave it
pending: an unnecessary implementer turn is cheap, a false green is not.
"""

        return f"""\
{preamble}

{prior_proposal_section}{committed_section}
# Action

1. Explore the codebase thoroughly — read relevant files, understand existing patterns and utilities.
2. Produce a TDD implementation plan using the plan-tools MCP tools:
   a. Call `create_plan(task_id, title, analysis, files)` with your analysis.
   b. Call `add_prerequisite(prereq_id, description)` for any setup work needed before TDD steps.
   c. Call `add_plan_step(step_id, step_type, description)` for each TDD step, in order. Alternate test/impl.
   d. Call `add_design_decision(decision, rationale)` for non-obvious choices.
   e. Call `add_reuse_item(what, where, how)` for existing code/patterns being reused.
   f. Call `confirm_plan()` as your FINAL action, once every step is added, to mark the plan complete. Without it the plan is treated as incomplete and will not advance.
3. List ALL files (or directory paths) you expect to create or modify in the `files` parameter of `create_plan` — this drives concurrency locks and the phantom-done gate, so be exhaustive and precise.
"""

    async def build_revalidation_prompt(
        self,
        task: dict,
        existing_plan: dict,
        changed_files: list[str],
        worktree: Path | None = None,
        context: str | None = None,
        *,
        text_drift: TaskTextDrift | None = None,
        integrity_refusal: DoneStepIntegrityVerdict | None = None,
    ) -> str:
        """Build prompt for the architect to revalidate a plan after blast-radius requeue.

        The task was planned in a prior session but requeued because module
        locks were contended.  Main has since advanced (the contending task
        merged).  The architect reviews the existing plan against the changes
        and either confirms, updates, or recreates it.
        """
        preamble = await self._task_preamble(task, 'architect', context)
        prior_proposal_section = self._format_prior_proposal(task)

        plan_files = set(existing_plan.get('files', []))
        overlapping = [f for f in changed_files if f in plan_files]
        non_overlapping = [f for f in changed_files if f not in plan_files]

        overlap_section = ''
        if overlapping:
            overlap_list = '\n'.join(f'- `{f}`' for f in overlapping)
            overlap_section = (
                f'### Overlapping with your plan (REVIEW THESE):\n{overlap_list}'
            )

        other_section = ''
        if non_overlapping:
            other_list = '\n'.join(f'- `{f}`' for f in non_overlapping[:30])
            suffix = ''
            if len(non_overlapping) > 30:
                suffix = f'\n- ... and {len(non_overlapping) - 30} more'
            other_section = (
                f'### Other changes on main:\n{other_list}{suffix}'
            )

        if not changed_files:
            files_section = '_No files changed on main since your plan was created._'
        else:
            files_section = f'{overlap_section}\n\n{other_section}'.strip()

        plan_json = json.dumps(existing_plan, indent=2)
        notice = _TEXT_DRIFT_NOTICES[text_drift] if text_drift is not None else ''
        text_drift_section = f'{notice}\n\n' if notice else ''
        refusal_section = _integrity_refusal_section(integrity_refusal)
        dispositions = ', '.join(f'`{d}`' for d in sorted(DESCOPE_DISPOSITIONS))

        return f"""\
{preamble}

{prior_proposal_section}
# Plan Revalidation

You created a plan for this task in a prior session, but the task was requeued
because another task held locks on modules you need. That task has since merged
to main, so the files it touched may have changed.

Your job: review your previous plan against the current state of the codebase
and either confirm it, update it, or recreate it from scratch.

{text_drift_section}## Your Previous Plan

```json
{plan_json}
```

## Files Changed on Main Since Your Plan

{files_section}

{refusal_section}## Action

1. Read the overlapping files (if any) to understand what changed.
2. Decide whether your plan is still valid.
3. Take ONE of these actions:
   a. **Plan still valid**: call `confirm_plan()`.
   b. **Plan needs updates**: use `update_plan_metadata(files=...)`,
      `remove_plan_step(step_id)`, `replace_plan_step(step_id, step_type, description)`,
      and/or `add_plan_step(step_id, step_type, description)` to adjust the plan in place.
   c. **Plan is invalid**: call `create_plan(...)` to start fresh, then add steps as usual.
4. Ensure the `files` list is exhaustive — it drives concurrency locks.
5. Keep every step with status "done" with its commit. If the revised plan
   genuinely drops one (e.g. `create_plan` replaced the steps), record it with
   `record_descoped_step(step_id, commit, disposition, authority, note)`:
   disposition is one of {dispositions}; authority is an escalation id or a
   human ratification marker. A revalidation that drops a done step without
   that record, or newly marks a step done at another done step's commit, is
   refused.
6. Once the plan is valid, after (a), (b) or (c) alike, call `confirm_plan()`
   as your FINAL action. Editing the step set (b) or calling `create_plan` (c)
   clears the plan's confirmation, so a revalidation that stops without it
   leaves an unconfirmed plan.
"""

    async def build_plan_completion_prompt(
        self,
        task: dict,
        partial_plan: dict,
        worktree: Path | None = None,  # noqa: ARG002 — interface symmetry
        context: str | None = None,
    ) -> str:
        """Build prompt to finish a partial plan left by an interrupted session.

        A prior planning session wrote some steps but never called
        ``confirm_plan`` (it ran out of budget/turns, or crashed). Rather than
        discard that work and re-plan from scratch, hand the partial plan back
        to the architect: verify the existing steps against current main, add
        whatever is missing, then finalize. The architect may also discard the
        partial entirely (``create_plan`` overwrites it) if it reflects a
        flawed approach.
        """
        preamble = await self._task_preamble(task, 'architect', context)

        existing_steps = [
            s for s in partial_plan.get('steps', []) if isinstance(s, dict)
        ]
        plan_json = json.dumps(partial_plan, indent=2)

        return f"""\
{preamble}

# Plan Completion — Finish an Interrupted Plan

A previous planning session for this task was cut short (it ran out of budget
or turns) before it finalized the plan. It left a PARTIAL plan on disk with
{len(existing_steps)} step(s) already recorded. Your job is to finish it — not
to start over from nothing.

## Partial Plan (already on disk)

```json
{plan_json}
```

## Action

1. Read the relevant files to re-ground yourself in the task and verify the
   existing steps are still correct against the CURRENT state of the codebase
   (main may have advanced since the partial was written).
2. Then take ONE of these paths:
   a. **Partial is sound, just unfinished** — keep the existing steps and add
      the remaining ones with `add_plan_step(...)` (continue the existing TDD
      ordering and step-id sequence), adjusting `files` via
      `update_plan_metadata(files=...)` if the scope grew. Fix individual steps
      with `replace_plan_step` / `remove_plan_step` as needed.
   b. **Partial reflects a flawed approach** — call `create_plan(...)` to
      replace it wholesale, then add steps as usual.
   c. **The task should not be planned at all** — use the rejection exits
      (`report_blocking_dependency` / `report_task_already_done` /
      `report_ready_to_merge` / `report_unactionable_task`) exactly as in
      fresh planning.

   If any existing or newly-added step is already satisfied by a commit this
   branch carries — confirmed by running that commit's tests and seeing them
   pass — call `mark_step_committed(step_id, <sha>)` for it instead of
   leaving it `pending`; do NOT drop or merge steps just because the work
   exists.
3. If you produced a plan (paths a or b), call `confirm_plan()` as your FINAL
   action to mark it complete. Without it the plan stays incomplete and will
   not advance.
4. Keep the `files` list exhaustive — it drives concurrency locks. Do NOT
   remove or replace steps with status "done".
"""

    async def build_plan_tightening_prompt(
        self,
        task: dict,
        plan: dict,
        not_touched: list[str],
        worktree: Path | None = None,  # noqa: ARG002 — interface symmetry
        context: str | None = None,
    ) -> str:
        """Architect narrowing pass after the pre-merge plan-files gate.

        The merge gate flagged ``not_touched``: plan-declared files that
        no commit on the branch actually touched.  Give the architect
        ONE bounded chance to reconcile the plan with branch reality.
        Option (c), ``drop_plan_file``, gives an over-declared file a
        truthful exit from the "drop = falsify provenance, confirm = mislabel
        complete work" dilemma named at
        ``orchestrator/src/orchestrator/merge_lane/gates.py::CROSS_REPO_DELIVERABLE_REASON_PREFIX``
        and ``orchestrator/src/orchestrator/merge_lane/gates.py::ALREADY_LANDED_REASON_PREFIX``.
        The exit reports an architect may file instead of narrowing are
        consumed by
        ``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._consume_narrowing_exit_report``;
        the prompt offers only the one the gate honours as a clean close
        (``report_task_already_done``).

        Lenient semantics — the architect may keep some flagged entries
        (treating them as genuinely needed; the gate's re-check is then
        the source of truth).  The only hard constraint is no NEW files:
        a post-pass subset check rejects any plan that added entries
        beyond the current ``plan.files`` set.
        """
        preamble = await self._task_preamble(task, 'architect', context)

        current_files = list(plan.get('files', []))
        files_list = '\n'.join(f'- `{f}`' for f in current_files) or '_(empty)_'
        not_touched_list = '\n'.join(f'- `{f}`' for f in not_touched) or '_(none)_'

        return f"""\
{preamble}

# Plan Tightening — Architect Narrowing Pass

The pre-merge gate flagged that some files you declared in this task's
plan were never touched by any commit on the branch.  Before the merge
can proceed, please narrow the plan against current branch state — drop
entries that turned out not to be needed, or confirm the plan if the
work is genuinely incomplete.

## Current plan files

{files_list}

## Flagged — declared but not touched on the branch

{not_touched_list}

## Action — choose ONE of (a), (b) or (d); (c) is per-entry and may be repeated

a. **Drop genuinely-unneeded entries**: call
   `update_plan_metadata(files=[<narrowed list>])` with a subset of the
   current plan files.  A flagged entry you LEAVE in the list
   will re-fire the gate's re-check and escalate — keeping one is a
   deliberate choice to escalate, not a neutral default.  Either way,
   the gate's re-check is the source of truth.
b. **Plan is honest as-is**: call `confirm_plan()` unchanged.  The
   workflow will then file a level-1 escalation (auto-watcher triages; promotes to L2 if a human is needed) — choose this only when the
   work is genuinely incomplete and the flagged files really do need
   edits.  A DELIVERED branch whose flagged file simply needed no edit
   is NOT this case — that belongs in (c) via `drop_plan_file`.
c. **Correctly declared, and correctly needed no change**: call
   `drop_plan_file(path, reason)` — ONCE PER SUCH ENTRY, repeating for
   each one.  Use this when you were right to declare the file AND the
   branch was right to leave it alone: the work landed, and this entry
   simply needed no change.  `reason` is one line saying why the entry
   was in scope and why no edit was needed.

   This is not falsification, and that is the whole point of the
   separate call.  The entry leaves the `files` list, so the gate's
   re-check can pass; the reason stays in the plan record, so why the
   file was ever declared remains auditable.  Dropping an entry with no
   recorded reason is what would falsify the plan — which is why (a)
   is the wrong tool for this case and `drop_plan_file` refuses a blank
   reason.
d. **The WHOLE task is already on main, including every commit this
   branch carries**: call `report_task_already_done(commit, evidence)`
   instead of narrowing.  The orchestrator re-verifies that the commit is on
   main (and attributes it to this task, and checks any declared
   delivered_checks) and closes the task; a claim that fails that
   verification blocks the task instead.  If this branch carries any patch
   not yet on main, the report is discarded and the gate decides.
   `report_unactionable_task` and `report_false_premise` are also honoured
   here, but each files a level-1 escalation carrying your reason.

## Forbidden for this pass

Do not call `report_ready_to_merge` or `report_blocking_dependency`: the
merge is already under way, so both are discarded unread.  A delivered
branch whose flagged file needed no edit is option (c).

Do not call `create_plan`, `add_plan_step`, or `replace_plan_step`.
This narrowing pass is scoped strictly to the `files` list; step
rewrites belong to a separate revalidation flow.

You must NOT add new files to the plan: the post-pass verifier rejects
any plan whose `files` list contains entries beyond the current set
above.  If the work needs new files, call `confirm_plan()` instead and
let a human triage the scope change.

The plan must also never be narrowed to an empty `files` list — an empty
list is not a narrowed plan, it is an unchecked one, because the gate has
nothing left to re-check.  `drop_plan_file` refuses to remove the last
entry; do not use (a) to empty the list either.  If nothing legitimately
remains, that is option (b): call `confirm_plan()` and let the escalation
triage it.
"""

    async def build_plan_schema_repair_prompt(
        self,
        task: dict,
        context: str | None = None,
    ) -> str:
        """Architect pass that restructures an on-disk plan missing ``steps``.

        The plan is never copied into the prompt: plan.json is its single home,
        read through the lane's ``.task/plan.json``. Completeness is attested
        only by the architect's own ``confirm_plan()``.
        """
        preamble = await self._task_preamble(task, 'architect', context)

        return f"""\
{preamble}

# Plan Schema Repair

The plan for this task is on disk at `.task/plan.json`, but it has no
non-empty top-level `steps` array, so it cannot advance. It is NOT reproduced
here — read the file in full.

## Action

1. Read `.task/plan.json` in full.
2. Record its existing step content with the plan-tools:
   `add_prerequisite(prereq_id, description)` for setup work and
   `add_plan_step(step_id, step_type, description)` for each TDD step, in the
   plan's own order, carrying each description across unchanged. If `files`
   is missing or empty, set it with `update_plan_metadata(files=[...])`.
3. Restructure only: do not redesign the plan or explore beyond what
   restructuring needs. Do not call `create_plan` — it replaces the document
   wholesale and would discard the analysis and decisions you are
   restructuring.
4. Call `confirm_plan()` as your FINAL action. Without it the repaired plan is
   treated as incomplete and will not advance.

If the file holds no step content to restructure, do not invent steps: stop
without calling `confirm_plan()` and the task will be re-planned.
"""

    async def build_replan_prompt(
        self,
        task: dict,
        review_feedback: str,
        context: str | None = None,
    ) -> str:
        """Architect pass that extends the plan to answer blocking review issues.

        The plan is never copied into the prompt: plan.json is its single home,
        read through the lane's ``.task/plan.json``, so a replayed prompt cannot
        show stale steps.
        """
        preamble = await self._task_preamble(task, 'architect', context)

        return f"""\
{preamble}

# Re-Plan After Review

The implementation was reviewed and blocking issues were found.

{review_feedback}

## Action

1. Read `.task/plan.json` for the current plan and each step's status — it
   is not reproduced here.
2. Add the steps that address the blocking issues with
   `add_plan_step(step_id, step_type, description)`, using step ids not
   already in the plan. New steps are recorded as pending.
3. If those steps touch files the plan does not list, call
   `update_plan_metadata(files=[...])` with the full widened list.
4. Keep every existing step in place, in order: do not remove or reorder
   steps, and do not call `create_plan` — it replaces the plan wholesale and
   would discard completed steps.
5. Call `confirm_plan()` as your FINAL action. Adding steps clears the plan's
   confirmation, so a replan that stops without it leaves an unconfirmed plan.
"""

    async def build_simple_task_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
    ) -> str:
        """Build prompt for the SIMPLE_TASK agent (Lever C).

        The workflow and its exits live in
        ``orchestrator/src/orchestrator/agents/roles.py::SIMPLE_TASK_WORKFLOW``,
        which ``SIMPLE_TASK.system_prompt`` carries, so this briefing adds only
        the task preamble and the listed files. Once the agent marks its step
        done, the orchestrator advances to VERIFY without invoking the
        implementer.
        """
        preamble = await self._task_preamble(task, 'simple_task', context)

        files =(task.get('metadata') or {}).get('files') or []
        if files:
            files_list = '\n'.join(f'- `{f}`' for f in files)
            files_section = f'## Listed files\n\n{files_list}'
        else:
            files_section = '_No files listed in task metadata — explore briefly to identify the target file(s)._'

        return f"""\
{preamble}

# Action

Make this change end-to-end in this one session, following the workflow in
your role instructions; it also says when to stop and which exit to report
instead.

{files_section}
"""

    async def build_implementer_prompt(
        self,
        plan: dict,
        context: str | None = None,
        rebase_notice: dict | None = None,
        task_id: str | None = None,
        wip_notice: list[dict] | None = None,
        *,
        scope_drift: ScopeDriftReport | None = None,
    ) -> str:
        """Build prompt for the implementer agent.

        Renders NO plan progress and NO iteration history (tasks 5728, 5744):
        plan.json and iterations.jsonl are their single homes, and the Session
        Startup Protocol below mandates reading both. A copy frozen into this
        string is the one that can lie, because the retry ladder replays an
        assembled prompt verbatim into a fresh session
        (shared/src/shared/cli_invoke.py::_reset_for_fresh_retry). Pinned by
        orchestrator/tests/test_briefing_progress_spot.py. The Action paragraph
        follows config.implementer_brief_variant and, for the same reason, names
        no step id.
        """
        scope = _plan_scope(plan, task_id)
        if context is None:
            context = await self._get_memory_context(scope, 'implementer')

        identity = self._agent_identity(scope.task_id, 'implementer')

        rebase_section = ''
        if rebase_notice:
            files_list = '\n'.join(
                f'- `{f}`' for f in rebase_notice['changed_files'][:30]
            )
            rebase_section = f"""
## Rebase Notice

Your worktree was rebased onto the latest main branch.

- **Previous base:** `{rebase_notice['old_base'][:12]}`
- **New base:** `{rebase_notice['new_base'][:12]}`
- **Files changed on main since last base:**

{files_list}

Review any overlap with your plan steps before continuing — file contents may have changed.
"""

        scope_drift_block = _format_scope_drift(scope_drift)
        scope_drift_section = f'\n{scope_drift_block}' if scope_drift_block else ''

        wip_section = ''
        if wip_notice:
            # No limit: _detect_tip_wip_commits is bounded by the contiguous
            # WIP run at HEAD, unlike the architect's whole-branch detector.
            commits_list = _format_commit_bullets(wip_notice)
            # Attribution protocol and why (task 5177): orchestrator/tests/test_harness_wip_step_detection.py::TestWipSectionStepAttribution
            wip_section = f"""
## Already-Committed WIP — Verify Before Re-Implementing

The harness auto-commits uncommitted work as a safety net before a
rebase/requeue/reclaim. The commit(s) below landed at branch HEAD this way,
which means one or more pending steps' implementation may already be sitting
there, complete, waiting on `mark_step_done`.

{commits_list}

A step's recorded commit is the commit whose diff carries that step's change.
Passing tests alone do not identify it: they pass just the same when the
change was committed earlier on this branch, below the commit(s) above.

Before writing any new code, walk the pending steps in plan order:

1. Run `git show <sha>` for each commit above to see what it contains.
2. For each step, run `git log --oneline {self.config.git.main_branch}..HEAD -- <path>`
   over the files it changes to find the commit on this branch that carries
   its change, and run its tests. Only a step whose tests behave as its spec
   says counts as carried.
   - **Carried by an earlier, non-WIP commit:** call
     `mark_step_done(step_id, <that commit's sha>)` and move to the next step.
   - **Carried by a WIP commit above:** call
     `mark_step_done(step_id, <that WIP sha>)` and move to the next step.
     Several steps citing one WIP sha is correct when its diff carries each of
     them. Do not split, amend or rewrite a WIP commit, and do not make an
     empty commit to get a distinct sha.
   - **Only partly carried, or not carried:** the walk ends here. Finish the
     step normally, commit, and cite that new commit.
"""

        return f"""\
{context}

{identity}

# Plan Overview

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

{rebase_section}
{wip_section}{scope_drift_section}
# Session Startup Protocol

1. Read `.task/plan.json` to see the full plan with current status.
2. Read `.task/iterations.jsonl` to see prior iteration details (absent until
   the first iteration is logged).
3. Run `git log --oneline -10` to see recent commits.
4. Identify the next pending step (prerequisites first, then steps).
5. **Mandatory pre-flight — before writing any code for that step:** run
   `git status` and `git diff HEAD` on the step's target files. A prior
   iteration may have already written — or fully implemented — this
   step's code and then crashed or ran out of context before staging,
   committing, or calling `mark_step_done`. That work shows up here as an
   uncommitted diff in the working tree, not as a WIP safety-commit (that
   narrower case is covered separately above when detected). If the diff
   already satisfies the step's spec, run its tests; if they pass, commit
   the existing work and call `mark_step_done(step_id, commit_sha)`
   instead of re-implementing from scratch. Only write new code if the
   step is genuinely unsatisfied by what's already on disk.

# Action

{_implementer_action(self.config.implementer_brief_variant)}
"""

    async def build_amender_prompt(
        self,
        plan: dict,
        suggestions: list[dict],
        locked_modules: list[str],
        context: str | None = None,
        task_id: str | None = None,
        *,
        scope_drift: ScopeDriftReport | None = None,
    ) -> str:
        """Build prompt for an amendment pass — implementer applies in-scope
        review suggestions without re-planning.

        ``suggestions`` is the pre-filtered in-scope list (already restricted
        to files inside ``locked_modules``). ``locked_modules`` is listed in
        the prompt so the agent can self-check scope before editing.

        The Scope Discipline block must stay true at EVERY ``lock_depth`` this
        orchestrator dispatches under, because one prompt string serves every
        project. Its original wording ("creating new files inside them is
        allowed") was authored against a package-shaped depth, where a module
        was a directory a new file could sit inside. At a file-granular depth
        the entries above ARE single files, so "inside them" is unsatisfiable
        and a created path is necessarily outside the list — which is expected
        and is not a scope violation (esc-3147-7, ruled 2026-08-24: act, then
        auto-widen). Phrase the rules around the EDIT/CREATE distinction, which
        holds at any depth, never around containment in a "module".

        Renders NO iteration history (task 5744), for the same reason as
        :meth:`build_implementer_prompt`: iterations.jsonl is its single home
        and the Action section below mandates reading it.

        Rule 3 tells the agent not to file a scope request in this pass; one
        filed anyway is still applied at the pass boundary
        (``workflow.py::TaskWorkflow._amend``), so none is orphaned.
        """
        scope = _plan_scope(plan, task_id)
        if context is None:
            context = await self._get_memory_context(scope, 'implementer')

        identity = self._agent_identity(scope.task_id, 'implementer')

        modules_list = '\n'.join(f'- `{m}`' for m in sorted(locked_modules))

        suggestion_blocks = []
        for i, s in enumerate(suggestions, 1):
            reviewer = s.get('reviewer', 'unknown')
            category = s.get('category', '')
            location = s.get('location', '')
            description = s.get('description', '')
            fix = s.get('suggested_fix', '')
            block = [
                f'### {i}. [{reviewer}] {category}',
            ]
            if location:
                block.append(f'**Location:** `{location}`')
            block.append(f'**Issue:** {description}')
            if fix:
                block.append(f'**Suggested fix:** {fix}')
            suggestion_blocks.append('\n'.join(block))
        suggestions_body = '\n\n'.join(suggestion_blocks)
        scope_drift_block = _format_scope_drift(scope_drift)
        scope_drift_section = f'{scope_drift_block}\n' if scope_drift_block else ''

        return f"""\
{context}

{identity}

{scope_drift_section}# Amendment Pass

The implementation for this task is complete and verification has passed.
A code reviewer surfaced the suggestions below, all scoped to modules this
task already holds locks for. Your job is to apply them as focused
amendments — small edits that address each point without re-planning or
expanding the task's concurrency footprint.

## Plan Overview

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

## Scope Discipline

This task holds locks for the following modules:

{modules_list}

1. Do NOT edit an existing file that is not covered above. CREATING a new
   file IS allowed, and is the right move when a suggestion calls for one
   (say, extracting a shared test helper): a path that did not exist cannot
   collide with another task's concurrent edits. Where the entries above are
   themselves single files, a new file is necessarily outside that list —
   that is expected, and not something to escalate.
2. Do NOT modify the plan (`.task/plan.json`) — it is frozen for this pass.
3. If a suggestion requires EDITING an existing file not covered above,
   skip it and note the reason in your commit message — it will be
   re-surfaced by the next review cycle or escalated as a follow-up task.
   This pass never widens the task's scope, so do NOT call
   `report_missing_plan_file` here, whatever your role's standing scope
   guidance says.
4. Prefix amendment commit messages with `amend:` so they're distinguishable
   from the main plan commits.

## Suggestions to Address

{suggestions_body}

# Action

1. Read `.task/plan.json` and `.task/iterations.jsonl` to refresh context on
   what was already done.
2. Run `git log --oneline -10` to see recent commits.
3. Apply each in-scope suggestion above. Commit amendments with `amend:`
   prefixes, grouping related fixes when sensible.
4. Run verification for the touched files before finishing.
{WAIT_PATTERN_REMINDER}
"""

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None,
        task_id: str | None = None,
        *, attempt_context: VerifyAttemptContext | None = None,
        scope_drift: ScopeDriftReport | None = None,
    ) -> str:
        """Build prompt for the debugger agent.

        *attempt_context* adds a ``# Verify Attempt`` section naming the
        failed attempt and the causes of earlier ones in the same loop; with
        ``None`` the prompt is byte-identical to the attempt-less form.
        """
        scope = _plan_scope(plan, task_id)
        if context is None:
            context = await self._get_memory_context(scope, 'debugger')

        identity = self._agent_identity(scope.task_id, 'debugger')
        verify_attempt = _format_verify_attempt(attempt_context) if attempt_context else ''
        scope_drift_block = _format_scope_drift(scope_drift)
        scope_drift_section = f'{scope_drift_block}\n' if scope_drift_block else ''

        return f"""\
{context}

{identity}

# Task Context

**Task:** {plan.get('title', 'Unknown')}
**Analysis:** {plan.get('analysis', 'N/A')}

{verify_attempt}{scope_drift_section}# Failures

```
{failures}
```

# Action

1. Analyze the root cause of each failure.
2. Make minimal, targeted fixes.
3. Run the verification commands to confirm fixes.
4. Commit your fixes.
"""

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        task: dict | None = None,
        diff_cap_chars: int = DIFF_ONLY_DIFF_CAP_CHARS,
        open_info_notes: Sequence[Escalation] | None = None,
    ) -> str:
        """Build prompt for a reviewer agent.

        *task* scopes the memory block to the work under review (D7). The
        reviewer is the highest-volume role in the fleet and had only the
        generic project-wide block, even though the workflow holds the task
        at every dispatch site. It stays optional so callers that hold no
        task — and every test that passes ``context`` directly — keep
        working, falling back to the generic conventions query.

        When *amendment_suggestions* is provided, this review immediately
        follows an in-workflow amendment round; an advisory "# Amendment
        Re-Review Scope" section is appended constraining the reviewer to
        verify those prior suggestions were addressed and to report only new
        findings within the amendment delta (task 2750).  When it is ``None``
        the returned prompt is byte-identical to the non-amendment path.  The
        advisory section is best-effort — the deterministic
        ``partition_suggestions_by_delta`` filter is the enforceable guarantee.

        *diff_cap_chars* is where the diff is cut; ``0`` inlines it whole.
        Production callers leave the default.

        *open_info_notes* lists the task's open info notes for the reviewer to
        disposition; none leaves the prompt byte-identical.
        """
        if context is None:
            context = await self._get_memory_context(BriefingScope.from_task(task), 'reviewer')

        if diff_cap_chars and len(diff) > diff_cap_chars:
            diff = diff[:diff_cap_chars] + '\n\n... [diff truncated] ...'

        action = review_action_section(
            reviewer_type,
            'Review the diff according to your specialization. Explore the codebase as needed for context.',
        )
        prompt = f"""\
{context}

# Code Diff to Review

```diff
{diff}
```

{action}"""

        if amendment_suggestions:
            prompt += '\n' + amendment_scope_section(amendment_suggestions)

        return self._with_open_info_notes(prompt, reviewer_type, open_info_notes)

    async def build_context_complete_reviewer_prompt(
        self, reviewer_type: str, subject: ReviewSubject, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
    ) -> str:
        """Build the context-complete (D2) reviewer prompt for *subject*.

        Memory is scoped to the task exactly as :meth:`build_reviewer_prompt`
        scopes it; the rendering lives in ``agents/review_briefing.py``.
        *open_info_notes* is appended exactly as :meth:`build_reviewer_prompt`
        appends it.
        """
        if context is None:
            context = await self._get_memory_context(BriefingScope.from_task(subject.task), 'reviewer')
        prompt = render_context_complete_review(
            reviewer_type=reviewer_type,
            subject=subject,
            context=context,
            task_block=self._format_task(subject.task, include_test_strategy=True),
            amendment_suggestions=amendment_suggestions,
        )
        return self._with_open_info_notes(prompt, reviewer_type, open_info_notes)

    def _with_open_info_notes(
        self, prompt: str, reviewer_type: str, notes: Sequence[Escalation] | None,
    ) -> str:
        """*prompt* with the reviewer's notes block appended; no notes leave it byte-identical."""
        if not notes:
            return prompt
        return prompt + '\n' + _open_info_notes_section(
            notes, reviewer_type=reviewer_type,
            detail_chars=self.config.info_l0_note_detail_chars,
        )

    async def build_completion_judge_prompt(
        self,
        plan: dict,
        iteration_log: list[dict],
        diff: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        """Build prompt for the completion judge agent."""
        scope = _plan_scope(plan, task_id)
        if context is None:
            context = await self._get_memory_context(scope, 'judge')

        identity = self._agent_identity(scope.task_id, 'judge')

        # Truncate diff (same cap as reviewer)
        if len(diff) > 50000:
            diff = diff[:50000] + '\n\n... [diff truncated] ...'

        # Last 5 iteration log entries (reviewer uses 3; judge benefits from
        # seeing more of the arc of work)
        log_section = ''
        if iteration_log:
            recent = iteration_log[-5:]
            lines = [
                f"- iter {e.get('iteration', '?')} [{e.get('agent', '?')}]: "
                f"{e.get('summary', 'N/A')}"
                for e in recent
            ]
            log_section = "## Recent Iterations\n\n" + '\n'.join(lines)

        # Serialize only the plan fields the judge needs
        plan_json = json.dumps({
            'task_id': plan.get('task_id'),
            'title': plan.get('title'),
            'analysis': plan.get('analysis'),
            'prerequisites': plan.get('prerequisites', []),
            'steps': plan.get('steps', []),
        }, indent=2)

        return f"""\
{context}

{identity}

# Plan

```json
{plan_json}
```

{log_section}

# Code Diff (worktree vs pre-task base)

```diff
{diff}
```

# Action

Read the code in the worktree as needed to verify behavior. Then return
your verdict as JSON matching the schema. Follow the safety rules: if the
diff is empty or trivial, `substantive_work=false` and `complete=false`.
"""

    async def build_merger_prompt(self, conflicts: str, task_intent: str) -> str:
        """Build prompt for the merger agent.

        Carries NO memory block, by decision (D7). Merging is mechanical —
        read both sides of a conflict, preserve both intents, test — and the
        role is rare (7 dispatches in 14 days, measured). What it had was the
        generic project-wide recall nobody could show helped it, so the block
        and its ``context`` parameter are gone rather than rescoped: this
        absence is deliberate, not an oversight.
        """
        return f"""\
# Task Intent

{task_intent}

# Merge Conflicts

{conflicts}

# Action

1. Read both sides of each conflict carefully.
2. Understand the intent of each change.
3. Resolve conflicts conservatively — preserve both sides' intent.
4. Run tests to verify the resolution.
5. If you cannot confidently resolve, call `submit_merge_disposition(blocked=true, reason="<why>")` and stop.
6. Once you have resolved, tested, and committed successfully, call `submit_merge_disposition(blocked=false, reason="")`.

Your disposition is read from the `submit_merge_disposition` tool call, not from your prose output — you MUST call it before finishing.
"""

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = None,
        *,
        scope_drift: ScopeDriftReport | None = None,
    ) -> str:
        """Build prompt for resuming after an escalation resolution."""
        context = await self._get_memory_context(BriefingScope.from_task(task), 'implementer')
        prior_proposal_section = self._format_prior_proposal(task)
        scope_drift_block = _format_scope_drift(scope_drift)
        scope_drift_section = f'{scope_drift_block}\n' if scope_drift_block else ''

        return f"""\
{context}

# Resuming After Escalation

This task was paused because an agent escalated a blocking issue.

## The Issue
{escalation_summary}

## Handler's Resolution
{resolution}

{prior_proposal_section}
{scope_drift_section}## Action
Resume the task applying the resolution(s) that answer this round's
escalation(s), each attributed to its escalation id. The prior agent's work
is preserved in the worktree. Read `.task/plan.json` and
`.task/iterations.jsonl` to understand current progress, then continue from
where the previous agent left off.
"""

    async def build_scope_drift_followup_prompt(
        self, plan: dict, scope_drift: ScopeDriftReport, task_id: str | None = None,
    ) -> str:
        """Prompt for the one session that briefs a contested scope request no writer saw.

        ``workflow.py::TaskWorkflow._brief_contested_scope_drift`` runs it when
        the session that filed the request was the last writer session due.
        That can be the judge's early exit with steps still pending, so the
        prompt claims nothing about the plan's progress.
        """
        scope = _plan_scope(plan, task_id)
        context = await self._get_memory_context(scope, 'implementer')
        identity = self._agent_identity(scope.task_id, 'implementer')
        return f"""\
{context}

{identity}

# Scope Request Follow-up

A file reported with `report_missing_plan_file` is locked by another task,
and no other writer session is due to hear about it. This session exists only
to settle it.

{_format_scope_drift(scope_drift)}
## Action
For each contested file: revert your edits to it and commit, keeping the
task's tests passing, or escalate as the section above says. Make no other
change. Do not edit `.task/plan.json`.
"""

    async def build_steward_initial_prompt(
        self,
        task: dict,
        escalation: dict,
        pending_escalations: list[dict],
        worktree: Path,
    ) -> str:
        """Build full briefing prompt for the steward's first invocation.

        Includes memory context, task details, escalation info, and action
        instructions.  Used for the initial session and after cap-hit resets.
        """
        preamble = await self._task_preamble(task, 'steward', None)
        esc_block = self._format_escalation(escalation)

        pending_block = ''
        other_pending = [e for e in pending_escalations if e.get('id') != escalation.get('id')]
        if other_pending:
            items = '\n'.join(
                f'- `{e.get("id")}` [{e.get("category")}]: {e.get("summary")}'
                for e in other_pending
            )
            pending_block = f'\n## Other Pending Escalations\n\n{items}\n'

        already_merged = already_merged_step(
            LandedShaSearches.for_task(str(task.get('id', '?')), self.config.git),
        )

        return f"""\
{preamble}

# Escalation

{esc_block}
{pending_block}
# Parameters

- **project_id:** `{self.project_id}`
- **project_root:** `{self.config.project_root}`
- **worktree:** `{worktree}`

# Action

1. Understand the escalation and the task context.
2. {already_merged}
3. Read the relevant code.
4. Handle the escalation — fix the issue, or triage suggestions.
5. Run tests to verify any code changes.
6. Call `resolve_issue` with a summary of what you did.
"""

    async def build_steward_continuation_prompt(
        self,
        task: dict,
        escalation: dict,
    ) -> str:
        """Build a minimal prompt for resuming the steward session.

        The session already has full context from the initial briefing,
        so this just provides the new escalation details.
        """
        esc_block = self._format_escalation(escalation)

        return f"""\
# New Escalation for Task {task.get('id', '?')} — {task.get('title', '')}

{esc_block}

Handle this escalation, then call `resolve_issue` with a summary.
"""

    @staticmethod
    def _format_escalation(escalation: dict) -> str:
        """Format an escalation dict (blocking or suggestion) into markdown."""
        lines = [
            f'- **ID:** `{escalation.get("id", "?")}`',
            f'- **Category:** {escalation.get("category", "unknown")}',
            f'- **Severity:** {escalation.get("severity", "unknown")}',
            f'- **Summary:** {escalation.get("summary", "N/A")}',
        ]
        if escalation.get('detail'):
            lines.append(f'- **Detail:** {escalation["detail"]}')
        if escalation.get('suggested_action'):
            lines.append(f'- **Suggested action:** {escalation["suggested_action"]}')
        return chr(10).join(lines)

    async def _get_memory_context(self, scope: BriefingScope, role: str) -> str:
        """The ``# Context`` block for a dispatch about *scope* by *role*."""
        return await self._memory_recall.context_block(scope, _caller_agent_id(scope.task_id, role))

    def _format_prior_proposal(self, task: dict) -> str:
        """Format the most recent dry-run block-time proposal, if any.

        Reads ``task.metadata.dry_run_proposals[-1]`` defensively (None-safe
        at every level) and renders it as a markdown block carrying an
        explicit provenance/verification line — the reader must not assume a
        persisted proposal still holds against the current tree. Returns ''
        when there is no proposal, the latest entry is not a dict, or the
        proposal predates the task's last block transition
        (``metadata.last_blocked_at``).

        The staleness comparison fails OPEN (i.e. includes the proposal)
        whenever either timestamp is absent or fails to parse via
        ``datetime.fromisoformat`` — including a naive-vs-aware mismatch,
        which raises ``TypeError`` on comparison — so a formatting hiccup
        never silently drops persisted analysis.

        Staleness compares ``timestamp`` (falling back to ``investigated_at``),
        which is re-stamped when a re-block is covered by the latest entry
        (task 3125), while the rendered
        'Investigated at' line shows ``investigated_at``, falling back to
        ``timestamp``.

        Called only from retry/resume prompt builders
        (``build_revalidation_prompt``, ``build_resume_prompt``) and, behind
        ``include_prior_proposals=True``, from ``build_architect_prompt``'s
        re-plan path — NEVER unconditionally from ``_format_task``, which
        would leak proposals into the first-dispatch anti-anchoring path
        (C-A1).
        """
        proposals = (task.get('metadata') or {}).get('dry_run_proposals') or []
        if not proposals:
            return ''
        proposal = proposals[-1]
        if not isinstance(proposal, dict):
            return ''

        proposal_text = proposal.get('proposal_text', '')
        risk_label = proposal.get('risk_label', '')
        files_referenced = proposal.get('files_referenced') or []
        stamped_at = proposal.get('timestamp') or proposal.get('investigated_at') or ''
        investigated_at = proposal.get('investigated_at') or proposal.get('timestamp') or ''

        last_blocked_at = (task.get('metadata') or {}).get('last_blocked_at')
        if stamped_at and last_blocked_at:
            try:
                is_stale = datetime.fromisoformat(stamped_at) < datetime.fromisoformat(last_blocked_at)
            except (ValueError, TypeError):
                pass  # fail open — never silently drop persisted analysis
            else:
                if is_stale:
                    return ''

        # Coerce defensively: files_referenced is persisted, untyped data, so
        # a stray non-string element must never crash the prompt build — the
        # same fail-open spirit as the timestamp comparison above. Deferred
        # until after the staleness early-return so a stale/omitted proposal
        # skips the work entirely.
        files_str = ', '.join(str(f) for f in files_referenced)

        return f"""\
## Prior Block-Time Investigation

A prior block-time investigation concluded the following; verify against the current tree before reusing — do NOT assume it still holds:

**Proposal:** {proposal_text}
**Risk:** {risk_label}
**Files referenced:** {files_str}
**Investigated at:** {investigated_at}
"""

    def _format_task(
        self, task: Mapping[str, Any], *,
        include_files: bool = True, include_test_strategy: bool = False,
    ) -> str:
        """Format a task dict as readable text.

        Args:
            task: Task dict with id, title, description, metadata, etc.
            include_files: When False, the ``metadata.files`` line is omitted
                from the output.  Defaults to True so all existing callers are
                unaffected.  Pass ``include_files=False`` from
                ``build_architect_prompt`` to anti-anchor the first plan
                derivation (C-A1): the architect must derive its own file
                footprint rather than echoing the queue-time metadata guess.
            include_test_strategy: When True, the task's ``testStrategy`` is
                rendered. Only the context-complete reviewer passes it, so
                ``review.reviewer_briefing`` stays the one switch that turns
                that briefing change on and off (task 5369).

        Appends the task's own ``metadata.delivered_checks`` as a capability-gate
        section via :func:`_format_delivered_checks` (task 5359) — the gate
        ``orchestrator/src/orchestrator/delivered_checks.py::gate_mark_done_on_delivered_checks``
        applies to THIS task's mark-done, not only to its dependents, so an agent
        that cannot see it is measured against a contract it was never shown.
        Deliberately NOT suppressed by ``include_files=False``: C-A1 hides a
        queue-time GUESS, whereas this is an authored acceptance contract the
        first derivation must plan against.

        That exemption knowingly covers a grep descriptor's ``paths``, which IS
        a file footprint and so does overlap C-A1's subject — the overlap was
        noticed, not missed. The two differ in what they BIND. ``metadata.files``
        binds nothing: the architect is told to derive its own footprint, so
        echoing the guess only anchors it. ``paths`` binds the capability: it is
        the scope ``git grep -E`` will actually search at mark-done, so an
        architect that plans the behaviour into a directory ``paths`` does not
        name has failed a check it was never shown. Hiding it would hand the
        first derivation a contract it cannot evaluate while still blocking the
        task on it — strictly the failure this section exists to cure. The
        residual anchoring cost (``paths`` does convey a location) is accepted
        as the cheaper half of that trade.

        The descriptor contract itself lives in ``docs/task-authoring.md`` §3.3
        — pointed at, never restated (INV-9).

        Then appends ``metadata.followup_items`` via :func:`_format_followup_items`, also exempt
        from ``include_files=False``: the items are the task's charter, not a footprint guess.
        """
        metadata = task.get('metadata') or {}
        lines = []
        if task.get('id'):
            lines.append(f'**ID:** {task["id"]}')
        if task.get('title'):
            lines.append(f'**Title:** {task["title"]}')
        if task.get('description'):
            lines.append(f'**Description:** {task["description"]}')
        if task.get('details'):
            lines.append(f'**Details:** {task["details"]}')
        if include_test_strategy and task.get('testStrategy'):
            lines.append(f'**Test strategy:** {task["testStrategy"]}')
        if include_files and metadata.get('files'):
            lines.append(f'**Files:** {", ".join(metadata["files"])}')
        deps = task.get('dependencies', [])
        if deps:
            dep_ids = [str(d.get('id', d)) if isinstance(d, dict) else str(d) for d in deps]
            lines.append(f'**Dependencies:** {", ".join(dep_ids)}')
        body = '\n'.join(lines) if lines else json.dumps(dict(task), indent=2)
        blocks = (
            body,
            _format_delivered_checks(metadata.get('delivered_checks')),
            _format_followup_items(metadata.get(FOLLOWUP_ITEMS_METADATA_KEY)),
        )
        return '\n\n'.join(block for block in blocks if block)

"""Verdict-tools MCP server -- stdio tool server for verdict-artifact writes.

Spawned per-agent invocation by the orchestrator (mirrors plan_tools.py).
A role-specific agent (reviewer/judge/triage/merger) submits its structured
verdict via the single tool registered for its ``--verdict-role``. The tool
writes a schema-versioned envelope to ``verdicts/<role>.json`` under the
task's ``TaskArtifacts`` root — the role-derived filename is authoritative
(never an agent-supplied field), so a reviewer cannot misname its artifact
onto a sibling's path. For reviewer verdicts, the ``reviewer`` payload field
is also validated to match ``--verdict-role``: a mismatch is rejected
(``status: error``) rather than written, so the payload can never disagree
with the filename it lands in either.

Usage (stdio transport, spawned by orchestrator):
    <sys.executable> -m orchestrator.mcp.verdict_tools --worktree /path/to/worktree \
        --verdict-role judge --session-id <session_id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Fully qualified rather than `from shared import ...`: the module is
# deliberately NOT re-exported from shared/__init__, so `import shared` does
# not pull fastmcp into every consumer of the base layer.
from shared.mcp_markup_middleware import (
    OUTCOMES as MARKUP_OUTCOMES,
)
from shared.mcp_markup_middleware import (
    MarkupGuardMiddleware,
    RepairPolicy,
    accepts_markup_override,
)
from shared.mcp_missing_arguments import MissingArgumentMiddleware

from orchestrator.artifacts import TaskArtifacts, _validate_verdict_role
from orchestrator.mcp import markup_journal, markup_sink
from orchestrator.review_issue_contract import (
    REVIEW_ISSUE_OFF_CONTRACT_FACT,
    IssueIntake,
    check_issues,
)
from orchestrator.review_payload import (
    ReviewPayloadError,
    note_dispositions_record,
    parse_note_dispositions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope helper
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def _envelope(role: str, session_id: str, payload: dict) -> dict:
    """Wrap *payload* in the schema-versioned verdict envelope."""
    return {
        'role': role,
        'schema_version': SCHEMA_VERSION,
        'session_id': session_id,
        'emitted_at': datetime.now(UTC).isoformat(),
        'verdict': payload,
    }


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
#
# Only _submit_review_verdict has constrained/validated fields (the
# reviewer==role identity check, the verdict enum, the note_dispositions
# shape below, and each issue checked against the contract in
# orchestrator/review_issue_contract.py) — the other three payloads
# (completion/triage/merge) have no enum-like fields today,
# so they write whatever they receive. Booleans and lists are still
# type-checked at the pydantic MCP tool boundary in create_server(). Add
# parallel validation here if any of those payloads gain a constrained
# field in the future.
# ---------------------------------------------------------------------------


def _write_and_ack(
    artifacts: TaskArtifacts, role: str, session_id: str, payload: dict
) -> dict[str, Any]:
    """Envelope *payload*, write it to ``verdicts/<role>.json``, and return
    the ``{'status': 'ok', 'role': role}`` acknowledgement shared by all
    four ``_submit_*`` functions below.
    """
    artifacts.write_verdict(role, _envelope(role, session_id, payload))
    return {'status': 'ok', 'role': role}


def _submit_review_verdict(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    reviewer: str,
    verdict: str,
    issues: list[dict],
    summary: str,
    note_dispositions: list[dict] | None = None,
) -> dict[str, Any]:
    if reviewer != role:
        return {
            'status': 'error',
            'message': (
                f'reviewer {reviewer!r} must match --verdict-role {role!r} — '
                'the artifact filename is authoritative for this role; a '
                'mismatched payload is rejected rather than written'
            ),
        }
    if verdict not in {'PASS', 'ISSUES_FOUND'}:
        return {
            'status': 'error',
            'message': f'verdict must be one of PASS/ISSUES_FOUND, got {verdict!r}',
        }
    try:
        parsed_notes = parse_note_dispositions(note_dispositions, reviewer=reviewer)
    except ReviewPayloadError as exc:
        return {'status': 'error', 'message': str(exc)}
    intake = check_issues(issues)
    if intake.unusable:
        return _issue_contract_refusal(intake)
    payload = {
        'reviewer': reviewer,
        'verdict': verdict,
        'issues': list(intake.issues),
        'summary': summary,
        **note_dispositions_record(parsed_notes),
    }
    ack = _write_and_ack(artifacts, role, session_id, payload)
    if intake.normalised:
        ack |= _issue_contract_normalised(intake)
    return ack


class IssueIntakeOutcome(StrEnum):
    """The ``issue_contract.outcome`` of a reviewer call and of its journal line.

    Disjoint from the markup guard's ``OUTCOMES``, so a journal line cut down
    to its floor keys still says which kind of record it is.
    """

    NORMALISED = 'normalised'
    REFUSED = 'refused'


def _issue_contract_refusal(intake: IssueIntake) -> dict[str, Any]:
    deviations = [d.describe() for d in intake.unusable]
    return {
        'status': 'error',
        'message': (
            'nothing was written: every issue must be an object carrying a '
            'severity and a description, and these are not: '
            + '; '.join(deviations) + '. Call submit_review_verdict again with '
            'severity and description on every issue (or drop the issue) and '
            'the verdict will land.'
        ),
        'issue_contract': {'outcome': IssueIntakeOutcome.REFUSED, 'deviations': deviations},
    }


def _issue_contract_normalised(intake: IssueIntake) -> dict[str, Any]:
    return {
        'message': (
            'the verdict was recorded with the normalisations listed in '
            'issue_contract.deviations. If one misstates a finding (for '
            'example a blocking finding recorded as a suggestion), call '
            'submit_review_verdict again with corrected fields: the latest '
            'call replaces this verdict.'
        ),
        'issue_contract': {
            'outcome': IssueIntakeOutcome.NORMALISED,
            'deviations': [d.describe() for d in intake.normalised],
        },
    }


def _submit_completion_verdict(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    complete: bool,
    reasoning: str,
    uncovered_plan_steps: list[str],
    substantive_work: bool,
) -> dict[str, Any]:
    payload = {
        'complete': complete,
        'reasoning': reasoning,
        'uncovered_plan_steps': uncovered_plan_steps,
        'substantive_work': substantive_work,
    }
    return _write_and_ack(artifacts, role, session_id, payload)


def _submit_triage(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    accepted: list[dict],
    skipped: list[dict],
    proposed_task_groups: list[dict],
) -> dict[str, Any]:
    payload = {
        'accepted': accepted,
        'skipped': skipped,
        'proposed_task_groups': proposed_task_groups,
    }
    return _write_and_ack(artifacts, role, session_id, payload)


def _submit_merge_disposition(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str,
    blocked: bool,
    reason: str,
) -> dict[str, Any]:
    payload = {
        'blocked': blocked,
        'reason': reason,
    }
    return _write_and_ack(artifacts, role, session_id, payload)


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------

# Roles that get a dedicated singleton tool. Any role NOT in this set is
# treated as a reviewer-panel member's name (defined in roles.py, task β) —
# verdict_tools stays decoupled from that list and falls back to
# submit_review_verdict for it.
_SINGLETON_ROLE_TOOLS = frozenset({'judge', 'triage', 'merger'})


_MARKUP_OUTCOMES_AS_JQ = ', '.join(f'"{outcome}"' for outcome in MARKUP_OUTCOMES)

#: verdict-tools' declaration of its own escalation channel (task 3690).
#: Every field is stated here rather than defaulted inside ``markup_sink``:
#: adding a server to the shared channel must be a decision at every axis, not
#: an inheritance of plan-tools' answers.
MARKUP_SINK_SPEC = markup_sink.MarkupSinkSpec(
    server_label='verdict-tools',
    agent_role='verdict-tools-markup-guard',
    residue_anchor_task_id='verdict-tools-markup-residue',
    storm_anchor_task_id='verdict-tools-markup-storm',
    refusal_consequence=(
        'NO verdict was written, so the review gate this call was supposed to '
        "close is still open and the reviewer's findings list below is the "
        'only copy of it that exists.'
    ),
    storm_consequence=(
        'further review gates strand on verdicts that never land.'
    ),
    #: The HONEST answer for this boundary, and still verdict-tools' OWN
    #: rather than plan-tools' — it names verdict-tools' file. It used to name
    #: transcript mining, because that was the only route that genuinely
    #: existed while this server's fact channel was a log line nobody retains;
    #: task 4917 gave the boundary the journal that follow-up promised, so this
    #: string now points at it and the transcript instruction is RETIRED rather
    #: than supplemented.
    #:
    #: COMPOSED from ``MARKUP_JOURNAL_DIRNAME`` rather than respelled, so the
    #: constant stays the single owner of the path an operator is told to read
    #: and the instruction cannot drift away from the artifact it names — which
    #: is the failure mode this whole leaf exists to close.
    attribution_source=(
        'every markup fact this server sees is journalled one line per event '
        f'to {markup_journal.MARKUP_JOURNAL_DIRNAME}/verdict-tools.jsonl under '
        'the main checkout — each line carries subject_task_id, tool, param, '
        'outcome and a UTC timestamp, so the leaking task is nameable from the '
        "burst window's own lines. The same file also journals reviewer "
        f'issue-contract facts (fact "{REVIEW_ISSUE_OFF_CONTRACT_FACT}"), so '
        'select the markup lines by their outcome, which the two kinds never '
        "share and which survives a line cut to its floor keys: jq 'select("
        f'.outcome | IN({_MARKUP_OUTCOMES_AS_JQ}))\'; note '
        'that under FORWARD_REPAIR a REPAIRED call is never bounced and never '
        'reaches this escalation channel at all, so the journal is the only '
        'place a repair is visible'
    ),
)


def _markup_subject_task_id(artifacts: TaskArtifacts) -> str:
    """Which task's verdict call leaked — the SUBJECT, not the routing key.

    Deliberately NOT the escalation's ``task_id`` field: that is the non-task
    residue anchor, because a level-2 record filed under a LIVE task id halts
    that task, and preserving a payload must not cost the run that produced it.

    The plan's own ``task_id`` when there is a plan to read (there always is by
    the time a verdict is submitted — the verdict is about that plan's diff);
    ``markup_sink`` falls back to the worktree name and then to an explicit
    "unattributed" when this returns empty. An unreadable plan degrades the
    same way rather than losing the record.
    """
    try:
        plan = artifacts.read_plan()
    except Exception:
        logger.warning(
            'markup guard: could not read the plan for attribution at %s; '
            'falling back to the worktree name', artifacts.artifact_path('plan.json'),
        )
        return ''
    task_id = plan.get('task_id') if isinstance(plan, dict) else None
    return task_id if isinstance(task_id, str) else ''


def _markup_fact_journal(
    artifacts: TaskArtifacts,
    resolve_project_root: Callable[[Path], Path | None],
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build verdict-tools' DURABLE fact channel (task 4917).

    The middleware emits one ``markup_detected`` fact on every outcome, and it
    is the only record anywhere that names WHICH call leaked. Before this it
    reached exactly one place: a ``logger.info`` in a per-agent stdio
    subprocess whose stderr the CLI agent that spawned it consumes. This gives
    it a consumer that survives the process — see ``markup_journal``, the
    server-agnostic channel task 4744 built for plan-tools and this boundary
    now reuses unchanged.

    ``server_label`` is read OFF the spec rather than respelled, so the spec
    stays the single owner of the label and ``verdict-tools.jsonl`` cannot come
    to disagree with the storm record that points at it.

    *resolve_project_root* is the caller's resolver, passed through unchanged.

    ``subject_task_id`` stays a THUNK, evaluated per record on the worker
    thread: a verdict can be refused before its plan is readable.

    The same journal also receives the review-issue contract facts, told apart
    by their ``fact`` key (``REVIEW_ISSUE_OFF_CONTRACT_FACT``) and, on a line
    cut to its floor keys, by their ``outcome`` (``IssueIntakeOutcome``).
    """
    return markup_journal.make_fact_journal(
        worktree=artifacts.worktree,
        server_label=MARKUP_SINK_SPEC.server_label,
        subject_task_id=lambda: _markup_subject_task_id(artifacts),
        resolve_root=resolve_project_root,
    )


def create_server(
    artifacts: TaskArtifacts,
    role: str,
    session_id: str = '',
    *,
    resolve_project_root: Callable[[Path], Path | None] = (
        markup_sink.resolve_project_root
    ),
) -> FastMCP:
    """Create the verdict-tools MCP server with EXACTLY ONE tool registered,
    selected by *role*.

    *resolve_project_root* maps the worktree to the checkout whose escalation
    queue and markup journal both channels file into; it runs lazily, on the
    first markup event, never at construction.
    """
    mcp = FastMCP('verdict-tools')

    # --- Leaked tool-call envelope markup (task 3690, PRD section 4 C2) ---
    #
    # Registered HERE, BEFORE the `if role == ...` chain below, so ONE
    # registration covers all four branches and a fifth branch added later
    # cannot be born unguarded. A per-branch registration would be exactly the
    # silent gap this task exists to close.
    #
    # FORWARD_REPAIR, not REJECT_WITH_REPAIR, and the reason is C2's own: a lost
    # submit_review_verdict STRANDS A REVIEW GATE (INV-6). The tier is passed
    # EXPLICITLY as a keyword because INV-1 makes it a registration-time
    # DECLARATION — never inferred per call from the shape of the damage or
    # from a tool's name.
    #
    # This server's leaks were LOUD, unlike escalation's. submit_review_verdict
    # declares FOUR required parameters, so an absorbed `issues` failed the call
    # outright with `Missing required argument` rather than landing silently
    # with an empty list; 19 corrupted calls of this shape sit in the committed
    # corpus. That is why the repair can only work from `on_call_tool`, which
    # runs BEFORE pydantic validation (PRD boundary row B14) — and why
    # strict_input_validation is deliberately NOT set (row B15): with it on the
    # SDK jsonschema-validates first, the middleware chain is never entered, and
    # every required-parameter leak becomes silently unrepairable.
    #
    # exempt_tools is written out even though frozenset() is the default: an
    # exemption is a declaration, and spelling it makes a future tool addition
    # here a DECISION rather than an omission. No tool on this server carries
    # envelope literals as data — the scan_memory_content case that motivates
    # exemptions lives on fused-memory (sibling task 4458). A name added here
    # would match BARE (`submit_review_verdict`, never the agent-facing
    # mcp__verdict-tools__submit_review_verdict spelling the corpus records).
    #
    # THE ESCALATION SINK IS THE SHARED ONE, and deliberately not a second
    # mechanism of this server's own. `orchestrator.mcp.markup_sink` is the one
    # orchestrator-side channel every boundary guard files through — promoted
    # out of plan_tools by this task once verdict-tools needed the identical
    # thing. A private, weaker preservation path per server is exactly the
    # INV-5 failure this PRD exists to rule against, and it very nearly shipped
    # here: the first cut of this leaf wrote residue to a worktree-local
    # `.task/markup_residue-<n>.json`, which dies with the lane at
    # `git worktree remove --force` and which nothing ever reads. The floor is
    # the queue or nothing, the same floor plan-tools declares, decided once
    # in markup_sink.
    #
    # Filing works from this process even though it is a standalone stdio
    # subprocess with no in-process queue: markup_sink resolves project_root
    # off the worktree's `--git-common-dir` and lazily opens the real
    # EscalationQueue on the 0.27% of calls that need it, so startup latency is
    # untouched. Residue is filed under a NON-TASK anchor -- at the level 2 the
    # middleware declares, a pending record carrying a live task id would halt
    # the very task whose verdict leaked.
    #
    # The state this preserves is worth MORE than plan-tools', not less: a lost
    # submit_review_verdict strands a review gate (INV-6) AND destroys a
    # reviewer's entire `issues` findings list, which is by construction text
    # the agent cannot re-emit identically.
    #
    # No try/except around the delegation: the sink's own contract is that it
    # never raises and never changes the caller's outcome, and the middleware's
    # `_call_sink` is a second floor under it. A record that cannot be filed
    # anywhere returns None, and the middleware's hint then tells the caller
    # the truth rather than claiming a preservation that did not happen.
    #
    # THE FACT SINK IS A DURABLE JOURNAL (task 4917), not a log line. The
    # middleware emits one `markup_detected` record on EVERY outcome, and it is
    # the only thing anywhere that names WHICH call leaked. This server is the
    # per-agent stdio subprocess the orchestrator spawns per invocation, so a
    # `logger.info` here reached only the stderr the spawning CLI agent
    # consumes — neither journald nor anything else that survives the process.
    # That was measured on plan-tools' identical boundary (task 4744): 0
    # journald lines against 35 real rejections over the same span, which made
    # the storm record's "identify the leaking caller" instruction
    # unfollowable. The same journal module serves both boundaries; this one
    # adds no line format, no size bounds and no failure handling of its own.
    #
    # AND IT MATTERS MORE HERE THAN THERE, because of FORWARD_REPAIR. A
    # repaired call is never bounced: the tool body runs, the verdict lands,
    # `escalation_sink` is never consulted (it sees only unrepairable residue
    # and window storms), and the only caller-visible trace is a
    # `meta['markup_repair']` block on a response nobody retains. So on this
    # server the journal is not merely the best durable record of a repair —
    # it is the ONLY one.
    #
    # WHAT THAT COSTS, and why it is paid lazily. The fact sink is awaited on
    # every outcome, so the FORWARD_REPAIR success path now hops to a worker
    # thread where a `logger.info` used to run inline; on the FIRST markup
    # event in a given server process that thread also pays one
    # `git rev-parse --git-common-dir` (markup_sink.resolve_project_root, 10s
    # timeout) before the append. Successes are memoized, so it is once per
    # process, and — like the escalation path above — it is reached only on the
    # measured 0.27% of calls that carry a leak. Resolving eagerly in
    # `create_server` instead would move that subprocess onto EVERY server
    # construction, which is one per agent invocation including the ~99.7% that
    # never leak, against the startup-latency constraint markup_sink's own
    # header records (an import/exec stall past MCP_TIMEOUT gets the server
    # silently dropped: tasks 1775 / 1776 / 2942). So the lazy shape is
    # deliberate here for the same reason it is on plan-tools (task 4744), and
    # a review that re-derives this trade should stop at this paragraph.
    #
    # One injected resolver feeds both channels, so a residue record and the
    # journal line about it land in the same checkout: two independently
    # resolved roots could silently disagree. The one journal instance also
    # serves the reviewer tool's issue-contract facts below and the escalation
    # sink's loss records (see
    # orchestrator/src/orchestrator/mcp/markup_sink.py::make_escalation_sink).
    fact_journal = _markup_fact_journal(artifacts, resolve_project_root)
    mcp.add_middleware(MarkupGuardMiddleware(
        policy=RepairPolicy.FORWARD_REPAIR,
        exempt_tools=frozenset(),
        fact_sink=fact_journal,
        escalation_sink=markup_sink.make_escalation_sink(
            worktree=artifacts.worktree,
            spec=MARKUP_SINK_SPEC,
            subject_task_id=lambda: _markup_subject_task_id(artifacts),
            loss_sink=fact_journal,
            resolve_root=resolve_project_root,
        ),
    ))
    mcp.add_middleware(MissingArgumentMiddleware())

    if role == 'judge':
        @mcp.tool()
        @accepts_markup_override
        def submit_completion_verdict(
            complete: bool,
            reasoning: str,
            uncovered_plan_steps: list[str],
            substantive_work: bool,
        ) -> dict[str, Any]:
            """Submit the completion-judge verdict for this task.

            Args:
                complete: Whether the plan's substantive work is complete.
                reasoning: Explanation for the verdict.
                uncovered_plan_steps: Plan step ids not covered by the diff.
                substantive_work: Whether the diff contains substantive work.
            """
            return _submit_completion_verdict(
                artifacts, role, session_id,
                complete, reasoning, uncovered_plan_steps, substantive_work,
            )
    elif role == 'triage':
        @mcp.tool()
        @accepts_markup_override
        def submit_triage(
            accepted: list[dict],
            skipped: list[dict],
            proposed_task_groups: list[dict],
        ) -> dict[str, Any]:
            """Submit the triage verdict for this task.

            Args:
                accepted: Suggestions accepted for follow-up tasks.
                skipped: Suggestions skipped, with reasons.
                proposed_task_groups: Proposed grouping of accepted suggestions.
            """
            return _submit_triage(
                artifacts, role, session_id,
                accepted, skipped, proposed_task_groups,
            )
    elif role == 'merger':
        @mcp.tool()
        @accepts_markup_override
        def submit_merge_disposition(
            blocked: bool,
            reason: str,
        ) -> dict[str, Any]:
            """Submit the merge disposition verdict for this task.

            Args:
                blocked: Whether the merge should be blocked.
                reason: Explanation for the disposition (may be empty when
                    not blocked).
            """
            return _submit_merge_disposition(
                artifacts, role, session_id, blocked, reason,
            )
    else:
        # A reviewer-panel member's name — see _SINGLETON_ROLE_TOOLS above.
        assert role not in _SINGLETON_ROLE_TOOLS

        @mcp.tool()
        @accepts_markup_override
        async def submit_review_verdict(
            reviewer: str,
            verdict: str,
            issues: list[dict],
            summary: str,
            note_dispositions: list[dict] | None = None,
        ) -> dict[str, Any]:
            """Submit a reviewer's verdict for this task.

            Args:
                reviewer: The reviewer's name. Must equal --verdict-role;
                    a mismatch is rejected (status: error) rather than
                    written.
                verdict: One of "PASS" or "ISSUES_FOUND".
                issues: Structured list of issues found (empty if PASS),
                    each shaped as the reviewer contract in your
                    instructions states. An issue lacking severity or
                    description is refused (status: error); other
                    off-contract fields are normalised and reported back
                    under issue_contract.
                summary: One-paragraph summary of the review.
                note_dispositions: One entry per info note your briefing lists, each
                    {escalation_id, disposition: "addressed" | "no_action" |
                    "work", rationale}. Omit it when the briefing lists no
                    notes. A malformed list is rejected (status: error)
                    rather than written.
            """
            response = _submit_review_verdict(
                artifacts, role, session_id, reviewer, verdict, issues, summary,
                note_dispositions,
            )
            if 'issue_contract' in response:
                await fact_journal({
                    'fact': REVIEW_ISSUE_OFF_CONTRACT_FACT,
                    'tool': 'submit_review_verdict',
                    'param': 'issues',
                    'role': role,
                    **response['issue_contract'],
                })
            return response

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _artifacts_from_args(
    argv: list[str] | None = None,
) -> tuple[TaskArtifacts, str, str]:
    """Parse ``--worktree``/``--meta-root``/``--task-root``/``--verdict-role``/
    ``--session-id`` and return the corresponding
    ``(TaskArtifacts, role, session_id)``.

    The root resolution mirrors plan_tools.py's ``_artifacts_from_args``
    byte-identically (parity signal) so both servers resolve the IDENTICAL
    roots for a given dispatch: each root must already exist, and omitting
    both root flags leaves the legacy ``<worktree>/.task`` single root.
    ``--verdict-role`` is validated eagerly
    here (``_validate_verdict_role``) so a malformed role aborts server
    startup with a clear message, instead of surfacing as an opaque
    ValueError on every subsequent tool call.
    """
    parser = argparse.ArgumentParser(description='Verdict-tools MCP server (stdio)')
    parser.add_argument(
        '--worktree', type=Path, required=True,
        help='Path to the git worktree containing .task/',
    )
    parser.add_argument(
        '--meta-root', type=Path, default=None,
        help=(
            'Optional `.task-meta` root (sibling of the worktree). When '
            'omitted, verdicts/ et al. are read/written at the legacy '
            '<worktree>/.task location.'
        ),
    )
    parser.add_argument(
        '--task-root', type=Path, default=None,
        help=(
            'Optional task-lifetime `.task-meta/task-<id>` root. When '
            'omitted, every artifact resolves against the lane root.'
        ),
    )
    parser.add_argument(
        '--verdict-role', type=str, required=True,
        help='The role submitting a verdict (judge/triage/merger, or a '
             'reviewer-panel name) — selects the one registered tool and '
             'the authoritative verdicts/<role>.json filename.',
    )
    parser.add_argument(
        '--session-id', type=str, default='',
        help='Optional session id recorded in the verdict envelope.',
    )
    args = parser.parse_args(argv)

    try:
        _validate_verdict_role(args.verdict_role)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    artifacts = TaskArtifacts(
        args.worktree.resolve(),
        args.meta_root.resolve() if args.meta_root is not None else None,
        args.task_root.resolve() if args.task_root is not None else None,
    )
    for root in (artifacts.lane_root, artifacts.task_root):
        if not root.is_dir():
            print(f'Error: {root} does not exist', file=sys.stderr)
            sys.exit(1)
    return artifacts, args.verdict_role, args.session_id


def main() -> None:
    """Parse CLI args and run the stdio MCP server."""
    artifacts, role, session_id = _artifacts_from_args()
    server = create_server(artifacts, role, session_id)
    # show_banner=False suppresses FastMCP's decorative startup banner and its
    # synchronous PyPI update-check (check_for_newer_version -> httpx.get). That
    # check runs on the stdio startup path BEFORE the MCP initialize handshake;
    # eliminating it removes a network call that could delay startup past the
    # CLI's MCP_TIMEOUT and get this server silently dropped (task 2942).
    server.run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    main()

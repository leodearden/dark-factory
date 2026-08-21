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
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Fully qualified rather than `from shared import ...`: the module is
# deliberately NOT re-exported from shared/__init__, so `import shared` does
# not pull fastmcp into every consumer of the base layer.
from shared.mcp_markup_middleware import (
    FACT_MARKUP_DETECTED,
    MarkupGuardMiddleware,
    RepairPolicy,
)

from orchestrator.artifacts import TaskArtifacts, _validate_verdict_role

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
# reviewer==role identity check and the verdict enum below) — the other
# three payloads (completion/triage/merge) have no enum-like fields today,
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
    payload = {
        'reviewer': reviewer,
        'verdict': verdict,
        'issues': issues,
        'summary': summary,
    }
    return _write_and_ack(artifacts, role, session_id, payload)


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


def _emit_markup_fact(fact: dict[str, Any]) -> None:
    """Emit the ``markup_detected`` record itself, as ONE structured line.

    INV-2: every outcome emits the fact, and no consumer re-derives it by
    log-scraping. The middleware already logs a human-readable WARNING for an
    operator reading the stream, so duplicating that prose here would add a
    second thing to read and still nothing to parse. This emits the RECORD —
    greppable by its own name, then ``json.loads``-able whole.

    Same emitter shape as ``escalation.server``'s, which is the only other
    registration site that wires one; the middleware owns the record, so
    neither builds it.

    Module level rather than a ``create_server`` closure: it captures nothing.
    """
    logger.info('%s %s', FACT_MARKUP_DETECTED, json.dumps(fact, sort_keys=True))


def create_server(artifacts: TaskArtifacts, role: str, session_id: str = '') -> FastMCP:
    """Create the verdict-tools MCP server with EXACTLY ONE tool registered,
    selected by *role*.
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
    # escalation_sink lands the residue of an UNREPAIRABLE call as a gitignored
    # `.task/markup_residue-<n>.json` under the TaskArtifacts root this server
    # already holds — durable, in the task's own meta root, and readable by an
    # operator without a queue. That is what makes refusing a corrupted call
    # non-destructive here (C2 L187 / INV-7): the refusal payload quotes the
    # filename, so a bounced reviewer can point an operator at its own data.
    #
    # It is NOT a queue submission and deliberately not equivalent to the
    # escalation server's. This server runs as a standalone stdio subprocess
    # (`python -m orchestrator.mcp.verdict_tools`, spawned from
    # mcp_lifecycle.py) inside the task worktree with no in-process escalation
    # queue and no escalation client, so nothing PROACTIVELY surfaces the file —
    # it waits to be found. Wiring proactive surfacing is the follow-up filed
    # alongside task 3690.
    #
    # The state this preserves is worth MORE than the escalation server's, not
    # less: a lost submit_review_verdict strands a review gate (INV-6) AND
    # destroys a reviewer's entire `issues` findings list, which is by
    # construction text the agent cannot re-emit identically.
    #
    # No try/except around the delegation: the middleware invokes sinks
    # defensively (`_call_sink` never raises and never changes the caller's
    # outcome), and a second guard here would only hide which layer failed. A
    # write that cannot land returns None, and the middleware's hint then tells
    # the caller the truth rather than claiming a preservation that did not
    # happen.
    mcp.add_middleware(MarkupGuardMiddleware(
        policy=RepairPolicy.FORWARD_REPAIR,
        exempt_tools=frozenset(),
        fact_sink=_emit_markup_fact,
        escalation_sink=artifacts.write_markup_residue,
    ))

    if role == 'judge':
        @mcp.tool()
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
        def submit_review_verdict(
            reviewer: str,
            verdict: str,
            issues: list[dict],
            summary: str,
        ) -> dict[str, Any]:
            """Submit a reviewer's verdict for this task.

            Args:
                reviewer: The reviewer's name. Must equal --verdict-role;
                    a mismatch is rejected (status: error) rather than
                    written.
                verdict: One of "PASS" or "ISSUES_FOUND".
                issues: Structured list of issues found (empty if PASS).
                summary: One-paragraph summary of the review.
            """
            return _submit_review_verdict(
                artifacts, role, session_id, reviewer, verdict, issues, summary,
            )

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _artifacts_from_args(
    argv: list[str] | None = None,
) -> tuple[TaskArtifacts, str, str]:
    """Parse ``--worktree``/``--meta-root``/``--verdict-role``/``--session-id``
    and return the corresponding ``(TaskArtifacts, role, session_id)``.

    The worktree/meta-root resolution mirrors plan_tools.py's
    ``_artifacts_from_args`` byte-identically (parity signal) so both sides
    resolve the IDENTICAL meta_root for a given worktree. When
    ``--meta-root`` is omitted, the root dir checked for existence is the
    legacy ``<worktree>/.task``. ``--verdict-role`` is validated eagerly
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

    worktree = args.worktree.resolve()
    meta_root = args.meta_root.resolve() if args.meta_root is not None else None
    root_to_check = meta_root if meta_root is not None else worktree / '.task'
    if not root_to_check.is_dir():
        print(f'Error: {root_to_check} does not exist', file=sys.stderr)
        sys.exit(1)

    return TaskArtifacts(worktree, meta_root), args.verdict_role, args.session_id


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

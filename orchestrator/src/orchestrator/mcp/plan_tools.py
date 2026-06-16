"""Plan-tools MCP server -- stdio tool server for safe plan.json operations.

Spawned per-agent invocation by the orchestrator.  The architect agent
builds plans via ``create_plan`` / ``add_plan_step`` / etc., and the
implementer marks steps done via ``mark_step_done``.  On revalidation
(blast-radius requeue), the architect uses ``update_plan_metadata``,
``remove_plan_step``, ``replace_plan_step``, and ``confirm_plan`` to
update an existing plan without recreating it from scratch.  All writes
go through ``TaskArtifacts`` methods, preserving ``_session_id`` and
enforcing correct schema.

Usage (stdio transport, spawned by orchestrator):
    # Direct-interpreter no-uv hot path (production, task 1776):
    <sys.executable> -m orchestrator.mcp.plan_tools --worktree /path/to/worktree

    # uv fallback with fast-start flags (task 1775, backward-compat):
    uv run --project <orchestrator-dir> --no-sync --frozen \
        python -m orchestrator.mcp.plan_tools --worktree /path/to/worktree
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from orchestrator.artifacts import TaskArtifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone implementation functions (testable without MCP transport)
# ---------------------------------------------------------------------------


def _create_plan(
    artifacts: TaskArtifacts,
    task_id: str,
    title: str,
    analysis: str,
    files: list[str],
) -> dict[str, Any]:
    plan = {
        'task_id': task_id,
        'title': title,
        'analysis': analysis,
        'files': files,
        'prerequisites': [],
        'steps': [],
        'design_decisions': [],
        'reuse': [],
    }
    artifacts.write_plan(plan)
    return {'status': 'ok', 'task_id': task_id}


def _add_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    # Reject duplicate step IDs
    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if step_id in existing_ids:
        return {'status': 'error', 'message': f'Step {step_id!r} already exists in plan.'}

    plan.setdefault('steps', []).append({
        'id': step_id,
        'type': step_type,
        'description': description,
        'status': 'pending',
        'commit': None,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'step_id': step_id, 'total_steps': len(plan['steps'])}


def _add_prerequisite(
    artifacts: TaskArtifacts,
    prereq_id: str,
    description: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    existing_ids = {
        s.get('id')
        for col in ('prerequisites', 'steps')
        for s in plan.get(col, [])
        if isinstance(s, dict)
    }
    if prereq_id in existing_ids:
        return {'status': 'error', 'message': f'Prerequisite {prereq_id!r} already exists.'}

    plan.setdefault('prerequisites', []).append({
        'id': prereq_id,
        'description': description,
        'status': 'pending',
        'commit': None,
        'tests': [],
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'prereq_id': prereq_id}


def _add_design_decision(
    artifacts: TaskArtifacts,
    decision: str,
    rationale: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('design_decisions', []).append({
        'decision': decision,
        'rationale': rationale,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'total_decisions': len(plan['design_decisions'])}


def _add_reuse_item(
    artifacts: TaskArtifacts,
    what: str,
    where: str,
    how: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    plan.setdefault('reuse', []).append({
        'what': what,
        'where': where,
        'how': how,
    })
    artifacts.write_plan(plan)
    return {'status': 'ok', 'total_reuse': len(plan['reuse'])}


def _mark_step_done(
    artifacts: TaskArtifacts,
    step_id: str,
    commit_sha: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    for collection in ('prerequisites', 'steps'):
        for item in plan.get(collection, []):
            if isinstance(item, dict) and item.get('id') == step_id:
                artifacts.update_step_status(step_id, 'done', commit=commit_sha)
                return {
                    'status': 'ok',
                    'step_id': step_id,
                    'new_status': 'done',
                    'commit': commit_sha,
                }
    return {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'}


# ---------------------------------------------------------------------------
# Revalidation helpers
# ---------------------------------------------------------------------------


def _update_plan_metadata(
    artifacts: TaskArtifacts,
    files: list[str] | None = None,
    analysis: str | None = None,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists. Call create_plan first.'}

    if files is not None:
        plan['files'] = files
    if analysis is not None:
        plan['analysis'] = analysis
    artifacts.write_plan(plan)
    return {
        'status': 'ok',
        'files': len(plan.get('files', [])),
    }


def _remove_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    for collection in ('prerequisites', 'steps'):
        items = plan.get(collection, [])
        for i, item in enumerate(items):
            if isinstance(item, dict) and item.get('id') == step_id:
                if item.get('status') == 'done':
                    return {
                        'status': 'error',
                        'message': f'Step {step_id!r} has status "done" and cannot be removed.',
                    }
                items.pop(i)
                artifacts.write_plan(plan)
                return {'status': 'ok', 'removed': step_id, 'collection': collection}

    return {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'}


def _replace_plan_step(
    artifacts: TaskArtifacts,
    step_id: str,
    step_type: str,
    description: str,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}

    for collection in ('prerequisites', 'steps'):
        for item in plan.get(collection, []):
            if isinstance(item, dict) and item.get('id') == step_id:
                if item.get('status') == 'done':
                    return {
                        'status': 'error',
                        'message': f'Step {step_id!r} has status "done" and cannot be replaced.',
                    }
                item['type'] = step_type
                item['description'] = description
                artifacts.write_plan(plan)
                return {'status': 'ok', 'replaced': step_id}

    return {'status': 'error', 'message': f'Step {step_id!r} not found in plan.'}


def _confirm_plan(
    artifacts: TaskArtifacts,
) -> dict[str, Any]:
    plan = artifacts.read_plan()
    if not plan:
        return {'status': 'error', 'message': 'No plan exists.'}
    if not plan.get('steps'):
        return {'status': 'error', 'message': 'Plan has no steps — cannot confirm.'}
    if not plan.get('files'):
        return {
            'status': 'error',
            'message': (
                'Plan has no files — cannot confirm. Call create_plan (or '
                'update_plan_metadata) with a non-empty files list first.'
            ),
        }

    now = datetime.now(UTC).isoformat()
    # ``_finalized_at`` is the durable completeness marker.  Its PRESENCE means
    # the architect declared the plan complete; the workflow requires it before
    # a plan may advance to execution (workflow._plan), and salvages a plan that
    # carries it even when the CLI run failed on a budget/turn cap.  Its ABSENCE
    # on a plan that already has steps means a prior session was interrupted
    # mid-build — the workflow routes that partial to a completion pass instead
    # of discarding it.  ``_revalidated_at`` is retained for revalidation-age
    # checks (see workflow._can_skip_revalidation).
    plan['_finalized_at'] = now
    plan['_revalidated_at'] = now
    artifacts.write_plan(plan)
    return {
        'status': 'ok',
        'finalized': True,
        'steps': len(plan['steps']),
        'files': len(plan.get('files', [])),
    }


# ---------------------------------------------------------------------------
# Missing-dependency reporting (architect escape hatch)
# ---------------------------------------------------------------------------


def _report_blocking_dependency(
    artifacts: TaskArtifacts,
    depends_on_task_id: str,
    reason: str,
    main_sha: str,
) -> dict[str, Any]:
    """Write the blocking-dependency artifact for the workflow to act on.

    The architect calls this when it has determined the task cannot be
    planned because it depends on work that has not yet landed on main.
    No Taskmaster mutation happens here — the workflow reads the artifact
    after the architect returns and registers the dependency
    deterministically.
    """
    artifacts.write_blocking_dependency(
        depends_on_task_id=depends_on_task_id,
        reason=reason,
        main_sha_at_report=main_sha,
    )
    return {
        'status': 'ok',
        'depends_on_task_id': depends_on_task_id,
    }


def _report_task_already_done(
    artifacts: TaskArtifacts,
    commit: str,
    evidence: str,
) -> dict[str, Any]:
    """Write the already-done artifact for the workflow to act on.

    The architect calls this when it has determined that the task's work
    is already on main at *commit*.  The workflow validates the claim
    (commit reachable from main) and sets task status to ``done``.
    """
    artifacts.write_already_done(commit=commit, evidence=evidence)
    return {'status': 'ok', 'commit': commit}


def _report_unactionable_task(
    artifacts: TaskArtifacts,
    reason: str,
    evidence: str,
) -> dict[str, Any]:
    """Write the unactionable-task artifact for the workflow to act on.

    The architect calls this when it has determined the task spec itself
    is unworkable — premises are contradictory, the goal is incompatible
    with current main, or no valid plan is possible as written.  The
    workflow short-circuits to a level-1 escalation, bypassing the steward.
    """
    artifacts.write_unactionable_task(reason=reason, evidence=evidence)
    return {'status': 'ok', 'reason': reason}


def _report_false_premise(
    artifacts: TaskArtifacts,
    classification: str,
    premise: str,
    evidence: str,
    proposed_resolution: str,
) -> dict[str, Any]:
    """Write the false-premise artifact for the workflow to act on.

    The architect calls this when it has determined that a RED-test premise
    is false, unreachable, or misattributed — before writing the doomed
    assertion.  The workflow short-circuits to a level-1 design_concern
    escalation, bypassing the steward.
    """
    artifacts.write_false_premise(
        classification=classification,
        premise=premise,
        evidence=evidence,
        proposed_resolution=proposed_resolution,
    )
    return {'status': 'ok', 'classification': classification}


def _resolve_main_sha(worktree: Path) -> str:
    """Return the current ``main`` HEAD SHA visible from *worktree*.

    Falls back to an empty string on git failure — the workflow side
    treats an empty ``main_sha_at_report`` as "advance check skipped".
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'main'],
            cwd=str(worktree),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            'Failed to resolve main SHA for blocking_dependency report: %s', exc
        )
        return ''


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_server(artifacts: TaskArtifacts) -> FastMCP:
    """Create the plan-tools MCP server with all tools registered."""
    mcp = FastMCP('plan-tools')

    @mcp.tool()
    def create_plan(
        task_id: str,
        title: str,
        analysis: str,
        files: list[str],
    ) -> dict[str, Any]:
        """Initialize a new implementation plan with metadata.

        Call this first, before adding steps or prerequisites.
        Creates .task/plan.json with the provided metadata and empty
        step/prerequisite arrays.

        Args:
            task_id: The task identifier (e.g. "df_task_13").
            title: Human-readable task title.
            analysis: Your analysis of the task, existing code, and approach.
            files: List of files (or directory paths) this task will create or modify.
                Drives concurrency locks and the phantom-done gate.
        """
        return _create_plan(artifacts, task_id, title, analysis, files)

    @mcp.tool()
    def add_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a TDD step to the plan. Steps execute in the order added.

        Args:
            step_id: Unique step ID (e.g. "step-1", "step-2").
            step_type: Either "test" or "impl".
            description: What this step does.
        """
        return _add_plan_step(artifacts, step_id, step_type, description)

    @mcp.tool()
    def add_prerequisite(
        prereq_id: str,
        description: str,
    ) -> dict[str, Any]:
        """Add a prerequisite to the plan. Prerequisites run before TDD steps.

        Use for setup work like config files, fixtures, etc.

        Args:
            prereq_id: Unique prerequisite ID (e.g. "pre-1").
            description: What this prerequisite sets up.
        """
        return _add_prerequisite(artifacts, prereq_id, description)

    @mcp.tool()
    def add_design_decision(
        decision: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Record a design decision and its rationale in the plan.

        Args:
            decision: The decision that was made.
            rationale: Why this decision was made over alternatives.
        """
        return _add_design_decision(artifacts, decision, rationale)

    @mcp.tool()
    def add_reuse_item(
        what: str,
        where: str,
        how: str,
    ) -> dict[str, Any]:
        """Record an existing code pattern or utility being reused.

        Args:
            what: What is being reused (e.g. "MergeResult dataclass pattern").
            where: Where it exists (e.g. "orchestrator/src/orchestrator/git_ops.py:14-18").
            how: How it will be reused in this task.
        """
        return _add_reuse_item(artifacts, what, where, how)

    @mcp.tool()
    def mark_step_done(
        step_id: str,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Mark a plan step or prerequisite as done with its commit SHA.

        Use this instead of directly editing .task/plan.json.
        Only the status and commit fields are updated; all other plan
        structure (including provenance metadata) is preserved.

        Args:
            step_id: The step or prerequisite ID (e.g. "step-1", "pre-1").
            commit_sha: The git commit SHA for this step's changes.
        """
        return _mark_step_done(artifacts, step_id, commit_sha)

    # --- Revalidation tools ---

    @mcp.tool()
    def update_plan_metadata(
        files: list[str] | None = None,
        analysis: str | None = None,
    ) -> dict[str, Any]:
        """Update the plan's top-level metadata without touching steps or prerequisites.

        Use during plan revalidation to update the file list or analysis
        after reviewing changes on main. All parameters are optional — only
        non-None values are updated.

        Args:
            files: Updated list of files (or directory paths) this task will
                create or modify. Drives concurrency locks and the phantom-done gate.
            analysis: Updated analysis text.
        """
        return _update_plan_metadata(artifacts, files, analysis)

    @mcp.tool()
    def remove_plan_step(
        step_id: str,
    ) -> dict[str, Any]:
        """Remove a pending step or prerequisite from the plan by ID.

        Use during plan revalidation when a step is no longer needed
        due to changes on main. Cannot remove steps with status "done".

        Args:
            step_id: The step or prerequisite ID to remove (e.g. "step-3").
        """
        return _remove_plan_step(artifacts, step_id)

    @mcp.tool()
    def replace_plan_step(
        step_id: str,
        step_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Replace a pending step's type and description in-place.

        Preserves the step's position, status, and commit fields.
        Use during plan revalidation when a step needs revision due
        to changes on main. Cannot replace steps with status "done".

        Args:
            step_id: The step ID to replace (e.g. "step-2").
            step_type: New step type — either "test" or "impl".
            description: New description of what this step does.
        """
        return _replace_plan_step(artifacts, step_id, step_type, description)

    @mcp.tool()
    def confirm_plan() -> dict[str, Any]:
        """Mark the plan COMPLETE. Call this as your final plan-tools action.

        This is the terminal "I am done building (or revising) this plan"
        signal, required in BOTH flows:
          - Initial planning: call it once every step and prerequisite has
            been added, immediately after the last ``add_plan_step``.
          - Revalidation: call it when you have reviewed the changes on main
            and the plan needs no further modification.

        Stamps a durable completeness marker on the plan. A plan WITHOUT this
        marker is treated as incomplete: it will not advance to execution, and
        a later session may resume and finish it. Requires a non-empty steps
        and files list.
        """
        return _confirm_plan(artifacts)

    @mcp.tool()
    def report_blocking_dependency(
        depends_on_task_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Report that this task cannot be planned because it depends on
        another task whose work has not yet landed on main.

        Use this INSTEAD of writing a plan when you discover, during the
        verify-premises phase, that a referenced file/symbol is missing
        because the sibling task that would create it has not yet merged.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will read the report,
        register the Taskmaster dependency, and re-queue this task to
        run again after the named task lands.

        Args:
            depends_on_task_id: The task id this task is blocked on
                (the one whose work is missing). Use the exact task id
                from the briefing or the task tree (e.g. "1042").
            reason: One-line explanation of what's missing — file/symbol
                names and where this task expected to find them.
        """
        # ``artifacts.root`` is ``worktree / '.task'`` so the worktree is its parent.
        worktree = artifacts.root.parent
        main_sha = _resolve_main_sha(worktree)
        return _report_blocking_dependency(
            artifacts, depends_on_task_id, reason, main_sha,
        )

    @mcp.tool()
    def report_task_already_done(
        commit: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Report that this task's work is already on main at ``commit``.

        Use this INSTEAD of writing a plan when you discover, during the
        verify-premises phase, that the work this task asks for is already
        present on main — typically because a sibling task direct-merged
        the change, or a prior orchestrator run landed it under a different
        task id.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will validate that ``commit``
        is reachable from main, then set this task to ``done`` with
        provenance pointing at ``commit``.

        Args:
            commit: The git SHA on main that contains this task's work.
                Must be reachable from main; the orchestrator verifies this.
            evidence: Brief description of what you found — file/symbol
                names that match this task's expected output, or why the
                commit appears to subsume this task's scope.
        """
        return _report_task_already_done(artifacts, commit, evidence)

    @mcp.tool()
    def report_unactionable_task(
        reason: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Report that this task spec is unworkable as written.

        Use this INSTEAD of writing a plan when you have determined the
        task itself cannot proceed — premises are contradictory, the goal
        is incompatible with current main, or no valid plan exists as
        written. Use this only when a human needs to rewrite or cancel
        the task; for risks/concerns where a plan is still possible, use
        ``escalate_blocker`` instead.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will mark this task BLOCKED
        and submit a level-1 escalation directly — the steward is bypassed
        because no automated handler can fix a broken spec.

        Args:
            reason: One-line summary of why the task is unworkable.
            evidence: Detail — the specific contradiction, false premise,
                or impossibility, with file:line references where possible.
        """
        return _report_unactionable_task(artifacts, reason, evidence)

    @mcp.tool()
    def report_false_premise(
        classification: str,
        premise: str,
        evidence: str,
        proposed_resolution: str,
    ) -> dict[str, Any]:
        """Report that a RED-test premise is false before writing the assertion.

        Use this INSTEAD of writing the test when you have determined that
        a RED test's substantive premise is false, unreachable, or
        misattributed.  The three premise classes are:

        - ``numeric_bound``: The test asserts a numeric threshold or tolerance
          (e.g. "within X%", "<= eps", ">= N dB", "to M digits") with no
          achievability basis.
        - ``exactness``: The test asserts closed-form exactness or reproduction
          (e.g. "exact within 1e-12", "reproduces P(t) exactly") but the
          stated configuration does not satisfy the identity that would make
          it true.
        - ``dependency_capability``: The test asserts that this task produces
          or evaluates a capability that is actually delivered by a task that
          DEPENDS ON this one, not a prerequisite.

        After calling this tool, stop. Do NOT call ``create_plan`` or
        ``escalate_blocker``. The orchestrator will mark this task BLOCKED
        and submit a level-1 design_concern escalation directly — the steward
        is bypassed because re-speccing a test premise requires human/curator
        judgment.

        Args:
            classification: One of ``numeric_bound``, ``exactness``, or
                ``dependency_capability``.
            premise: The exact assertion or claim that is false.
            evidence: The mathematical identity, method-error estimate,
                dependency-graph trace, or other proof that the premise fails.
            proposed_resolution: Suggested fix — move signal to producing task /
                weaken bound + file follow-up / change asserted configuration.
        """
        return _report_false_premise(
            artifacts, classification, premise, evidence, proposed_resolution
        )

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse --worktree and run the stdio MCP server."""
    parser = argparse.ArgumentParser(description='Plan-tools MCP server (stdio)')
    parser.add_argument(
        '--worktree', type=Path, required=True,
        help='Path to the git worktree containing .task/',
    )
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    if not (worktree / '.task').is_dir():
        print(f'Error: {worktree / ".task"} does not exist', file=sys.stderr)
        sys.exit(1)

    artifacts = TaskArtifacts(worktree)
    server = create_server(artifacts)
    server.run(transport='stdio')


if __name__ == '__main__':
    main()

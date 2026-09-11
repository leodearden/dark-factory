"""Live shadow-eval fixture builder (PRD ``plans/live-shadow-eval-prd.md``, C3).

Renders a LIVE production task record — one the factory is about to dispatch,
or has just dispatched — as an eval fixture pinned at that task's dispatch
base, so a candidate config can be run against real in-flight work and scored
against whatever production eventually lands.

``build_live_fixture`` is the sibling of ``task_sampler.build_fixture_record``
over the same fixture schema: the sampler mints a fixture from a task that has
already LANDED (so it can carry ``post_task_commit``, a ``reference`` diff stat
and a landed ``verify_outcome``), while this builder mints one at DISPATCH time,
when none of that exists yet. Neither wraps the other.

Purity invariant (C3). The builder reads nothing but its arguments: no git, no
task store, no ``plan.json`` lookup. The base is the caller's ``base_sha`` —
never "HEAD now" — and the plan is a dict the caller already read (production's
copy lives at ``<worktree_base>/.task-meta/<worktree_name>/plan.json``, keyed by
worktree name rather than task id, and resolving it belongs to the coordinator
that already owns that worktree). A builder that went looking would bind the
fixture's base to whenever it happened to run, making a cell's inputs
unreproducible from its ``shadow_cells`` row.

Emitted keys, and what the omissions mean
-----------------------------------------

``runner.load_task``'s consumers read far more keys than this builder emits.
Every omission is deliberate — the runner's documented default stands in — and
the table below is the interface contract, pinned executably by
``test_eval_live_fixture.py``'s absence assertion:

===========================  ===============================================
Emitted                      Value
===========================  ===============================================
``id``                       ``shadow_<task_id>_<cell_id>``
``name``                     the task title
``project_root``             ``str(project_root)`` — see the field note below
``pre_task_commit``          ``base_sha``, byte-equal, unnormalised
``task_definition``          ``{title, description, details}``, always all 3
``verify_commands``          the caller's gates (a copy)
``modules``                  ``task.metadata.modules`` (a copy)
``plan``                     the caller's plan (a deep copy)
===========================  ===============================================

===============================  ===========================================
Omitted                          What stands in its place
===============================  ===========================================
``reference``                    settle-time data a cell does not have yet
``post_task_commit``             ditto — production has not landed
``setup_commands``               ``None``; ``uv run`` self-syncs, and the
                                 live fixture corpus minted by
                                 ``build_fixture_record`` sets it null
``timeout_minutes``              every run path takes ``timeout_override``,
                                 so wall clock is the coordinator's call
``max_execute_iterations``       runner default 20
``max_review_cycles``            runner default 1
``judge_after_each_iteration``   runner default True
``max_architect_turns``          runner default 50
``adversarial``                  a live task is never an adversarial fixture
``complexity`` / ``project``     provenance only; read by
``cohort`` / ``provenance``      ``task_sampler.audit_fixture_corpus``, which
``verify_outcome``               audits the on-disk corpus a live cell never
                                 joins
===============================  ===========================================

The four runner knobs are omitted rather than pinned because no live task
determines them and the runner defaults are the standard the INCUMBENT is also
measured under — writing them here would silently fork shadow cells from the
corpus baseline the comparison depends on.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from orchestrator.landing_evidence import is_valid_sha_40

__all__ = [
    'ShadowShape',
    'build_live_fixture',
]


class ShadowShape(StrEnum):
    """The closed vocabulary of shadow-cell shapes (C1, ``shadow_cells.shape``).

    A ``StrEnum`` rather than a bare ``str`` parameter: the vocabulary is
    closed, so a member IS the value stored in the TEXT column and read back
    with ``ShadowShape(raw)``, while the plan rule below lives beside the
    vocabulary it constrains instead of being re-derived at each call site.
    """

    IMPLEMENTER = 'implementer'
    ARCHITECT = 'architect'
    ARCHITECT_CONSEQUENCE = 'architect-consequence'
    END_TO_END = 'end-to-end'

    @property
    def requires_plan(self) -> bool:
        """Whether a cell of this shape must be built with a plan.

        Tracks C1's ``plan_source`` column: ``'production'`` for the
        ``implementer`` shape (the cell runs production's own accepted plan),
        ``'candidate'`` for ``architect-consequence`` — consequence leg 2, the
        incumbent implementer run over the candidate architect's plan — and
        NULL for the two shapes whose architect is live, ``architect`` (the
        leg-1 architect cell) and ``end-to-end``, neither of whose runners
        reads a plan at all.

        So plan presence is a total function of shape, which is what lets
        ``build_live_fixture`` refuse in both directions rather than letting a
        stray plan be silently ignored.
        """
        return self in (ShadowShape.IMPLEMENTER, ShadowShape.ARCHITECT_CONSEQUENCE)


def _resolved_shape(shape: str) -> ShadowShape:
    """Coerce *shape* to a :class:`ShadowShape`, naming the vocabulary on miss.

    Mirrors ``task_sampler.default_verify_commands``'s message style: the
    offending value and the full set of valid ones, so a caller reading the
    error never has to go find the enum.
    """
    try:
        return ShadowShape(shape)
    except ValueError:
        valid = ', '.join(member.value for member in ShadowShape)
        raise ValueError(
            f'build_live_fixture: unknown shape {shape!r} (expected one of {valid})'
        ) from None


def build_live_fixture(
    task: dict,
    *,
    base_sha: str,
    project_root: Path | str,
    plan: dict | None,
    verify_commands: dict[str, str],
    shape: str,
    cell_id: str,
) -> dict:
    """Render the LIVE *task* as an eval fixture pinned at *base_sha*.

    Returns a dict accepted unchanged by ``runner.load_task``'s consumers.
    The eight emitted keys and every deliberate omission are tabled in the
    module docstring; the builder reads nothing but these arguments.

    *base_sha* is the task's ``metadata.branch_base_sha`` — the commit the
    production worktree was cut from, and the commit the shadow cell's own
    worktree is created at, so the two runs start from the same tree.

    *plan* is production's accepted plan for the ``implementer`` shape and the
    candidate architect's for consequence leg 2; the two live-architect shapes
    take ``None``. *cell_id* is the ``shadow_cells`` row's ulid, which makes
    the fixture id unique across the cells opened for one task.

    Raises ``ValueError`` — never logs and continues — on an unrecognised
    *shape*, a *plan* that disagrees with the shape's ``requires_plan``, a
    *base_sha* that is not a 40-hex sha, an empty *cell_id*, or a record with
    no task id. All of these are data-plumbing bugs in the caller, and every
    one of them survives to a LATER failure that has already created an eval
    worktree if it is not refused here.
    """
    resolved_shape = _resolved_shape(shape)
    if resolved_shape.requires_plan and not plan:
        raise ValueError(
            f'build_live_fixture: shape {resolved_shape.value!r} requires a '
            f'plan (got {plan!r}); it runs the frozen-plan implementer path, '
            f'which needs the accepted plan to execute'
        )
    if not resolved_shape.requires_plan and plan is not None:
        raise ValueError(
            f'build_live_fixture: shape {resolved_shape.value!r} forbids a '
            f'plan (got a plan); its runner plans live and never reads one, '
            f'so a plan here would be silently ignored'
        )
    if not is_valid_sha_40(base_sha):
        raise ValueError(
            f'build_live_fixture: base_sha must be a 40-char lowercase hex '
            f'sha (got {base_sha!r}); the eval worktree is created at this '
            f'commit and compared against `git rev-parse HEAD` literally'
        )
    if not str(cell_id or '').strip():
        raise ValueError(
            f'build_live_fixture: cell_id must be non-empty (got {cell_id!r}); '
            f'it is what distinguishes the cells opened for one task'
        )
    task_id = str(task.get('id') or '').strip()
    if not task_id:
        raise ValueError(
            f'build_live_fixture: task record has no id (got '
            f'{task.get("id")!r}); the id is half the fixture id'
        )

    metadata = task.get('metadata') or {}
    return {
        'id': f'shadow_{task_id}_{cell_id}',
        'name': str(task.get('title') or ''),
        'project_root': project_root,
        'pre_task_commit': base_sha,
        'task_definition': {
            'title': str(task.get('title') or ''),
            'description': str(task.get('description') or ''),
            'details': str(task.get('details') or ''),
        },
        'verify_commands': verify_commands,
        'modules': metadata.get('modules') or [],
        'plan': plan,
    }

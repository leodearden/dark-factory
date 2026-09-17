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

Emitted keys, and why the omissions are omissions
-------------------------------------------------

===========================  ===============================================
Emitted                      Value
===========================  ===============================================
``id``                       ``shadow_<task_id>_<cell_id>``
``name``                     the task title
``project_root``             ``str(project_root)``, absolute
``pre_task_commit``          ``base_sha``, byte-equal, unnormalised
``task_definition``          ``{title, description, details}``, always all 3
``verify_commands``          the caller's gates (a copy)
``modules``                  ``task.metadata.modules`` (a copy)
``plan``                     the caller's plan (a deep copy)
===========================  ===============================================

Every other key ``runner.load_task``'s consumers read is omitted, so the
runner's own documented default stands. ``test_eval_live_fixture.py``'s
exact-set assertion is what PINS that; the grouping below is the reasoning
behind it, and deliberately restates none of the default VALUES —
``runner.py`` is their SPOT.

- *Not knowable yet* — ``reference`` and ``post_task_commit`` are the
  settle-time reference a live cell does not have at build time (C3 forbids
  emitting them here).
- *Not the fixture's to decide* — ``timeout_minutes`` (every run path takes a
  ``timeout_override``, so wall clock is the coordinator's call-site concern)
  and the four runner knobs ``max_execute_iterations`` /
  ``max_review_cycles`` / ``judge_after_each_iteration`` /
  ``max_architect_turns``. No live task determines any of them, and the runner
  defaults are the standard the INCUMBENT is also measured under — pinning
  them here would silently fork shadow cells from the corpus baseline the
  whole comparison depends on.
- *Parity with the corpus* — ``setup_commands`` is omitted, as the ζ-minted
  corpus records also omit it; ``uv run`` self-syncs, so emitting a guessed
  sync command would be new behaviour rather than parity.
- *Harmful if present* — ``adversarial``: a live production task is never an
  adversarial fixture, and ``scoring.py::compute_recovery_score`` indexes
  ``adversarial['recovery_rubric']`` unguarded once the key is truthy.
- *No runtime consumer at all* — ``complexity`` / ``project`` / ``cohort`` /
  ``provenance`` / ``verify_outcome`` are provenance read only by
  ``task_sampler.audit_fixture_corpus``, which audits the on-disk corpus a
  live cell never joins.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from orchestrator.evals.task_sampler import (
    Cell,
    CompletedTaskCandidate,
    cell_of,
    default_verify_commands,
    repo_of_project,
)
from orchestrator.landing_evidence import is_valid_sha_40

__all__ = [
    'ShadowShape',
    'build_live_fixture',
    'live_stratum',
    'live_verify_commands',
]

# The gates every cell must carry explicitly. A cell missing one silently
# inherits the BASE config's command from
# ``runner.py::build_eval_orch_config`` — a different gate than the corpus
# baseline it is scored against.
_REQUIRED_GATES = ('test', 'lint', 'typecheck')


class ShadowShape(StrEnum):
    """The closed vocabulary of shadow-cell shapes (C1, ``shadow_cells.shape``).

    A ``StrEnum`` rather than a bare ``str`` parameter: the vocabulary is
    closed, so a member IS the value stored in the TEXT column and read back
    with ``ShadowShape(raw)``, while the plan rule below lives beside the
    vocabulary it constrains instead of being re-derived at each call site.

    Sited in this module deliberately, and not to be re-litigated: leaf α
    (C1's ``shadow_cells`` store in ``run_store.py``) and this leaf are both
    ``Prereqs: none``, so this one cannot import α, and it is the vocabulary's
    first consumer. When α lands it imports this enum rather than restating
    the four values (heuristic 11, SPOT) — accepting that ``run_store`` then
    reaches up into ``orchestrator.evals`` and drags ``task_sampler``'s git
    and sqlite glue along with it. If α would rather not carry that import,
    relocating the enum down beside its ``ShadowCell`` record (the way
    ``LandingReason`` lives beside its consumers) is α's call to make, with
    this module importing it back. What must not happen either way is a second
    copy of the four values.
    """

    IMPLEMENTER = 'implementer'
    ARCHITECT = 'architect'
    ARCHITECT_CONSEQUENCE = 'architect-consequence'
    END_TO_END = 'end-to-end'

    @property
    def requires_plan(self) -> bool:
        """Whether a cell of this shape must be built with a plan.

        Tracks C1's ``plan_source`` column: ``'production'`` for
        ``implementer`` (the cell runs production's own accepted plan),
        ``'candidate'`` for ``architect-consequence`` (consequence leg 2, the
        incumbent implementer run over the candidate architect's plan), and
        NULL for the two shapes whose architect is live — ``architect`` (the
        leg-1 architect cell) and ``end-to-end`` — neither of whose runners
        reads a plan at all.

        Plan presence is therefore a total function of shape, which is what
        lets ``build_live_fixture`` refuse in both directions rather than
        letting a stray plan be silently ignored.
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


def _resolved_project_root(project_root: Path | str) -> str:
    """Coerce *project_root* to an absolute path string, refusing a bad one.

    ``str``, not ``Path``: the fixture must be JSON-serialisable to round-trip
    at all, and ``runner.py::load_task`` reads the key as
    ``raw_root.startswith(...)``, where a ``Path`` (or a null) is an
    ``AttributeError``. Normalised once, here, so no caller has to know that.
    Refused rather than passed through because, uniquely among the emitted
    paths, a bad value fails SILENTLY downstream — each raise below names how.
    """
    if not project_root:
        raise ValueError(
            f'build_live_fixture: project_root must be non-empty (got '
            f'{project_root!r}); an empty one reaches build_eval_orch_config '
            f'as Path("") == cwd and the cell is scored against that tree'
        )
    text = str(project_root)
    if not Path(text).is_absolute():
        raise ValueError(
            f'build_live_fixture: project_root must be an absolute path (got '
            f'{text!r}); load_task silently rewrites a project_root that does '
            f'not exist to the repo root above the fixture file, so a bad one '
            f'reruns the cell against the wrong checkout instead of failing'
        )
    return text


def _resolved_gates(verify_commands: Mapping[str, str]) -> dict[str, str]:
    """Copy *verify_commands*, refusing a set that does not carry all 3 gates.

    ``runner.py::build_eval_orch_config`` resolves each gate as
    ``task.get('verify_commands', {}).get(<gate>, base.<command>)``, so a
    missing gate is the one bad input that degrades into a PLAUSIBLE
    measurement rather than a failure. The copy follows
    ``build_fixture_record``'s own ``dict(verify_commands)`` (heuristic 8).
    """
    if not isinstance(verify_commands, Mapping):
        raise ValueError(
            f'build_live_fixture: verify_commands must be a mapping of '
            f'{"/".join(_REQUIRED_GATES)} to command strings (got '
            f'{type(verify_commands).__name__})'
        )
    missing = sorted(
        gate
        for gate in _REQUIRED_GATES
        if not str(verify_commands.get(gate) or '').strip()
    )
    if missing:
        raise ValueError(
            f'build_live_fixture: verify_commands must carry '
            f'{"/".join(_REQUIRED_GATES)} (missing or blank: {missing}); a '
            f'missing gate silently falls back to the base config\'s command, '
            f'which would score the cell against different gates than the '
            f'corpus baseline it is compared against'
        )
    return dict(verify_commands)


def _resolved_modules(metadata: Mapping[str, object]) -> list[str]:
    """Copy ``metadata.modules``, refusing anything but strings in a sequence.

    An absent or empty value is a thin task, not a malformed one, and yields
    ``[]``. ``runner.py::run_eval`` hands the result straight to
    ``TaskAssignment(modules=...)`` and the eval scheduler, which scope the
    cell by it and score it either way.
    """
    modules = metadata.get('modules')
    if not modules:
        return []
    if isinstance(modules, str) or not isinstance(modules, (list, tuple)):
        raise ValueError(
            f'build_live_fixture: task.metadata.modules must be a list of '
            f'module-path strings (got {modules!r}); a bare string would be '
            f'exploded into one module per character and scoped the cell to '
            f'nonsense'
        )
    non_strings = [entry for entry in modules if not isinstance(entry, str)]
    if non_strings:
        raise ValueError(
            f'build_live_fixture: task.metadata.modules must contain only '
            f'module-path strings (got {non_strings!r} in {modules!r})'
        )
    return list(modules)


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
    *base_sha* that is not a 40-hex sha, an empty *cell_id*, a record with no
    task id, a relative or empty *project_root*, a *verify_commands* missing
    any of the three gates, or a malformed ``metadata.modules``. Every one of
    those is a data-plumbing bug in the caller that, unrefused, survives into
    a LATER failure with an eval worktree already created — or worse into a
    cell that runs and is SCORED against the wrong tree, the wrong gates or
    nonsense module scoping, with nothing in the record to show it.
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
    resolved_root = _resolved_project_root(project_root)
    resolved_gates = _resolved_gates(verify_commands)
    resolved_modules = _resolved_modules(task.get('metadata') or {})

    return {
        'id': f'shadow_{task_id}_{cell_id}',
        'name': str(task.get('title') or ''),
        'project_root': resolved_root,
        'pre_task_commit': base_sha,
        'task_definition': {
            'title': str(task.get('title') or ''),
            'description': str(task.get('description') or ''),
            'details': str(task.get('details') or ''),
        },
        'verify_commands': resolved_gates,
        'modules': resolved_modules,
        # Deep, not shallow: run_eval hands this plan to a real workflow that
        # flips step status in place, and the dict the coordinator passed in is
        # production's freshly-read plan.json. A shadow cell must never write
        # through to the live task's plan. The fixture stays `==` to the
        # caller's plan, which is all C3's "plan as given" asks.
        'plan': copy.deepcopy(plan),
    }


# ---------------------------------------------------------------------------
# The two derivations a caller needs to OPEN a cell, alongside the builder
# ---------------------------------------------------------------------------
#
# C3 takes ``verify_commands`` as an INPUT and says nothing about the stratum,
# so the reuse this leaf owes ``task_sampler`` has nowhere to live inside the
# builder itself. Without these two, every caller would re-derive
# ``repo_of_project`` + ``default_verify_commands`` and re-adapt a task record
# onto the classifiers — the duplication SPOT forbids.


def live_verify_commands(project: str) -> dict[str, str]:
    """Return the ``{test,lint,typecheck}`` gates a cell on *project* runs.

    The single definition of "the gates a live shadow cell runs", so a cell and
    the ``evals/tasks/`` corpus it is compared against can never disagree about
    them. Deliberately thin: the command strings live in ``task_sampler`` and
    are never restated here, and ``repo_of_project``'s ``ValueError`` on an
    unrecognised project propagates unchanged rather than defaulting — a
    silent fallback would run (say) a Rust task's cell under pytest and score
    the resulting red gate as a candidate failure.
    """
    return default_verify_commands(repo_of_project(project))


def _candidate_from_live_task(task: dict, project: str) -> CompletedTaskCandidate:
    """Adapt a LIVE task record onto the classifiers' existing input type.

    Populates only the four fields the classifiers read — ``project``,
    ``title``, ``description``, ``complexity`` — plus ``task_id``, which
    ``repo_of`` names in its error. ``project_root`` and the landed-commit
    fields are left at their defaults on purpose: a stratum derivation has no
    business holding a checkout path, and a live cell has no landed provenance
    at build time.
    """
    metadata = task.get('metadata') or {}
    return CompletedTaskCandidate(
        task_id=str(task.get('id') or ''),
        project=project,
        project_root='',
        title=str(task.get('title') or ''),
        description=str(task.get('description') or ''),
        complexity=metadata.get('complexity'),
    )


def live_stratum(task: dict, *, project: str) -> Cell:
    """Return the ``(repo, kind, path)`` stratification cell for a LIVE *task*.

    The sampler's own three axes applied to a live record, so a shadow cell and
    a corpus fixture land in the same cell for the same reasons. That
    comparability is the point, and it fixes the reading of the path axis:
    ``classify_path`` strata the way the SAMPLER strata the corpus, which is a
    deliberate conservative superset of how production actually routed the
    task — it applies ``has_simple_task_blocker`` to ``title + description``
    where ``triage.is_declared_simple_task`` applies it to the description
    alone (see ``task_sampler.py::classify_path``). So a task carrying a
    blocker token only in its TITLE strata as ``'full'`` here while production
    dispatched it down the ``'simple'`` path. Asking production's predicate
    directly would be more faithful to that one task and would make the cell
    incomparable with every corpus fixture, which is the worse trade.

    Returns the structured triple rather than the ``<repo>×<kind>×<path>``
    text: rendering it as a delimited string and splitting it back would be an
    ad-hoc parser at every consumer. The single text rendering belongs with
    the ``shadow_cells.stratum`` column that stores it.

    Raises ``ValueError`` on an unrecognised *project* (propagated from
    ``repo_of``) — repo is a hard axis, not a defaultable one.
    """
    return cell_of(_candidate_from_live_task(task, project))

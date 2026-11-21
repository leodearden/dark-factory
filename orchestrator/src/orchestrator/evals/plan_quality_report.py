"""Per-cell column surfaces over persisted eval cells: plan quality (θ), with its
per-config aggregate, and recovery score (η).

Rationale for the plan-quality surface: docs/plan-scoring-and-judge.md, "The
plan-quality report: four treatments (tasks 3118, 3302, 3628, 4760)".
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .cell_admission import (
    judged_blind_count,
    judged_without_reference,
    mean_plan_quality,
    plan_quality_score,
    plan_rate,
)
from .measurement import UnmeasuredCause, format_invocation_errors
from .metrics import DECLINE_KINDS, RoleUnderTest, terminal_kind_of
from .plan_predicates import produced_a_plan
from .table_render import render_fixed_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .results import EvalResult


_RECOVERY_COLUMNS = ('task_id', 'config_name', 'adversarial', 'recovery_score')


def build_recovery_report(results: list[EvalResult]) -> dict[str, Any]:
    """The recovery-score column over *results*, as ``{'rows': [...]}``.

    Each row carries ``task_id``, ``config_name``, ``recovery_score`` (a float for
    an adversarial fixture, ``None`` otherwise) and ``adversarial``. Rows are
    sorted by ``(task_id, config_name)``.

    ``adversarial`` is the explicit ``metrics['adversarial']`` flag
    ``collect_metrics`` stamps, NOT ``recovery_score is not None``: recovery
    scoring can fail on an adversarial fixture and leave the score ``None``, and
    inferring the flag would hide that failure as ``adversarial: false``. A cell
    persisted before the flag existed falls back to that inference, so old
    reports render unchanged.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        recovery_score = result.metrics.get('recovery_score')
        explicit_adversarial = result.metrics.get('adversarial')
        adversarial = (
            bool(explicit_adversarial)
            if explicit_adversarial is not None
            else recovery_score is not None
        )
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'adversarial': adversarial,
            'recovery_score': recovery_score,
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))
    return {'rows': rows}


def format_recovery_table(report: dict[str, Any]) -> str:
    """Render :func:`build_recovery_report` output as a deterministic table.

    A ``recovery_score`` column shows the populated float (4 dp) for adversarial
    rows and ``-`` (the null sentinel) for ordinary rows. The same report always
    renders byte-identically (no wall-clock or dict-order dependence).
    """
    rows = report.get('rows', [])

    def _score_cell(value: float | None) -> str:
        return '-' if value is None else f'{value:.4f}'

    rendered = [
        {
            'task_id': str(r['task_id']),
            'config_name': str(r['config_name']),
            'adversarial': 'yes' if r['adversarial'] else 'no',
            'recovery_score': _score_cell(r['recovery_score']),
        }
        for r in rows
    ]
    return '\n'.join(render_fixed_table(
        _RECOVERY_COLUMNS, rendered, header='recovery_score report:',
    ))


_PLAN_QUALITY_COLUMNS = (
    'task_id', 'config_name', 'role_under_test', 'plan_quality',
    'invocation_error',
)
_PLAN_QUALITY_MEAN_COLUMNS = (
    # ``declined`` sits right after the ``no_plan`` it explains (task 4760).
    'config_name', 'n', 'cap_excluded', 'no_plan', 'declined',
    'judged_without_reference', 'plan_rate', 'mean_plan_quality',
)
# A cap-tainted row's plan_quality cell: neither a number nor a non-architect
# row's '-', and cause-neutral (the cause is in the invocation_error column).
_CAP_EXCLUDED_CELL = 'excluded'
_PLAN_QUALITY_MEAN_HEADER = 'plan_quality by config:'


def _merge_counts(per_key: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Fold per-config ``{kind: count}`` tallies into ONE campaign-wide tally.

    Pure: reads the mapping, mutates nothing, and returns insertion-ordered
    counts the caller key-sorts for determinism.
    """
    merged: dict[str, int] = defaultdict(int)
    for counts in per_key.values():
        for key, n in counts.items():
            merged[key] += n
    return dict(merged)


def _exclusion_cause(row: dict[str, Any]) -> UnmeasuredCause:
    """Why a cap-tainted *row* was excluded: its first record's cause.

    Architect records precede judge ones and a judge-only failure never
    taints, so the first record is the one that drove the exclusion.
    """
    cell = f'{row["task_id"]} x {row["config_name"]}'
    records = row['invocation_errors']
    if not records:
        raise ValueError(
            f'{cell}: a cap_tainted cell must carry at least one '
            f'invocation_errors record naming why it was not measured, and '
            f'this one carries none. Load cells through ResultStore.load_all, '
            f'which supplies the record for pre-6734 cells.'
        )
    cause = records[0]['cause']
    try:
        return UnmeasuredCause(cause)
    except ValueError:
        raise ValueError(
            f'{cell}: invocation_errors[0] names cause {cause!r}, which is not '
            f'an UnmeasuredCause ({", ".join(c.value for c in UnmeasuredCause)}).'
        ) from None


def build_plan_quality_report(results: list[EvalResult]) -> dict[str, Any]:
    """The θ plan-quality report over *results*: per-cell rows and a per-config aggregate.

    ``rows``, sorted by ``(task_id, config_name)``, carry ``task_id``,
    ``config_name``, ``role_under_test``, the persisted ``plan_quality`` and
    ``plan_steps``, ``cap_tainted``, ``invocation_errors``,
    ``judged_without_reference``, ``terminal_kind``, ``scored_plan_quality`` (the
    :func:`plan_quality_score` the aggregate averages) and ``produced_a_plan``. A
    field a cell predates reads back ``None`` (``False`` for ``cap_tainted``).

    ``configs``, one per config with an architect row and sorted by name, carry
    ``n`` (the admitted, scored cells), ``cap_excluded``, ``no_plan``,
    ``declined``, ``declined_by_kind``, ``no_plan_declined``,
    ``terminal_kind_unmeasured``, ``judged_without_reference`` and its
    ``_unmeasured`` count, ``total``, ``plan_rate`` and ``mean_plan_quality``. A
    cap-tainted cell is left out of ``n`` and counted in ``cap_excluded``; a
    no-plan cell is scored ``0.0`` and counted in ``no_plan``.
    ``mean_plan_quality`` and ``plan_rate`` are ``None`` for an empty pool, and
    ``judged_without_reference`` is ``None`` when an admitted cell predates its
    marker.

    The report level carries ``cap_excluded``, ``cap_excluded_by_cause`` (keyed by
    the first record's :class:`UnmeasuredCause`), ``declined``,
    ``declined_by_kind``, ``no_plan_declined``, ``terminal_kind_unmeasured``,
    ``judged_without_reference`` (``None`` when any admitted cell predates the
    marker) and its ``_unmeasured`` count. Every by-cause and by-kind dict is
    key-sorted. A cap-tainted row without a valid record raises ``ValueError``
    naming the cell.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'role_under_test': result.metrics.get('role_under_test'),
            'plan_quality': result.metrics.get('plan_quality'),
            'plan_steps': result.metrics.get('plan_steps'),
            'cap_tainted': bool(result.metrics.get('cap_tainted')),
            'invocation_errors': result.metrics.get('invocation_errors'),
            # Three-valued: ``None`` means never measured, not False (task 3628).
            'judged_without_reference': judged_without_reference(
                result.metrics
            ),
            # ``None`` (implementer or pre-field cell) is terminal_kind_unmeasured.
            'terminal_kind': terminal_kind_of(result.metrics),
            # Beside the raw plan_quality, so the judge's number a floor overrode shows.
            'scored_plan_quality': plan_quality_score(
                result.metrics,
                where=f'{result.task_id} x {result.config_name}',
            ),
            # From the metrics dict, never re-derived from the row: a renamed row key
            # would silently answer False for every row.
            'produced_a_plan': produced_a_plan(result.metrics),
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))

    # Architect rows only: an implementer run's null score must not dilute the mean.
    scored: dict[str, list[float]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    no_plan: dict[str, int] = defaultdict(int)
    judged_blind: dict[str, int] = defaultdict(int)
    judged_blind_unmeasured: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    by_cause: dict[UnmeasuredCause, int] = defaultdict(int)
    # The task-4760 decline split, over the admitted pool ``scored`` is over.
    declined: dict[str, int] = defaultdict(int)
    declined_by_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    no_plan_declined: dict[str, int] = defaultdict(int)
    terminal_unmeasured: dict[str, int] = defaultdict(int)
    for row in rows:
        if row['role_under_test'] != RoleUnderTest.ARCHITECT:
            continue
        cfg = row['config_name']
        totals[cfg] += 1
        if row['cap_tainted']:
            excluded[cfg] += 1
            by_cause[_exclusion_cause(row)] += 1
        elif row['scored_plan_quality'] is not None:
            # Subscript, not ``.get``: a dropped key must raise, not read as "no plan".
            if not row['produced_a_plan']:
                no_plan[cfg] += 1
            # Inside the admitted branch, so a subset of ``n``; a keyless cell is
            # unmeasured, not False (task 3628).
            if row['judged_without_reference'] is None:
                judged_blind_unmeasured[cfg] += 1
            elif row['judged_without_reference']:
                judged_blind[cfg] += 1
            # Same branch, same reason (task 4760); ``None`` is unmeasured, never
            # "did not decline".
            kind = row['terminal_kind']
            if kind is None:
                terminal_unmeasured[cfg] += 1
            elif kind in DECLINE_KINDS:
                declined[cfg] += 1
                declined_by_kind[cfg][kind] += 1
                # The intersection with ``no_plan``; a plan-then-decline cell is not in it.
                if not row['produced_a_plan']:
                    no_plan_declined[cfg] += 1
            # The accessor's floored score, the pool build_composite_report averages too.
            scored[cfg].append(row['scored_plan_quality'])

    configs = [
        {
            'config_name': cfg,
            'n': len(scored[cfg]),
            'cap_excluded': excluded[cfg],
            'no_plan': no_plan[cfg],
            'declined': declined[cfg],
            'declined_by_kind': {
                kind: declined_by_kind[cfg][kind]
                for kind in sorted(declined_by_kind[cfg])
            },
            'no_plan_declined': no_plan_declined[cfg],
            'terminal_kind_unmeasured': terminal_unmeasured[cfg],
            'judged_without_reference': judged_blind_count(
                judged_blind[cfg], judged_blind_unmeasured[cfg],
            ),
            'judged_without_reference_unmeasured': judged_blind_unmeasured[cfg],
            'total': totals[cfg],
            'plan_rate': plan_rate(len(scored[cfg]), no_plan[cfg]),
            'mean_plan_quality': mean_plan_quality(scored[cfg]),
        }
        for cfg in sorted(totals)
    ]
    return {
        'rows': rows,
        'configs': configs,
        'cap_excluded': sum(excluded.values()),
        'cap_excluded_by_cause': {c.value: by_cause[c] for c in sorted(by_cause)},
        # Summed from the per-config counters. Unmeasured cells make these PARTIAL,
        # not unknown, and terminal_kind_unmeasured says by how much.
        'declined': sum(declined.values()),
        'declined_by_kind': {
            kind: total
            for kind, total in sorted(_merge_counts(declined_by_kind).items())
        },
        'no_plan_declined': sum(no_plan_declined.values()),
        'terminal_kind_unmeasured': sum(terminal_unmeasured.values()),
        # One keyless admitted cell anywhere makes the campaign-wide bound unknown.
        'judged_without_reference': judged_blind_count(
            sum(judged_blind.values()), sum(judged_blind_unmeasured.values()),
        ),
        'judged_without_reference_unmeasured': sum(
            judged_blind_unmeasured.values()
        ),
    }


def _format_plan_quality_mean_section(report: dict[str, Any]) -> list[str]:
    """The per-config mean block, in :data:`_PLAN_QUALITY_MEAN_COLUMNS` order.

    ``mean_plan_quality`` is printed beside the ``n`` it averages and the counts
    that qualify it. ``judged_without_reference``, ``plan_rate`` and
    ``mean_plan_quality`` render ``-`` for ``None``; a measured zero count,
    ``declined`` included, renders ``0``.
    """
    configs = report.get('configs', [])
    rendered = [
        {
            'config_name': str(c['config_name']),
            'n': str(c['n']),
            'cap_excluded': str(c['cap_excluded']),
            'no_plan': str(c['no_plan']),
            'declined': str(c['declined']),
            'judged_without_reference': (
                '-' if c.get('judged_without_reference') is None
                else str(c['judged_without_reference'])
            ),
            'plan_rate': (
                '-' if c.get('plan_rate') is None
                else f'{float(c["plan_rate"]):.4f}'
            ),
            'mean_plan_quality': (
                '-' if c['mean_plan_quality'] is None
                else f'{float(c["mean_plan_quality"]):.4f}'
            ),
        }
        for c in configs
    ]
    return render_fixed_table(
        _PLAN_QUALITY_MEAN_COLUMNS, rendered, header=_PLAN_QUALITY_MEAN_HEADER,
    )


def format_plan_quality_table(report: dict[str, Any]) -> str:
    """Render :func:`build_plan_quality_report` output as a deterministic table.

    One row per cell. ``plan_quality`` is the persisted float at 4 dp for an
    architect row, ``-`` for a non-architect row and ``excluded`` for a
    cap-tainted one. ``role_under_test`` and ``invocation_error`` (the
    :func:`format_invocation_errors` rendering) render ``-`` for ``None``.

    Trailer lines follow the rows: the exclusion count broken out by cause, then
    the judged-without-reference bound and the decline split, each stated against
    the scored cells and printed unless measured and zero, then the per-config
    mean block. The same report always renders byte-identically.
    """
    rows = report.get('rows', [])

    def _score_cell(row: dict[str, Any]) -> str:
        if row.get('cap_tainted'):
            return _CAP_EXCLUDED_CELL
        value = row['plan_quality']
        return '-' if value is None else f'{value:.4f}'

    def _text_cell(value: str | None) -> str:
        return '-' if value is None else str(value)

    rendered = [
        {
            'task_id': str(r['task_id']),
            'config_name': str(r['config_name']),
            'role_under_test': _text_cell(r['role_under_test']),
            'plan_quality': _score_cell(r),
            'invocation_error': _text_cell(
                format_invocation_errors(r.get('invocation_errors'))
            ),
        }
        for r in rows
    ]
    lines = render_fixed_table(
        _PLAN_QUALITY_COLUMNS, rendered, header='plan_quality report:',
    )

    architect_cells = sum(c['total'] for c in report.get('configs', []))
    # By cause, so a permanent config error cannot read as a transient cap window.
    by_cause = report.get('cap_excluded_by_cause') or {}
    causes = ', '.join(f'{cause}: {n}' for cause, n in sorted(by_cause.items()))
    lines.append('')
    lines.append(
        f'excluded: {report.get("cap_excluded", 0)} unmeasurable cell(s) of '
        f'{architect_cells} architect cell(s)'
        + (f' ({causes})' if causes else '')
    )
    # Against the scored cells, and suppressed only when measured and zero: an
    # UNMEASURED bound always prints (task 3628).
    judged_blind = report.get('judged_without_reference', 0)
    unmeasured = report.get('judged_without_reference_unmeasured', 0) or 0
    scored_cells = sum(c['n'] for c in report.get('configs', []))
    if judged_blind is None or unmeasured:
        lines.append(
            f'judged without reference: UNMEASURED — {unmeasured} of '
            f'{scored_cells} scored cell(s) predate the marker, so their '
            f'plan_quality validity is UNKNOWN (not bounded, not clean)'
        )
    elif judged_blind:
        lines.append(
            f'judged without reference: {judged_blind} of {scored_cells} '
            f'scored cell(s) (plan_quality bounded)'
        )
    # The decline split (task 4760), against the scored cells: it explains the
    # ``no_plan`` count below, and is suppressed only when measured and empty.
    declined_total = report.get('declined', 0) or 0
    declined_by_kind = report.get('declined_by_kind') or {}
    no_plan_cells = sum(c['no_plan'] for c in report.get('configs', []))
    no_plan_declined = report.get('no_plan_declined', 0) or 0
    terminal_unmeasured = report.get('terminal_kind_unmeasured', 0) or 0
    if declined_total:
        kinds = ', '.join(
            f'{kind}: {n}' for kind, n in sorted(declined_by_kind.items())
        )
        lines.append(
            f'declined: {declined_total} of {scored_cells} scored cell(s) '
            f'ended on an explicit decline exit'
            + (f' ({kinds})' if kinds else '')
            + f' — {no_plan_declined} of {no_plan_cells} no-plan cell(s) were '
            f'a refusal, not a planning failure'
        )
    # An UNMEASURED population always prints: a silent '0 declines' over a
    # pre-4760 corpus would read as "every candidate failed to plan".
    if terminal_unmeasured:
        lines.append(
            # "no resolved terminal kind", not "predate the field": terminal_kind_of
            # reads an absent key and a written null alike.
            f'terminal kind: UNMEASURED — {terminal_unmeasured} of '
            f'{scored_cells} scored cell(s) carry no resolved terminal kind, '
            f'so the decline split is PARTIAL (not measured-clean)'
        )
    lines.append('')
    lines.extend(_format_plan_quality_mean_section(report))
    return '\n'.join(lines)

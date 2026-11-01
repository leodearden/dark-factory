"""Per-cell column surfaces over persisted eval cells: plan quality (θ), with its
per-config aggregate, and recovery score (η).
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


# ---------------------------------------------------------------------------
# recovery_score surface (eval-revival η)
#
# An ADDITIVE, interim per-(task_id, config_name) column distinct from the Elo
# leaderboard above. It surfaces the ``recovery_score`` that
# ``runner.collect_metrics`` writes into ``EvalResult.metrics`` — a populated
# float for adversarial fixtures, the ``None`` null sentinel otherwise. It does
# NOT touch the Elo ``build_report`` / ``format_markdown`` schema: the full C4
# per-config composite+cost+latency+recovery report row is owned by Phase-3
# task λ, which η does not depend on; this is the distinct-column surface μ/λ
# consume in the interim.
# ---------------------------------------------------------------------------

_RECOVERY_COLUMNS = ('task_id', 'config_name', 'adversarial', 'recovery_score')


def build_recovery_report(results: list[EvalResult]) -> dict[str, Any]:
    """Build the recovery-score column from a list of :class:`EvalResult`.

    Each row carries ``task_id`` / ``config_name``, the ``recovery_score``
    pulled from ``EvalResult.metrics`` (a populated float for adversarial
    fixtures, ``None`` — the C4 ``recovery_score | null`` sentinel — otherwise),
    and an ``adversarial`` bool.

    The ``adversarial`` flag is taken from the explicit ``metrics['adversarial']``
    boolean that ``collect_metrics`` stamps from the task record — NOT inferred
    from ``recovery_score is not None``. That distinction matters: recovery
    scoring can fail on a genuinely adversarial fixture, in which case
    ``collect_metrics``' guard leaves ``recovery_score=None``; inferring the flag
    from the score would then silently mislabel that run ``adversarial: false``
    and hide the scoring failure. Keying on the explicit flag renders such a run
    as ``adversarial: true`` with a null score instead. For results persisted
    BEFORE the flag existed (no ``adversarial`` key in ``metrics``), we fall
    back to the old ``recovery_score is not None`` inference so old reports
    render unchanged. Rows are sorted by ``(task_id, config_name)`` so the
    surface is deterministic.
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


# ---------------------------------------------------------------------------
# plan_quality surface (eval-revival θ)
#
# An ADDITIVE, interim per-(task_id, config_name, role_under_test) column
# distinct from the Elo leaderboard above — the architect-eval analogue of the
# recovery_score surface. It surfaces the ``plan_quality`` that
# ``run_architect_eval`` writes into ``EvalResult.metrics`` — a populated float
# for architect runs (role_under_test ARCHITECT), the ``None`` null sentinel
# otherwise (ordinary implementer runs never invoke the plan judge). It does NOT
# touch the Elo ``build_report`` / ``format_markdown`` schema: the full C4
# per-config composite+cost+latency+plan_quality report row is owned by Phase-3
# task λ, which θ does not depend on; this is the distinct-column surface μ/λ
# consume in the interim, mirroring η's recovery surface.
# ---------------------------------------------------------------------------

_PLAN_QUALITY_COLUMNS = (
    'task_id', 'config_name', 'role_under_test', 'plan_quality',
    'invocation_error',
)
_PLAN_QUALITY_MEAN_COLUMNS = (
    # ``declined`` sits IMMEDIATELY AFTER ``no_plan`` (task 4760) because it is
    # the explanation OF that number — how many of the cells that scored zero
    # got there by taking an explicit plan-tools decline exit rather than by
    # failing to plan. Appending it at the far end would separate the count from
    # the one it qualifies, which is the whole defect this column removes.
    'config_name', 'n', 'cap_excluded', 'no_plan', 'declined',
    'judged_without_reference', 'plan_rate', 'mean_plan_quality',
)
# The plan_quality cell of a cap-tainted row. Deliberately NOT a number and NOT
# the bare '-' null sentinel (which already means "not an architect run"): a
# reader must be able to tell "this candidate scored nothing" from "we could not
# measure this candidate" at a glance.
#
# Spelled 'excluded', not 'cap-excluded' (reviewer: design-coherence): the
# ``cap_tainted`` flag covers every unmeasurable cause — cap hit, auth failure,
# model-not-found, server error (5xx), zero-output wedge, harness error — and a
# PERMANENT config error ("this candidate can never run") must not render as a
# TRANSIENT one ("rerun after the cap window"). The cause is always carried in
# the adjacent ``invocation_error`` column (the rendered ``invocation_errors``
# records) and broken out by cause in the summary line, so the marker itself
# stays cause-neutral rather than asserting a cap.
_CAP_EXCLUDED_CELL = 'excluded'
_PLAN_QUALITY_MEAN_HEADER = 'plan_quality by config:'


def _merge_counts(per_key: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Fold per-config ``{kind: count}`` tallies into ONE campaign-wide tally.

    THE reduction behind ``build_plan_quality_report``'s report-level
    ``declined_by_kind`` (task 4760), so that total is summed from the very
    counters the per-config entries are emitted from rather than re-derived from
    the rendered ``configs`` list — the :func:`mean_plan_quality` discipline
    applied to a histogram: one arithmetic, so the two surfaces cannot disagree
    about the same quantity.

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
    """Build the plan-quality column from a list of :class:`EvalResult` (θ).

    Each row carries ``task_id`` / ``config_name`` / ``role_under_test`` and the
    ``plan_quality`` pulled from ``EvalResult.metrics`` — a populated float for
    architect runs (``role_under_test`` ARCHITECT), ``None`` (the C4
    ``plan_quality | null`` sentinel) otherwise. Both are read directly from the
    persisted metrics dict; a result predating the θ fields (no ``plan_quality``
    / ``role_under_test`` key) reads back as ``None`` for each, so old reports
    render unchanged. Rows are sorted by ``(task_id, config_name)`` so the
    surface is deterministic.

    Rows additionally carry the task-3118 infra markers ``cap_tainted`` /
    ``invocation_errors`` (likewise default-safe: ``False`` / ``None`` for
    metrics predating them), and the report gains a per-config aggregate over
    ARCHITECT rows plus a report-level ``cap_excluded`` total.

    THE INVARIANT: a cap-tainted cell is an infra failure — we never got to ask
    the model — so it is EXCLUDED from ``mean_plan_quality`` and COUNTED as
    excluded, never averaged in as a zero. Averaging it in would penalise
    whichever candidate happened to be scheduled inside a cap window, which is a
    property of the schedule, not of the candidate. A config with no scored cells
    reports ``mean_plan_quality=None`` rather than ``0.0``, so "we measured
    nothing" can never read as "it scored nothing".

    ITS COUNTERPART (task 3302): a cell whose architect ran fine but produced NO
    PLAN (``not plan_predicates.produced_a_plan``) is the opposite case — a genuine
    content measurement — so it is FLOORED to ``0.0`` by
    :func:`plan_quality_score`, KEPT in the mean, and counted separately as
    ``no_plan``. Two causes, two treatments, two counts, neither silent:
    transport refusal → excluded + ``cap_excluded``; content failure → floored +
    ``no_plan``. Conflating them would either let the candidate that failed to
    plan escape the pool entirely or penalise the one that merely hit a cap.

    That ``no_plan`` count now also drives a per-config ``plan_rate`` (task
    3379) — the fraction of admitted cells that emitted a plan at all, reduced
    by THE shared :func:`plan_rate` that :func:`build_composite_report`'s row
    calls, over the SAME admitted ``n`` ``mean_plan_quality`` averages. So the
    reliability figure and the quality figure describe the same pool, and the
    two tables the CLI prints adjacently cannot disagree about either. No new
    per-ROW field: the aggregate reads counts it already holds.

    ``cap_excluded_by_cause`` breaks that total out by CAUSE
    (``{'cap_hit': 2, 'model_not_found': 1}``, key-sorted for determinism)
    because the causes are not interchangeable: a cap hit is transient and
    schedule-attributable ("rerun after the window"), whereas a model-not-found
    or auth failure is a PERMANENT, candidate-specific configuration error. A
    single "cap-excluded" total would let the latter masquerade as the former
    and hide a config that can never run behind ``n=0, mean=None``.

    A THIRD treatment, disjoint from both (eval-revival σ, task 3628):
    ``judged_without_reference`` counts the admitted cells whose score is the
    LLM judge's own number produced from an EMPTY reference diff — graded on
    plausibility, never against the landed change. Unlike the other two this
    changes NO number: the cell is KEPT AND SCORED, stays in ``n``, and its
    score stays in ``mean_plan_quality``. It is a VALIDITY bound, not an
    exclusion. Averaging these cells out would discard real measurements, while
    reporting them silently is the defect σ exists to remove — the v1 campaign
    judged half its hard subset this way and it was findable only by
    archaeology. So: transport refusal → EXCLUDED + ``cap_excluded``; content
    failure → FLOORED + ``no_plan``; judged blind → KEPT, SCORED, and FLAGGED.

    That third count is ``None``, never ``0``, whenever any admitted cell
    predates the marker (:func:`judged_without_reference`), with
    ``judged_without_reference_unmeasured`` carrying how many did — an
    unmeasured bound and a measured-clean one are opposite findings and must
    not render alike.

    A FOURTH treatment (task 4760), which — like the third — subtracts nothing
    and floors nothing, and unlike the third does not even bound a number:
    ``declined`` / ``declined_by_kind`` / ``no_plan_declined`` are EXPLANATORY.
    They move no cell between pools and change no existing figure; they say WHY
    the ``no_plan`` cells are there. ``no_plan`` has always answered HOW MANY
    architect cells emitted no plan and never WHY, and the two causes it
    conflates are opposite verdicts on the candidate: a genuine "could not
    plan" is a content failure, whereas ``report_false_premise`` and its four
    siblings are the architect CORRECTLY refusing an unplannable task. THE
    tranche-1 misreading is the case in point: 47 of 47 ``plan_steps = 0``
    architect cells were explicit, adversarially-verified-true declines, and the
    campaign readout could not say so — it reported an 89% planning failure that
    the transcripts flatly contradicted, and separating the two took manual
    forensics over destroyed-on-cleanup artifacts.

    So: transport refusal → EXCLUDED + ``cap_excluded``; content failure →
    FLOORED + ``no_plan``; judged blind → KEPT, SCORED, FLAGGED; terminal
    decline → KEPT, SCORED, UNCHANGED, and EXPLAINED. ``declined`` counts
    admitted cells whose ``terminal_kind`` is one of the five explicit decline
    exits — including a PLAN-THEN-DECLINE cell, which is in ``declined`` but not
    in ``no_plan``, so ``declined - no_plan_declined`` recovers that population
    exactly. ``declined_by_kind`` is key-sorted for the same reason
    ``cap_excluded_by_cause`` is: the kinds are not interchangeable.
    ``terminal_kind_unmeasured`` counts the admitted cells whose terminal
    statement was never resolved — read through ``terminal_kind_of``, so an
    ABSENT key (the cell predates the field, the only cause the live instrument
    can produce) and an explicit ``None`` count ALIKE — so a PARTIAL split never
    wears the appearance of a complete one.

    ``plan_rate`` and ``mean_plan_quality`` are DELIBERATELY untouched by all of
    this: planRate keeps its exact historical derivation (``plan_steps > 0``
    over the admitted pool) so every campaign artifact already committed stays
    comparable, and the split is reported ALONGSIDE it.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'role_under_test': result.metrics.get('role_under_test'),
            'plan_quality': result.metrics.get('plan_quality'),
            # The PLAN-PRODUCTION predicate's input (task 3302), carried
            # verbatim so the row is self-describing: a reader can see the step
            # count the adjacent plan_quality was — or was not — scored over.
            'plan_steps': result.metrics.get('plan_steps'),
            'cap_tainted': bool(result.metrics.get('cap_tainted')),
            'invocation_errors': result.metrics.get('invocation_errors'),
            # The σ validity marker (task 3628) — read through THE shared
            # accessor, NOT like cap_tainted above. cap_tainted's absence
            # genuinely means "no infra failure was recorded"; this field's
            # absence means the cell's validity was never measured at all, and
            # collapsing that to False is the one mistake that would make the
            # count reassuring instead of true. ``None`` is therefore a third
            # value here and travels all the way to the rendered '-'.
            'judged_without_reference': judged_without_reference(
                result.metrics
            ),
            # The TERMINAL STATEMENT this cell ended on (task 4760), read
            # through THE shared accessor rather than a bare ``.get`` for the
            # same reason ``judged_without_reference`` above is: one expression,
            # one contract, so the report layer and the campaign driver can
            # never disagree about what the field means. ``None`` — an
            # implementer cell (the field does not apply) or a cell persisted
            # before the field existed — travels through to the aggregate's
            # ``terminal_kind_unmeasured``, never collapsed to "did not
            # decline".
            'terminal_kind': terminal_kind_of(result.metrics),
            # The θ score this cell contributes to the aggregate, through THE
            # shared accessor (task 3302): the persisted float for a real plan,
            # a floored 0.0 for a cell that produced none, None when the cell is
            # excluded outright. Kept beside the RAW persisted ``plan_quality``
            # rather than replacing it — the per-row table reports what was
            # persisted, and an operator must still be able to see the judge's
            # number that the floor overrode.
            'scored_plan_quality': plan_quality_score(
                result.metrics,
                where=f'{result.task_id} x {result.config_name}',
            ),
            # The predicate's ANSWER, computed here from the metrics dict its
            # contract is written against — never re-derived downstream from the
            # row (reviewer: robustness). ``plan_steps`` above is carried for the
            # READER; asking ``produced_a_plan(row)`` instead would work only
            # while the row happens to spell that key identically to the metrics
            # dict, and a rename would silently answer False for EVERY row —
            # flooring every scored cell to 0.0 with no exception, no log and no
            # failing test. Sourced beside ``scored_plan_quality`` so the count
            # below and the score above can never describe different cells.
            'produced_a_plan': produced_a_plan(result.metrics),
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))

    # Per-config aggregate over ARCHITECT rows only — an implementer run never
    # invokes the plan judge, so its null score is a different thing entirely
    # and must not dilute the architect mean.
    scored: dict[str, list[float]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    no_plan: dict[str, int] = defaultdict(int)
    judged_blind: dict[str, int] = defaultdict(int)
    judged_blind_unmeasured: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    by_cause: dict[UnmeasuredCause, int] = defaultdict(int)
    # The task-4760 decline split. Four counters, all over the SAME admitted
    # pool ``scored`` / ``no_plan`` / ``judged_blind`` are over.
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
            # Subscript, not ``.get``: the answer was computed from the metrics
            # dict when the row was built, so a dropped key must raise here
            # rather than read back as "no config produced a plan".
            if not row['produced_a_plan']:
                # DISJOINT from the exclusion above: this cell was measured and
                # kept, at the 0.0 the accessor floored it to (task 3302).
                no_plan[cfg] += 1
            # Counted inside THIS branch on purpose (task 3628): the count is a
            # strict subset of the admitted pool ``n`` reports, never of the
            # excluded cells — a cap-tainted cell has no plan_quality to bound,
            # so it belongs to cap_excluded alone. The keyless case is tallied
            # separately rather than falling into the False arm: it is the pool
            # membership that is measured here, not the validity.
            if row['judged_without_reference'] is None:
                judged_blind_unmeasured[cfg] += 1
            elif row['judged_without_reference']:
                judged_blind[cfg] += 1
            # The decline split (task 4760), counted INSIDE this same branch and
            # for the same reason the two counts above are: it describes the
            # ADMITTED pool ``n`` reports. A cap-tainted cell may well carry a
            # stale terminal kind, but it belongs to ``cap_excluded`` alone —
            # counting it here would describe a cell the mean beside it never
            # saw. The UNMEASURED case is tallied separately rather than falling
            # into the "not a decline" arm: a cell whose terminal kind reads
            # ``None`` — key absent (it predates the field) or written null,
            # which ``terminal_kind_of`` deliberately does not distinguish —
            # never had its terminal statement resolved, and reading that as
            # "did not decline" is precisely the
            # partial-count-wearing-a-complete-count's-clothes defect this
            # split exists to remove.
            kind = row['terminal_kind']
            if kind is None:
                terminal_unmeasured[cfg] += 1
            elif kind in DECLINE_KINDS:
                declined[cfg] += 1
                declined_by_kind[cfg][kind] += 1
                # The INTERSECTION with ``no_plan`` — THE tranche-1 number.
                # Gated on the same predicate ``no_plan`` above is gated on, so
                # the two counts cannot describe different cells; a
                # plan-then-decline cell increments ``declined`` and NOT this.
                if not row['produced_a_plan']:
                    no_plan_declined[cfg] += 1
            # THE shared accessor's answer (task 3302), not the raw persisted
            # score: a cell whose architect produced no plan lands here as a
            # floored 0.0, identically to build_composite_report's pool, so
            # mean_plan_quality and the composite row cannot drift.
            scored[cfg].append(row['scored_plan_quality'])

    configs = [
        {
            'config_name': cfg,
            'n': len(scored[cfg]),
            'cap_excluded': excluded[cfg],
            # How many of those ``n`` scored cells produced NO plan and were
            # floored to 0.0 (task 3302). Disjoint from ``cap_excluded``: those
            # cells are not in ``n`` at all.
            'no_plan': no_plan[cfg],
            # WHY those cells are there (task 4760). ``declined`` counts the
            # admitted cells whose terminal statement was one of the five
            # explicit plan-tools decline exits — a CORRECT refusal, not a
            # planning failure — and subtracts nothing: every one of them is
            # still in ``n``, in ``plan_rate`` and in ``mean_plan_quality`` at
            # the score it earned.
            'declined': declined[cfg],
            # …broken out by KIND, key-sorted for determinism, exactly as
            # ``cap_excluded_by_cause`` is: 'already_done' and 'false_premise'
            # are opposite findings about the fixture and must not merge.
            'declined_by_kind': {
                kind: declined_by_kind[cfg][kind]
                for kind in sorted(declined_by_kind[cfg])
            },
            # The intersection of ``declined`` and ``no_plan`` — THE tranche-1
            # sentence ("47 of 47 no-plan cells were explicit declines"). A
            # plan-then-decline cell is in ``declined`` but NOT here, so
            # ``declined - no_plan_declined`` recovers that population exactly.
            'no_plan_declined': no_plan_declined[cfg],
            # How many of those same ``n`` cells carry NO resolved terminal
            # kind — in practice, cells that predate the field — so a PARTIAL
            # split is never read as a complete one. Its siblings above are
            # counts over the MEASURED sub-population only.
            'terminal_kind_unmeasured': terminal_unmeasured[cfg],
            # How many of those same ``n`` scored cells were judged WITHOUT a
            # reference diff (task 3628). Unlike the two counts above this one
            # subtracts nothing and floors nothing — the cells are in ``n`` and
            # in the mean at their real scores; only the confidence is bounded.
            # ``None`` when ANY of those cells predates the marker: the bound
            # is then unknown, and 0 would assert a validity nobody measured.
            'judged_without_reference': judged_blind_count(
                judged_blind[cfg], judged_blind_unmeasured[cfg],
            ),
            # …with the size of the unmeasured population beside it, so the
            # None above is diagnosable from the row alone.
            'judged_without_reference_unmeasured': judged_blind_unmeasured[cfg],
            'total': totals[cfg],
            # The RELIABILITY figure over the very same admitted pool ``n``
            # counts and ``mean_plan_quality`` averages (task 3379), through THE
            # shared :func:`plan_rate` build_composite_report's row also calls —
            # no re-derived arithmetic, so the two adjacent tables cannot drift.
            'plan_rate': plan_rate(len(scored[cfg]), no_plan[cfg]),
            # THE ONE reduction, shared with build_composite_report's row so the
            # two surfaces cannot drift (task 3099, :func:`mean_plan_quality`).
            'mean_plan_quality': mean_plan_quality(scored[cfg]),
        }
        for cfg in sorted(totals)
    ]
    return {
        'rows': rows,
        'configs': configs,
        'cap_excluded': sum(excluded.values()),
        'cap_excluded_by_cause': {c.value: by_cause[c] for c in sorted(by_cause)},
        # Report-level decline totals (task 4760), summed from the very same
        # per-config counters above so the two surfaces cannot drift. No
        # absence rule of their own: unlike ``judged_without_reference``, an
        # unmeasured cell here does not make the campaign-wide count UNKNOWN —
        # it makes it PARTIAL, and ``terminal_kind_unmeasured`` beside it says
        # by how much, which is the honest shape for a count whose
        # sub-population is nameable.
        'declined': sum(declined.values()),
        'declined_by_kind': {
            kind: total
            for kind, total in sorted(_merge_counts(declined_by_kind).items())
        },
        'no_plan_declined': sum(no_plan_declined.values()),
        'terminal_kind_unmeasured': sum(terminal_unmeasured.values()),
        # Report-level total, with the same absence rule as the per-config
        # counts (task 3628): one keyless admitted cell ANYWHERE makes the
        # campaign-wide bound unknown, so the total is None rather than a
        # partial sum wearing the appearance of a complete one.
        'judged_without_reference': judged_blind_count(
            sum(judged_blind.values()), sum(judged_blind_unmeasured.values()),
        ),
        'judged_without_reference_unmeasured': sum(
            judged_blind_unmeasured.values()
        ),
    }


def _format_plan_quality_mean_section(report: dict[str, Any]) -> list[str]:
    """Render the per-config mean block: how many cells actually scored.

    The point of the block is that ``mean_plan_quality`` is reported ALONGSIDE
    the ``n`` it was computed over, the ``cap_excluded`` count it left out and
    the ``no_plan`` count it scored as zeros, so a mean over 19 of 22 cells —
    four of which produced no plan at all — reads as exactly that instead of
    looking like a mean over 22 comparable plans. A config with nothing scored
    renders ``-``, never ``0.0000``.

    ``judged_without_reference`` (task 3628) is the same discipline applied to
    VALIDITY rather than to membership: a mean reported without saying how many
    of its cells were graded with no landed diff to grade against is the same
    silent degradation one layer up. Those cells ARE in ``n`` and in the mean.

    ``declined`` (task 4760) is the same discipline applied to CAUSE: it sits
    immediately after the ``no_plan`` it explains and says how many of those
    zero-scoring cells got there by an explicit plan-tools refusal rather than
    by failing to plan. It adjusts nothing — those cells are in ``n``, in
    ``plan_rate`` and in the mean exactly as before. A measured zero renders
    ``0``, not the ``-`` its neighbours use for "unmeasurable".
    """
    configs = report.get('configs', [])
    rendered = [
        {
            'config_name': str(c['config_name']),
            'n': str(c['n']),
            'cap_excluded': str(c['cap_excluded']),
            'no_plan': str(c['no_plan']),
            # How many of those n cells ended on an explicit decline exit (task
            # 4760). A measured ZERO renders '0', never the '-' sentinel its
            # neighbours use: on this surface '-' already means "not an
            # architect run" / "we could not measure this", and a config that
            # simply never declined is a MEASURED fact about it.
            'declined': str(c['declined']),
            # How many of those n cells the judge scored with NO reference diff
            # (task 3628). Rendered here rather than only in JSON because a
            # mean reported without saying how many of its cells were judged
            # blind is the same silent degradation one layer up. ``-`` — the
            # same sentinel the mean beside it uses — when the count is None
            # because some cell predates the marker; printing 0 there would be
            # the reassuring answer to a question nobody asked the instrument.
            'judged_without_reference': (
                '-' if c.get('judged_without_reference') is None
                else str(c['judged_without_reference'])
            ),
            # The same '-'-when-None rule the mean beside it uses: a config
            # whose every cell was refused has no rate, and a fabricated 0.0
            # would assert a reliability failure we never observed (task 3379).
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

    A ``plan_quality`` column shows the populated float (4 dp) for architect rows
    and ``-`` (the null sentinel) for non-architect rows; ``role_under_test``
    and ``invocation_error`` (the :func:`format_invocation_errors` rendering of
    the row's records) render ``-`` when ``None``. A CAP-TAINTED row shows the
    explicit ``excluded`` marker in place of a score, alongside the records
    that caused it — an infra failure must never render as
    ``0.0000``, and must stay distinguishable from the ``-`` a non-architect row
    uses.

    Sections follow the rows: the exclusion count — how many architect cells
    were not measurable, BROKEN OUT BY CAUSE so a permanent config error is not
    read as a transient cap window — the task-3628 validity bound, the task-4760
    DECLINE SPLIT (how many of the scored cells ended on an explicit plan-tools
    refusal, broken out by kind, and how much of the ``no_plan`` band that
    accounts for), and the per-config mean block. All are computed from the
    report, so the CLI caller picks them up with no change of its own. The same
    report always renders byte-identically (no wall-clock or dict-order
    dependence).
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
    # Broken out by CAUSE so a permanent config error (model_not_found /
    # auth_failed — "this candidate can never run") cannot masquerade as a
    # transient schedule artifact (cap_hit — "rerun after the window").
    by_cause = report.get('cap_excluded_by_cause') or {}
    causes = ', '.join(f'{cause}: {n}' for cause, n in sorted(by_cause.items()))
    lines.append('')
    lines.append(
        f'excluded: {report.get("cap_excluded", 0)} unmeasurable cell(s) of '
        f'{architect_cells} architect cell(s)'
        + (f' ({causes})' if causes else '')
    )
    # The σ validity bound beside the exclusion count (task 3628), stated
    # against ITS OWN pool: the SCORED cells, not every architect cell, because
    # that is the population whose plan_quality it bounds. Suppressed only when
    # the bound is MEASURED and zero — a clean campaign must not grow a scary
    # line — which is also what keeps this byte-deterministic.
    #
    # An UNMEASURED bound is emphatically not that case and must print (task
    # 3628 amendment): a rebuild over the pre-σ corpus has no measured cell at
    # all, and suppressing the line there would make "nothing bounded these
    # scores" look exactly like "everything was graded against a landed diff".
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
    # The decline split (task 4760), modelled on the ``cap_excluded_by_cause``
    # section above and stated against ITS OWN pool — the SCORED cells, which is
    # the population ``terminal_kind`` was resolved over. It explains the
    # ``no_plan`` count printed below rather than adjusting it: a decline is the
    # architect CORRECTLY refusing an unplannable task, and the tranche-1
    # readout could not say so.
    #
    # Suppressed only when the split is MEASURED and empty (no declines, nothing
    # unmeasured) — the judged-without-reference discipline: a campaign where
    # every candidate planned must not grow a line about refusals it never made,
    # and that is also what keeps this byte-deterministic.
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
    # An UNMEASURED population is emphatically NOT the suppressible case: a
    # rebuild over the pre-4760 corpus resolved no terminal kind at all, and a
    # silent '0 declines' there would read as "every candidate tried and
    # failed to plan" — the exact reading this split exists to refute.
    if terminal_unmeasured:
        lines.append(
            # "carry no resolved terminal kind", not "predate the field": the
            # count is over ``terminal_kind_of``'s answer, which reads an
            # absent key and an explicit null alike, so naming only the first
            # cause would state more than the number knows. Predating the
            # field IS the only cause the live instrument can produce — the
            # write is unconditional — but the sentence must stay true of the
            # count, not of today's producer.
            f'terminal kind: UNMEASURED — {terminal_unmeasured} of '
            f'{scored_cells} scored cell(s) carry no resolved terminal kind, '
            f'so the decline split is PARTIAL (not measured-clean)'
        )
    lines.append('')
    lines.extend(_format_plan_quality_mean_section(report))
    return '\n'.join(lines)

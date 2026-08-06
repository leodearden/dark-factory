#!/usr/bin/env python3
"""Campaign driver for the fable-architect trial v2 screen (β2).

PRD: ``plans/fable-architect-trial-v2-prd.md``, task β2, decision D10 — the v1
campaign ran off a GITIGNORED scratch script
(``data/eval-campaign/fable_architect_only.py``), so the numbers in the v1
verdict cannot be recomputed from anything the repo carries. This driver is the
committed replacement: it parameterizes that script's shape (candidate filter →
fixture glob → ``run_ofat_stage`` → plan-quality report → raw per-cell dump) and
adds the per-candidate accounting γ1's calibration report is COMPUTED from
rather than hand-derived.

PLACEMENT (answers PRD open question 4). All logic lives HERE, in
``scripts/``, with no new module inside ``orchestrator/evals/``. The
``run_<campaign>.py`` spelling matches the committed eval-driver siblings
``run_judge_ofat_pilot.py`` and ``run_vllm_eval.py``, but this file deliberately
does NOT copy their logic/CLI split: PRD D1 states this PRD's own tasks "edit
nothing in the instrument", so adding a module inside the instrument package
would cross the eval-framework-revival lane's single-ownership boundary. The
instrument (``run_ofat_stage``, ``build_plan_quality_report``,
``produced_a_plan``, ``get_config_by_name``) is CONSUMED UNMODIFIED.

TWO RULES A LATER READER MUST NOT UNDO
--------------------------------------
1. ``--q-ceiling`` is never defaulted. The plan-quality threshold that decides
   which fixtures are DISCARDED as ceiling-saturated is empirically anchored:
   derived in γ1 from v1 incumbent cells on validly-referenced fixtures,
   recorded provisional, and ratified or adjusted by Leo at γ2. A default here
   would silently become the de facto threshold and pre-empt that ruling (G6).
2. ``judged_without_reference`` reads UNMEASURED, never ``0``, whenever ANY
   architect cell lacks the key. Its producer is eval-revival σ (task 3628);
   until that lands, reporting ``0`` would let ``plan_quality`` read as fully
   validity-bounded when nothing bounded it. Mirrors ``report.py``'s own
   ``mean_plan_quality=None`` convention — "we measured nothing" must never read
   as "it scored nothing".
3. Reference validity is read PER CELL (``MARKER_KEY not in metrics`` -> not
   known-good), never from a run-level flag. The run-level
   :func:`marker_available` survives only as a DISPLAY flag on the bands dict
   and in the renderer legend. This is not a stylistic preference: answering the
   per-cell question with the run-level fact made one post-σ sibling cell
   silently discard a fixture whose validity was never measured — the one
   direction PRD D6 calls permanently lossy.

Additionally: ``--candidate`` is REQUIRED with no default, so this driver
carries no forward reference to ``architect-fable-max`` (eval-revival ρ, task
3627, unlanded) and needs no edit when ρ lands; and ``--tasks-dir`` exits
non-zero naming the path when absent rather than falling back to the standing
``evals/tasks`` corpus, which is the incumbent-success-biased pool the v2 screen
exists to escape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# --- self-locating import bootstrap -----------------------------------------
# scripts/ is not on sys.path when run standalone; add the sibling package srcs
# (idempotent — a no-op under pytest, where conftest already inserted them).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _rel in ('orchestrator/src', 'shared/src', 'escalation/src'):
    _p = str(_REPO_ROOT / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# β1's (task 3631) v2 hard-fixture pool. Absent until β1 lands — which is why
# :func:`resolve_fixture_paths` fails LOUDLY rather than falling back.
DEFAULT_TASKS_DIR = _REPO_ROOT / 'orchestrator' / 'src' / 'orchestrator' / 'evals' / 'tasks_hard_v2'


def resolve_fixture_paths(tasks_dir: Path) -> list[Path]:
    """Sorted ``*.json`` fixtures in *tasks_dir*, or a loud ``SystemExit``.

    NO CORPUS FALLBACK, deliberately. ``tasks_hard_v2/`` is β1's unlanded
    deliverable, so "the dir is not there yet" is the common early case and must
    be loud. Silently falling back to the standing ``evals/tasks`` corpus would
    run a several-hundred-dollar campaign over the WRONG fixture pool — the
    incumbent-success-biased set (PRD P1) the v2 screen exists to escape — and
    emit a plausible-looking report answering a different question.

    The glob is non-recursive and suffix-filtered, matching
    ``cli._load_fixture_dir``, so the two notions of "a fixture dir" agree.
    """
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise SystemExit(
            f'Error: fixture dir not found: {tasks_dir}\n'
            '  This driver never falls back to the standing evals/tasks corpus: that '
            'pool is the incumbent-success-biased set the v2 screen exists to escape, '
            'and running the campaign over it would spend real money answering a '
            'different question. Pass --tasks-dir explicitly, or wait for the v2 hard '
            'pool (fable-trial-v2 β1, task 3631) to land.'
        )
    paths = sorted(tasks_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'Error: no *.json eval fixtures found in {tasks_dir}')
    return paths


def enumerate_cells(
    fixture_paths: list[Path], candidates: list[str], trials: int,
) -> list[dict[str, Any]]:
    """The ``(fixture × candidate × trial)`` product this campaign will run.

    Ordered fixture-then-candidate-then-trial and derived only from its
    arguments, so the enumerated matrix is byte-stable across invocations and a
    committed γ1 artifact diffs cleanly. ``task_id`` is the fixture STEM, which
    is what ``run_architect_eval`` stamps onto each ``EvalResult``, so the
    expected matrix and the returned cells are directly comparable.
    """
    return [
        {'task_id': path.stem, 'config_name': name, 'trial': trial}
        for path in fixture_paths
        for name in candidates
        for trial in range(1, trials + 1)
    ]


def resolve_candidates(names: list[str]) -> list[Any]:
    """Resolve candidate NAMES to ``EvalConfig``s, rejecting any non-architect.

    Two loud failures, both structural rather than advisory:

    * an unresolvable name exits naming itself;
    * a RESOLVABLE config whose ``role != 'architect'`` exits too.

    The second is the load-bearing one. ``run_ofat_stage`` dispatches BY ROLE:
    an implementer candidate goes through ``run_eval``, a full agentic workflow
    cell at roughly 10x the cost of the plan-only ``run_architect_eval`` cell
    this campaign is made of. The PRD forbids ``eval-ofat`` for exactly that
    reason (it runs all 8 candidates including the implementer/judge cells), so
    a typo that happened to name a real implementer config would otherwise turn
    into hundreds of dollars of the wrong spend, silently and with no error.
    ``ofat_candidates()`` is never called anywhere in this driver.
    """
    from orchestrator.evals.configs import get_config_by_name

    resolved = []
    for name in names:
        cfg = get_config_by_name(name)
        if cfg is None:
            raise SystemExit(
                f'Error: unknown eval config: {name!r}\n'
                '  Architect candidates live in ARCHITECT_EVAL_CONFIGS '
                '(orchestrator/src/orchestrator/evals/configs.py).'
            )
        if cfg.role != 'architect':
            raise SystemExit(
                f'Error: candidate {name!r} has role={cfg.role!r}, not \'architect\'.\n'
                '  run_ofat_stage dispatches BY ROLE, so this candidate would be routed '
                'through run_eval — a full agentic workflow cell at roughly 10x the cost '
                'of the plan-only architect cell this campaign measures. That is the spend '
                'the PRD forbids when it says "Do NOT use eval-ofat". Name an '
                'ARCHITECT_EVAL_CONFIGS entry instead.'
            )
        resolved.append(cfg)
    return resolved


def _parse_budget_spec(spec: str) -> tuple[str, float]:
    """Parse a ``NAME=AMOUNT`` ``--budget`` spec, or exit naming the bad spec."""
    name, sep, raw = spec.partition('=')
    if not sep or not name:
        raise SystemExit(
            f'Error: malformed --budget spec {spec!r}: expected NAME=AMOUNT '
            '(e.g. architect-opus-max=15).'
        )
    try:
        amount = float(raw)
    except ValueError as e:
        raise SystemExit(
            f'Error: malformed --budget spec {spec!r}: {raw!r} is not a number.'
        ) from e
    return name, amount


def apply_budgets(candidates: list[Any], budgets: dict[str, float]) -> list[Any]:
    """Return *candidates* with ``max_budget_usd`` overridden per *budgets*.

    ``dataclasses.replace``, never in-place mutation: the module-level
    ``ARCHITECT_EVAL_CONFIGS`` entries are shared with every other eval caller
    and must stay byte-unchanged, which is the parity discipline the revival
    lane enforces. ``run_architect_eval`` binds ``max_budget_usd`` straight off
    the config, so γ2's comparison-regime ruling — equal-cost $15/$15,
    equal-turns with fable's budget lifted, or both arms — is a pure per-config
    parameter choice needing no runner surgery.

    A budget naming a candidate that is NOT being run exits non-zero: silently
    ignoring it would run an arm at the wrong price and invalidate the very
    comparison the ruling rests on.
    """
    import dataclasses

    selected = {c.name for c in candidates}
    unknown = sorted(set(budgets) - selected)
    if unknown:
        raise SystemExit(
            f'Error: --budget names candidate(s) not being run: {", ".join(unknown)}\n'
            f'  Selected candidates: {", ".join(sorted(selected))}. A silently-ignored '
            'budget would run a comparison arm at the wrong price.'
        )
    return [
        dataclasses.replace(c, max_budget_usd=budgets[c.name]) if c.name in budgets else c
        for c in candidates
    ]


# The per-candidate fields copied through from the report layer's accumulator.
# Named once so the summary and the renderer cannot disagree about the schema.
_SUMMARY_FIELDS = (
    'config_name', 'n', 'total', 'cap_excluded', 'no_plan', 'plan_rate',
    'mean_plan_quality',
)

# The per-cell reference-validity marker. Producer is eval-revival σ (task
# 3628); see :func:`marker_available` for why its ABSENCE is a measurable fact
# rather than a guess.
MARKER_KEY = 'judged_without_reference'

# How many architect cells carried no marker at all — the companion that makes a
# PARTIALLY measured corpus legible instead of merely unknown.
UNMEASURED_CELLS_KEY = 'judged_without_reference_unmeasured_cells'

# What an unmeasured marker renders as. Deliberately a WORD, never ``0`` and
# never the ``-`` report.py uses for an empty mean: this one has to survive
# being skimmed.
UNMEASURED = 'unmeasured'


def marker_available(results: list[Any]) -> bool:
    """True iff ANY cell carries :data:`MARKER_KEY` — i.e. the instrument emits it.

    DISPLAY ONLY. This answers exactly one question — "does this instrument emit
    the marker at all?" — for the renderer's NOTE line and the bands artifact's
    schema. It must NEVER be used to decide a PER-CELL question. It once was,
    and in a mixed corpus that made a keyless cell score as though its reference
    had been verified good, discarding a fixture that was never measured.
    Per-cell validity is read off each cell: ``MARKER_KEY not in metrics``.

    The test is EXACT, not heuristic. ``EvalMetrics.to_dict()`` is an
    ``asdict``, so it emits every DECLARED field regardless of value — which
    means a cell whose reference WAS valid still carries the key, set ``False``.
    "Key absent on every cell" therefore has exactly one cause: the instrument
    predates eval-revival σ (task 3628) and never measured the question at all.
    """
    return any(MARKER_KEY in (r.metrics or {}) for r in results)


def _architect_cells(results: list[Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """``(task_id, config_name, metrics)`` for architect cells only.

    THE ROLE FILTER RUNS FIRST, everywhere — and it is single-sourced HERE so
    that "everywhere" is checkable rather than aspirational. An implementer run
    never invokes the plan judge, so it is out of scope by construction; and if
    the filter ran after the marker check instead, a single implementer cell
    riding along under the same config name (carrying no marker key, because it
    never could) would erase a candidate's genuine, complete measurement.

    ``task_id`` rides along so :func:`partition_bands` consumes this same filter
    rather than re-implementing the ``role_under_test`` test inline. Two copies
    of a filter documented as load-bearing is one copy too many: a change to what
    counts as an architect cell would otherwise have to be made twice, with only
    one of the sites carrying the argument for why it matters.
    """
    cells = []
    for result in results:
        metrics = result.metrics or {}
        if metrics.get('role_under_test') != 'architect':
            continue
        cells.append((result.task_id, result.config_name, metrics))
    return cells


def count_judged_without_reference(
    results: list[Any], unmeasured: dict[str, int] | None = None,
) -> dict[str, int | None]:
    """Per candidate: how many architect cells were judged without a valid reference.

    ``None`` — never ``0`` — for any candidate with even ONE architect cell
    lacking :data:`MARKER_KEY`. The distinction is the point: ``0`` asserts that
    we looked at every cell and found none, which would let ``plan_quality``
    read as fully validity-bounded when in fact nothing bounded it. ``None``
    says we never measured. This mirrors ``report.py``'s own
    ``mean_plan_quality=None`` convention, where "we measured nothing" is
    likewise kept distinct from "it scored nothing".

    PARTIAL MEASUREMENT IS NOT MEASUREMENT, and this is decided PER CELL. An
    earlier version short-circuited on the run-level :func:`marker_available`,
    so in a mixed corpus a keyless cell silently scored as ``False`` and the
    candidate reported a fabricated ``0``. One unmeasured cell means the
    candidate's bound is unknown, so the honest value is ``None``. Use
    :func:`count_unmeasured_marker_cells` to see how much of it was unmeasured.

    No edit is needed here when σ lands. The key simply starts appearing on each
    cell and these ``None``s become real counts.

    *unmeasured* accepts an already-computed
    :func:`count_unmeasured_marker_cells` map so a caller needing both (as
    :func:`summarize_candidates` does) scans the cells once instead of three
    times. It is an optimisation only: omitting it computes the same map here.
    """
    counts: dict[str, int | None] = {}
    if unmeasured is None:
        unmeasured = count_unmeasured_marker_cells(results)
    for _task_id, name, metrics in _architect_cells(results):
        if unmeasured.get(name):
            counts[name] = None
            continue
        counts[name] = (counts.get(name) or 0) + (1 if metrics.get(MARKER_KEY) else 0)
    return counts


def count_unmeasured_marker_cells(results: list[Any]) -> dict[str, int]:
    """Per candidate: how many architect cells carry NO :data:`MARKER_KEY` at all.

    ``None`` from :func:`count_judged_without_reference` cannot distinguish "1
    of 50 cells unmeasured" from "50 of 50" — a huge difference to an operator
    reading γ1's artifact. This count makes a partially-instrumented corpus
    LEGIBLE rather than merely unknown, which is the loud-over-silent norm
    applied to the transition state.

    That state is normal, not exotic: γ1's recipe re-runs cap-tainted fixtures,
    and a re-run after eval-revival σ (task 3628) lands writes cells carrying
    the key alongside older ones that never could.
    """
    counts: dict[str, int] = {}
    for _task_id, name, metrics in _architect_cells(results):
        counts[name] = counts.get(name, 0) + (0 if MARKER_KEY in metrics else 1)
    return counts


def summarize_candidates(results: list[Any]) -> list[dict[str, Any]]:
    """Per-candidate rows, SURFACED from the report layer — never recomputed here.

    THE LOAD-BEARING RULE: none of these counts are derived in this driver.
    ``cap_excluded``, ``no_plan`` and ``plan_rate`` are
    :func:`build_plan_quality_report`'s per-config accumulator (tasks
    3118/3302/3379) and ``mean_plan_quality`` is its shared reduction. Both
    reductions were deliberately made SHARED with ``build_composite_report``'s
    row so the two tables the CLI prints adjacently cannot give contradictory
    answers about the same quantity. Re-deriving any of them here would create
    exactly that second free-to-disagree surface — so this function copies the
    fields through verbatim and adds nothing arithmetic of its own.

    Why ``plan_rate`` is trustworthy on the no-plan band specifically: it is
    JUDGE-FREE by construction, computed from :func:`produced_a_plan` over the
    PERSISTED ``plan_steps``, so it stays valid on exactly the fixtures where no
    reference diff exists and ``plan_quality`` therefore cannot be interpreted.

    Rows are sorted by ``config_name`` so a committed report artifact diffs
    cleanly.
    """
    from orchestrator.evals.report import build_plan_quality_report

    report = build_plan_quality_report(results)
    # Scanned ONCE and handed to both consumers: count_judged_without_reference
    # needs exactly this map to decide unmeasured-vs-counted, and the renderer
    # needs it to say "unmeasured (N of M cells)".
    unmeasured_counts = count_unmeasured_marker_cells(results)
    marker_counts = count_judged_without_reference(results, unmeasured_counts)
    rows = []
    for entry in report['configs']:
        row = {field: entry[field] for field in _SUMMARY_FIELDS}
        name = entry['config_name']
        row[MARKER_KEY] = marker_counts.get(name)
        # How much of the above was never measured. Carried alongside rather
        # than folded in, because `None` alone cannot distinguish 1-of-50
        # unmeasured cells from 50-of-50.
        row[UNMEASURED_CELLS_KEY] = unmeasured_counts.get(name, 0)
        rows.append(row)
    rows.sort(key=lambda r: r['config_name'])
    return rows


# The raw per-cell dump. v1's discipline (task_id / config_name / outcome /
# trial / plan_quality / plan_steps / cost_usd) extended with judge_cost_usd and
# cap_tainted, so the verdict is recomputable without re-parsing a log.
_CELL_FIELDS = ('plan_quality', 'plan_steps', 'cost_usd', 'judge_cost_usd', 'cap_tainted')


def find_missing_cells(
    results: list[Any], cell_matrix: list[dict[str, Any]], candidates: list[Any],
) -> dict[str, Any]:
    """Diff the ENUMERATED matrix against the cells that actually came back.

    WHY THE MATRIX IS NOT ENOUGH ON ITS OWN. ``cell_matrix`` is carried in the
    report so a cell that never returned "is visible as a gap rather than
    silently absent" — but visibility requires something to actually perform the
    comparison, and nothing did. ``run_ofat_stage`` delegates to
    ``_bounded_fanout``, which LOGS a non-cancel cell failure and CONTINUES: the
    failed cell is simply absent from the returned list.
    ``build_plan_quality_report`` then derives its ``configs`` from the cells
    that returned, so a candidate whose every cell failed (an auth error or a
    model-not-found on one arm, say) produces NO ROW AT ALL. The report shows one
    arm, the table prints one row, and ``main`` returns 0 — a vanished comparison
    arm being the single gap that most invalidates a screen whose entire thesis
    is that γ1's calibration is COMPUTED rather than hand-derived. Partial
    failures are equally invisible: ``n``/``total`` just read low, with nothing
    saying how low they should have been.

    ``candidates_absent`` is called out separately from the cell list because it
    is a categorically worse failure: a partial shortfall weakens a comparison,
    an absent arm means there is NO comparison, and :func:`main` exits non-zero
    on it after writing the artifact so the gap is both signalled and recorded.
    """
    returned = {(r.task_id, r.config_name, r.trial) for r in results}
    missing = [
        cell for cell in cell_matrix
        if (cell['task_id'], cell['config_name'], cell['trial']) not in returned
    ]
    missing.sort(key=lambda c: (c['task_id'], c['config_name'], c['trial']))
    present_configs = {r.config_name for r in results}
    return {
        'count': len(missing),
        'expected': len(cell_matrix),
        'cells': missing,
        'candidates_absent': sorted(
            c.name for c in candidates if c.name not in present_configs
        ),
    }


def build_campaign_report(
    results: list[Any],
    candidates: list[Any],
    cell_matrix: list[dict[str, Any]],
    q_ceiling: float | None = None,
    expect_cells: bool = False,
) -> dict[str, Any]:
    """Assemble the campaign report: candidates, raw cells, matrix, optional bands.

    Every per-cell field is read with ``.get`` off the metrics dict — the
    ``build_plan_quality_report`` convention — so a result persisted before a
    field existed loads rather than raising.

    ``bands`` is OMITTED, not emptied, when *q_ceiling* is None: an empty band
    dict would read as "we banded the pool and found nothing", which is a
    different claim from "banding was not requested". ``missing_cells`` follows
    the SAME rule under *expect_cells*: on a dry run nothing was dispatched, so
    "every enumerated cell is missing" is not a finding, and reporting it would
    be as misleading as suppressing a real one. Callers that actually ran (or
    re-analyzed) cells pass ``expect_cells=True``.

    EVERY ROW MUST TRACE TO A RESOLVED ``EvalConfig``. A cell naming a config
    that is not among *candidates* RAISES rather than emitting a row with null
    model/effort/budget. This is defense in depth behind
    :func:`filter_campaign_results`: ``getattr(None, 'model', None)`` silently
    yielding ``None`` is the exact mechanism that turned a wrong-pool load into
    a plausible-looking table, and a null-config row is a silent degradation
    rather than a tolerable gap.
    """
    config_by_name = {c.name: c for c in candidates}
    rows = []
    for row in summarize_candidates(results):
        name = row['config_name']
        cfg = config_by_name.get(name)
        if cfg is None:
            raise SystemExit(
                f'Error: result cell names config {name!r}, which is not one of this '
                f'campaign\'s candidates: {", ".join(sorted(config_by_name)) or "(none)"}.\n'
                '  Every reported row must trace to a resolved EvalConfig — emitting one '
                'with a null model/effort/budget would present an out-of-campaign cell as '
                'a plausible result. If this came from --results-dir, the dir holds cells '
                'from another campaign; if from --run, the runner dispatched a config that '
                'was never requested.'
            )
        rows.append({
            **row,
            'model': cfg.model,
            'effort': cfg.effort,
            'max_budget_usd': cfg.max_budget_usd,
        })

    cells = [
        {
            'task_id': r.task_id,
            'config_name': r.config_name,
            'trial': r.trial,
            'outcome': r.outcome,
            **{field: (r.metrics or {}).get(field) for field in _CELL_FIELDS},
        }
        for r in results
    ]
    cells.sort(key=lambda c: (c['task_id'], c['config_name'], c['trial']))

    report: dict[str, Any] = {
        'candidates': rows,
        'cells': cells,
        'cell_matrix': cell_matrix,
    }
    if expect_cells:
        report['missing_cells'] = find_missing_cells(results, cell_matrix, candidates)
    if q_ceiling is not None:
        report['bands'] = partition_bands(results, q_ceiling)
    return report


def _fmt(value: Any) -> str:
    """Render one summary value: ``unmeasured`` for ``None``, else the number.

    ``None`` NEVER renders as ``0``. Both ``mean_plan_quality`` and
    :data:`MARKER_KEY` use ``None`` for "not measured", and the whole reason
    those fields are nullable is that a zero would be read as a measurement.
    """
    if value is None:
        return UNMEASURED
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def format_campaign_report(report: dict[str, Any]) -> str:
    """Render the campaign report as deterministic text.

    No timestamps and no dict-iteration-order dependence: the same report
    formats byte-identically every time, so a committed γ1 artifact diffs
    cleanly.
    """
    lines = ['per-candidate summary:']
    header = (
        f'{"candidate":<26} {"n":>4} {"total":>6} {"cap_excl":>9} {"no_plan":>8} '
        f'{"plan_rate":>10} {"mean_pq":>10} {MARKER_KEY:>26}'
    )
    lines.append(header)
    lines.append('-' * len(header))
    unmeasured_marker = False
    for row in report.get('candidates', []):
        marker = _fmt(row.get(MARKER_KEY))
        if row.get(MARKER_KEY) is None:
            unmeasured_marker = True
            # A PARTIALLY measured corpus reads as "unmeasured (3 of 12 cells)"
            # rather than a bare word, so an operator can tell a fully pre-σ run
            # from one re-run cell short of a complete bound.
            missing = row.get(UNMEASURED_CELLS_KEY)
            if missing:
                marker = f'{UNMEASURED} ({missing} of {row["total"]} cells)'
        lines.append(
            f'{row["config_name"]:<26} {_fmt(row["n"]):>4} {_fmt(row["total"]):>6} '
            f'{_fmt(row["cap_excluded"]):>9} {_fmt(row["no_plan"]):>8} '
            f'{_fmt(row["plan_rate"]):>10} {_fmt(row["mean_plan_quality"]):>10} '
            f'{marker:>26}'
        )
    missing = report.get('missing_cells')
    if missing and missing['count']:
        # Every missing cell is listed, never truncated to a top-N: a silent cap
        # would let a partial listing read as the complete gap.
        lines += [
            '',
            f'MISSING CELLS: {missing["count"]} of {missing["expected"]} enumerated '
            'cells never returned.',
            '  A cell that fails is LOGGED AND SKIPPED by the runner\'s fan-out, so it '
            'is absent from',
            '  the results rather than reported — which makes n/total read low with no '
            'indication of',
            '  how low they should have been. Each gap below is (fixture, candidate, '
            'trial):',
        ]
        lines += [
            f'    {cell["task_id"]} / {cell["config_name"]} / trial {cell["trial"]}'
            for cell in missing['cells']
        ]
        if missing['candidates_absent']:
            absent = ', '.join(missing['candidates_absent'])
            lines += [
                f'  *** ENTIRE CANDIDATE ARM ABSENT: {absent} ***',
                '  Not one cell returned for the arm(s) above, so they carry NO row in '
                'the table and',
                '  this campaign has no comparison to make. This is what an auth error '
                'or a',
                '  model-not-found on a single arm looks like. Exiting non-zero.',
            ]

    bands = report.get('bands')
    if bands:
        lines += ['', f'PRD-D6 banding (q_ceiling={bands["q_ceiling"]}):']
        for band in BANDS:
            lines.append(f'  {band:<14} {bands["counts"][band]}')
        lines += [
            f'  retained  ({len(bands["retained"])}): {", ".join(bands["retained"]) or "-"}',
            f'  discarded ({len(bands["discarded"])}): {", ".join(bands["discarded"]) or "-"}',
        ]
        if not bands['marker_available']:
            lines.append(
                '  NOTE: reference validity is UNMEASURED, so the ceiling band is '
                'unsatisfiable and nothing can be discarded (D6: ambiguity -> retain).'
            )

    if unmeasured_marker:
        lines += [
            '',
            f'LEGEND — {MARKER_KEY} = {UNMEASURED}: at least one architect cell carries '
            'no',
            '  reference-validity marker, so NO bound was placed on how far plan_quality '
            'may be',
            '  trusted for that candidate. It is NOT a count of zero. The producer is '
            'eval-revival σ',
            '  (task 3628); once it lands the key appears on every cell and these read '
            'as real counts.',
            f'  "{UNMEASURED} (N of M cells)" is the MIXED case: only N cells lacked the '
            'marker. That is',
            '  the normal transition state — a γ1 re-run of cap-tainted fixtures after σ '
            'lands writes',
            '  keyed cells alongside older keyless ones — not an exotic one. Partial '
            'measurement is',
            '  not measurement, so the bound stays unknown until every cell carries the '
            'key; and each',
            '  keyless cell bands intermittent (RETAINED) on its own, never ceiling.',
        ]
    return '\n'.join(lines)


BANDS = ('ceiling', 'intermittent', 'no_plan', 'unmeasured')

# ``ceiling`` is the ONLY discarded band. Everything else is retained.
RETAINED_BANDS = tuple(b for b in BANDS if b != 'ceiling')

# Which band wins when one fixture has SEVERAL admitted cells — most-retaining
# first, so ``ceiling`` is last and a fixture is discarded only when EVERY cell
# says ceiling. Among the retained bands the order is a labelling choice with
# one consequence: γ1's recipe re-runs ``unmeasured`` fixtures to get an
# admissible cell, so a fixture that already has a genuine measurement must not
# be labelled ``unmeasured`` and re-run pointlessly. ``no_plan`` leads because a
# candidate that could not plan at all is the strongest evidence of headroom
# this pool is being selected for.
_BAND_PRECEDENCE = ('no_plan', 'intermittent', 'unmeasured', 'ceiling')


def band_for_cell(metrics: dict[str, Any], q_ceiling: float) -> str:
    """Band ONE cell per PRD D6, reading every fact off THAT CELL's metrics.

    Precedence, and why each rung sits where it does:

    1. ``cap_tainted`` -> ``unmeasured``. A transport refusal: we never got to
       ask the model. γ1's recipe re-runs these so every fixture gets one
       admissible cell, so the driver NAMES them rather than banding on a
       refusal — banding one would penalise whichever candidate happened to be
       scheduled inside a session-cap window, a property of the schedule.
    2. ``not produced_a_plan`` -> ``no_plan``. THE plan-production predicate
       (``metrics.py:191``), used directly and never re-implemented, and never
       replaced by ``plan_quality > 0``: the two plan scorers disagreed exactly
       on a stepless artifact, so a nonzero score is not evidence a plan exists.
    3. reference validity not KNOWN-GOOD -> ``intermittent``. Either THIS CELL
       carries no :data:`MARKER_KEY` (it predates σ, so its validity was never
       measured) or it is marked as judged without a reference. The two cases
       are deliberately indistinguishable here: both mean "we cannot interpret
       this cell's ``plan_quality``", and ``plan_quality`` is meaningful only
       where a real reference block exists.
    4. ``plan_quality`` missing or below *q_ceiling* -> ``intermittent``.
    5. otherwise -> ``ceiling``, the sole DISCARDED band.

    PER CELL, NOT PER RUN — the rule a later reader must not undo. This function
    once took a run-level ``marker_available`` third argument, which answered
    the per-cell question "was THIS cell's plan_quality bounded by a real
    reference diff?" with the run-level fact "does ANY cell carry the key?". In
    a MIXED corpus those diverge, and the failure was silent and in the
    forbidden direction: a keyless cell alongside one post-σ sibling scored as
    if its reference had been verified good, flipping a never-measured fixture
    from RETAINED to DISCARDED. The parameter is GONE rather than ignored,
    because an argument that looks load-bearing but is not is exactly the trap
    that produced that bug.

    Rungs 3-4 encode D6's asymmetry: ambiguity resolves to RETAIN, because
    misbanding-to-retain costs ~$20 of stage-2 spend while misbanding-to-discard
    loses the signal permanently. Concretely, on a pre-σ cell rung 3 always
    fires, so such a fixture can never be discarded — the conservative
    direction, by construction rather than by care.
    """
    from orchestrator.evals.metrics import produced_a_plan

    if metrics.get('cap_tainted'):
        return 'unmeasured'
    if not produced_a_plan(metrics):
        return 'no_plan'
    if MARKER_KEY not in metrics or metrics.get(MARKER_KEY):
        return 'intermittent'
    quality = metrics.get('plan_quality')
    if quality is None or quality < q_ceiling:
        return 'intermittent'
    return 'ceiling'


def partition_bands(results: list[Any], q_ceiling: float) -> dict[str, Any]:
    """Band every fixture in the pool and split it into retained vs discarded.

    Stage 1 is one trial per fixture per candidate, but a re-run (or a second
    arm) can leave a fixture with several admitted cells. Those collapse via
    :data:`_BAND_PRECEDENCE`, most-retaining first, so a fixture is discarded
    only when EVERY one of its cells says ``ceiling`` — the D6 ambiguity rule
    holding at the fixture level as well as the cell level.

    ``retained`` and ``discarded`` partition the pool EXACTLY: disjoint, and
    their union is every fixture that produced a cell. ``counts`` always carries
    all four bands, zeros included, so the artifact's schema does not shift with
    its contents.

    The returned ``marker_available`` is a run-level DISPLAY flag ONLY: it
    answers "does this instrument emit the marker at all", which the renderer's
    NOTE line reports and the artifact's schema keeps stable. It must NEVER
    again be consulted to decide a per-cell question — doing exactly that is
    what let one post-σ sibling discard a never-measured fixture. Banding reads
    :data:`MARKER_KEY` off each cell, via :func:`band_for_cell`.

    The architect-only filter comes from :func:`_architect_cells` rather than
    being repeated inline, so the rule that it must run first has exactly one
    implementation.
    """
    ranked: dict[str, int] = {}
    for task_id, _config_name, metrics in _architect_cells(results):
        band = band_for_cell(metrics, q_ceiling)
        rank = _BAND_PRECEDENCE.index(band)
        ranked[task_id] = min(rank, ranked.get(task_id, rank))

    by_fixture = {task_id: _BAND_PRECEDENCE[rank] for task_id, rank in sorted(ranked.items())}
    counts = {band: 0 for band in BANDS}
    for band in by_fixture.values():
        counts[band] += 1
    return {
        'q_ceiling': q_ceiling,
        'marker_available': marker_available(results),
        'by_fixture': by_fixture,
        'counts': counts,
        'retained': sorted(t for t, b in by_fixture.items() if b in RETAINED_BANDS),
        'discarded': sorted(t for t, b in by_fixture.items() if b == 'ceiling'),
    }


def _run_campaign(
    fixture_paths: list[Path],
    candidates: list[Any],
    *,
    trials: int,
    max_parallel: int | None,
    config_path: Path | None,
    timeout: int | None,
) -> list[Any]:
    """Drive the campaign LIVE over ``run_ofat_stage`` and persist every cell.

    THE single real-API-spend seam, and deliberately not unit-tested end to end
    — the ``run_judge_ofat_pilot.py`` precedent. Its WIRING is covered instead by
    tests that monkeypatch ``run_ofat_stage`` / ``save_result`` / ``load_config``
    and assert exactly what this function hands over.

    ``run_ofat_stage``'s architect branch already routes every ``role ==
    'architect'`` candidate to ``run_architect_eval`` and handles trials,
    ``max_parallel`` and per-cell failure isolation, so this driver only
    assembles the arguments and fans out. Consumed UNMODIFIED — PRD D1 forbids
    instrument edits from this lane. ``ofat_candidates()`` is never called: the
    candidate list is exactly what :func:`resolve_candidates` returned, every
    entry already checked to be an architect.

    ``timeout`` passes through as ``timeout_override``, whose unit is MINUTES —
    verified against the consumer rather than assumed: ``run_architect_eval``
    does ``timeout_minutes = timeout_override or task.get('timeout_minutes', 60)``
    and then ``timeout=timeout_minutes * 60`` (``runner.py:598-647``). A unit
    mismatch here would silently make every cell's bound 60x wrong in one
    direction or the other.
    """
    import asyncio

    from orchestrator.config import ConfigRequiredError, load_config
    from orchestrator.evals.runner import run_ofat_stage, save_result

    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        raise SystemExit(f'Error: {e} (pass --config or set ORCH_CONFIG_PATH)') from e

    results = asyncio.run(run_ofat_stage(
        fixture_paths, candidates, base_config,
        max_parallel=max_parallel, trials=trials, timeout_override=timeout,
    ))
    for result in results:
        save_result(result)
    return results


def filter_campaign_results(
    results: list[Any], candidate_names: set[str], fixture_stems: set[str],
) -> tuple[list[Any], list[Any]]:
    """Split loaded cells into ``(kept, dropped)`` by THIS campaign's scope.

    A cell is kept iff its ``config_name`` is one of *candidate_names* AND its
    ``task_id`` is one of *fixture_stems*.

    WHY THIS IS MANDATORY, not defensive tidiness. ``save_result`` persists into
    the SHARED packaged ``orchestrator/src/orchestrator/evals/results/``
    (``runner.py:50``), alongside every eval that has ever run — 586 cells in the
    main checkout today, 51 of them v1 ``architect-fable-high`` runs over the
    incumbent-success-biased standing corpus that the v2 screen exists to
    escape. The documented round-trip (``--run`` persists, a later
    ``--results-dir`` re-reads) points a re-analysis straight at that pool. Left
    unfiltered it silently answers a DIFFERENT question over the WRONG fixture
    set — the same failure :func:`resolve_fixture_paths`'s loud no-fallback rule
    prevents on the run path, re-entering through the analyze path.

    BOTH AXES MATTER INDEPENDENTLY. Filtering only on candidate name would let
    every v1 cell for a reused candidate through, since ``architect-fable-high``
    ran in v1 too; a right-candidate/wrong-fixture cell is precisely the
    contamination case, and it is caught on the fixture axis alone.
    """
    kept, dropped = [], []
    for result in results:
        in_scope = (
            result.config_name in candidate_names and result.task_id in fixture_stems
        )
        (kept if in_scope else dropped).append(result)
    return kept, dropped


def _load_results_from_dir(results_dir: Path) -> list[Any]:
    """Load every persisted ``EvalResult`` JSON in *results_dir*.

    Filters each loaded dict to the known dataclass fields, mirroring
    ``runner.load_results``, so a cell persisted before a field existed still
    loads instead of raising on an unexpected keyword.

    TWO LOUD FAILURES, both because this path is aimed by default at the SHARED
    packaged results dir holding cells written by many campaigns over time:

    * A MISSING DIR exits naming it. ``Path.glob`` returns EMPTY for a
      nonexistent dir rather than raising, so without this check a plain typo
      falls through to :func:`_load_campaign_results`'s zero-survivors branch and
      gets diagnosed as contamination ("the dir predates this campaign or belongs
      to another one") when the real problem is the path. Mirrors
      :func:`resolve_fixture_paths`, which already applies exactly this rule to
      the other input path.
    * A MALFORMED FILE exits NAMING THE FILE. One truncated or partial write
      among hundreds of cells would otherwise take down the whole re-analysis
      with a bare ``JSONDecodeError`` traceback naming neither the offending file
      nor the dir — unactionable. Failing loudly rather than skipping is
      deliberate: a silently skipped cell shifts every count in the report, and
      this driver's whole thesis is that γ1's calibration is COMPUTED from a
      pool whose membership is known exactly.

    KNOWN DUPLICATION (deliberate, recorded rather than hidden): the load-and-
    filter body is a near-verbatim sibling of ``scripts/run_judge_ofat_pilot.py``
    ``_load_results_from_dir``, which does NOT carry the hardening above. The two
    copies can now drift. Extracting a shared ``scripts/`` helper would mean
    editing that sibling script, which is outside this task's locked module set,
    so it is filed as a follow-up instead of done here.
    """
    from orchestrator.evals.runner import EvalResult

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit(
            f'Error: results dir not found (or is not a directory): {results_dir}\n'
            '  This is a path problem, not a contamination problem: Path.glob returns '
            'empty for a nonexistent dir rather than raising, so without this check the '
            'typo would be reported downstream as "no in-campaign cells found".'
        )
    known = {f.name for f in EvalResult.__dataclass_fields__.values()}
    results = []
    for path in sorted(results_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            results.append(EvalResult(**{k: v for k, v in data.items() if k in known}))
        except (ValueError, OSError, TypeError, AttributeError) as e:
            raise SystemExit(
                f'Error: could not load persisted eval cell {path}: '
                f'{type(e).__name__}: {e}\n'
                '  Expected a JSON object shaped like EvalResult.to_dict(). A truncated '
                'or partial write in the shared packaged results dir is the usual cause. '
                'The file is named rather than skipped: dropping a cell silently would '
                'shift every count in the report.'
            ) from e
    return results


def _load_campaign_results(
    results_dir: Path, candidates: list[Any], fixture_paths: list[Path],
) -> tuple[list[Any], dict[str, Any]]:
    """Load persisted cells, scope them to THIS campaign, and account for the drops.

    LOUD, NEVER SILENT. Filtering silently would be nearly as bad as not
    filtering: the committed γ1 artifact must carry its own exclusion record so
    the calibration is auditable, and an operator who pointed at the wrong
    directory must be told rather than handed a plausible report.

    Zero survivors EXITS rather than rendering an empty-pool report, because "we
    measured nothing" and "you pointed at the wrong directory" must not share a
    shape — and the latter is by far the likelier cause, given the shared
    packaged results dir this path is aimed at by default.
    """
    loaded = _load_results_from_dir(results_dir)
    candidate_names = {c.name for c in candidates}
    fixture_stems = {p.stem for p in fixture_paths}
    kept, dropped = filter_campaign_results(loaded, candidate_names, fixture_stems)

    filtered = {
        'dropped': len(dropped),
        'dropped_config_names': sorted({r.config_name for r in dropped}),
        'dropped_task_ids': sorted({r.task_id for r in dropped}),
    }
    if dropped:
        print(
            f'dropped {len(dropped)} out-of-campaign cell(s) loaded from {results_dir}: '
            f'config(s) {", ".join(filtered["dropped_config_names"])}; '
            f'fixture(s) {", ".join(filtered["dropped_task_ids"])}'
        )
    if not kept:
        raise SystemExit(
            f'Error: no in-campaign cells found in {results_dir} '
            f'({len(loaded)} loaded, all {len(dropped)} out of campaign).\n'
            f'  Selected candidates: {", ".join(sorted(candidate_names))}\n'
            f'  Fixture pool: {", ".join(sorted(fixture_stems))}\n'
            '  This usually means the dir predates this campaign or belongs to another '
            'one — note that save_result writes into the SHARED packaged evals/results/ '
            'dir alongside every eval that has ever run. Reporting over an empty pool '
            'would give "we measured nothing" the same shape as a real result.'
        )
    return kept, filtered


def _positive_int(raw: str) -> int:
    """An ``int >= 1`` argparse type, or a loud parse-time rejection.

    Every degenerate value here is a SILENT no-op rather than an error, which is
    why it is rejected at the flag instead of discovered in the artifact:

    * ``--trials 0`` makes :func:`enumerate_cells` return ``[]``, so
      ``run_ofat_stage`` builds zero thunks and ``_bounded_fanout``
      short-circuits — a "campaign" that dispatched nothing, printed an empty
      table and exited 0. A negative value behaves identically.
    * ``--max-parallel 0`` constructs ``asyncio.Semaphore(0)`` and DEADLOCKS the
      fan-out: no cell can ever acquire a slot.
    * ``--timeout 0`` would cap every cell at zero minutes.

    argparse renders the flag name alongside this message, so the failure names
    both the flag and the offending value.
    """
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'{raw!r} is not an integer') from e
    if value < 1:
        raise argparse.ArgumentTypeError(
            f'must be >= 1, got {value} — a zero or negative value here is a silent '
            'no-op (zero cells dispatched, or a zero-slot semaphore that deadlocks '
            'the fan-out), not a smaller campaign'
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run_fable_trial_v2_campaign',
        description='fable-architect trial v2 (β2): drive the architect OFAT screen over '
                    'the v2 hard pool and report per-candidate planRate, plan_quality, '
                    'cap_excluded and judged_without_reference.',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--dry-run', action='store_true',
        help='Enumerate the cell matrix and the comparison regime. Zero spend.',
    )
    mode.add_argument(
        '--run', action='store_true',
        help='Drive the live campaign (run_ofat_stage), persist each cell, then report.',
    )
    mode.add_argument(
        '--results-dir', type=Path, default=None,
        help='Re-analyze persisted EvalResult JSONs from this dir (no live run).',
    )
    parser.add_argument(
        '--tasks-dir', type=Path, default=DEFAULT_TASKS_DIR,
        help=f'Fixture dir for the campaign (default: {DEFAULT_TASKS_DIR}).',
    )
    parser.add_argument(
        '--candidate', action='append', required=True, metavar='NAME',
        help='Architect candidate config name; repeat for each arm. REQUIRED — there is '
             'no default candidate list, so this driver carries no forward reference to '
             'an unlanded config and the comparison arms stay auditable on the CLI.',
    )
    parser.add_argument(
        '--budget', action='append', default=None, metavar='NAME=AMOUNT',
        help='Override a candidate\'s max_budget_usd; repeat per candidate. This is how '
             'the γ2 comparison-regime ruling is applied (equal-cost, equal-turns, or '
             'both arms) — a pure parameter choice, no runner surgery.',
    )
    parser.add_argument(
        '--stage1', action='store_true',
        help='Compute the PRD-D6 banding partition (retained vs ceiling-discarded). '
             'Requires --q-ceiling.',
    )
    parser.add_argument(
        '--q-ceiling', type=float, default=None, metavar='FLOAT',
        help='plan_quality at or above which a validly-referenced planned fixture is '
             'banded ceiling and DISCARDED. No default, deliberately — see --stage1.',
    )
    parser.add_argument('--trials', type=_positive_int, default=1,
                        help='Trials per (fixture, candidate) cell (>= 1).')
    parser.add_argument('--max-parallel', type=_positive_int, default=None,
                        help='Max concurrent eval cells (>= 1).')
    parser.add_argument('--config', dest='config_path', type=Path, default=None,
                        help='Orchestrator config YAML for --run.')
    parser.add_argument('--timeout', type=_positive_int, default=None,
                        help='Per-cell timeout override (minutes) for --run.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Write the campaign report as JSON to this path.')
    return parser


def _format_dry_run(
    fixture_paths: list[Path], candidates: list[Any], trials: int, cells: list[dict[str, Any]],
) -> str:
    """Render the zero-spend matrix preview AND the comparison regime.

    The per-candidate ``model / effort / max_budget_usd`` line is the point: the
    regime γ2 rules on is auditable on the terminal before a single dollar is
    spent, rather than reconstructed from a log afterwards.
    """
    lines = [
        f'fable-trial-v2 campaign (DRY RUN — no spend): {len(candidates)} candidates '
        f'x {len(fixture_paths)} fixtures x {trials} trials = {len(cells)} cells',
    ]
    for cfg in candidates:
        lines.append(
            f'  candidate: {cfg.name} (model={cfg.model} effort={cfg.effort} '
            f'max_budget_usd={cfg.max_budget_usd})'
        )
    for path in fixture_paths:
        lines.append(f'  fixture:   {path.stem}')
    return '\n'.join(lines)


def _resolve_q_ceiling(args: argparse.Namespace) -> float | None:
    """The banding threshold, or a loud exit — NEVER a guessed default (G6).

    PRD D6 makes Q_ceiling empirically anchored: derived in γ1 from v1 incumbent
    cells on validly-referenced fixtures, recorded provisional, then ratified or
    adjusted by Leo at γ2. A default baked in here would silently become the de
    facto threshold and pre-empt that ruling — and it decides which fixtures are
    DISCARDED, the one direction D6 calls permanently lossy.

    THREE rejections, because every way of asking for a banding that cannot be
    computed must fail at the flag rather than in the artifact:

    1. ``--stage1`` with ``--dry-run``: a dry run measures nothing, so banding an
       empty result set would emit a fully-populated all-zero ``bands`` block —
       exactly the reading :func:`build_campaign_report` omits ``bands`` to
       prevent ("we banded the pool and found nothing" is a different claim from
       "banding was not requested"), reintroduced through the one mode where by
       construction nothing was measured.
    2. ``--stage1`` without ``--q-ceiling``: no guessed threshold (above).
    3. ``--q-ceiling`` without ``--stage1``: the threshold would be SILENTLY
       IGNORED — no bands computed, no warning, a bands-free report — leaving an
       operator who typed the empirically-anchored number no way to know why. The
       reverse direction is already loud, and the asymmetry is what makes the
       silent one dangerous.
    """
    if args.stage1 and args.dry_run:
        raise SystemExit(
            'Error: --stage1 cannot be combined with --dry-run.\n'
            '  A dry run enumerates the cell matrix and spends nothing, so there are no '
            'measured cells to band: every band would read 0 over an empty pool, and '
            'marker_available would report False having inspected nothing. That is a '
            'fabricated measurement, not a preview. Run --run or --results-dir to band.'
        )
    if args.stage1 and args.q_ceiling is None:
        raise SystemExit(
            'Error: --stage1 requires --q-ceiling, which has no default.\n'
            '  PRD D6: the plan-quality ceiling is empirically anchored — derived in γ1 '
            'from v1 incumbent cells on validly-referenced fixtures, recorded '
            'provisional, and ratified or adjusted by Leo at γ2. It decides which '
            'fixtures are DISCARDED, and misbanding-to-discard loses signal '
            'permanently, so it is never guessed here.'
        )
    if args.q_ceiling is not None and not args.stage1:
        raise SystemExit(
            f'Error: --q-ceiling {args.q_ceiling} was given without --stage1.\n'
            '  Nothing would be banded and the threshold would be silently ignored, '
            'handing back a bands-free report with no indication why. Pass --stage1 to '
            'compute the PRD-D6 partition, or drop --q-ceiling.'
        )
    return args.q_ceiling if args.stage1 else None


def _write_report(report: dict[str, Any], out: Path | None) -> None:
    """Write the report JSON to *out*, creating parent dirs. No-op when unset.

    ``sort_keys`` so the artifact is byte-stable for the same report, which is
    what lets a committed γ1 result diff cleanly against its successor.
    """
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'campaign report written to {out}')


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Validated FIRST, before fixture resolution and long before any spend: a
    # campaign that would end in an un-computable banding must not be launched.
    q_ceiling = _resolve_q_ceiling(args)
    fixture_paths = resolve_fixture_paths(args.tasks_dir)
    candidates = resolve_candidates(list(args.candidate))
    budgets = dict(_parse_budget_spec(s) for s in (args.budget or []))
    candidates = apply_budgets(candidates, budgets)
    cell_matrix = enumerate_cells(fixture_paths, [c.name for c in candidates], args.trials)

    # Three mutually-exclusive modes, all converging on the SAME report builder
    # and renderer, so a dry-run preview, a live campaign and a re-analysis of
    # persisted cells cannot describe the run in three different schemas.
    filtered: dict[str, Any] | None = None
    if args.dry_run:
        print(_format_dry_run(fixture_paths, candidates, args.trials, cell_matrix))
        results: list[Any] = []
    elif args.run:
        # DELIBERATELY UNFILTERED. These cells are in-campaign by construction,
        # so an unexpected one means the runner dispatched something never
        # requested — a real bug that must surface through the unknown-config
        # guard, not be silently dropped behind a clean-looking report.
        results = _run_campaign(
            fixture_paths, candidates,
            trials=args.trials, max_parallel=args.max_parallel,
            config_path=args.config_path, timeout=args.timeout,
        )
    else:
        results, filtered = _load_campaign_results(
            args.results_dir, candidates, fixture_paths,
        )

    report = build_campaign_report(
        results, candidates, cell_matrix,
        q_ceiling=q_ceiling, expect_cells=not args.dry_run,
    )
    if filtered is not None:
        report['filtered'] = filtered
    if not args.dry_run:
        print(format_campaign_report(report))
    _write_report(report, args.out)

    # A vanished comparison arm exits NON-ZERO, but only AFTER the report is
    # printed and the artifact written: the diagnosis is the most valuable thing
    # this run produced, so it must survive the failure signal rather than be
    # replaced by it. A partial shortfall stays exit 0 with a loud banner — it
    # weakens the comparison; an absent arm means there is no comparison.
    absent = (report.get('missing_cells') or {}).get('candidates_absent')
    if absent:
        # Flushed before the stderr line so "see MISSING CELLS above" stays true
        # under a pipe, where stdout is block-buffered and would otherwise land
        # after it — i.e. in exactly the CI log where the pointer matters most.
        sys.stdout.flush()
        print(
            f'ERROR: no cells returned for candidate(s): {", ".join(absent)}. The '
            'campaign produced no comparison; see MISSING CELLS above and the '
            "report's missing_cells block.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

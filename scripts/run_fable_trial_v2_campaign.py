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
2. ``judged_without_reference`` reads UNMEASURED, never ``0``, when no cell
   carries the key. Its producer is eval-revival σ (task 3628); until that
   lands, reporting ``0`` would let ``plan_quality`` read as fully
   validity-bounded when nothing bounded it. Mirrors ``report.py``'s own
   ``mean_plan_quality=None`` convention — "we measured nothing" must never read
   as "it scored nothing".

Additionally: ``--candidate`` is REQUIRED with no default, so this driver
carries no forward reference to ``architect-fable-max`` (eval-revival ρ, task
3627, unlanded) and needs no edit when ρ lands; and ``--tasks-dir`` exits
non-zero naming the path when absent rather than falling back to the standing
``evals/tasks`` corpus, which is the incumbent-success-biased pool the v2 screen
exists to escape.
"""

from __future__ import annotations

import argparse
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

# What an unmeasured marker renders as. Deliberately a WORD, never ``0`` and
# never the ``-`` report.py uses for an empty mean: this one has to survive
# being skimmed.
UNMEASURED = 'unmeasured'


def marker_available(results: list[Any]) -> bool:
    """True iff ANY cell carries :data:`MARKER_KEY` — i.e. the instrument emits it.

    This test is EXACT, not heuristic. ``EvalMetrics.to_dict()`` is an
    ``asdict``, so it emits every DECLARED field regardless of value — which
    means a cell whose reference WAS valid still carries the key, set ``False``.
    "Key absent on every cell" therefore has exactly one cause: the instrument
    predates eval-revival σ (task 3628) and never measured the question at all.
    """
    return any(MARKER_KEY in (r.metrics or {}) for r in results)


def count_judged_without_reference(results: list[Any]) -> dict[str, int | None]:
    """Per candidate: how many architect cells were judged without a valid reference.

    ``None`` for EVERY candidate when the marker is unavailable — never ``0``.
    The distinction is the point: ``0`` asserts that we looked and found none,
    which would let ``plan_quality`` read as fully validity-bounded when in fact
    nothing bounded it. ``None`` says we never measured. This mirrors
    ``report.py``'s own ``mean_plan_quality=None`` convention, where "we
    measured nothing" is likewise kept distinct from "it scored nothing".

    ARCHITECT CELLS ONLY, matching ``build_plan_quality_report``'s aggregate: an
    implementer run never invokes the plan judge, so it cannot have been judged
    without a reference, and counting one would inflate the very number that
    bounds how far ``plan_quality`` may be trusted.

    No edit is needed here when σ lands. The key simply starts appearing on each
    cell and these ``None``s become real counts.
    """
    available = marker_available(results)
    counts: dict[str, int | None] = {}
    for result in results:
        metrics = result.metrics or {}
        if metrics.get('role_under_test') != 'architect':
            continue
        name = result.config_name
        if not available:
            counts[name] = None
            continue
        counts[name] = (counts.get(name) or 0) + (1 if metrics.get(MARKER_KEY) else 0)
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
    marker_counts = count_judged_without_reference(results)
    rows = []
    for entry in report['configs']:
        row = {field: entry[field] for field in _SUMMARY_FIELDS}
        row[MARKER_KEY] = marker_counts.get(entry['config_name'])
        rows.append(row)
    rows.sort(key=lambda r: r['config_name'])
    return rows


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
        if row.get(MARKER_KEY) is None:
            unmeasured_marker = True
        lines.append(
            f'{row["config_name"]:<26} {_fmt(row["n"]):>4} {_fmt(row["total"]):>6} '
            f'{_fmt(row["cap_excluded"]):>9} {_fmt(row["no_plan"]):>8} '
            f'{_fmt(row["plan_rate"]):>10} {_fmt(row["mean_plan_quality"]):>10} '
            f'{_fmt(row.get(MARKER_KEY)):>26}'
        )
    if unmeasured_marker:
        lines += [
            '',
            f'LEGEND — {MARKER_KEY} = {UNMEASURED}: this instrument does not emit the '
            'marker,',
            '  so NO bound was placed on how far plan_quality may be trusted. It is NOT '
            'a count of',
            '  zero. The producer is eval-revival σ (task 3628); once it lands the key '
            'appears on',
            '  every cell and these read as real counts. Until then, treat every '
            'plan_quality here',
            '  as unvalidated against a reference diff.',
        ]
    return '\n'.join(lines)


def _run_campaign(*args: Any, **kwargs: Any) -> list[Any]:
    """The LIVE real-API-spend seam — wired in a later step of this task.

    Named and referenced from ``main`` already so ``--dry-run``'s
    never-reach-the-live-path guarantee is testable by monkeypatching THIS
    attribute rather than a heavyweight transitive import.
    """
    raise NotImplementedError('_run_campaign is wired by the --run step of task 3632')


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
    parser.add_argument('--trials', type=int, default=1, help='Trials per (fixture, candidate) cell.')
    parser.add_argument('--max-parallel', type=int, default=None, help='Max concurrent eval cells.')
    parser.add_argument('--config', dest='config_path', type=Path, default=None,
                        help='Orchestrator config YAML for --run.')
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    fixture_paths = resolve_fixture_paths(args.tasks_dir)
    candidates = resolve_candidates(list(args.candidate))
    budgets = dict(_parse_budget_spec(s) for s in (args.budget or []))
    candidates = apply_budgets(candidates, budgets)
    cells = enumerate_cells(fixture_paths, [c.name for c in candidates], args.trials)

    if args.dry_run:
        print(_format_dry_run(fixture_paths, candidates, args.trials, cells))
        return 0

    raise NotImplementedError('--run / --results-dir are wired by a later step of task 3632')


if __name__ == '__main__':
    sys.exit(main())

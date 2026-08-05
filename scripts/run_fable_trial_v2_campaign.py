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
    parser.add_argument('--trials', type=int, default=1, help='Trials per (fixture, candidate) cell.')
    parser.add_argument('--max-parallel', type=int, default=None, help='Max concurrent eval cells.')
    parser.add_argument('--config', dest='config_path', type=Path, default=None,
                        help='Orchestrator config YAML for --run.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Write the campaign report as JSON to this path.')
    return parser


def _format_dry_run(
    fixture_paths: list[Path], candidates: list[str], trials: int, cells: list[dict[str, Any]],
) -> str:
    """Render the zero-spend matrix preview."""
    lines = [
        f'fable-trial-v2 campaign (DRY RUN — no spend): {len(candidates)} candidates '
        f'x {len(fixture_paths)} fixtures x {trials} trials = {len(cells)} cells',
    ]
    for name in candidates:
        lines.append(f'  candidate: {name}')
    for path in fixture_paths:
        lines.append(f'  fixture:   {path.stem}')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    fixture_paths = resolve_fixture_paths(args.tasks_dir)
    candidate_names: list[str] = list(args.candidate)
    cells = enumerate_cells(fixture_paths, candidate_names, args.trials)

    if args.dry_run:
        print(_format_dry_run(fixture_paths, candidate_names, args.trials, cells))
        return 0

    raise NotImplementedError('--run / --results-dir are wired by a later step of task 3632')


if __name__ == '__main__':
    sys.exit(main())

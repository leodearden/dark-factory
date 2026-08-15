#!/usr/bin/env python3
"""Operator CLI for the Routing κ judge-model OFAT pilot (task 2815).

Thin wiring over ``orchestrator.evals.judge_pilot`` (all decision/render logic
lives there; this file only parses args, drives I/O, and maps the verdict to a
process exit code). Two modes:

* ``--analyze-only`` — load persisted judge-OFAT ``EvalResult`` JSONs (from
  ``--results-dir`` or the packaged ``evals/results`` dir), analyze, write the
  markdown report to ``--out``, and exit ``0`` iff the verdict is ``adopt``.
* ``--run`` — invoke the eval-revival ο OFAT seam
  (``run_ofat_stage(task_paths, JUDGE_EVAL_CONFIGS, trials=N)``) over the fixtures
  to PRODUCE results (a live, hours-long, real-API-spend run of a Sonnet-max
  implementer × fixtures × trials — NOT unit-tested by design), persist them, then
  fall through to the same analysis.

Exit codes gate the decide-and-act flip: ``0`` adopt · ``1`` marginal · ``2``
reject · ``2`` on any other verdict. On a clear ``adopt`` an operator/CI applies
``models.judge=haiku`` per ``plans/judge-ofat-pilot-runbook.md``; otherwise keep
sonnet and escalate with the generated report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- self-locating import bootstrap -----------------------------------------
# scripts/ is not on sys.path when run standalone; add the sibling package srcs
# (idempotent — a no-op under pytest, where conftest already inserted them).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _rel in ('orchestrator/src', 'shared/src', 'escalation/src'):
    _p = str(_REPO_ROOT / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)
# scripts/ itself: implicit on sys.path[0] when run standalone (Python adds the
# main script's own dir), and explicit via scripts/tests/conftest.py under
# pytest — but not guaranteed when this module is loaded via
# importlib.util.spec_from_file_location (orchestrator/tests/test_judge_ofat_pilot.py),
# so make it explicit here too (idempotent).
_p = str(_REPO_ROOT / 'scripts')
if _p not in sys.path:
    sys.path.insert(0, _p)

from _eval_results_io import load_results_from_dir  # noqa: E402
from orchestrator.evals.judge_pilot import (  # noqa: E402
    CANDIDATE_JUDGE,
    DEFAULT_NON_INFERIORITY_MARGIN,
    INCUMBENT_JUDGE,
    analyze_judge_ofat,
    format_judge_ofat_report,
)

# adopt → 0 so `run_judge_ofat_pilot.py --analyze-only && apply-flip` gates
# cleanly; every non-adopt verdict is a distinct non-zero code for triage.
_EXIT_BY_VERDICT = {'adopt': 0, 'marginal': 1, 'reject': 2}


def _load_results(results_dir: Path | None) -> list:
    """Persisted results from *results_dir*, or the packaged ``RESULTS_DIR``.

    The ``--results-dir`` branch is :func:`_eval_results_io.load_results_from_dir`
    (task 3743) — shared with ``scripts/run_fable_trial_v2_campaign.py`` so the
    missing-dir / malformed-file hardening lands once for both drivers, both of
    which read the same shared packaged results dir by default.
    """
    if results_dir is not None:
        return load_results_from_dir(results_dir)
    from orchestrator.evals.runner import load_results

    return load_results()


def _run_pilot(
    *,
    trials: int,
    tasks_dir: Path | None,
    config_path: Path | None,
    max_parallel: int | None,
    timeout: int | None,
) -> list:
    """Drive the ο judge-OFAT seam LIVE and persist the results.

    Loads the base orchestrator config, resolves the fixture corpus, and fans
    ``run_ofat_stage`` over ``JUDGE_EVAL_CONFIGS`` × fixtures × trials — the ο seam
    consumed UNMODIFIED (decision 11). Each result is persisted via
    ``save_result`` so a later ``--analyze-only`` can re-read it. Not unit-tested:
    it spawns full agentic workflows with real API spend.
    """
    import asyncio

    from orchestrator.config import ConfigRequiredError, load_config
    from orchestrator.evals.configs import JUDGE_EVAL_CONFIGS
    from orchestrator.evals.runner import run_ofat_stage, save_result

    try:
        base_config = load_config(config_path)
    except ConfigRequiredError as e:
        raise SystemExit(f'Error: {e} (pass --config or set ORCH_CONFIG_PATH)') from e

    from orchestrator.evals import runner as _runner

    packaged_tasks = Path(_runner.__file__).parent / 'tasks'
    resolved_tasks_dir = tasks_dir or packaged_tasks
    if not resolved_tasks_dir.exists():
        resolved_tasks_dir = Path(base_config.project_root) / 'orchestrator' / 'evals' / 'tasks'
    task_paths = sorted(Path(resolved_tasks_dir).glob('*.json'))
    if not task_paths:
        raise SystemExit(f'No eval fixtures found in {resolved_tasks_dir}')

    results = asyncio.run(run_ofat_stage(
        task_paths, JUDGE_EVAL_CONFIGS, base_config,
        max_parallel=max_parallel, trials=trials, timeout_override=timeout,
    ))
    for r in results:
        save_result(r)
    return results


def _default_out() -> Path:
    from orchestrator.evals.runner import RESULTS_DIR

    return RESULTS_DIR / 'judge_ofat_pilot_report.md'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run_judge_ofat_pilot',
        description='Routing κ judge-model OFAT pilot: run and/or analyze the '
                    'judge-haiku-vs-judge-sonnet OFAT and decide adoption.',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--run', action='store_true',
        help='Run the live judge OFAT (run_ofat_stage over JUDGE_EVAL_CONFIGS) then analyze.',
    )
    mode.add_argument(
        '--analyze-only', action='store_true',
        help='Analyze already-persisted judge-OFAT results (no live run).',
    )
    parser.add_argument(
        '--results-dir', type=Path, default=None,
        help='Dir of persisted EvalResult JSONs (default: packaged evals/results).',
    )
    parser.add_argument(
        '--out', type=Path, default=None,
        help='Where to write the markdown report (default: evals/results/judge_ofat_pilot_report.md).',
    )
    parser.add_argument(
        '--margin', type=float, default=DEFAULT_NON_INFERIORITY_MARGIN,
        help=f'Non-inferiority margin in composite points (default: {DEFAULT_NON_INFERIORITY_MARGIN}).',
    )
    parser.add_argument('--incumbent', default=INCUMBENT_JUDGE, help='Incumbent judge config name.')
    parser.add_argument('--candidate', default=CANDIDATE_JUDGE, help='Candidate judge config name.')
    parser.add_argument('--trials', type=int, default=3, help='Trials per (fixture, candidate) cell for --run.')
    parser.add_argument('--tasks-dir', type=Path, default=None, help='Fixture dir for --run.')
    parser.add_argument('--config', dest='config_path', type=Path, default=None, help='Orchestrator config YAML for --run.')
    parser.add_argument('--max-parallel', type=int, default=None, help='Max concurrent eval runs for --run.')
    parser.add_argument('--timeout', type=int, default=None, help='Per-run timeout (minutes) for --run.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.run:
        results = _run_pilot(
            trials=args.trials, tasks_dir=args.tasks_dir, config_path=args.config_path,
            max_parallel=args.max_parallel, timeout=args.timeout,
        )
    else:
        results = _load_results(args.results_dir)

    decision, composite_report = analyze_judge_ofat(
        results, incumbent=args.incumbent, candidate=args.candidate, margin=args.margin,
    )
    report_text = format_judge_ofat_report(decision, composite_report)

    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)

    print(f'verdict: {decision.verdict} (escalate: {decision.escalate})')
    for reason in decision.reasons:
        print(f'  - {reason}')
    print(f'report written to {out_path}')
    if decision.verdict == 'adopt':
        print('RECOMMENDATION: adopt — flip models.judge: sonnet -> haiku (see report + runbook)')
    else:
        print('RECOMMENDATION: keep models.judge=sonnet and escalate with the report')

    return _EXIT_BY_VERDICT.get(decision.verdict, 2)


if __name__ == '__main__':
    sys.exit(main())

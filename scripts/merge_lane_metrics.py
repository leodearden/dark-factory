#!/usr/bin/env python3
"""Command line for the merge-lane quality ratchet: report, check, or write the baseline.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. It fronts two modules:
``scripts/merge_lane_cluster.py`` takes the measurement, and
``scripts/merge_lane_ratchet.py`` judges it against the committed baseline and
writes the baseline and its authorized-raise ledger.

Exit codes
----------
0  clean -- measures taken, and (under ``--check``) no ratchet violation.
1  ratchet violations -- one or more measures rose above the baseline, or a new
   file/function exceeded a ceiling. Violations are printed to stderr. This
   covers BOTH faces of the gate: ``--check`` finding a raise, and
   ``--write-baseline`` REFUSING to absorb one for want of an
   ``--authorize-raise`` (``ratchet.UnauthorizedRaise``, which deliberately does
   not subclass ``MetricsError`` so it cannot be reclassified as a broken tool).
2  instrument failure (``source_measures.MetricsError``, or its subclasses
   ``lane_cluster.ClusterSpecError`` for a cluster path it cannot read and
   ``ratchet.RatchetError`` for the ratchet's own faults) -- an unparseable cluster file,
   complexipy missing or out of range, a missing/malformed baseline, or a
   baseline whose recorded parameters no longer match this tree. Deliberately
   distinct from 1 so a broken instrument is never mistaken either for a clean
   tree or for a real regression.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import merge_lane_cluster as lane_cluster
import merge_lane_ratchet as ratchet
import source_measures

# ---------------------------------------------------------------------------
# CLI, in the house shape of scripts/scan_task_toolcall_leaks.py: _build_parser()
# / _render_table() / main(argv) -> int, with the exit ladder documented in the
# module docstring above (0 clean, 1 ratchet violations, 2 instrument failure).


def _render_table(report: dict, root: Path) -> str:
    """The human-readable measure table printed by ``--report``.

    ``mi`` (radon's maintainability index) is computed HERE rather than stored in
    the report: it is reported, never ratcheted, so putting it in the baseline
    would freeze a number nothing enforces and churn the file whenever radon
    changed its formula.
    """
    files = report.get('files', {})
    totals = ratchet.derive_totals(report)
    params = report.get('params', {})
    width = max([len(p) for p in files] + [len('TOTALS')])

    lines = [
        f'merge-lane metrics -- {len(files)} cluster paths, '
        f'complexipy {params.get("complexipy_version")}',
        '',
        f'{"path":<{width}}  {"lines":>7}  {"prose":>7}  {"cognitive":>9}  {"mi":>6}',
        '-' * (width + 36),
    ]
    for path, entry in sorted(files.items()):
        source = source_measures.read_source(root, path)
        mi = source_measures.maintainability_index(source, path=path)
        lines.append(
            f'{path:<{width}}  {entry["lines"]:>7}  {entry["prose_lines"]:>7}  '
            f'{entry["cognitive"]:>9}  {mi:>6.2f}'
        )
    lines.extend(
        [
            '-' * (width + 36),
            f'{"TOTALS":<{width}}  {totals["lines"]:>7}  {totals["prose_lines"]:>7}  '
            f'{totals["cognitive"]:>9}',
            '',
            f'function_local_imports  {totals["function_local_imports"]:>7}',
            f'reexport_names          {totals["reexport_names"]:>7}',
            '',
            f'test suite -- {len(report.get("tests", {}))} lane-importing files '
            f'under {lane_cluster.TESTS_ROOT}',
            f'  patch_targets (distinct)  {totals["patch_targets"]:>7}',
            f'  private_reads             {totals["private_reads"]:>7}',
            '',
        ]
    )

    # INV-11's user-observable signal: completeness is legible in the RESULT,
    # not only in a log line. A partial sweep measures LOW, so a reader who
    # cannot see this row cannot tell an improvement from a skipped file.
    #
    # THE TEST-TREE DENOMINATOR IS ONLY HERE. It is reported, never ratcheted --
    # the same call this function already makes for `mi` -- so the committed
    # baseline carries no test-tree number at all and this row is the one place
    # the sweep's breadth is visible. Its NUMERATOR is deliberately absent: that
    # is ``len(report['tests'])``, already rendered under a better label by the
    # "test suite --" line above, and two copies of one number in one table only
    # invite a reader to wonder whether they can disagree (SPOT). `.get` is
    # tolerant on purpose: a stored baseline block has no `test_tree` key, and
    # rendering one must not raise.
    enumeration = report.get('enumeration', {})
    unreadable = list(enumeration.get('unreadable', ()))
    test_tree = enumeration.get('test_tree', {})
    lines.append(
        f'enumeration: complete={enumeration.get("complete")}  '
        f'cluster requested={len(enumeration.get("requested", ()))} '
        f'resolved={len(enumeration.get("resolved", ()))}  '
        f'test tree requested={test_tree.get("requested")}  '
        f'unreadable={len(unreadable)}'
    )
    if unreadable:
        lines.extend(f'  UNREADABLE: {path}' for path in unreadable)
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Measure the merge-lane cluster (PRD plans/merge-lane-quality-prd.md '
            'task alpha) and ratchet its structural measures against the '
            'committed baseline. READ-ONLY except under --write-baseline.'
        ),
        epilog=(
            'Exit codes: 0 clean; 1 ratchet violations (a measure rose above '
            'the baseline, or a new file/function breached a ceiling); '
            '2 instrument failure (an unparseable cluster file, complexipy '
            f'missing or outside {source_measures.COMPLEXIPY_REQUIRED}, a missing or malformed '
            'baseline, or a baseline whose recorded parameters no longer match '
            'this tree). 1 and 2 are deliberately distinct: a broken instrument '
            'must never read as a clean tree or as a real regression.'
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--report', action='store_true', help='Print the measure table.'
    )
    mode.add_argument(
        '--json', action='store_true', help='Print the report as JSON.'
    )
    mode.add_argument(
        '--check', action='store_true',
        help='Compare a fresh measurement against --baseline (the ratchet).',
    )
    mode.add_argument(
        '--write-baseline', metavar='PATH',
        help='Measure and write a baseline to PATH. REFUSES (exit 1) to absorb '
        'a measure that rose unless --authorize-raise names the task doing it, '
        'so a regeneration can never silently widen the ratchet for every '
        'downstream task -- see the file\'s own _README.',
    )
    parser.add_argument(
        '--root', default=str(ratchet.repo_root()),
        help='Repo root to measure (default: this script\'s own checkout). '
        'Pair it with --baseline; on its own it would compare another tree '
        'against THIS checkout\'s baseline.',
    )
    parser.add_argument(
        '--baseline', default=str(ratchet.repo_root() / ratchet.BASELINE_RELPATH),
        help='Baseline JSON for --check (default: %(default)s).',
    )
    parser.add_argument(
        '--authorize-raise', metavar='TASK_ID',
        help='Permit --write-baseline to absorb the raises in THIS measurement, '
        'recording them in --ledger against this task id. Requires --reason. '
        'Authorization is an act, not a standing entry: a recorded raise never '
        'permits a later one.',
    )
    parser.add_argument(
        '--reason', metavar='TEXT',
        help='Why this raise is net-additive work rather than a refactor that '
        'failed. Requires --authorize-raise; recorded verbatim in the ledger.',
    )
    parser.add_argument(
        '--ledger', default=str(ratchet.repo_root() / ratchet.LEDGER_DIR_RELPATH),
        help='Authorized-raise ledger DIRECTORY --authorize-raise records one new '
        'entry file in (default: %(default)s). Pair it with --write-baseline when '
        'writing outside this checkout; on its own it would record in THIS '
        'checkout\'s ledger.',
    )
    return parser


def _resolve_authorization(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> ratchet.RaiseAuthorization | None:
    """Validate the authorization flags against each other, or exit 2 saying why.

    argparse cannot express "these two imply each other, and both require that
    mode", so it is done here rather than left unchecked. A flag that silently
    did nothing would read, to the agent who typed it, exactly like an
    authorization that was granted.

    The value object's own non-blank invariant lands here as the THIRD usage
    error rather than as a broken instrument: ``--reason ""`` is a flag typed
    wrong, so it prints like the two above and exits on the same code, before
    anything is measured.
    """
    if (args.authorize_raise is None) != (args.reason is None):
        parser.error(
            '--authorize-raise and --reason require each other: a raise with no '
            'recorded reason is not an authorized raise, it is an unattributed '
            'one'
        )
    if args.authorize_raise is None:
        return None
    if not args.write_baseline:
        parser.error(
            '--authorize-raise is a modifier of --write-baseline, which is the '
            'only mode that can absorb a raise. --check reports raises; it '
            'never records them'
        )
    try:
        return ratchet.RaiseAuthorization(task_id=args.authorize_raise, reason=args.reason)
    except ratchet.RatchetError as exc:
        parser.error(str(exc))


def _write(
    args: argparse.Namespace, report: dict, authorization: ratchet.RaiseAuthorization | None
) -> int:
    """The --write-baseline arm, reporting what this invocation actually did.

    Without an authorization the ledger is neither read nor written -- an
    unauthorized raise is refused before it is reached -- so a plain
    regeneration never depends on that file's health, and --ledger can be passed
    through unconditionally to keep one call site rather than two arms.

    WHAT IS PRINTED COMES FROM THE WRITE, never from re-reading the ledger, so
    the operator's log cannot claim a raise the committed file does not carry.
    """
    ledger = Path(args.ledger)
    written = ratchet.write_baseline(
        Path(args.write_baseline), report, authorization=authorization, ledger=ledger
    )
    print(f'wrote {written.path}')
    if authorization is None:
        return 0
    if written.record is None:
        # An authorization that recorded nothing must SAY so. Printing only
        # "wrote ..." would read, to the agent who typed --authorize-raise,
        # exactly like an authorization that was granted and recorded -- the
        # very failure _resolve_authorization rejects flag shapes to avoid
        # (INV-11, no silent fail-soft).
        print(f'no measure rose; nothing recorded in {ledger}')
        return 0
    print(
        f'recorded {len(written.record["measures"])} authorized raise(s) for '
        f'task {written.record["task_id"]} in {ledger}'
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See the module docstring for the exit-code contract."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    authorization = _resolve_authorization(parser, args)
    root = Path(args.root)
    try:
        report = lane_cluster.build_report(root)
        if args.json:
            print(json.dumps(report))
            return 0
        if args.report:
            print(_render_table(report, root))
            return 0
        if args.write_baseline:
            return _write(args, report, authorization)
        violations = ratchet.check_against_baseline(
            report, ratchet.load_baseline(Path(args.baseline))
        )
        if not violations:
            return 0
        print(
            f'{len(violations)} merge-lane ratchet violation(s) against '
            f'{args.baseline}:',
            file=sys.stderr,
        )
        for violation in violations:
            print(f'  {violation.message}', file=sys.stderr)
        print(ratchet.RAISE_REMEDY, file=sys.stderr)
        return 1
    except ratchet.UnauthorizedRaise as exc:
        # Exit 1, not 2: a refused regeneration is a measure that ROSE, found at
        # write time instead of at check time. Exit 2 stays reserved for a
        # broken instrument, and both faces of the gate print the same lines.
        print(
            f'{len(exc.violations)} merge-lane ratchet violation(s) would be '
            f'absorbed by writing {args.write_baseline}:',
            file=sys.stderr,
        )
        for violation in exc.violations:
            print(f'  {violation.message}', file=sys.stderr)
        print(ratchet.RAISE_REMEDY, file=sys.stderr)
        return 1
    except source_measures.MetricsError as exc:
        print(f'merge_lane_metrics: {exc}', file=sys.stderr)
        return 2




if __name__ == '__main__':
    sys.exit(main())

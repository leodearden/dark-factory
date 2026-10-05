"""The store-only test census (task 5414): one dated markdown report per project.

Part 1 ranks never-failed tests by per-run cost from retained verify evidence;
Part 2 counts pinning (Python) or duplication (Rust) across the whole test
tree. A report, never a gate: exit 0 when the report is written, 2 when it
cannot be (a missing directory, or a tool fault such as a tree that is not the
top of a git work tree). There is no exit 1.

    python scripts/suite_census.py --ecosystem pytest --root . --project dark-factory \\
        --out plans/test-census-<date>-dark-factory.md
"""
from __future__ import annotations

import argparse
import datetime
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import suite_census_pinning
import suite_census_rust
from merge_lane_metrics import MetricsError
from suite_census_evidence import nextest_evidence, pytest_evidence
from suite_census_outcomes import Evidence, OutcomeCensus, census_outcomes
from suite_census_outcomes import render_markdown as render_outcomes

from shared import safe_io

_INVOCATION = ('python', 'scripts/suite_census.py')


@dataclass(frozen=True)
class Ecosystem:
    evidence: Callable[[Path, Path], Evidence]
    pinning_markdown: Callable[[Path], str]


def _python_pinning(tree: Path) -> str:
    return suite_census_pinning.render_markdown(suite_census_pinning.measure_python_tree(tree))


def _rust_duplication(tree: Path) -> str:
    return suite_census_rust.render_markdown(suite_census_rust.measure_rust_tree(tree))


ECOSYSTEMS: MappingProxyType[str, Ecosystem] = MappingProxyType({
    'pytest': Ecosystem(evidence=pytest_evidence, pinning_markdown=_python_pinning),
    'nextest': Ecosystem(evidence=nextest_evidence, pinning_markdown=_rust_duplication),
})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Write the store-only test census report for one project (task 5414).',
    )
    parser.add_argument('--ecosystem', required=True, choices=sorted(ECOSYSTEMS))
    parser.add_argument(
        '--root', required=True, type=Path, help='project root holding data/ and .worktrees/',
    )
    parser.add_argument(
        '--tree', type=Path, help='git work tree whose tests are resolved and measured (default: --root)',
    )
    parser.add_argument('--project', required=True, help='project label for the report header')
    parser.add_argument('--out', required=True, type=Path, help='markdown report to write')
    parser.add_argument('--top', type=int, default=100, help='ranked rows to show (default 100)')
    parser.add_argument(
        '--date', default=datetime.datetime.now(datetime.UTC).date().isoformat(),
        help='measurement date for the header (default: today, UTC)',
    )
    return parser


def _head(tree: Path) -> str:
    try:
        done = subprocess.run(
            ['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise MetricsError(f'could not run git in {tree}: {exc}') from exc
    if done.returncode != 0:
        raise MetricsError(f'`git rev-parse HEAD` failed in {tree}: {done.stderr.strip() or "no stderr"}')
    return done.stdout.strip()


def _document(
    args: argparse.Namespace, argv: Sequence[str], tree: Path, sha: str,
    census: OutcomeCensus, part_2: str,
) -> str:
    sources = '\n'.join(f'- {w.source}: `{w.pattern}`' for w in census.windows)
    return (
        f'# Test census: {args.project}, {args.date}\n\n'
        f'- Measured tree: `{tree}` at `{sha}`\n'
        f'- Evidence root: `{args.root}`\n'
        f'- Ecosystem: {args.ecosystem}\n'
        '- Task 5414: a store-only census. No test is retired or changed here.\n\n'
        '## Commands used\n\n'
        f'```\n{shlex.join([*_INVOCATION, *argv])}\n```\n\n'
        f'Evidence read, per source:\n\n{sources}\n\n'
        '## Part 1: never failed × per-run cost\n\n'
        f'{render_outcomes(census, top=args.top)}\n'
        '## Part 2: pinning and duplication across the whole test tree\n\n'
        f'{part_2}'
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    tree: Path = args.tree or args.root
    for flag, path in (('--root', args.root), ('--tree', tree)):
        if not path.is_dir():
            print(f'suite_census: {flag} {path} is not a directory', file=sys.stderr)
            return 2
    ecosystem = ECOSYSTEMS[args.ecosystem]
    try:
        sha = _head(tree)
        census = census_outcomes(ecosystem.evidence(args.root, tree))
        part_2 = ecosystem.pinning_markdown(tree)
    except (MetricsError, ValueError) as exc:
        print(f'suite_census: {exc}', file=sys.stderr)
        return 2
    safe_io.atomic_write_text(args.out, _document(args, argv, tree, sha, census, part_2), mkdir=True)
    print(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

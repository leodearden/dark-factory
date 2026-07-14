"""CLI for the T5 curator replay corpus builder (task 2496 / PRD D-6).

Usage:
    cd orchestrator && uv run python -m orchestrator.evals.prompt_opt <command>

Commands:
    build-curator-corpus   Build the labeled curator replay corpus from
                            tickets.db. See curator_replay_corpus/README.md
                            for the schema, split, and human spot-check
                            sign-off protocol.

Mirrors orchestrator/evals/reviewer_trial/__main__.py's click pattern. This
module is operator-facing only -- every function it calls
(build_curator_corpus, propose_curator_label_frontier, audit_curator_corpus)
is already unit-tested hermetically against synthetic fixtures
(test_prompt_opt_curator_corpus.py, never the real DB or a real LLM). This
file adds no new runtime behavior of its own: it is thin wiring that adds
real I/O -- a real tickets.db path and real frontier-model calls -- around
already-tested machinery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click

LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'

_PKG_DIR = Path(__file__).parent
_CURATOR_CORPUS_DIR = _PKG_DIR / 'curator_replay_corpus'
_CURATOR_MANIFEST_PATH = _CURATOR_CORPUS_DIR / 'manifest.json'

# Overridable via env var for portability across checkouts -- tickets.db is
# offline, gitignored operational data that lives in the MAIN repo, not a
# task worktree (mirrors reviewer_trial's REVIEWER_TRIAL_RUNS_DB pattern;
# see corpus/README.md's "Regenerating / extending the corpus").
_DEFAULT_TICKETS_DB = Path(os.environ.get(
    'CURATOR_TICKETS_DB', '/home/leo/src/dark-factory/data/reconciliation/tickets.db',
))
# ~100 labeled decisions matches D-6's "make a 2:1:7 split meaningful"
# floor for the reviewer corpus (target ~100, >=~50 minimum); curator
# labels are cheaper per-item (one frontier call, no diff text to read) so
# 100 is a conservative default an operator can raise via --n.
_DEFAULT_N = 100
_DEFAULT_SEED = 2496
# Mirrors curator_corpus.py's own _DEFAULT_SPOT_CHECK_CAP -- the fraction/
# minimum defaults inside select_spot_check_subset already bound the
# per-stratum sample; this cap only bites for a much larger --n than the
# default above.
_DEFAULT_SPOT_CHECK_SIZE = 200

# (check key, human-readable description) -- order matches
# curator_corpus.audit_curator_corpus's check ordering so PASS/FAIL output
# reads top-to-bottom in the same sequence the report's `failures` would.
_AUDIT_CHECKS = (
    ('item_count', 'item count meets the minimum floor'),
    ('missing_split', 'every item has a split assigned'),
    ('split_ratio', 'train/selection/test ratio approximates 2:1:7'),
    ('missing_gold_label', 'every item carries a gold label'),
    ('adjudication_coverage', 'adjudication_log covers every ticket_id'),
    ('spot_check_subset_empty', 'a non-empty documented spot-check subset exists'),
)


def _print_check(passed: bool, message: str) -> bool:
    tag = click.style('[PASS]', fg='green') if passed else click.style('[FAIL]', fg='red')
    click.echo(f'{tag} {message}')
    return passed


@click.group()
@click.option('--verbose', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """T5 curator replay corpus operator CLI."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt='%H:%M:%S', stream=sys.stderr)


@cli.command('build-curator-corpus')
@click.option('--db-path', default=str(_DEFAULT_TICKETS_DB), type=click.Path(path_type=Path),
              help='Path to the offline tickets.db (env: CURATOR_TICKETS_DB)')
@click.option('--n', default=_DEFAULT_N, type=int,
              help='Max labeled decisions to sample (stratified by recorded action)')
@click.option('--seed', default=_DEFAULT_SEED, type=int, help='Deterministic sampling/split seed')
@click.option('--spot-check-size', default=_DEFAULT_SPOT_CHECK_SIZE, type=int,
              help='Cap on the human spot-check subset size')
@click.option('--out', default=str(_CURATOR_MANIFEST_PATH), type=click.Path(path_type=Path),
              help="Where to write manifest.json ('annotations/' and adjudication_log.jsonl "
                   'are written alongside it)')
def build_curator_corpus_cmd(db_path: Path, n: int, seed: int, spot_check_size: int, out: Path) -> None:
    """Build the T5 curator replay corpus from tickets.db (task 2496 / PRD D-6).

    Thin wiring over already-tested machinery (curator_corpus.py): reads
    recorded curator decisions from *db_path*, samples up to *n* stratified
    by recorded action, obtains a GOLD label for each sampled candidate from
    a real frontier-model call (propose_curator_label_frontier -- NEVER the
    recorded decision; PRD D-6: decisions != ground truth), flags a
    deterministic human spot-check subset capped at *spot_check_size*,
    assigns a 2:1:7 split, and writes manifest.json + annotations/*.json +
    adjudication_log.jsonl under *out*'s directory. Prints an
    audit_curator_corpus PASS/FAIL summary and exits non-zero on FAIL.
    """
    from .curator_corpus import (
        audit_curator_corpus,
        build_curator_corpus,
        propose_curator_label_frontier,
    )

    async def _run() -> int:
        if not db_path.is_file():
            click.echo(
                f'tickets.db not found at {db_path}\n'
                '  This is an offline generation tool that reads the gitignored '
                'reconciliation ticket store (data/reconciliation/tickets.db in the '
                'MAIN repo checkout, not a task worktree -- see '
                'curator_replay_corpus/README.md). Pass --db-path, or set '
                'CURATOR_TICKETS_DB, to point at a real tickets.db.',
                err=True,
            )
            return 1

        click.echo(click.style(f'Build Curator Corpus: n={n} seed={seed} db={db_path}', bold=True))
        click.echo('=' * 60)

        manifest, log = await build_curator_corpus(
            db_path, n=n, seed=seed, spot_check_size=spot_check_size,
            frontier_proposer=propose_curator_label_frontier,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(out)
        adjudication_log_path = out.parent / 'adjudication_log.jsonl'
        log.save(adjudication_log_path)

        click.echo(f'  {len(manifest.items)} items labeled and written to {out}')
        click.echo(f'  Adjudication log: {adjudication_log_path}')

        report = audit_curator_corpus(manifest, log)
        click.echo()
        click.echo(click.style(f'Corpus Audit: {report.item_count} items', bold=True))
        checks_passed = 0
        for key, message in _AUDIT_CHECKS:
            if _print_check(key not in report.failures, message):
                checks_passed += 1

        click.echo()
        color = 'green' if report.ok else 'red'
        status = 'PASS' if report.ok else 'FAIL'
        click.echo(click.style(
            f'Result: {status} ({checks_passed}/{len(_AUDIT_CHECKS)} checks passed)', fg=color, bold=True,
        ))
        return 0 if report.ok else 1

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo('\nInterrupted.')
        sys.exit(130)


if __name__ == '__main__':
    cli()

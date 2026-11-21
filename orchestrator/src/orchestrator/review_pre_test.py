"""Run a project's declared pre-test setup commands before a review checkpoint's Phase 1, and describe the outcome."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from orchestrator.verify import _run_cmd

logger = logging.getLogger(__name__)

_OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class PreTestCommandResult:
    command: str
    returncode: int
    timed_out: bool
    duration_secs: float
    output_tail: str

    @property
    def failed(self) -> bool:
        return self.timed_out or self.returncode != 0


def _failure_reason(result: PreTestCommandResult) -> str:
    if result.timed_out:
        return f'timed out after {result.duration_secs:.0f}s'
    return f'exit code {result.returncode}'


async def _run_one(
    command: str, *, cwd: Path, timeout_secs: float, env: dict[str, str] | None,
) -> PreTestCommandResult:
    logger.info('Review pre-test setup: running %r in %s', command, cwd)
    start = time.monotonic()
    rc, out, timed_out = await _run_cmd(command, cwd, timeout_secs, env=env)
    result = PreTestCommandResult(
        command=command,
        returncode=rc,
        timed_out=timed_out,
        duration_secs=time.monotonic() - start,
        output_tail=out[-_OUTPUT_TAIL_CHARS:],
    )
    if result.failed:
        logger.warning(
            'Review pre-test setup: %r failed (%s); the review proceeds regardless\n'
            'output tail:\n%s',
            command, _failure_reason(result), result.output_tail,
        )
    else:
        logger.info(
            'Review pre-test setup: %r ok in %.1fs', command, result.duration_secs,
        )
    return result


async def run_pre_test_commands(
    commands: Sequence[str],
    *,
    cwd: Path,
    timeout_secs: float,
    env: Mapping[str, str] | None,
) -> tuple[PreTestCommandResult, ...]:
    """Run *commands* in order in *cwd*; a failure is logged and recorded, never raised.

    Every command runs even when an earlier one fails, so the caller gets the
    outcome of each.  Chain dependent steps with ``&&`` inside one entry when
    a later step must not run after an earlier failure.
    """
    overlay = dict(env) if env else None
    results = []
    for command in tuple(commands):
        results.append(
            await _run_one(command, cwd=cwd, timeout_secs=timeout_secs, env=overlay),
        )
    return tuple(results)


def _status_line(result: PreTestCommandResult) -> str:
    if result.failed:
        return f'- `{result.command}` — FAILED ({_failure_reason(result)})'
    return f'- `{result.command}` — ok ({result.duration_secs:.1f}s)'


def _failure_detail(result: PreTestCommandResult) -> str:
    return f"""
#### Output tail of `{result.command}` (last {_OUTPUT_TAIL_CHARS} chars)
```
{result.output_tail}
```
"""


def render_pre_test_section(results: Sequence[PreTestCommandResult]) -> str:
    """Markdown sub-section for the reviewer prompt's Phase 1 block; '' when nothing ran."""
    if not results:
        return ''
    status_lines = '\n'.join(_status_line(r) for r in results)
    section = f"""
### Pre-test Setup

These project-declared setup commands ran in the project root, in order,
before Phase 1:

{status_lines}
"""
    failed = [r for r in results if r.failed]
    if failed:
        section += """
A setup command failed. Phase 1 reds that depend on what a failed setup step
builds (e.g. stale frontend bundles served to e2e tests) are attributable to
the setup failure, not to the code under review — report them as such.
"""
        section += ''.join(_failure_detail(r) for r in failed)
    return section

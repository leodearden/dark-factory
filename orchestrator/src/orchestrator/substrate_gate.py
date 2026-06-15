"""substrate_gate — dispatch-time substrate re-diff gate.

Re-runs a committed probe set against current ``main`` before an agent spins
up; a PASS→FAIL flip (e.g. a sibling deleted ``Type::Real`` between author
and dispatch) blocks dispatch and triggers a human-reviewed L1 escalation
rather than wasting an agent spin-up and an L2.

PRD: prd-gate-exec D4.
Design: plans/prd-gate-exec-d4.md

Dependency-light: stdlib + logging only.  Mirrors b3_gate.py's structure
(injectable run_subprocess, frozen verdict dataclass, exit-code→verdict
mapping).  No CLI — this module runs in-process inside the harness's
``_run_slot``, unlike b3_gate which is invoked cross-process by the steward.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS = 'pass'
FLIP = 'flip'
SKIP = 'skip'

EXIT_ALL_PASS = 0      # checker: all probes passed
EXIT_FAIL = 1          # checker: ≥1 FAIL
EXIT_UNPROVABLE = 2    # checker: ≥1 UNPROVABLE

DEFAULT_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubstrateVerdict:
    """Result of a dispatch-time substrate re-check.

    Attributes:
        verdict: One of PASS / FLIP / SKIP.
        exit_code: Raw exit code from the checker process (None for SKIP or
            when the checker was not invoked).
        checker_argv: The full argv passed to the checker (None when no
            resolvable checker command).
        probe_set: Path to the probe-set file (relative to the gate worktree).
        reason: Human-readable explanation of the verdict.
        stdout: Truncated checker stdout (first 2000 chars).
        stderr: Truncated checker stderr (first 2000 chars).
    """

    verdict: str
    exit_code: int | None
    checker_argv: list[str] | None
    probe_set: str | None
    reason: str
    stdout: str = ''
    stderr: str = ''

    @property
    def flipped(self) -> bool:
        """True when the verdict is FLIP (gate should block dispatch)."""
        return self.verdict == FLIP


# ---------------------------------------------------------------------------
# extract_probe_set
# ---------------------------------------------------------------------------


def extract_probe_set(task: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the ``substrate_probe`` descriptor from a task dict, or None.

    Mirrors ``Scheduler._normalize_task_metadata``'s JSON-string→dict
    coercion so the read is robust to the fused-memory wire format (metadata
    may arrive as a dict, a JSON string, or absent/None).

    Returns the descriptor dict when:
    - ``task['metadata']`` is a dict (or JSON string that decodes to a dict)
    - the dict contains ``'substrate_probe'`` as a nested dict
    - that nested dict contains a non-empty ``'probe_set'`` key

    Returns None in all other cases (no metadata, malformed, no probe_set).
    """
    raw = task.get('metadata')

    # Coerce JSON string to dict (mirrors Scheduler._normalize_task_metadata)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        raw = parsed

    if not isinstance(raw, dict):
        return None

    descriptor = raw.get('substrate_probe')
    if not isinstance(descriptor, dict):
        return None

    probe_set = descriptor.get('probe_set')
    if not probe_set:  # None or empty string
        return None

    return descriptor


# ---------------------------------------------------------------------------
# build_checker_argv
# ---------------------------------------------------------------------------


def build_checker_argv(descriptor: Any) -> list[str] | None:
    """Build the full checker argv from a probe descriptor.

    Concatenates ``descriptor['checker']`` with ``descriptor['probe_set']``,
    with optional ``{probe_set}`` placeholder substitution inside the checker
    template items (so callers can embed the path as a flag value, e.g.
    ``--probes={probe_set}``).

    Returns None when:
    - ``descriptor`` is None or not a dict
    - ``descriptor['checker']`` is missing, None, or empty
    - ``descriptor['probe_set']`` is missing or falsy
    """
    if not isinstance(descriptor, dict):
        return None

    checker = descriptor.get('checker')
    if not checker:  # None or empty list
        return None

    probe_set = descriptor.get('probe_set')
    if not probe_set:
        return None

    # Apply {probe_set} substitution in each checker template token
    expanded = [token.replace('{probe_set}', probe_set) for token in checker]

    # Append probe_set as a trailing positional arg when no {probe_set}
    # placeholder was used anywhere in the template.
    if all(token == orig for token, orig in zip(expanded, checker)):
        return expanded + [probe_set]
    return expanded


# ---------------------------------------------------------------------------
# Default subprocess runner
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 2000  # chars to keep from stdout/stderr


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Any,
    timeout: int,
) -> tuple[int, str, str]:
    """Run ``argv`` in ``cwd``; return (returncode, stdout, stderr).

    On ``subprocess.TimeoutExpired`` or any other exception, returns a
    synthetic non-{0,1,2} exit code (99) so the caller maps it to the
    "substrate unverifiable" FLIP branch.  The 99 sentinel is distinct from
    checker exit codes 0/1/2 and from typical shell error codes (126/127/255).
    """
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.warning('substrate checker timed out (argv=%s)', argv)
        return 99, '', 'timeout'
    except Exception as exc:
        logger.warning('substrate checker raised: %s', exc)
        return 99, '', str(exc)


# ---------------------------------------------------------------------------
# run_substrate_recheck
# ---------------------------------------------------------------------------


def run_substrate_recheck(
    *,
    task: dict[str, Any],
    worktree: Any,
    run_subprocess: Any = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> SubstrateVerdict:
    """Re-run the committed probe set against a gate worktree at current main.

    Exit-code → verdict mapping (per PRD §10 / SEAM CONTRACT):
    - rc=0  → PASS  (all probes still pass; dispatch allowed)
    - rc=1  → FLIP  (≥1 FAIL; PASS→FAIL drift detected)
    - rc=2  → FLIP  (≥1 UNPROVABLE; treated as blocked premise)
    - other non-zero / timeout / missing checker → FLIP ("substrate unverifiable")
    - no descriptor → SKIP (gate is a no-op; non-probe task)

    Args:
        task: Task dict (may contain metadata as dict or JSON string).
        worktree: Path to the ephemeral gate worktree (checked out at main SHA).
        run_subprocess: Injectable callable ``(argv, *, cwd, timeout) -> (rc,
            stdout, stderr)``; defaults to the stdlib-backed ``_run_subprocess``.
        timeout: Checker process timeout in seconds.
    """
    if run_subprocess is None:
        run_subprocess = _run_subprocess

    descriptor = extract_probe_set(task)

    # --- No descriptor → SKIP (gate no-op) ---
    if descriptor is None:
        return SubstrateVerdict(
            verdict=SKIP,
            exit_code=None,
            checker_argv=None,
            probe_set=None,
            reason='no substrate_probe descriptor — gate skipped',
        )

    probe_set = descriptor['probe_set']
    argv = build_checker_argv(descriptor)

    # --- No resolvable checker → FLIP ---
    if not argv:
        return SubstrateVerdict(
            verdict=FLIP,
            exit_code=None,
            checker_argv=None,
            probe_set=probe_set,
            reason='probe set declared but no checker command resolvable',
        )

    # --- Run the checker ---
    logger.info(
        'substrate_gate: running checker %s in %s (probe_set=%s)',
        argv,
        worktree,
        probe_set,
    )
    try:
        rc, stdout, stderr = run_subprocess(argv, cwd=worktree, timeout=timeout)
    except Exception as exc:
        return SubstrateVerdict(
            verdict=FLIP,
            exit_code=None,
            checker_argv=argv,
            probe_set=probe_set,
            reason=f'substrate unverifiable / run_subprocess raised: {exc}',
        )

    stdout_trunc = (stdout or '')[:_MAX_OUTPUT]
    stderr_trunc = (stderr or '')[:_MAX_OUTPUT]

    # --- Map exit code to verdict ---
    if rc == EXIT_ALL_PASS:
        return SubstrateVerdict(
            verdict=PASS,
            exit_code=rc,
            checker_argv=argv,
            probe_set=probe_set,
            reason='all probes PASS on current main — dispatch allowed',
            stdout=stdout_trunc,
            stderr=stderr_trunc,
        )
    elif rc == EXIT_FAIL:
        return SubstrateVerdict(
            verdict=FLIP,
            exit_code=rc,
            checker_argv=argv,
            probe_set=probe_set,
            reason='PASS→FAIL flip detected: ≥1 probe FAIL on current main (4352 drift case)',
            stdout=stdout_trunc,
            stderr=stderr_trunc,
        )
    elif rc == EXIT_UNPROVABLE:
        return SubstrateVerdict(
            verdict=FLIP,
            exit_code=rc,
            checker_argv=argv,
            probe_set=probe_set,
            reason='PASS→FAIL flip detected: ≥1 probe UNPROVABLE on current main',
            stdout=stdout_trunc,
            stderr=stderr_trunc,
        )
    else:
        return SubstrateVerdict(
            verdict=FLIP,
            exit_code=rc,
            checker_argv=argv,
            probe_set=probe_set,
            reason=f'substrate unverifiable / checker error rc={rc}',
            stdout=stdout_trunc,
            stderr=stderr_trunc,
        )

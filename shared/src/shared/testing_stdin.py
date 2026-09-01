"""Stdin-delivery test stubs, shared by BOTH spawn runners (task 3147).

There are two independent implementations of the spawn path that carry an
agent payload on stdin, and both had the same starvation defect:

  * ``shared.cli_invoke._run_subprocess``            (claude, sandboxed + direct)
  * ``orchestrator.agents.invoke._run_subprocess_local``  (codex)

Their regression suites live in different packages
(``shared/tests/test_cli_invoke.py`` and ``orchestrator/tests/test_invoke.py``)
and ``shared.tests`` is not an importable package from the orchestrator pytest
rootdir, so the stubs live HERE — in the shipped ``shared`` package, the same
convention as :mod:`shared.testing` — rather than being copy-pasted into each
suite.  One definition, so an edit to the rejection signature cannot land in
one suite and silently miss the other.

**The stubs' stderr must stay verbatim-equal to the production rejection
signature.**  The second line reproduces
:data:`shared.invocation_outcome.CLI_INPUT_REQUIRED_MARKERS` exactly, which is
what makes the tests pin the real ``CLI_INPUT_REJECTED`` classification rather
than a lookalike.  That coupling is enforced at import time below, not merely
documented: editing the marker table without editing the stubs raises here
instead of quietly degrading every test that uses them.
"""

from __future__ import annotations

import asyncio
import textwrap
import time

from shared.invocation_outcome import CLI_INPUT_REQUIRED_MARKERS

__all__ = [
    'STDIN_DIGEST_STUB',
    'STDIN_STARVATION_STUB',
    'make_starving_exec',
]


STDIN_STARVATION_STUB = textwrap.dedent(
    """
    import sys, select

    # Mimics the claude CLI's stdin deadline, shortened so the test is fast.
    # The stderr text below reproduces CLI_INPUT_REQUIRED_MARKERS
    # (shared/src/shared/invocation_outcome.py) VERBATIM, so these tests pin
    # the exact production rejection signature rather than a lookalike.
    r, _, _ = select.select([sys.stdin], [], [], 0.3)
    if not r:
        sys.stderr.write('Warning: no stdin data received in 3s, proceeding without it.\\n')
        sys.stderr.write('Error: Input must be provided either through stdin or as a prompt argument when using --print\\n')
        sys.exit(1)
    data = sys.stdin.buffer.read()
    sys.stdout.write('GOT:%d' % len(data))
    """
)


STDIN_DIGEST_STUB = textwrap.dedent(
    """
    import sys, select, hashlib

    r, _, _ = select.select([sys.stdin], [], [], 0.3)
    if not r:
        sys.stderr.write('Warning: no stdin data received in 3s, proceeding without it.\\n')
        sys.stderr.write('Error: Input must be provided either through stdin or as a prompt argument when using --print\\n')
        sys.exit(1)
    data = sys.stdin.buffer.read()
    # Length AND digest, so a concurrent result can be matched back to its own
    # invocation and cross-wiring cannot hide behind an equal length.
    sys.stdout.write('GOT:%d:%s' % (len(data), hashlib.sha256(data).hexdigest()[:16]))
    """
)


# Structural coupling, not a comment.  CLI_INPUT_REQUIRED_MARKERS is matched
# case-insensitively in production (see shared.invocation_outcome), so the
# stubs are checked the same way.  A raise, not an ``assert``: asserts are
# stripped under ``python -O`` and this guard must not evaporate.
for _stub_name, _stub in (
    ('STDIN_STARVATION_STUB', STDIN_STARVATION_STUB),
    ('STDIN_DIGEST_STUB', STDIN_DIGEST_STUB),
):
    _missing = [m for m in CLI_INPUT_REQUIRED_MARKERS if m not in _stub.lower()]
    if _missing:
        raise RuntimeError(
            f'{_stub_name} no longer emits the production CLI-input-rejected '
            f'signature; missing {_missing!r}.  Update the stub to match '
            f'shared.invocation_outcome.CLI_INPUT_REQUIRED_MARKERS verbatim — '
            f'tests using it are meant to pin the real rejection text, not a '
            f'lookalike.'
        )
del _stub_name, _stub, _missing


def make_starving_exec(block_secs: float):
    """Return a ``create_subprocess_exec`` fake that starves the loop after spawn.

    Delegates to the REAL ``asyncio.create_subprocess_exec`` — a genuine child
    is required to observe stdin delivery at the OS level — then blocks the
    event loop with a SYNCHRONOUS ``time.sleep`` before returning the proc.

    That blocking sleep is the whole point: it stalls the loop in exactly the
    production window (between the child's exec and the parent's pipe write),
    making the race deterministic instead of probabilistic.  In production the
    same stall is produced by 48 concurrent agents sharing one event loop plus
    multi-MB synchronous transcript parsing and fsync-per-commit SQLite writes.
    """
    real_exec = asyncio.create_subprocess_exec  # capture BEFORE patching

    async def starving_exec(*args, **kwargs):
        proc = await real_exec(*args, **kwargs)
        time.sleep(block_secs)  # SYNCHRONOUS — blocks the event loop on purpose
        return proc

    return starving_exec

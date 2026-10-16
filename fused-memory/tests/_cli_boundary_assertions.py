"""Read what ``shared.cli_boundary.run_cli`` reports, for every script suite that adopts it.

The boundary reports a failed stdout as exactly ONE ``error: ...`` line on
stderr and exits ``EXIT_STDOUT_FAILED``. Each adoption of ``run_cli`` under
``fused-memory/scripts/`` pins that contract for its own script, so the reading
of it lives here once rather than in a copy per test module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shared.cli_boundary import EXIT_STDOUT_FAILED, run_cli
from shared.testing_streams import closed_pipe_stdout


def error_lines(stderr: str) -> list[str]:
    """Only the ``error: `` lines of *stderr*.

    Logging may or may not reach stderr under pytest's root handlers, so the
    whole stream is not a stable count.
    """
    return [line for line in stderr.splitlines() if line.startswith('error: ')]


def closed_pipe_error_line(
    monkeypatch: Any, capsys: Any, main: Callable[[], int], *, buffering: int | None,
) -> str:
    """Run *main* through ``run_cli`` into a closed pipe; return the one ``error:`` line.

    Asserts the boundary's contract on the way: ``EXIT_STDOUT_FAILED`` and
    exactly one line. *buffering* selects the regime, as in
    ``shared.testing_streams.closed_pipe_stdout``: ``None`` defers the failure
    to the boundary's flush, ``1`` raises it in-band from the report's print.
    """
    with closed_pipe_stdout(monkeypatch, buffering=buffering, quiet_close=True):
        code = run_cli(main)
    errors = error_lines(capsys.readouterr().err)
    assert code == EXIT_STDOUT_FAILED, errors
    assert len(errors) == 1, errors
    return errors[0]

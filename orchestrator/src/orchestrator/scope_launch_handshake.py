"""Proof that a launcher such as ``systemd-run --scope`` exec'd the bash payload it was handed.

The payload's first line writes one byte to an inherited pipe and closes it;
the command follows on its own line. ``bash -c`` runs line 1 before parsing
line 2, so even a syntax error in the command cannot suppress the byte (see
``orchestrator/tests/test_scope_launch_handshake.py``). EOF with no byte once
the launcher has exited therefore proves the payload never started; a byte, or
a writer still open, proves nothing either way. Measured on systemd 255:
``systemd-run --user --scope`` keeps an fd passed through ``pass_fds`` open
across its exec, and a launch it refuses exits with the pipe at EOF.
"""

from __future__ import annotations

import os
from types import TracebackType


class ScopeLaunchHandshake:
    def __init__(self) -> None:
        self._reader, self._writer = os.pipe()
        os.set_blocking(self._reader, False)
        self._reader_open = True
        self._writer_open = True

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self._writer,)

    def payload(self, cmd: str) -> str:
        return f'printf . >&{self._writer}; exec {self._writer}>&-\n{cmd}'

    def release_writer(self) -> None:
        if self._writer_open:
            self._writer_open = False
            os.close(self._writer)

    def launcher_exited_before_exec(self) -> bool:
        self.release_writer()
        try:
            return os.read(self._reader, 1) == b''
        except BlockingIOError:
            return False

    def close(self) -> None:
        self.release_writer()
        if self._reader_open:
            self._reader_open = False
            os.close(self._reader)

    def __enter__(self) -> ScopeLaunchHandshake:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

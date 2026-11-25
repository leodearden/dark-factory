"""The defect filer's own state dir S (plans/client-feed-prd.md §14.4): apart from out_dir, in_dir and project_root after every symlink, held by one filer at a time, and the cursor it keeps."""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from client_feed.contract import NonNegativeInt, PositiveInt, Token
from client_feed.filer_tasks import FilerFallback
from client_feed.policy import Policy

logger = logging.getLogger(__name__)

LOCK_NAME: Final = 'filer.lock'
CURSOR_NAME: Final = 'cursor.json'
_STATE_DIR_MODE: Final = 0o700
_LOCK_MODE: Final = 0o600


class RefRecord(BaseModel):
    """An unconverged ref's failed attempts at its fix submission, and the fallback once one applies."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    attempts: PositiveInt
    last_code: Token
    fallback: FilerFallback | None = None


class FilerCursor(BaseModel):
    """Where the filer's inbox tail resumes, and its unconverged refs; an optimisation only, never needed for correctness."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    inbox_offset: NonNegativeInt = 0
    refs: dict[str, RefRecord] = Field(default_factory=dict)


class StateDirRefused(Exception):
    """A filer.state_dir that reaches out_dir, in_dir or project_root once its symlinks are resolved."""

    def __init__(self, state_dir: Path, detail: str) -> None:
        self.state_dir = state_dir
        super().__init__(f'state dir {state_dir} refused: {detail}')


class FilerLockHeld(Exception):
    """Another filer holds this state dir's lock."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        super().__init__(f'another defect filer holds {lock_path}')


@dataclass(frozen=True)
class StateDir:
    """S, while held."""

    path: Path

    def read_cursor(self) -> FilerCursor:
        """The stored cursor; a missing or unusable one reads as the inbox start, since every ref re-derives from the store."""
        path = self.path / CURSOR_NAME
        try:
            return FilerCursor.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return FilerCursor()
        except (OSError, ValidationError) as exc:
            logger.warning('filer cursor %s is unusable (%s); reading the inbox from its start', path, type(exc).__name__)
            return FilerCursor()

    def write_cursor(self, cursor: FilerCursor) -> None:
        """Replace the cursor atomically: a temporary file in S, fsynced, then renamed over it."""
        fd, temporary = tempfile.mkstemp(dir=self.path, prefix=f'.{CURSOR_NAME}.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(cursor.model_dump_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path / CURSOR_NAME)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise


def _require_apart(policy: Policy, state_dir: Path) -> None:
    try:
        policy.require_filer_state_dir_apart(resolved=True)
    except ValueError as exc:
        raise StateDirRefused(state_dir, str(exc)) from exc


@contextlib.contextmanager
def held_state_dir(policy: Policy) -> Iterator[StateDir]:
    """S checked apart before it is made and again after, then locked for this filer alone until the block exits."""
    if policy.filer is None:
        raise ValueError(f'the policy of {policy.project_id} has no filer block')
    state_dir = policy.filer.state_dir
    _require_apart(policy, state_dir)
    state_dir.mkdir(mode=_STATE_DIR_MODE, parents=True, exist_ok=True)
    _require_apart(policy, state_dir)
    lock_path = state_dir / LOCK_NAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, _LOCK_MODE)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FilerLockHeld(lock_path) from None
        yield StateDir(state_dir)
    finally:
        os.close(fd)

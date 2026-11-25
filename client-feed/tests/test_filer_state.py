"""The filer's state dir S (plans/client-feed-prd.md §14.4): apart from O, I and R after every symlink, held by one filer, and its cursor."""

import logging
import stat
from pathlib import Path

import pytest

from client_feed.filer_state import (
    CURSOR_NAME,
    LOCK_NAME,
    FilerCursor,
    FilerLockHeld,
    RefRecord,
    StateDirRefused,
    held_state_dir,
)
from client_feed.filer_tasks import FilerFallback
from client_feed.filer_tools import RefusalKind
from client_feed.policy import Policy


def _policy(tmp_path: Path, state_dir: Path) -> Policy:
    return Policy.model_validate({
        'project_id': 'acme-widgets',
        'project_root': str(tmp_path / 'root'),
        'out_dir': str(tmp_path / 'out'),
        'in_dir': str(tmp_path / 'in'),
        'visible_globs': ['src/**'],
        'filer': {'state_dir': str(state_dir)},
    })


def test_a_missing_state_dir_is_made_private_with_its_lock(tmp_path: Path) -> None:
    state_dir = tmp_path / 'state' / 'filer'

    with held_state_dir(_policy(tmp_path, state_dir)) as state:
        assert state.path == state_dir
        assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
        assert (state_dir / LOCK_NAME).is_file()


def test_a_second_holder_is_refused_while_the_first_holds_the_lock(tmp_path: Path) -> None:
    policy = _policy(tmp_path, tmp_path / 'state')

    with held_state_dir(policy):
        with pytest.raises(FilerLockHeld) as caught, held_state_dir(policy):
            pass
        assert str(tmp_path / 'state' / LOCK_NAME) in str(caught.value)

    with held_state_dir(policy) as state:
        assert state.path == tmp_path / 'state'


@pytest.mark.parametrize('container', ['out', 'in', 'root'])
def test_a_state_dir_reaching_a_publisher_dir_through_a_symlink_is_refused_untouched(
    tmp_path: Path, container: str
) -> None:
    target = tmp_path / container
    target.mkdir()
    (tmp_path / 'link').symlink_to(target)

    with pytest.raises(StateDirRefused) as caught, held_state_dir(_policy(tmp_path, tmp_path / 'link' / 'filer')):
        pass

    assert 'filer.state_dir' in str(caught.value)
    assert list(target.iterdir()) == []


def test_the_cursor_round_trips_with_no_temporary_file_left(tmp_path: Path) -> None:
    cursor = FilerCursor(
        inbox_offset=120,
        refs={
            'portal:acme-widgets:2:3': RefRecord(
                attempts=2,
                last_code='TaskNotFound',
                fallback=FilerFallback(kind=RefusalKind.TOOL_ERROR, code='is_error'),
            )
        },
    )

    with held_state_dir(_policy(tmp_path, tmp_path / 'state')) as state:
        state.write_cursor(cursor)
        read = state.read_cursor()

    assert read == cursor
    assert sorted(path.name for path in (tmp_path / 'state').iterdir()) == sorted([CURSOR_NAME, LOCK_NAME])


def test_a_missing_cursor_reads_as_the_inbox_start(tmp_path: Path) -> None:
    with held_state_dir(_policy(tmp_path, tmp_path / 'state')) as state:
        assert state.read_cursor() == FilerCursor()
    assert FilerCursor() == FilerCursor(inbox_offset=0, refs={})


@pytest.mark.parametrize('text', ['{not json', '{"inbox_offset": -1}', '{"inbox_offset": 0, "extra": 1}'])
def test_an_unusable_cursor_reads_as_the_inbox_start_and_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, text: str
) -> None:
    (tmp_path / 'state').mkdir()
    (tmp_path / 'state' / CURSOR_NAME).write_text(text)

    with caplog.at_level(logging.WARNING, logger='client_feed.filer_state'), held_state_dir(
        _policy(tmp_path, tmp_path / 'state')
    ) as state:
        assert state.read_cursor() == FilerCursor()

    assert any(str(tmp_path / 'state' / CURSOR_NAME) in record.getMessage() for record in caplog.records)

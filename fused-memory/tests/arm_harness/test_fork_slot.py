"""Which child process holds a Redis server's fork slot, read from an INFO persistence reply.

The flag names are written here as literal strings, not taken from ForkHolder,
so these tests are the oracle for the field names Redis actually reports.
"""

from collections.abc import Mapping

import pytest

from fused_memory.arm_harness.fork_slot import ForkHolder, ForkSlotUnreadable, read_fork_holders

QUIET_PERSISTENCE: Mapping[str, int] = {
    'loading': 0,
    'rdb_changes_since_last_save': 6,
    'rdb_bgsave_in_progress': 0,
    'rdb_last_bgsave_status': 1,
    'rdb_current_bgsave_time_sec': -1,
    'aof_enabled': 0,
    'aof_rewrite_in_progress': 0,
    'aof_current_rewrite_time_sec': -1,
    'module_fork_in_progress': 0,
    'module_fork_last_cow_size': 0,
}

FLAG_HOLDERS = [
    ('rdb_bgsave_in_progress', ForkHolder.RDB_SAVE),
    ('aof_rewrite_in_progress', ForkHolder.AOF_REWRITE),
    ('module_fork_in_progress', ForkHolder.MODULE_FORK),
]


class RecordingServer:
    def __init__(self, reply: Mapping[str, object]) -> None:
        self.reply = reply
        self.sections: list[str] = []

    async def info(self, section: str, /) -> Mapping[str, object]:
        self.sections.append(section)
        return self.reply


@pytest.mark.asyncio
async def test_a_quiet_server_has_no_child_holding_the_fork() -> None:
    assert await read_fork_holders(RecordingServer(QUIET_PERSISTENCE)) == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize(('flag', 'holder'), FLAG_HOLDERS)
async def test_each_set_flag_names_its_child(flag: str, holder: ForkHolder) -> None:
    server = RecordingServer({**QUIET_PERSISTENCE, flag: 1})

    assert await read_fork_holders(server) == frozenset({holder})


@pytest.mark.asyncio
async def test_two_set_flags_name_both_children() -> None:
    server = RecordingServer(
        {**QUIET_PERSISTENCE, 'rdb_bgsave_in_progress': 1, 'module_fork_in_progress': 1}
    )

    assert await read_fork_holders(server) == frozenset(
        {ForkHolder.RDB_SAVE, ForkHolder.MODULE_FORK}
    )


@pytest.mark.asyncio
async def test_the_holders_are_read_from_the_persistence_section() -> None:
    server = RecordingServer(QUIET_PERSISTENCE)

    await read_fork_holders(server)

    assert server.sections == ['persistence']


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', [flag for flag, _ in FLAG_HOLDERS])
async def test_a_reply_missing_a_flag_is_unreadable_never_free(flag: str) -> None:
    reply = {key: value for key, value in QUIET_PERSISTENCE.items() if key != flag}

    with pytest.raises(ForkSlotUnreadable) as raised:
        await read_fork_holders(RecordingServer(reply))

    assert raised.value.missing == (flag,)
    assert flag in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', [flag for flag, _ in FLAG_HOLDERS])
@pytest.mark.parametrize('value', ['0', '1', 2, -1, 1.0, None, {'a': 1}])
async def test_a_flag_that_is_not_0_or_1_is_unreadable_never_free_or_held(
    flag: str, value: object
) -> None:
    server = RecordingServer({**QUIET_PERSISTENCE, flag: value})

    with pytest.raises(ForkSlotUnreadable) as raised:
        await read_fork_holders(server)

    assert raised.value.malformed == ((flag, value),)
    assert raised.value.missing == ()
    assert f'{flag}={value!r}' in str(raised.value)

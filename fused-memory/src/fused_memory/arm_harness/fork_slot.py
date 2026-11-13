"""Which child process holds a Redis server's single fork slot, read from INFO persistence."""

from collections.abc import Awaitable, Mapping
from enum import StrEnum
from typing import Protocol

PERSISTENCE_SECTION = 'persistence'


class ForkHolder(StrEnum):
    """A child process that can hold the fork; each value is the INFO flag naming it."""

    RDB_SAVE = 'rdb_bgsave_in_progress'
    AOF_REWRITE = 'aof_rewrite_in_progress'
    MODULE_FORK = 'module_fork_in_progress'


class ServerInfo(Protocol):
    """The slice of redis.asyncio.Redis (falkordb.asyncio.FalkorDB.connection) that reads INFO."""

    def info(self, section: str, /) -> Awaitable[Mapping[str, object]]: ...


class ForkSlotUnreadable(RuntimeError):
    """The INFO reply does not report every fork-holder flag as 0 or 1, so whether the fork is free is unknown."""

    def __init__(
        self, missing: tuple[str, ...], malformed: tuple[tuple[str, object], ...]
    ) -> None:
        self.missing = missing
        self.malformed = malformed
        unread = [f'{flag} (absent)' for flag in missing] + [
            f'{flag}={value!r}' for flag, value in malformed
        ]
        super().__init__(
            f'INFO {PERSISTENCE_SECTION} did not report {", ".join(unread)} as 0 or 1, so which '
            'child holds the fork cannot be read'
        )


async def read_fork_holders(server: ServerInfo) -> frozenset[ForkHolder]:
    reply = await server.info(PERSISTENCE_SECTION)
    missing = tuple(holder.value for holder in ForkHolder if holder.value not in reply)
    malformed = tuple(
        (holder.value, reply[holder.value])
        for holder in ForkHolder
        if holder.value in reply and not _is_flag(reply[holder.value])
    )
    if missing or malformed:
        raise ForkSlotUnreadable(missing, malformed)
    return frozenset(holder for holder in ForkHolder if reply[holder.value] == 1)


def _is_flag(value: object) -> bool:
    return isinstance(value, int) and value in (0, 1)

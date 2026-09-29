"""A Mem0 stand-in for the flag_for_stage2 relay pool (task 4376).

It honours the metadata filters the retirement path sends, AND-ed within one
dict as Qdrant's are, and its pool really shrinks on delete. So a test can
observe which markers a path read and retired, and a repeat run sees what an
earlier one did. A missing id deletes quietly, as ``delete_memory`` does.

Shared by ``test_stages.py`` (the retirement seam against the sweep) and
``test_targeted.py`` (the done hook that calls it).
"""

from unittest.mock import AsyncMock


class LiveFlagPool:
    recon_ledger = None

    def __init__(self, members: list[dict]):
        self.members = {m['id']: m for m in members}
        self.get_memories_by_metadata = AsyncMock(side_effect=self._scroll)
        self.count_memories_by_metadata = AsyncMock(side_effect=self._count)
        self.delete_memory = AsyncMock(side_effect=self._delete)

    def _matching(self, filters: dict) -> list[dict]:
        return [
            m for m in self.members.values()
            if all(m['metadata'].get(key) == value for key, value in filters.items())
        ]

    async def _scroll(self, *, project_id: str, filters: dict, limit: int) -> list[dict]:
        return [dict(m) for m in self._matching(filters)][:limit]

    async def _count(self, *, project_id: str, filters: dict) -> int:
        return len(self._matching(filters))

    async def _delete(self, *, memory_id: str, **_: object) -> None:
        self.members.pop(memory_id, None)

    def deleted_ids(self) -> list[str]:
        return [c.kwargs['memory_id'] for c in self.delete_memory.await_args_list]

    def filters_read(self) -> list[dict]:
        calls = (
            self.get_memories_by_metadata.await_args_list
            + self.count_memories_by_metadata.await_args_list
        )
        return [c.kwargs['filters'] for c in calls]

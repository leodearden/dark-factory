"""The on-disk temp files a Claude CLI spawn is handed by argv path: their names, and the
reclamation of the ones a killed spawner orphaned."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpawnTempName:
    """One kind of spawn temp file, named the way ``tempfile.mkstemp`` names it.

    ``matches`` mirrors CPython's mkstemp naming (8 chars of ``[a-z0-9_]``), so if
    CPython ever changes it the sweep under-deletes rather than over-deletes, and
    shared/tests/test_spawn_temp_files.py fails loudly against the real mkstemp.
    """

    prefix: str
    suffix: str
    _shape: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = re.compile(re.escape(self.prefix) + '[a-z0-9_]{8}' + re.escape(self.suffix))
        object.__setattr__(self, '_shape', shape)

    def mkstemp(self) -> tuple[int, str]:
        return tempfile.mkstemp(prefix=self.prefix, suffix=self.suffix)

    def matches(self, name: str) -> bool:
        return self._shape.fullmatch(name) is not None


SYSPROMPT_TEMP_NAME = SpawnTempName('sysprompt_', '.txt')
MCP_CONFIG_TEMP_NAME = SpawnTempName('mcp_', '.json')
SPAWN_TEMP_NAMES = (SYSPROMPT_TEMP_NAME, MCP_CONFIG_TEMP_NAME)

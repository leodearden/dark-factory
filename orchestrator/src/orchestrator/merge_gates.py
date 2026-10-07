"""``orchestrator.merge_gates`` is ``orchestrator.merge_lane.gates`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import gates
from orchestrator.merge_lane.gates import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = gates

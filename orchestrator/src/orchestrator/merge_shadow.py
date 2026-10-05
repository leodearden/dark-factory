"""``orchestrator.merge_shadow`` is ``orchestrator.merge_lane.shadow`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import shadow
from orchestrator.merge_lane.shadow import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = shadow

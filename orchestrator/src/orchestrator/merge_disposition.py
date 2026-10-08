"""``orchestrator.merge_disposition`` is ``orchestrator.merge_lane.disposition`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import disposition
from orchestrator.merge_lane.disposition import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = disposition

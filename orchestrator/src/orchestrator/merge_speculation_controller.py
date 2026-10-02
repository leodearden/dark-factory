"""``orchestrator.merge_speculation_controller`` is ``orchestrator.merge_lane.speculation_controller`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import speculation_controller
from orchestrator.merge_lane.speculation_controller import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = speculation_controller

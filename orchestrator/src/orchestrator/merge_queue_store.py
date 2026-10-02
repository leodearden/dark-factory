"""``orchestrator.merge_queue_store`` is ``orchestrator.merge_lane.queue_store`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import queue_store
from orchestrator.merge_lane.queue_store import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = queue_store

"""``orchestrator.merge_skew_tripwire`` is ``orchestrator.merge_lane.skew_tripwire`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import skew_tripwire
from orchestrator.merge_lane.skew_tripwire import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = skew_tripwire

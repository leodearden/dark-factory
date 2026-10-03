"""``orchestrator.landing_evidence`` is ``orchestrator.merge_lane.landing_evidence`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import landing_evidence
from orchestrator.merge_lane.landing_evidence import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = landing_evidence

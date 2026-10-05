"""``orchestrator.merge_completion`` is ``orchestrator.merge_lane.completion`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys

from orchestrator.merge_lane import completion
from orchestrator.merge_lane.completion import *  # noqa: F403

sys.modules[__name__] = completion

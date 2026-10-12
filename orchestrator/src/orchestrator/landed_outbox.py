"""``orchestrator.landed_outbox`` is ``orchestrator.merge_lane.landed_outbox`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys

from orchestrator.merge_lane import landed_outbox
from orchestrator.merge_lane.landed_outbox import *  # noqa: F403

sys.modules[__name__] = landed_outbox

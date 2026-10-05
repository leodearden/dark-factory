"""``orchestrator.merge_request_ledger`` is ``orchestrator.merge_lane.request_ledger`` under its pre-package name; task 5037 migrates its importers and deletes this file."""
import sys
from typing import Any

from orchestrator.merge_lane import request_ledger
from orchestrator.merge_lane.request_ledger import *  # noqa: F403


def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = request_ledger

"""Which escalation store a ``create_server`` instance serves.

The γ1 store-identity substrate from ``plans/escalation-store-ambiguity-prd.md``
§6.1: ``escalation.server::create_server`` takes one of these so that γ2 can
reject a ``project_root`` assertion naming a different store and γ3 can render
the store into every tool description.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class StoreIdentity:
    kind: Literal['project', 'reconciliation']
    queue_dir: Path
    project_id: str | None
    project_root: Path | None

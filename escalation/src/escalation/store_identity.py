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
    """Both path fields are made absolute and resolved at construction, so
    ``queue_dir`` compares directly against any other resolved path —
    ``escalation.queue::EscalationQueue.__init__`` keeps its ``queue_dir``
    as given.
    """

    kind: Literal['project', 'reconciliation']
    queue_dir: Path
    project_id: str | None
    project_root: Path | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'queue_dir', Path(self.queue_dir).resolve())
        if self.project_root is not None:
            object.__setattr__(self, 'project_root', Path(self.project_root).resolve())

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
    """Both path fields are made absolute and resolved at construction.

    A ``'project'`` identity must carry both ``project_id`` and
    ``project_root``; a ``'reconciliation'`` identity must carry neither.
    Constructing an incoherent identity raises ``ValueError``.
    """

    kind: Literal['project', 'reconciliation']
    queue_dir: Path
    project_id: str | None
    project_root: Path | None

    def __post_init__(self) -> None:
        self._require_project_fields_match_kind()
        object.__setattr__(self, 'queue_dir', Path(self.queue_dir).resolve())
        if self.project_root is not None:
            object.__setattr__(self, 'project_root', Path(self.project_root).resolve())

    def _require_project_fields_match_kind(self) -> None:
        serves_a_project = self.kind == 'project'
        for field_name in ('project_id', 'project_root'):
            value = getattr(self, field_name)
            if serves_a_project and value is None:
                raise ValueError(f"StoreIdentity kind='project' requires {field_name}; got None")
            if not serves_a_project and value is not None:
                raise ValueError(
                    f'StoreIdentity kind={self.kind!r} must not carry {field_name}; got {value!r}'
                )

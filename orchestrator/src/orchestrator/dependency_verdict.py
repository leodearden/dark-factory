"""The tri-state answer to "are this task's dependencies satisfied?" (task 6518).

Dispatch reads only ``.satisfied``, so an UNKNOWN dependency waits exactly as
an UNSATISFIED one does.  Park GC expires a park only on UNSATISFIED and keeps
it on UNKNOWN: expiry needs affirmative evidence (esc-6040-7 D2, INV-3
corroborate before acting).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class DepKind(StrEnum):
    """Which dependency arm produced a non-satisfied verdict."""

    LOCAL = 'local'
    EXTERNAL = 'external'
    DELIVERED_CHECK = 'delivered_check'


class DepState(StrEnum):
    SATISFIED = 'satisfied'
    UNSATISFIED = 'unsatisfied'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class DepsVerdict:
    """A dependency answer; ``kind`` names the arm iff the answer is not SATISFIED."""

    state: DepState
    kind: DepKind | None = None

    def __post_init__(self) -> None:
        if (self.kind is None) != (self.state is DepState.SATISFIED):
            raise ValueError(
                f'a {self.state} dependency verdict cannot carry kind {self.kind}: '
                'a kind names the arm of an unsatisfied or unknown verdict only'
            )

    @classmethod
    def unsatisfied(cls, kind: DepKind) -> DepsVerdict:
        return cls(DepState.UNSATISFIED, kind)

    @classmethod
    def unknown(cls, kind: DepKind) -> DepsVerdict:
        return cls(DepState.UNKNOWN, kind)

    @property
    def satisfied(self) -> bool:
        return self.state is DepState.SATISFIED

    @staticmethod
    def combine(verdicts: Iterable[DepsVerdict]) -> DepsVerdict:
        """The first UNSATISFIED verdict, else the first UNKNOWN, else SATISFIED."""
        first_unknown: DepsVerdict | None = None
        for verdict in verdicts:
            if verdict.state is DepState.UNSATISFIED:
                return verdict
            if verdict.state is DepState.UNKNOWN and first_unknown is None:
                first_unknown = verdict
        return DEPS_SATISFIED if first_unknown is None else first_unknown


DEPS_SATISFIED = DepsVerdict(DepState.SATISFIED)

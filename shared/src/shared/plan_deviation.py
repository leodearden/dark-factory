"""The closed vocabulary and record shape for per-step plan deviations.

This is the one home that the plan-tools server (validates), ``roles.py``
(renders the doctrine, leaf γ) and the leaf-δ census (reads) import; see
``plans/plan-deviation-recording-prd.md`` D4. It is intentionally NOT
re-exported from ``shared/__init__.py``, following the ``governed_exceptions``
sub-module convention that ``shared/tests/test_public_api.py`` pins.
"""
from __future__ import annotations

import dataclasses

__all__ = ['DEVIATION_KINDS', 'DeviationRecord', 'UnknownDeviationKind']

DEVIATION_KINDS: frozenset[str] = frozenset({
    'never_checked_fact',
    'wrong_inference',
    'plan_internal_contradiction',
    'inherited_from_task_text',
    'narrowing_refuted_by_measurement',
    'prediction_falsified',
    'other',
})


class UnknownDeviationKind(ValueError):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.allowed = tuple(sorted(DEVIATION_KINDS))
        super().__init__(
            f'kind must be one of the closed deviation vocabulary; got {kind!r}. '
            f'Allowed kinds: {", ".join(self.allowed)}.'
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeviationRecord:
    """One per-step deviation entry.

    ``recorded_by`` is best-effort: the session identity the writer had in
    hand, which may be the planning session's rather than the recorder's.
    """

    kind: str
    observed: str
    expected: str
    action_taken: str
    prose_only: bool
    note: str
    recorded_at: str
    recorded_by: str | None

    def __post_init__(self) -> None:
        if self.kind not in DEVIATION_KINDS:
            raise UnknownDeviationKind(self.kind)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

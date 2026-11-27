"""The ``defect_tests:`` config group (plans/defect-tests-prd.md, sections Contract / Config and I5)."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

__all__ = ['DefectTestsConfig', 'DefectTestsMode']

_M1_TEST_FILE_PATTERN = (
    r'(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|_test\.go$'
    r'|\.test\.tsx?$|\.spec\.tsx?$|(^|/)conftest\.py$'
)


class DefectTestsMode(StrEnum):
    """Whether the defect-test gates act, record only, or are off; each member IS the YAML value."""

    OFF = 'off'
    SHADOW = 'shadow'
    ENFORCE = 'enforce'


class DefectTestsConfig(BaseModel):
    """Knobs for the defect-test plan, proof and merge gates."""

    mode: DefectTestsMode = Field(
        default=DefectTestsMode.SHADOW,
        description=(
            'off / shadow / enforce. Governs only the plan refusal, the '
            'proof-driven replan and the merge refusal (PRD I5); in shadow they '
            'record and do not act. off also stops the verify legs injecting '
            'the -p _df_defect_plugin pytest plugin.'
        ),
    )
    max_plan_repairs: int = Field(
        default=1,
        ge=0,
        description='Architect repairs allowed for a plan that does not reproduce its declared defects.',
    )
    max_proof_replans: int = Field(
        default=1,
        ge=0,
        description='Replans allowed after a proof that does not prove its declared defects.',
    )
    proof_timeout_secs: float = Field(
        default=1800.0,
        gt=0,
        description=(
            'Wall-clock bound for one proof side: one verify command\'s warm '
            'budget (verify_command_timeout_secs). The proof stage measures it '
            'and may retune it (PRD open question 1).'
        ),
    )
    streak_threshold: int = Field(
        default=5,
        ge=1,
        description='Consecutive unproven tasks before one L2 escalation is filed for the streak.',
    )
    test_file_pattern: str = Field(
        default=_M1_TEST_FILE_PATTERN,
        description=(
            'Regex naming a test file. Consumers apply it with re.search to '
            'repo-relative POSIX paths. The default is the M1 rule of '
            'plans/test-per-bug-investigation-2026-10-15.md section 3.'
        ),
    )

    @field_validator('test_file_pattern')
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f'test_file_pattern {value!r} does not compile: {exc}') from exc
        return value

    @property
    def injects_plugin(self) -> bool:
        """Whether verify's test legs inject ``-p _df_defect_plugin``: the single home of that off-switch.

        The mode otherwise governs only the three I5 gates (plan refusal,
        proof-driven replan, merge refusal); this switch is the one other
        thing it turns off.
        """
        return self.mode is not DefectTestsMode.OFF

"""Anchor/contract test for the writer-facing metadata-vocabulary section
(task 3202, PRD ``docs/prds/memory-metadata-vocabulary.md`` leaf ι).

The PRD's §6 gap was measurable: the word ``metadata`` did not appear anywhere
in ``_MEMORY_INSTRUCTIONS``, so every agent role told to write memories was
told nothing about the reserved keys the writer path actually validates
(``fused-memory/src/fused_memory/memory_metadata.py``, leaf β / task 3195).
Leaf ι closes that gap by appending ``METADATA_VOCABULARY_INSTRUCTIONS`` to the
memory block.

This module is the ORCHESTRATOR-LOCAL anchor: presence, brace-safety, and the
role-specificity split, all with no cross-package import.  The registry pin —
"the keys documented here are exactly ``RESERVED_VOCABULARY_KEYS``" — lives in
``fused-memory/tests/test_metadata_vocabulary_prompt_pinning.py`` instead,
because only that suite has both packages on its pythonpath (fused-memory's
``[tool.pytest.ini_options] pythonpath = ["src", "../orchestrator/src"]``), so
the drift guard is a hard import there rather than a silently-skipped no-op.

SEQUENCING SEAM — task 3131 (dep-gated behind 3169) inverts the write-eagerness
guidance in the PRECEDING part of ``_MEMORY_INSTRUCTIONS`` (the "Write when you
discover…" / "Write immediately…" prose).  This module deliberately pins NONE of
those sentences: a prose pin would collide head-on with 3131.  The additive
contract is encoded STRUCTURALLY instead —
``_MEMORY_INSTRUCTIONS.endswith(METADATA_VOCABULARY_INSTRUCTIONS)`` with a
non-empty prefix — which says "ι only ADDS, at the end" while leaving every
sentence 3131 owns free to change.
"""

from __future__ import annotations

import pytest

from orchestrator.agents.roles import (
    _MEMORY_INSTRUCTIONS,
    METADATA_VOCABULARY_INSTRUCTIONS,
    ROLES,
)

# The roles whose system_prompt splices the memory block, and therefore must
# carry the vocabulary section.  Derived-then-asserted rather than trusted: the
# literal tuple below is the CONTRACT, and `test_memory_roles_split_is_exact`
# checks it still matches what `ROLES` actually splices.
_MEMORY_ROLES = ('architect', 'debugger', 'deep_reviewer', 'implementer', 'simple_task')

# The complement.  Role-specificity is part of the contract (mirroring
# test_standing_decision_prompt_drift.py's present/absent split): these roles do
# not write memories from a task briefing, and silently widening the splice is
# the kind of prompt-bloat regression this guard exists to catch.
_NON_MEMORY_ROLES = ('judge', 'merger', 'reviewer_comprehensive', 'steward')


def test_vocabulary_section_is_nonempty() -> None:
    """The mandated constant is a non-empty string."""
    assert isinstance(METADATA_VOCABULARY_INSTRUCTIONS, str)
    assert METADATA_VOCABULARY_INSTRUCTIONS.strip()


def test_vocabulary_section_is_brace_free() -> None:
    """No literal `{`/`}` — the roles.py convention for spliced constants.

    Role prompts are plain `+`-concatenated literals, deliberately NOT
    f-strings, precisely because braces would be mangled if a splice site ever
    interpolated them (see the MANDATED_STAGING_COMMAND and
    BACKGROUND_TASK_WARNING notes in roles.py).  Staying brace-free keeps this
    section safe under a future interpolating splice.
    """
    assert '{' not in METADATA_VOCABULARY_INSTRUCTIONS
    assert '}' not in METADATA_VOCABULARY_INSTRUCTIONS


def test_memory_roles_split_is_exact() -> None:
    """The 5/4 split is asserted, not assumed.

    `ROLES` has 9 entries; exactly 5 splice `_MEMORY_INSTRUCTIONS`.  If a role
    is added, or the splice widens/narrows, this fails before the parametrized
    presence/absence tests below can go quietly stale.
    """
    spliced = {name for name, role in ROLES.items() if _MEMORY_INSTRUCTIONS in role.system_prompt}
    assert spliced == set(_MEMORY_ROLES)
    assert set(ROLES) - spliced == set(_NON_MEMORY_ROLES)
    assert set(ROLES) == set(_MEMORY_ROLES) | set(_NON_MEMORY_ROLES)


@pytest.mark.parametrize('role_name', _MEMORY_ROLES)
def test_vocabulary_section_present_in_memory_roles(role_name: str) -> None:
    """Each memory-writing role's assembled system_prompt carries the section
    VERBATIM (a byte-identical `in` check, not a prose paraphrase)."""
    assert METADATA_VOCABULARY_INSTRUCTIONS in ROLES[role_name].system_prompt


@pytest.mark.parametrize('role_name', _NON_MEMORY_ROLES)
def test_vocabulary_section_absent_from_non_memory_roles(role_name: str) -> None:
    """The roles that do not splice the memory block must not carry it."""
    assert METADATA_VOCABULARY_INSTRUCTIONS not in ROLES[role_name].system_prompt


def test_vocabulary_section_is_appended_not_interleaved() -> None:
    """ADDITIVE SEAM GUARD for task 3131.

    The vocabulary section is the TAIL of the memory block, and the prose
    before it is non-empty.  This is the whole additive contract: ι appends,
    3131 rewrites what comes before, and neither edit needs to know the other's
    wording.  Do NOT strengthen this into a pin on any preceding sentence.
    """
    assert _MEMORY_INSTRUCTIONS.endswith(METADATA_VOCABULARY_INSTRUCTIONS)
    prefix = _MEMORY_INSTRUCTIONS[: -len(METADATA_VOCABULARY_INSTRUCTIONS)]
    assert prefix.strip(), 'the memory block must still carry its pre-existing guidance'

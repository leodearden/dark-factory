"""Anchor/contract test for the Layer-1 background-task guard (task 2761).

Models test_roles_staging_command.py: assert the mandated
``BACKGROUND_TASK_WARNING`` constant is present verbatim in every assembled
prompt that must carry it — the ``implementer`` and ``debugger`` role system
prompts. The built-amender-turn-prompt reinforcement check (the amender has no
dedicated role, so it inherits IMPLEMENTER's system prompt; the turn-prompt
injection reinforces the guard at the exact failure site — its "Run
verification before finishing" step is where Reify-5164's amender backgrounded
a 2700s verify and ended its turn) now lives in
``test_roles_wait_pattern.py::test_amender_prompt_reinforces_the_wait_rules``,
consolidated there with its near-duplicate sibling (task 3747) — see the
comment at the bottom of this file.

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine safety regression — the repo
sanctions exactly this kind of "mandated token present in each role prompt"
guard.  Each assertion is a one-line existence check, not a prose pin.
"""

from __future__ import annotations

from orchestrator.agents.roles import BACKGROUND_TASK_WARNING, ROLES


def test_background_warning_is_nonempty() -> None:
    """The mandated constant is a non-empty string."""
    assert BACKGROUND_TASK_WARNING.strip()


def test_implementer_system_prompt_carries_warning() -> None:
    """IMPLEMENTER.system_prompt embeds the warning verbatim (also covers the
    amender, which is invoked under the IMPLEMENTER role)."""
    assert BACKGROUND_TASK_WARNING in ROLES['implementer'].system_prompt


def test_debugger_system_prompt_carries_warning() -> None:
    """DEBUGGER.system_prompt embeds the warning verbatim."""
    assert BACKGROUND_TASK_WARNING in ROLES['debugger'].system_prompt


# The amender-turn-prompt reinforcement test that used to live here
# (``test_amender_prompt_reinforces_warning_at_failure_site``) was a
# near-duplicate of ``test_roles_wait_pattern.py::
# test_amender_prompt_reinforces_the_wait_rules`` — both built the same
# ``build_amender_prompt`` call over the same stub ``briefing`` fixture and
# asserted overlapping things at the same "Run verification before finishing"
# failure site. Task 3747 consolidated the two into that single test (which
# now also carries this file's ``BACKGROUND_TASK_WARNING in
# ROLES['implementer'].system_prompt`` assertion) and the shared ``briefing``
# fixture into ``conftest.py``, rather than leaving byte-similar copies to
# drift apart in these two sibling files.

"""Shared shape behind the role-prompt splice-contract anchor tests.

A "splice contract" is the family of invariants that hold when a named prompt
constant is spliced into a machine-derived set of role prompts: the constant is
non-empty, it reaches every role in the set, it reaches no role outside it, it
appears exactly once per role, it lands in a structurally defined spot, and the
hand-maintained role set still equals the set derived from the capability that
justifies the splice. Two anchor-test modules assert that family today:

- ``test_roles_wait_pattern.py`` — task 3607, ``BACKGROUND_WAIT_GUIDANCE``.
- ``test_roles_tool_call_rejection.py`` — tasks 4273/4578,
  ``TOOL_CALL_REJECTION_GUIDANCE``.

Both grew the same ~240-line shape independently. This module holds it once, so
a THIRD prompt constant costs ~10 lines of contract construction plus one-line
test bodies rather than a third clone.

THE STANDING RULE THIS MODULE MUST NOT WEAKEN, carried over from both consumers:
every assertion here is an existence / containment / count / index check against
a NAMED CONSTANT — never a string literal asserted against a constant's prose,
never a regex over wording, never a byte-size figure. A prose pin has no
correctness content in either direction: it passes on prose reworded to say the
opposite and fails on a legitimate tightening, so it only taxes future prompt
edits. Two such literal pins were tried in the wait file and removed (task 3607
review). Do not add a prose-pinning assertion to this module, and do not
"strengthen" any index check into a regex.

THE ``capability`` PREDICATE IS DELIBERATELY A PARAMETER, not a unified rule.
The two consumers ask genuinely different questions and both are correct for
their constant — see ``SpliceContract``'s class docstring.

EVERY ASSERTION TAKES A CALLER-SUPPLIED ``remedy``. This module formats only the
mechanical half of a failure message (the offender list or mapping, the
constant's name, the role-set's name, the rule that was applied); the
remediation prose stays at the call site, because that prose is where the
diagnostic value of these messages actually lives and it differs per constant.
``test_role_splice_contract.py`` pins that the ``remedy`` is rendered, so it
cannot be silently dropped.

Flat, underscore-prefixed test helper: no ``__init__.py`` in ``tests/``, imported
by bare module name (``from _role_splice_contract import SpliceContract``) off the
``sys.path`` entry ``conftest.py`` inserts. Same convention as
``_orch_helpers.py``, ``_workflow_helpers.py`` and ``_serial_merge_worker.py``.
The leading underscore also keeps pytest from collecting it as a test module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from orchestrator.agents.roles import ROLES, AgentRole

# A splice unit opens with its own ``\n## `` heading, so "spliced between the
# identity paragraph and the role's first real section" is exactly "its heading
# IS the first ``##`` heading in the prompt". That is a STRUCTURAL property of
# the splice, not a prose pin: it keeps holding when every heading in roles.py
# is renamed. It replaces an earlier ``'## Escalation'`` landmark that silently
# no-opped when the heading was not found.
MARKDOWN_HEADING = '\n## '


def assert_nonempty(name: str, value: str, *, remedy: str) -> None:
    """Assert ``value`` has non-whitespace content.

    NOT redundant with the containment assertions, though it reads that way: the
    empty string is a substring of every string, so every ``constant in prompt``
    check written against an emptied constant holds vacuously. This is the sole
    guard against a mandated block being silently dropped in a prompt refactor.

    ``.strip()`` rather than a bare truthiness check, because a constant emptied
    to ``'\\n'`` contributes nothing to a prompt while still being truthy.
    """
    assert value.strip(), (
        f'{name} is empty. Every containment assertion written against it still '
        'passes when it is — the empty string is a substring of anything — so '
        f'this assertion is the sole guard against it being silently dropped. '
        f'{remedy}'
    )


def assert_brace_free(name: str, value: str, *, remedy: str) -> None:
    """Assert ``value`` contains no literal ``{`` or ``}``.

    A constant that reaches a prompt by plain ``+`` concatenation is brace-safe
    by construction, but one that reaches an f-string or ``str.format`` site is
    not: a literal brace raises at format time or silently mangles the rendered
    prompt. Held brace-free defensively so a constant stays interpolation-safe
    if a future splice site needs it.
    """
    assert '{' not in value and '}' not in value, (
        f'{name} contains a literal brace, so it is not safe at an interpolating '
        f'splice site (it raises at format time or mangles the rendered prompt). '
        f'{remedy}'
    )


@dataclass(frozen=True)
class SpliceContract:
    """One prompt constant, the role set it is spliced into, and why.

    Frozen so a consumer's module-level ``_CONTRACT`` cannot be mutated by one
    test and silently change what a later test in the same module asserts.

    THE ASYMMETRY THIS DATACLASS EXISTS TO HOLD. ``capability`` is injected
    rather than unified because the two consumers ask genuinely DIFFERENT
    questions, and each is correct for its own constant:

    - ``'Bash' in role.allowed_tools`` — can this role launch a long-running
      command, so the wait guidance is not dead weight? (Excludes
      ``reviewer_comprehensive`` and ``judge``, which hold only
      ``'Bash(git:*)'``.)
    - ``role.prompt_spec is None`` — is this role's ``system_prompt`` literal
      text a splice can reliably reach, rather than a ``PromptSpec`` whose
      pinned artifact may override it at runtime? (Adds ``judge``.)

    Unifying them would either splice the wait block into ``judge``, where it is
    dead weight the wait file's comment explicitly justifies excluding, or drop
    the rejection guidance from ``judge``, where it is needed. The duplication
    this module removes is in the DERIVATION SHAPE, not the predicate.

    Attributes:
        constant_name: The constant's Python name, for failure messages.
        constant: The splice unit itself.
        roles: The hand-maintained role set the constant is spliced into.
        role_set_name: That set's Python name, so a failure names the variable
            a reader has to edit.
        capability: The property that justifies the splice, applied over
            ``all_roles`` to derive the set ``roles`` is checked against.
        capability_description: Prose naming that property, for messages.
        all_roles: The role mapping to derive over. Defaults to the real
            ``ROLES``; injectable so this module's own contract test can drive
            synthetic roles through the failure branches without mutating a
            production prompt.
    """

    constant_name: str
    constant: str
    roles: frozenset[str]
    role_set_name: str
    capability: Callable[[AgentRole], bool]
    capability_description: str
    # ``default_factory``, not a bare default: ``dict`` is unhashable and
    # ``dataclasses`` rejects a mutable default outright.
    all_roles: Mapping[str, AgentRole] = field(default_factory=lambda: ROLES)

    def assert_role_set_matches_capability(self, *, remedy: str) -> None:
        """Drift tripwire: the hand-maintained role set equals the derived one.

        Catches a role GAINING or LOSING the capability that justifies the
        splice, which is the edit that silently leaves a newly-eligible role
        without the constant (or leaves a no-longer-eligible role carrying dead
        weight). Reported as ``gained=``/``lost=`` rather than a bare inequality
        so the failure names the roles a reader has to act on.
        """
        derived = {name for name, role in self.all_roles.items() if self.capability(role)}

        assert derived == self.roles, (
            f'A role gained or lost {self.capability_description}, so the set of '
            f'roles eligible for {self.constant_name} has changed: '
            f'gained={sorted(derived - self.roles)} '
            f'lost={sorted(self.roles - derived)}. '
            f'The hand-maintained set is {self.role_set_name}. {remedy}'
        )

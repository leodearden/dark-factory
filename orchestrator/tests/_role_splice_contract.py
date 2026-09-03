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

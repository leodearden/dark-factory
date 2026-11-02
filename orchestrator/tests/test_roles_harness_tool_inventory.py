"""Inventory guard: a role prompt may not name a harness tool the harness lacks.

The built-in twin of `test_roles_ancestry_check.py`'s
`test_role_holds_every_mcp_tool_its_prompt_names`. That test asks whether a
named MCP tool is GRANTED to the naming role; this one asks the prior
question for built-ins -- whether the named tool EXISTS at all.

THE DEFECT IT HOLDS (task 5332). `BACKGROUND_TASK_WARNING` told every
Bash-capable role to "poll it to completion with `BashOutput` (or terminate
it with `KillShell`)". Neither tool is in the harness registry, and neither
ever was in this fleet -- `MEASURED_ABSENT_TOOLS` below carries that
measurement, its dates and the fleet census that corroborates it. An agent
that reached that sentence had no reachable next step, so it improvised one
-- a foreground `sleep`/`tail` poll chain, which is what the NEVER list two
paragraphs below it forbids. The prompt was steering agents into its own
prohibition.

WHAT IS SCANNED -- two surfaces, and BOTH are required.
  (a) every role's `system_prompt` in `ROLES`;
  (b) every module-level `str` constant in `roles.py` and
      `harness_guidance.py` (`_PROMPT_MODULES`) whose NAME matches
      `_?[A-Z][A-Z0-9_]*`, reached by introspection over each module's `vars()`.
Surface (a) alone is a false green, measured: `WAIT_PATTERN_REMINDER`
carried `BashOutput` and appears in NO role system_prompt -- `briefing.py`
interpolates it into `build_amender_prompt` only -- so a system-prompt-only
scan reports green while the defect survives at the exact failure site the
reminder exists to cover. The optional leading underscore in the constant
pattern is required too: the two modules hold a family of PRIVATE prompt constants
(`_GREP_ENGINE_LIMITS`, `_TOOL_CALL_REJECTION_KNOWN_SHAPES`,
`_SCOPE_BOUNDARY_FACTS`, `_MEMORY_INSTRUCTIONS`, the `_REVIEWER_*_TEMPLATE`s)
that a pattern anchored at `[A-Z]` would skip. Introspection rather than a
hand-listed import set means a prompt constant added later to a listed module
is covered the day it is added, with no second place to remember;
`test_scan_reaches_the_turn_prompt_only_constant` fails if the witness's home
module drops out of `_PROMPT_MODULES`.

COMMENTS ARE NOT SCANNED, and that is deliberate rather than an accepted
gap. A comment may correctly name a phantom tool, as the cautionary precedent
for naming a tool that is not there. A scan over source TEXT would fail on
that correct comment and pressure a future editor into deleting the citation
-- the guard would attack the documentation of the very defect it polices.
Scanning constant VALUES gets the comment/prompt distinction right for free,
with no exclusion list.

THE MAINTENANCE CONTRACT. `HARNESS_TOOLS` is hand-maintained from live
measurement; extend it when a prompt legitimately names a newly-shipped tool,
and extend `NON_TOOL_TERMS` when a prompt backticks a CamelCase word that is
not a tool at all. Reword prompts freely -- this guard pins no phrasing, only
the invariant that a named tool is a real tool.

ITS KNOWN HOLE, stated rather than glossed because it is the hole this task
fell into. The check is asymmetric: it catches a prompt naming a tool that is
NOT in the list (the add-a-phantom direction), but a tool DELETED upstream
stays in `HARNESS_TOOLS` and a prompt naming it stays GREEN. That is not
hypothetical: this task's own first plan proposed replacing
`BashOutput`/`KillShell` with `TaskOutput`/`TaskStop`, and `TaskOutput` was
measured live, then gone ten days later (`MEASURED_ABSENT_TOOLS`). Nothing
importable from Python can see the harness registry, so this cannot be
automated away. The two mitigations are: re-derive the list rather than trust
it (`MEASURED_ABSENT_TOOLS` carries the query), and -- the one that actually
carries the weight -- keep the guidance itself depending on as few volatile
tool names as possible, which is why the wait block is now written around
`Bash`'s own `run_in_background` parameter, `Read` on the returned output
file path, and `ToolSearch` as the way to discover the current termination
tool. The hole has little left to bite on.

WHAT THIS DOES NOT COVER, deliberately:
  * prompt text assembled outside those two modules -- `briefing.py`'s
    f-strings, the `skills/` sources -- since the scan reads only their
    module-level constants;
  * comments, per the paragraph above;
  * a tool named in PLAIN PROSE. Only backtick spans are scanned, so "poll it
    with BashOutput" escapes. Dropping the backtick requirement is not the fix
    -- every sentence-initial word becomes a candidate. Stated because ITS
    KNOWN HOLE above is not the only hole, and reading it as exhaustive is
    what would let this one through unnoticed;
  * tool GRANTS. Unlike its MCP sibling this guard asserts EXISTENCE only,
    and the reason is measured and recorded in the MEASURED REACHABLE comment
    below: `--allowed-tools` is a PERMISSION allowlist, not a
    tool-registry filter, so built-ins stay present and callable whether or
    not a role lists them. An MCP tool missing from `allowed_tools` really is
    a permission denial, which is why the sibling checks grants; a built-in
    missing from it is nothing at all. Adding a grant check here would fail on
    every CORRECT prompt -- no role grants `ToolSearch` or `Monitor` today,
    and both are reachable -- and would re-assert the exact mistaken theory
    that comment exists to forbid.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable
from typing import NamedTuple

import pytest

from orchestrator.agents import harness_guidance as harness_guidance_module
from orchestrator.agents import roles as roles_module
from orchestrator.agents.roles import ROLES

# Harness tools measured PRESENT on 2026-09-20, from this dispatched session's
# own tool list (Agent .. Write below) plus its deferred-tool list, which
# `ToolSearch` loads on demand (CronCreate .. WebSearch below).  Re-derive with
# the query recorded at `MEASURED_ABSENT_TOOLS` below; do NOT add a name on the
# strength of remembering it.  The asymmetry that governs edits here: a name wrongly
# OMITTED costs a future editor one spurious failure and a one-line fix, while
# a name wrongly INCLUDED silently un-covers the exact defect this module
# exists to catch.
#
# Session-to-session variance is real and worth knowing about before you
# "clean up" this list: the architect that planned task 5332 measured the same
# registry hours earlier and did NOT see DesignSync, EnterWorktree,
# ExitWorktree, ListMcpResourcesTool, ReadMcpResourceDirTool,
# ReadMcpResourceTool or RemoteTrigger in its deferred list. They are kept
# because this session did observe them; a present-but-unlisted tool is the
# cheap direction to be wrong in.
HARNESS_TOOLS = frozenset({
    'Agent',
    'Bash',
    'CronCreate',
    'CronDelete',
    'CronList',
    'DesignSync',
    'Edit',
    'EnterWorktree',
    'ExitWorktree',
    'Glob',
    'Grep',
    'ListAgents',
    'ListMcpResourcesTool',
    'Monitor',
    'NotebookEdit',
    'PushNotification',
    'Read',
    'ReadMcpResourceDirTool',
    'ReadMcpResourceTool',
    'RemoteTrigger',
    'ReportFindings',
    'ScheduleWakeup',
    'SendMessage',
    'Skill',
    'TaskStop',
    'ToolSearch',
    'WebFetch',
    'WebSearch',
    'Workflow',
    'Write',
})

# THE SINGLE HOME OF THE TOOL-REGISTRY MEASUREMENT.  The query, its dates and
# the census figure live HERE and are cited from everywhere else -- this
# module's docstring, `_failure_message`, cli_invoke.py's
# `_BACKGROUND_REAP_TOOLS`, test_cli_invoke_background.py.  An earlier revision
# hand-copied them to six sites and two had already drifted apart in shape, the
# same failure test_roles_wait_pattern.py's docstring records for its size
# figures: a prose copy does not move when the string it describes does.  Do
# NOT restate a date, the query or the census figure elsewhere -- cite this
# comment instead.
#
# THE QUERY, from a live dispatched session:
#     ToolSearch("select:BashOutput,KillShell,KillBash,TaskOutput,TaskStop,Monitor")
# plus that session's own tool list and its deferred-tool list.  `select:` is an
# exact-name lookup, so a name it does not return is ABSENT from the registry,
# not merely low-ranked.  This re-derives `HARNESS_TOOLS` as well as this set.
#
# WHAT IT RETURNED.  Two measurements, and the second is the one that justifies
# the whole module:
#   2026-09-10 -- TaskOutput, TaskStop and Monitor; NOT BashOutput or KillShell.
#                 Corroborated fleet-wide by census 2026-09-10 sec 1.1: zero
#                 BashOutput/KillShell tool_use across 8,546 archived
#                 transcripts, and 138 ToolSearch requests naming them answered
#                 with other names.
#   2026-09-20 -- only TaskStop and Monitor; `select:TaskOutput` answered "No
#                 matching deferred tools found".  TaskOutput was live ten days
#                 earlier.
#
# The names below are kept as a diagnostic arm, so this regression reports that
# dated evidence instead of a generic "unknown token".
MEASURED_ABSENT_TOOLS = frozenset({
    'BashOutput',
    'KillBash',
    'KillShell',
    'TaskOutput',
})

# MEASURED REACHABLE, 2026-08-05 -- why this guard checks existence, never
# grants.  A dispatched implementer whose cmdline was exactly
# `--allowed-tools Read Edit Write Bash Glob Grep mcp__...` had `ToolSearch` and
# `ScheduleWakeup` live in its tool list, and a `ToolSearch` select query
# returned deferred tools' schemas.  So `--allowed-tools` is a PERMISSION
# allowlist, not a registry filter: a built-in stays callable whether or not a
# role lists it.  Never trim a built-in from a prompt because a role's
# allowed_tools omits it.

# Backticked CamelCase words the prompts legitimately use that name no tool.
# Because the scan takes the LEADING identifier of a span (below), this holds
# both bare words and the head of a longer span -- a prose label (`Hypothesis:`)
# or a symbol citation (`GitOps.advance_main`).
NON_TOOL_TERMS = frozenset({
    'GitOps',
    'Hypothesis',
    'InputValidationError',
    'Memory',
    'NotImplementedError',
})

# The LEADING identifier of a backtick span, so a tool named in CALL form is
# caught too: these prompts routinely write `ToolSearch("select:Monitor")` and
# `Bash(run_in_background=true)`, so an anchored `...`-only pattern would have
# let "collect it with `BashOutput(id)`" reinstate this task's defect GREEN.
# Requiring the backtick is what keeps the scan usable -- without it every
# sentence-initial word (`Never`, `Polling`, `Foreground`) becomes a candidate.
# Widening cost exactly two NON_TOOL_TERMS entries when measured against every
# prompt surface, both above.
_BACKTICKED = re.compile(r'`([A-Za-z][A-Za-z0-9_]*)[^`]*`')

# Module-level prompt constants, private ones included -- see the module
# docstring on why the leading underscore is not optional.
_PROMPT_CONSTANT_NAME = re.compile(r'^_?[A-Z][A-Z0-9_]*$')

# The modules whose namespaces scan surface (b) introspects, lowest layer
# first, so a constant roles.py imports is reported under harness_guidance.
_PROMPT_MODULES = (harness_guidance_module, roles_module)


class _PromptConstant(NamedTuple):
    module: str
    name: str
    text: str

    @property
    def where(self) -> str:
        return f'{self.module}.{self.name}'


def _is_camel_case(token: str) -> bool:
    """A leading capital plus at least one lowercase letter.

    Excludes all-caps words (SIGTERM, EACCES, SPOT) and lowercase ones
    (timeout, append), neither of which can be a tool name.
    """
    return token[0].isupper() and any(char.islower() for char in token)


def _prompt_constants(modules: Iterable[types.ModuleType]) -> list[_PromptConstant]:
    """Every module-level string constant in `modules`, by introspection.

    One entry per distinct binding of a name to an object: an import is the
    same object under the same name, so it is scanned once, under the first
    module that binds it; a shadowing value is another object, so it is
    scanned too.
    """
    first_binding: dict[tuple[str, int], _PromptConstant] = {}
    for module in modules:
        module_name = module.__name__.rpartition('.')[2]
        for name, value in vars(module).items():
            if _PROMPT_CONSTANT_NAME.match(name) and isinstance(value, str):
                first_binding.setdefault(
                    (name, id(value)), _PromptConstant(module_name, name, value),
                )
    return sorted(first_binding.values())


_SCANNED_CONSTANTS = _prompt_constants(_PROMPT_MODULES)


def _unknown_tool_tokens(text: str) -> list[str]:
    """Backticked CamelCase tokens in `text` that name no known harness tool."""
    return sorted({
        token
        for token in _BACKTICKED.findall(text)
        if _is_camel_case(token)
        and token not in HARNESS_TOOLS
        and token not in NON_TOOL_TERMS
    })


def _failure_message(where: str, unknown: list[str]) -> str:
    phantom = [token for token in unknown if token in MEASURED_ABSENT_TOOLS]
    unrecognised = [token for token in unknown if token not in MEASURED_ABSENT_TOOLS]
    lines = [f'{where} names harness tools that do not exist: {unknown}.']
    if phantom:
        lines.append(
            f'{phantom} are MEASURED ABSENT -- see MEASURED_ABSENT_TOOLS in this module '
            'for the query and its dates. An agent following this prompt has no '
            'reachable next step and will improvise a forbidden poll loop. Name a tool '
            'that exists, and do NOT reach for another poll-tool name: what agents are '
            'told to do instead is stated once, in '
            'orchestrator/src/orchestrator/agents/harness_guidance.py::WAIT_PATTERN_GUIDANCE.',
        )
    if unrecognised:
        lines.append(
            f'{unrecognised} are unrecognised. Verify each from a live dispatched session '
            'with ToolSearch("select:<name>") -- an exact-name lookup, so an empty answer '
            'means ABSENT, not low-ranked -- then add it to HARNESS_TOOLS with its '
            'measurement date. If it is not a tool at all, add it to NON_TOOL_TERMS '
            'instead.',
        )
    return ' '.join(lines)


def test_detector_detects() -> None:
    """The detector's own smoke test, over literal fixtures rather than prompts.

    Every other assertion in this module is `assert not unknown` -- an ABSENCE
    check. Absence checks pass just as cheerfully when the detector is broken:
    a typo in `_BACKTICKED`, an inverted `_is_camel_case`, or an
    `_unknown_tool_tokens` that returned `[]` unconditionally would leave every
    parametrized case GREEN and the invariant silently un-enforced. That is the
    false-green shape this module's docstring narrates for scan surface (a),
    one level further down. So: prove the guard can fail before trusting it to
    pass. Fixtures are literals, so this pins no prompt phrasing.
    """
    # The task-5332 defect verbatim: both phantoms named, both reported.
    assert _unknown_tool_tokens(
        'poll it to completion with `BashOutput` (or terminate it with `KillShell`)'
    ) == ['BashOutput', 'KillShell']

    # The corrected guidance's shape: real tools, nothing reported.
    assert _unknown_tool_tokens('`Read` the output file path `Bash` returned') == []

    # CALL form, the idiom these prompts actually use -- see `_BACKTICKED`.
    assert _unknown_tool_tokens('collect it with `BashOutput(bash_id)`') == ['BashOutput']

    # Non-candidates: unbackticked (the documented residual hole), all-caps,
    # lowercase, and an allowlisted non-tool.
    assert _unknown_tool_tokens('poll it with BashOutput') == []
    assert _unknown_tool_tokens('killed by `SIGTERM`, sized by `timeout`') == []
    assert _unknown_tool_tokens('prefix it `Hypothesis:` so a reviewer can tell') == []


def test_scan_keeps_a_shadowed_name_and_folds_an_import() -> None:
    """Two listed modules binding one name to different text are both scanned.

    Keyed by name alone, the later module's value would silently replace the
    earlier one's and the earlier text would never be scanned -- the
    false-green shape this module exists to rule out. An import binds the
    same object under the same name, so it is scanned once, under the first
    module listed. Fixtures are synthetic modules, so this pins no prompt.
    """
    imported = 'imported text'
    lower = types.ModuleType('pkg.lower')
    upper = types.ModuleType('pkg.upper')
    vars(lower).update(SHARED=imported, SHADOWED='lower text')
    vars(upper).update(SHARED=imported, SHADOWED='upper text')

    assert _prompt_constants((lower, upper)) == [
        _PromptConstant('lower', 'SHADOWED', 'lower text'),
        _PromptConstant('lower', 'SHARED', 'imported text'),
        _PromptConstant('upper', 'SHADOWED', 'upper text'),
    ]


def test_scan_reaches_the_turn_prompt_only_constant() -> None:
    """The scan surface still holds this module's false-green witness.

    WAIT_PATTERN_REMINDER reaches agents only through briefing.py and is in no
    role system_prompt, so scan surface (b) is its only cover. Dropping its
    home module from `_PROMPT_MODULES` would silently un-scan it; this test
    fails first. It is the detector-coverage twin of `test_detector_detects`,
    not a prose pin.
    """
    scanned = {constant.name for constant in _SCANNED_CONSTANTS}
    assert 'WAIT_PATTERN_REMINDER' in scanned, (
        'WAIT_PATTERN_REMINDER is no longer among the scanned prompt constants. '
        'Add the module that defines it to _PROMPT_MODULES.'
    )


@pytest.mark.parametrize('role_key', sorted(ROLES))
def test_role_prompt_names_no_phantom_harness_tool(role_key: str) -> None:
    """Every built-in tool a role's system_prompt prescribes must exist.

    Parametrized over every role rather than only the offending one, for the
    same reason its MCP sibling is: any role's prompt can acquire the mistake.
    Deliberately generic -- it pins no particular tool and no phrasing, so a
    rewording that keeps naming real tools still passes.
    """
    unknown = _unknown_tool_tokens(ROLES[role_key].system_prompt)
    assert not unknown, _failure_message(f"{role_key}'s system_prompt", unknown)


@pytest.mark.parametrize(
    'constant', _SCANNED_CONSTANTS, ids=lambda constant: constant.where,
)
def test_prompt_constant_names_no_phantom_harness_tool(constant: _PromptConstant) -> None:
    """Same invariant over the scanned modules' prompt constants, spliced or not.

    The surface that catches what the system_prompt scan cannot:
    WAIT_PATTERN_REMINDER reaches an agent only through briefing.py's
    build_amender_prompt, so it is absent from every role's system_prompt
    while still being prompt text an agent is expected to act on.
    """
    unknown = _unknown_tool_tokens(constant.text)
    assert not unknown, _failure_message(constant.where, unknown)

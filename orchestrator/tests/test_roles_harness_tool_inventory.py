"""Inventory guard: a role prompt may not name a harness tool the harness lacks.

The built-in twin of `test_roles_ancestry_check.py`'s
`test_role_holds_every_mcp_tool_its_prompt_names`. That test asks whether a
named MCP tool is GRANTED to the naming role; this one asks the prior
question for built-ins -- whether the named tool EXISTS at all.

THE DEFECT IT HOLDS (task 5332). `BACKGROUND_TASK_WARNING` told every
Bash-capable role to "poll it to completion with `BashOutput` (or terminate
it with `KillShell`)". Neither tool is in the harness registry, and neither
ever was in this fleet: census 2026-09-10 §1.1 found ZERO `BashOutput` /
`KillShell` tool_use across 8,546 archived transcripts, while 138
`ToolSearch` requests naming them were answered with other names. An agent
that reached that sentence had no reachable next step, so it improvised one
-- a foreground `sleep`/`tail` poll chain, which is what the NEVER list two
paragraphs below it forbids. The prompt was steering agents into its own
prohibition.

WHAT IS SCANNED -- two surfaces, and BOTH are required.
  (a) every role's `system_prompt` in `ROLES`;
  (b) every module-level `str` constant in `roles.py` whose NAME matches
      `_?[A-Z][A-Z0-9_]*`, reached by introspection over `vars(roles)`.
Surface (a) alone is a false green, measured: `WAIT_PATTERN_REMINDER`
carried `BashOutput` and appears in NO role system_prompt -- `briefing.py`
interpolates it into `build_amender_prompt` only -- so a system-prompt-only
scan reports green while the defect survives at the exact failure site the
reminder exists to cover. The optional leading underscore in the constant
pattern is required too: roles.py holds a family of PRIVATE prompt constants
(`_GREP_ENGINE_LIMITS`, `_TOOL_CALL_REJECTION_KNOWN_SHAPES`,
`_SCOPE_BOUNDARY_FACTS`, `_MEMORY_INSTRUCTIONS`, the `_REVIEWER_*_TEMPLATE`s)
that a pattern anchored at `[A-Z]` would skip. Introspection rather than a
hand-listed import set means a prompt constant added later is covered the day
it is added, with no second place to remember.

COMMENTS ARE NOT SCANNED, and that is deliberate rather than an accepted
gap. roles.py's `_GREP_ENGINE_LIMITS` comment cites `BashOutput` CORRECTLY,
as the cautionary precedent for naming a tool that is not there (task 5331).
A scan over roles.py's source TEXT would fail on that correct comment and
pressure a future editor into deleting the citation -- the guard would attack
the documentation of the very defect it polices. Scanning constant VALUES
gets the comment/prompt distinction right for free, with no exclusion list.

THE MAINTENANCE CONTRACT. `HARNESS_TOOLS` is hand-maintained from live
measurement; extend it when a prompt legitimately names a newly-shipped tool,
and extend `NON_TOOL_TERMS` when a prompt backticks a CamelCase word that is
not a tool at all. Reword prompts freely -- this guard pins no phrasing, only
the invariant that a named tool is a real tool.

ITS KNOWN HOLE, stated rather than glossed because it is the hole this task
fell into. The check is asymmetric: it catches a prompt naming a tool that is
NOT in the list (the add-a-phantom direction), but a tool DELETED upstream
stays in `HARNESS_TOOLS` and a prompt naming it stays GREEN. That is not
hypothetical. The first revision of this task's plan proposed replacing
`BashOutput`/`KillShell` with `TaskOutput`/`TaskStop`, measured live on
2026-09-10; re-measured on 2026-09-20 the SAME query returned only `TaskStop`
and `Monitor`, and `select:TaskOutput` answered "No matching deferred tools
found". Ten days. Nothing importable from Python can see the harness
registry, so this cannot be automated away. The two mitigations are: re-derive
the list rather than trust it (query below), and -- the one that actually
carries the weight -- keep the guidance itself depending on as few volatile
tool names as possible, which is why the wait block is now written around
`Bash`'s own `run_in_background` parameter, `Read` on the returned output
file path, and `ToolSearch` as the way to discover the current termination
tool. The hole has little left to bite on.

RE-DERIVE `HARNESS_TOOLS` from a live dispatched session with:
    ToolSearch("select:BashOutput,KillShell,KillBash,TaskOutput,TaskStop,Monitor")
plus that session's own tool list and its deferred-tool list. `select:` is an
exact-name lookup, so a name it does not return is ABSENT from the registry,
not merely low-ranked.

WHAT THIS DOES NOT COVER, deliberately:
  * prompt text assembled outside roles.py -- `briefing.py`'s f-strings, the
    `skills/` sources -- since the scan reads roles.py's own constants;
  * comments, per the paragraph above;
  * a tool named in PLAIN PROSE. Only backtick spans are scanned, so "poll it
    with BashOutput" escapes. Dropping the backtick requirement is not the fix
    -- every sentence-initial word becomes a candidate -- and this hole is
    stated here because the deleted-upstream one below it is NOT the only one;
  * tool GRANTS. Unlike its MCP sibling this guard asserts EXISTENCE only,
    and the reason is measured and recorded in roles.py's own TOOL
    AVAILABILITY comment: `--allowed-tools` is a PERMISSION allowlist, not a
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

import pytest

from orchestrator.agents import roles as roles_module
from orchestrator.agents.roles import ROLES

# Harness tools measured PRESENT on 2026-09-20, from this dispatched session's
# own tool list (Agent .. Write below) plus its deferred-tool list, which
# `ToolSearch` loads on demand (CronCreate .. WebSearch below).  Re-derive with
# the query in the module docstring; do NOT add a name on the strength of
# remembering it.  The asymmetry that governs edits here: a name wrongly
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

# Names measured ABSENT, kept as a diagnostic arm so this regression reports
# its own dated evidence instead of a generic "unknown token".  Two
# measurements, and the second is the one that justifies the whole module:
#   2026-09-10 -- ToolSearch("select:BashOutput,KillShell,KillBash,TaskOutput,
#                 TaskStop,Monitor") returned TaskOutput, TaskStop and Monitor,
#                 and NOT BashOutput or KillShell.  Corroborated fleet-wide by
#                 census 2026-09-10 §1.1: zero BashOutput/KillShell tool_use
#                 across 8,546 archived transcripts, and 138 ToolSearch
#                 requests naming them answered with other names.
#   2026-09-20 -- the SAME query returned only TaskStop and Monitor;
#                 `select:TaskOutput` answered "No matching deferred tools
#                 found".  TaskOutput was live ten days earlier.
MEASURED_ABSENT_TOOLS = frozenset({
    'BashOutput',
    'KillBash',
    'KillShell',
    'TaskOutput',
})

# Backticked CamelCase words the prompts legitimately use that name no tool.
# Because the scan takes the LEADING identifier of a span (below), this holds
# both bare words and the head of a longer span -- a prose label (`Hypothesis:`)
# or a symbol citation (`GitOps.advance_main`).
NON_TOOL_TERMS = frozenset({
    'GitOps',
    'Hypothesis',
    'InputValidationError',
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


def _is_camel_case(token: str) -> bool:
    """A leading capital plus at least one lowercase letter.

    Excludes all-caps words (SIGTERM, EACCES, SPOT) and lowercase ones
    (timeout, append), neither of which can be a tool name.
    """
    return token[0].isupper() and any(char.islower() for char in token)


def _prompt_constants() -> list[tuple[str, str]]:
    """Every module-level string constant in roles.py, by introspection."""
    return sorted(
        (name, value)
        for name, value in vars(roles_module).items()
        if _PROMPT_CONSTANT_NAME.match(name) and isinstance(value, str)
    )


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
            f'{phantom} are MEASURED ABSENT. ToolSearch("select:BashOutput,KillShell,'
            'KillBash,TaskOutput,TaskStop,Monitor") returned TaskOutput/TaskStop/Monitor '
            'on 2026-09-10 and only TaskStop/Monitor on 2026-09-20, where '
            '"select:TaskOutput" answered "No matching deferred tools found"; census '
            '2026-09-10 §1.1 found zero BashOutput/KillShell tool_use across 8,546 '
            'archived transcripts. An agent following this prompt has no reachable next '
            'step and will improvise a forbidden poll loop. Name a tool that exists -- '
            'and do NOT just substitute another poll-tool name, because there is NO poll '
            'tool in this build: Bash(run_in_background=true) returns an output FILE '
            'PATH, and Read on that path yields the output plus an "[exited with code N]" '
            'trailer. That is the whole collection mechanism.',
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


@pytest.mark.parametrize('const_name', [name for name, _ in _prompt_constants()])
def test_prompt_constant_names_no_phantom_harness_tool(const_name: str) -> None:
    """Same invariant over roles.py's prompt constants, spliced or not.

    The surface that catches what the system_prompt scan cannot:
    WAIT_PATTERN_REMINDER reaches an agent only through briefing.py's
    build_amender_prompt, so it is absent from every role's system_prompt
    while still being prompt text an agent is expected to act on.
    """
    unknown = _unknown_tool_tokens(getattr(roles_module, const_name))
    assert not unknown, _failure_message(f'roles.{const_name}', unknown)

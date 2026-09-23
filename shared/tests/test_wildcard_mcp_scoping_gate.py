"""Gate: a wildcard tool-deny under an ``output_schema`` must also close MCP.

The invariant
-------------
``shared/src/shared/cli_invoke.py::build_claude_argv`` does NOT forward a
``disallowed_tools=['*']`` wildcard verbatim when an ``output_schema`` is also
present.  It substitutes ``_REAL_BUILTIN_TOOLS_DENYLIST`` — a list of BUILT-INS
ONLY, carrying no MCP tool pattern — so the schema's synthetic
``StructuredOutput`` tool survives.  MCP tools are therefore still REACHABLE at
such a call, even though the call reads as "deny everything".

The CLI ambient-merges the ``.mcp.json`` found at ``cwd``, and this repo's root
holds a live one (servers ``escalation``, ``fused-memory``).  Under
``permission_mode='bypassPermissions'`` — which every one of these callers uses
— that is unreviewed MCP **write** access, blast radius including
``halt_scheduler`` and ``delete_memory``.

A caller closes the hole in one of exactly two ways:

* ``mcp_config=no_mcp_servers_config()`` **together with**
  ``strict_mcp_config=True`` — see
  ``shared/src/shared/cli_invoke.py::no_mcp_servers_config``.  The config must
  stay TRUTHY: ``--strict-mcp-config`` is emitted inside ``build_claude_argv``'s
  ``if mcp_config:`` block, so a bare ``{}`` silently no-ops and reinstates the
  hole while looking correct.
* run at ``shared/src/shared/neutral_cwd.py::neutral_cli_cwd()`` — an empty
  scratch directory, so there is no ambient ``.mcp.json`` to merge.

Tasks 4145 and 4242 each closed one instance of this by hand.  This gate
generalises those two fixes so a third instance cannot be introduced silently.

The pair
--------
``shared/tests`` has a settled triad idiom — ``<name>_scan.py`` +
``<name>_allowlist.py`` + ``test_<name>_gate.py`` (see ``silent_fallthrough``,
``loop_blocking``, ``config_dir_archival``).  This guard is deliberately a PAIR:
the scanner and this file, with no allowlist.  The tree is measured at 5
matching sites, all 5 compliant, 0 needing exemption — a ledger plus its
disposition vocabulary and stale-entry hygiene would be machinery with no
subject.  A future caller that genuinely needs ambient MCP therefore cannot
self-bless; the whole-tree failure message says so and directs them to
escalate.  Given the blast radius above, a human look at the first such caller
is the right default.

The synthetic fixtures below carry the real weight.  This gate's live tree is
already GREEN, so the ratchet passes trivially and would keep passing if the
detector silently stopped detecting; only these fixtures prove it still fires.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from silent_fallthrough_scan import ParsedFile
from wildcard_mcp_scoping_scan import (
    EXEMPT_NEUTRAL_CWD,
    EXEMPT_STRICT_MCP,
    NOT_EXEMPT,
    WildcardMcpScopingSite,
    find_wildcard_mcp_scoping_sites,
    is_violation,
    records_worth_scanning,
)

_SYNTHETIC = 'synthetic/module.py'
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _scan(src: str, filename: str = _SYNTHETIC) -> list[WildcardMcpScopingSite]:
    """Parse one synthetic module source and scan it.

    ``textwrap.dedent`` is load-bearing, not tidiness: every fixture below is an
    indented triple-quoted literal, which ``ast.parse`` rejects with
    ``IndentationError`` verbatim.  The ``assert tree.body`` is the companion
    guard — it keeps the "not a site" assertions honest, since an empty parse
    would satisfy every one of them without exercising the detector at all.
    """
    tree = ast.parse(textwrap.dedent(src).strip('\n'))
    assert tree.body, f'synthetic module parsed empty; fixture lost its body: {src!r}'
    return find_wildcard_mcp_scoping_sites(tree, filename)


class TestDetectorFires:
    """A wildcard-deny + output_schema call at an unscoped cwd is reported."""

    def test_a_plain_target_call_is_one_violation(self) -> None:
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    prompt='hi',
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        (site,) = sites
        assert site.filename == _SYNTHETIC, f'got {site!r}'
        assert site.qualname == 'call_the_model', f'got {site!r}'
        assert site.callee == 'invoke_with_cap_retry', f'got {site!r}'
        assert site.lineno == 2, f'got {site!r}'
        assert site.exemption == NOT_EXEMPT, f'got {site!r}'
        assert is_violation(site), f'got {site!r}'
        assert site.message, 'a reported site must explain itself'

    def test_the_other_target_name_is_flagged(self) -> None:
        """Both entry points reach the same argv builder, so both are scanned."""
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_claude_agent(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].callee == 'invoke_claude_agent', f'got {sites[0]!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_an_attribute_spelling_is_flagged(self) -> None:
        """A module-qualified call is the same call; the callee is the attr."""
        sites = _scan('''
            def call_the_model(some_root):
                return cli_invoke.invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].callee == 'invoke_with_cap_retry', f'got {sites[0]!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_tuple_display_wildcard_is_flagged(self) -> None:
        """The builder tests ``'*' in disallowed_tools``, which holds for a
        tuple exactly as for a list.
        """
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=('*',),
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_an_awaited_call_in_an_async_method_is_flagged(self) -> None:
        """The shape all five real sites use: awaited inside an async method."""
        sites = _scan('''
            class Curator:
                async def _call_llm(self, prompt):
                    return await invoke_with_cap_retry(
                        prompt=prompt,
                        disallowed_tools=['*'],
                        output_schema=SCHEMA,
                        cwd=self._project_root,
                    )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].qualname == 'Curator._call_llm', f'got {sites[0]!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'


class TestStrictScopingIsCompliant:
    """Only an mcp_config KNOWN to be truthy, with a literal
    strict_mcp_config=True, closes MCP.
    """

    def test_zero_server_config_plus_strict_flag_is_compliant(self) -> None:
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=no_mcp_servers_config(),
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_STRICT_MCP, f'got {sites[0]!r}'
        assert not is_violation(sites[0]), f'got {sites[0]!r}'

    def test_the_strict_flag_without_a_config_is_still_a_violation(self) -> None:
        """The flag is inert alone: build_claude_argv emits it only inside
        ``if mcp_config:``, so with no config there is nothing to strict-scope.
        """
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=None,
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_config_without_the_strict_flag_is_still_a_violation(self) -> None:
        """--mcp-config ADDS servers; it does not displace the ambient merge.
        Only --strict-mcp-config makes the config the exclusive server set.
        """
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=no_mcp_servers_config(),
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_truthy_non_bool_strict_flag_is_still_a_violation(self) -> None:
        """Deliberately conservative: ``build_claude_argv`` would emit the flag
        for ``1``, but the parameter is a ``bool`` and every real caller spells
        it ``True``, so no other spelling exempts.
        """
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=no_mcp_servers_config(),
                    strict_mcp_config=1,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_an_empty_dict_config_is_still_a_violation(self) -> None:
        """The exact MUST-STAY-TRUTHY footgun ``no_mcp_servers_config``'s own
        docstring warns about: ``{}`` is falsy, so ``build_claude_argv`` emits
        NEITHER flag and the call silently keeps ambient MCP while reading as
        scoped.  ``no_mcp_servers_config()`` returns ``{'mcpServers': {}}``,
        which is truthy, precisely to avoid this.
        """
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config={},
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_config_passed_by_name_is_still_a_violation(self) -> None:
        """The same footgun arriving by another route: a name may hold ``{}``
        at runtime, and a static scan cannot see what it holds. A config the
        scanner cannot read exempts nothing; only the factory call spelled AT
        the call site does.
        """
        sites = _scan('''
            def call_the_model(some_root, cfg):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=cfg,
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_dict_display_with_a_key_is_compliant(self) -> None:
        """The factory's return value written inline is known truthy on sight."""
        sites = _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config={'mcpServers': {}},
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_STRICT_MCP, f'got {sites[0]!r}'

    def test_a_dict_display_of_only_a_splat_is_still_a_violation(self) -> None:
        """``{**base}`` is only as truthy as ``base``: a display counts only
        when it spells a key of its own.
        """
        sites = _scan('''
            def call_the_model(some_root, base):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config={**base},
                    strict_mcp_config=True,
                    cwd=some_root,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'


class TestNonMatchingCallsAreNotSites:
    """A call outside the invariant is reported as nothing at all, not as safe.

    A whole-tree gate that reports near-misses becomes noise, and a false RED
    blocks every merge until someone blesses a non-defect — which trains
    reviewers to bless rows unread and destroys the gate's value.  So these
    assert an EMPTY scan, not a compliant site.
    """

    def test_a_wildcard_without_a_schema_is_not_a_site(self) -> None:
        """With no schema the ``'*'`` is forwarded VERBATIM, so every tool —
        MCP included — really is denied.  There is nothing to guard.
        """
        assert _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    cwd=some_root,
                )
        ''') == []

    def test_an_explicitly_none_schema_is_not_a_site(self) -> None:
        assert _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=None,
                    cwd=some_root,
                )
        ''') == []

    def test_a_named_deny_list_is_not_a_site(self) -> None:
        """No wildcard, no substitution: the deny-list is forwarded as written."""
        assert _scan('''
            def call_the_model(some_root):
                return invoke_with_cap_retry(
                    disallowed_tools=['Bash'],
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''') == []

    def test_an_unrelated_callee_is_not_a_site(self) -> None:
        """The kwargs alone mean nothing; only these two entry points reach
        ``build_claude_argv`` and its wildcard substitution.
        """
        assert _scan('''
            def call_the_model(some_root):
                return some_other_helper(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=some_root,
                )
        ''') == []

    def test_an_unresolvable_kwargs_splat_is_not_a_site(self) -> None:
        """Unresolvable means SILENCE, never a speculative finding."""
        assert _scan('''
            def call_the_model(**kwargs):
                return invoke_with_cap_retry(**kwargs)
        ''') == []



class TestNeutralCwdExemption:
    """The cwd exemption checks WHAT was assigned, not that an assignment exists.

    Without these, step-5's one-hop resolution could decay into a blanket pass
    — any local name, any attribute — and the whole-tree gate would stay green
    the whole way down, because its tree is green either way.
    """

    def test_a_direct_neutral_cwd_call_is_compliant(self) -> None:
        sites = _scan('''
            def call_the_model():
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=neutral_cli_cwd(),
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_NEUTRAL_CWD, f'got {sites[0]!r}'

    def test_the_attribute_spelling_of_the_call_is_compliant(self) -> None:
        """``neutral_cwd.neutral_cli_cwd()`` is the same call, module-qualified."""
        sites = _scan('''
            def call_the_model():
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=neutral_cwd.neutral_cli_cwd(),
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_NEUTRAL_CWD, f'got {sites[0]!r}'

    def test_a_local_name_assigned_from_the_call_is_compliant(self) -> None:
        """The synthetic twin of all three real neutral-cwd sites.

        Without this fixture, a refactor that dropped the local-name resolution
        would red the whole-tree gate with no synthetic test explaining why.
        """
        sites = _scan('''
            class Curator:
                async def _call_llm(self, prompt):
                    cwd = neutral_cli_cwd()
                    return await invoke_with_cap_retry(
                        prompt=prompt,
                        disallowed_tools=['*'],
                        output_schema=SCHEMA,
                        cwd=cwd,
                    )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_NEUTRAL_CWD, f'got {sites[0]!r}'

    def test_an_annotated_assignment_from_the_call_is_compliant(self) -> None:
        sites = _scan('''
            def call_the_model():
                cwd: Path = neutral_cli_cwd()
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=cwd,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_NEUTRAL_CWD, f'got {sites[0]!r}'

    def test_every_target_of_a_chained_assignment_is_bound(self) -> None:
        """``cwd`` is the SECOND target, so reading only the first misses it."""
        sites = _scan('''
            def call_the_model():
                scratch = cwd = neutral_cli_cwd()
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=cwd,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert sites[0].exemption == EXEMPT_NEUTRAL_CWD, f'got {sites[0]!r}'

    def test_a_local_name_assigned_from_something_else_is_a_violation(self) -> None:
        """The single most important negative case: it proves the resolution
        checks WHAT was assigned, not merely that a local assignment exists.
        """
        sites = _scan('''
            def call_the_model(self):
                cwd = some_other_root()
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=cwd,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_an_attribute_cwd_is_a_violation(self) -> None:
        """Function-local ONLY, pinned together with the attribute exclusion:
        the enclosing function DOES bind a neutral cwd, to a different name and
        never to this argument. A guard that accepted ``cwd=self._cwd`` would
        accept any instance attribute whatsoever — the exact silent-exposure
        shape this gate exists to catch.
        """
        sites = _scan('''
            class Runner:
                async def go(self):
                    neutral = neutral_cli_cwd()
                    return await invoke_with_cap_retry(
                        disallowed_tools=['*'],
                        output_schema=SCHEMA,
                        cwd=self._cwd,
                    )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_neutral_binding_of_a_different_name_is_a_violation(self) -> None:
        """The bound name must actually be the one passed."""
        sites = _scan('''
            def call_the_model(cwd_param):
                other = neutral_cli_cwd()
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    cwd=cwd_param,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_a_binding_in_an_enclosing_function_is_a_violation(self) -> None:
        """No transitive chains and no enclosing-scope reads — deliberate, not
        accidental. Resolution searches the NEAREST enclosing function's own
        body; a closure over an outer binding does not carry the exemption in.
        """
        sites = _scan('''
            def outer():
                cwd = neutral_cli_cwd()

                def inner():
                    return invoke_with_cap_retry(
                        disallowed_tools=['*'],
                        output_schema=SCHEMA,
                        cwd=cwd,
                    )
                return inner
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert is_violation(sites[0]), f'got {sites[0]!r}'

    def test_both_protections_report_the_explicit_one(self) -> None:
        """A doubly-protected call is compliant, and the precedence is PINNED
        rather than incidental: the exemption named is the one the call STATES,
        not the one it inherits from where it happens to run.
        """
        sites = _scan('''
            def call_the_model():
                cwd = neutral_cli_cwd()
                return invoke_with_cap_retry(
                    disallowed_tools=['*'],
                    output_schema=SCHEMA,
                    mcp_config=no_mcp_servers_config(),
                    strict_mcp_config=True,
                    cwd=cwd,
                )
        ''')
        assert len(sites) == 1, f'got {sites!r}'
        assert not is_violation(sites[0]), f'got {sites[0]!r}'
        assert sites[0].exemption == EXEMPT_STRICT_MCP, f'got {sites[0]!r}'

# --------------------------------------------------------------------------- #
# The whole-tree gate
# --------------------------------------------------------------------------- #

#: Every matching call site in the first-party tree, measured at HEAD
#: 89e37fd6fb on 2026-09-22: 5 matched, 0 violations. Three are protected by
#: running at ``neutral_cli_cwd()``, two by ``no_mcp_servers_config()`` +
#: ``strict_mcp_config=True``. Kept as an exact SET, not a count: a new
#: matching caller anywhere in the tree must be looked at even when it is
#: correctly scoped, and none of these five may silently vanish.
_KNOWN_SITES: frozenset[tuple[str, str]] = frozenset({
    ('fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py',
     'PathScopeAdjudicator.adjudicate'),
    ('fused-memory/src/fused_memory/middleware/task_curator.py',
     'TaskCurator._call_llm'),
    ('fused-memory/src/fused_memory/middleware/task_curator.py',
     'TaskCurator._call_llm_batch'),
    ('fused-memory/src/fused_memory/reconciliation/agent_loop.py',
     'AgentLoop._call_claude_cli'),
    ('fused-memory/src/fused_memory/reconciliation/judge.py',
     'Judge._call_judge_cli'),
})


def _sites_in(records: Iterable[ParsedFile]) -> list[WildcardMcpScopingSite]:
    """Scan *records*, skipping any whose parse failed.

    A file that failed to parse carries ``tree is None`` and contributes
    nothing; ``test_silent_fallthrough_gate.test_no_unparseable_files`` is what
    reports those, so this gate does not duplicate the complaint.
    """
    sites: list[WildcardMcpScopingSite] = []
    for record in records:
        if record.tree is None:
            continue
        sites.extend(find_wildcard_mcp_scoping_sites(record.tree, record.relpath))
    return sites


@pytest.fixture(scope='session')
def tree_sites(first_party_tree: Sequence[ParsedFile]) -> list[WildcardMcpScopingSite]:
    """Every matching call site in the first-party tree.

    Takes the session-scoped ``first_party_tree`` fixture (``conftest.py``) so
    the ASTs ``parse_first_party_tree`` already built are WALKED, never
    re-parsed — and so this module never names the enumerator, which is what
    keeps ``test_tree_scan_sharing.py::TestNoRegrownWholeTreeParse`` green
    while ``_scan`` above is still free to parse synthetic fixtures.

    The ASTs are shared with every other gate in this directory and are walked
    READ-ONLY. ``records_worth_scanning`` drops the files whose source never
    spells a target name; ``TestPrefilterParity`` below is what proves that is
    an optimisation and not a policy.
    """
    return _sites_in(records_worth_scanning(first_party_tree))


class TestWholeTreeGate:
    """No first-party caller reaches the CLI with ambient MCP still exposed."""

    def test_no_unprotected_wildcard_schema_callers(
        self, tree_sites: list[WildcardMcpScopingSite]
    ) -> None:
        offenders = sorted(
            (site for site in tree_sites if is_violation(site)),
            key=lambda site: (site.filename, site.lineno),
        )
        if not offenders:
            return
        listing = '\n'.join(
            f'  {site.filename}::{site.qualname} -> {site.callee}  ~L{site.lineno}'
            for site in offenders
        )
        raise AssertionError(
            f"{len(offenders)} call site(s) pass disallowed_tools=['*'] with an "
            f"output_schema, at a cwd whose ambient MCP servers are not closed:\n"
            f'{listing}\n'
            f'(~L is a hint for finding the call, never its identity — it drifts '
            f'on any edit above the site.)\n'
            f'\n'
            f"THE SUBTLETY: that call reads as \"deny every tool\", and it is not. "
            f"shared/src/shared/cli_invoke.py::build_claude_argv silently replaces "
            f"the '*' with _REAL_BUILTIN_TOOLS_DENYLIST whenever an output_schema "
            f'is present, because the schema rides on a synthetic StructuredOutput '
            f'tool a wildcard would block. That list is BUILT-INS ONLY and carries '
            f'no MCP pattern, so MCP tools stay REACHABLE — and the CLI '
            f'ambient-merges the .mcp.json at cwd. Under bypassPermissions, which '
            f'every one of these callers uses, that is unreviewed MCP WRITE access; '
            f'halt_scheduler and delete_memory are in the blast radius.\n'
            f'\n'
            f'TO FIX, either close MCP explicitly at the call:\n'
            f'    mcp_config=no_mcp_servers_config(),\n'
            f'    strict_mcp_config=True,\n'
            f'  — both spelled literally AT the call; the gate does not look '
            f'through a name, which could hold {{}} at runtime. The config must '
            f'stay TRUTHY: --strict-mcp-config is emitted inside '
            f'build_claude_argv\'s `if mcp_config:` block, so a bare {{}} emits '
            f'NEITHER flag and silently reinstates the hole while looking '
            f'correct. See cli_invoke.py::no_mcp_servers_config.\n'
            f'OR run somewhere with no ambient .mcp.json to merge:\n'
            f'    cwd=neutral_cli_cwd(),   # shared/src/shared/neutral_cwd.py\n'
            f'\n'
            f'shared/tests is the FIRST segment of the repo test_command, so this '
            f'red blocks verify for EVERY subsequent task until it is resolved. '
            f'Fix the call site; do not weaken this assertion.\n'
            f'\n'
            f'If a caller GENUINELY needs ambient MCP, there is deliberately no '
            f'allowlist to add it to — ESCALATE instead. Given the blast radius '
            f'above, the first such caller gets a human look.'
        )

    def test_the_sweep_is_not_vacuous(
        self, tree_sites: list[WildcardMcpScopingSite]
    ) -> None:
        """Anti-vacuity: the ratchet above passes trivially over a green tree.

        This tree IS green, so a detector that silently stopped detecting — a
        renamed kwarg, a changed call spelling, a refactor of the match logic —
        would leave the suite green and the protection gone. This floor is what
        keeps it honest, and it is set AT the measured value rather than below
        it: unlike the sibling gates, which scan for defects whose count
        legitimately falls as fixes land, this one scans for load-bearing
        production callers, none of which may silently vanish.
        """
        assert len(tree_sites) >= len(_KNOWN_SITES), (
            f'the sweep found only {len(tree_sites)} matching call site(s); '
            f'{len(_KNOWN_SITES)} were measured at HEAD 89e37fd6fb on 2026-09-22:\n'
            + '\n'.join(f'  {relpath}::{qualname}'
                        for relpath, qualname in sorted(_KNOWN_SITES))
            + f'\n\nEither the detector stopped detecting, or a load-bearing '
              f'caller was deleted. Neither is a reason to lower this floor.\n'
              f'Repo root resolved to: {_REPO_ROOT}'
        )

    def test_every_known_site_is_accounted_for(
        self, tree_sites: list[WildcardMcpScopingSite]
    ) -> None:
        """A NEW matching caller is surfaced for review even when it is scoped
        correctly, so nobody adds a sixth one without reading this gate.
        """
        found = {(site.filename, site.qualname) for site in tree_sites}
        assert found == _KNOWN_SITES, (
            f'the matching call sites have changed.\n'
            f'  NEW (add to _KNOWN_SITES once reviewed): '
            f'{sorted(found - _KNOWN_SITES) or "none"}\n'
            f'  GONE (a load-bearing caller vanished, or the detector broke): '
            f'{sorted(_KNOWN_SITES - found) or "none"}\n'
            f'Repo root resolved to: {_REPO_ROOT}'
        )


class TestPrefilterParity:
    """The source-substring prefilter drops files, and drops nothing that matters.

    Walking all 524 parsed ASTs for Call nodes costs ~10x what walking only the
    files whose source spells a target name costs, and ``shared/tests`` is the
    FIRST segment of the repo test_command — so its runtime is charged to every
    subsequent task. The filter is sound by construction (a call cannot appear
    in a file whose source never spells the name), and these two tests are what
    keep that claim honest rather than merely asserted.
    """

    def test_prefiltered_scan_equals_unfiltered_scan(
        self, first_party_tree: Sequence[ParsedFile]
    ) -> None:
        assert sorted(_sites_in(records_worth_scanning(first_party_tree))) == sorted(
            _sites_in(first_party_tree)
        ), 'the prefilter changed the result; it is an optimisation, not a policy'

    def test_the_prefilter_actually_drops_something(
        self, first_party_tree: Sequence[ParsedFile]
    ) -> None:
        """A parity test alone passes vacuously if the filter keeps everything,
        silently costing the ~10x it exists to buy. The floor below it is the
        other direction: it cannot drop so much that the known sites vanish.
        """
        survivors = records_worth_scanning(first_party_tree)
        assert len(survivors) < len(first_party_tree), (
            f'the prefilter kept all {len(first_party_tree)} records — it is '
            f'filtering on nothing, and the scan is paying full price'
        )
        assert len(survivors) >= len(_KNOWN_SITES), (
            f'only {len(survivors)} record(s) survived the prefilter; the '
            f'{len(_KNOWN_SITES)} known sites live in '
            f'{len({relpath for relpath, _ in _KNOWN_SITES})} files, so every '
            f'one of those must survive'
        )


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))

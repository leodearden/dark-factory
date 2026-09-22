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

import pytest
from wildcard_mcp_scoping_scan import (
    EXEMPT_STRICT_MCP,
    NOT_EXEMPT,
    WildcardMcpScopingSite,
    find_wildcard_mcp_scoping_sites,
    is_violation,
)

_SYNTHETIC = 'synthetic/module.py'


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
    """Only a TRUTHY mcp_config paired with strict_mcp_config=True closes MCP."""

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


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))

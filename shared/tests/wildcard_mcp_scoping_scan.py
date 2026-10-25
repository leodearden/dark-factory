"""AST scanner for registry-filter call sites that leave MCP reachable.

Why this exists
---------------
Two call shapes reach the CLI through a built-in tool REGISTRY filter
(``--tools``), and a registry filter does **not** filter MCP:

* A ``disallowed_tools=['*']`` wildcard with an ``output_schema``.
  ``shared/src/shared/cli_invoke.py::build_claude_argv`` does NOT forward the
  wildcard verbatim then.  The schema is delivered through a synthetic
  ``StructuredOutput`` tool that a ``'*'`` deny would block, failing every
  structured-output call, so the builder emits ``--tools ''`` instead.  At a
  call that reads as "deny every tool", MCP tools are still reachable.
* An ``available_tools`` argument, with or without a schema.  It emits
  ``--tools <list>``, which names the built-in tools that exist and says
  nothing about MCP (task 4344).

In both cases MCP tools stay reachable — the CLI ambient-merges whatever
``.mcp.json`` sits at ``cwd``, and account-scoped claude.ai connectors connect
wherever ``cwd`` is.  This repo's root holds a live ``.mcp.json`` (servers
``escalation``, ``fused-memory``), and the wildcard callers
run at ``permission_mode='bypassPermissions'``, so the result is unreviewed MCP
**write** access: ``halt_scheduler`` and ``delete_memory`` are in the blast
radius.  A registry-filter caller's permission mode is no substitute for
closing MCP either: the mode can change at the call, and this gate reads only
what closes the hole.

A caller closes the hole one way, and it is the only one this scanner
recognises: ``mcp_config=no_mcp_servers_config()`` together with
``strict_mcp_config=True``.  See
``shared/src/shared/cli_invoke.py::no_mcp_servers_config``: it returns
``{'mcpServers': {}}``, a TRUTHY zero-server config, and the truthiness is
load-bearing — ``--strict-mcp-config`` is emitted inside
``build_claude_argv``'s ``if mcp_config:`` block, so a bare ``{}`` silently
emits NEITHER flag and reinstates the hole while looking correct.

Running at ``shared/src/shared/neutral_cwd.py::neutral_cli_cwd()`` removes only
the ambient project ``.mcp.json``.  It does not close MCP: an account-scoped
claude.ai connector still connected at an empty neutral cwd and exposed write
tools (CLI 2.1.283 system/init inventory, task 3995), so this scanner does not
accept it.

Tasks 4145 and 4242 each found and hand-fixed one instance of this.  Two
hand-fixes of one shape is the point at which a guard is cheaper than a third,
so this module generalises them.

Design
------
* Pure ``ast``, no I/O, and it **never parses**.  It takes an already-parsed
  ``ast.Module``, mirroring
  ``silent_fallthrough_scan.find_violations_in_tree``'s ``(tree, filename)``
  shape, so the whole-tree gate walks the ASTs the session fixture already
  built.  Keeping the parse out of this module (and the enumerator's name out
  of the gate) is what satisfies
  ``test_tree_scan_sharing.py::TestNoRegrownWholeTreeParse`` by construction
  rather than by exemption — task 4520 added that ratchet after two private
  whole-tree parses collided with the pytest-timeout budget under load.
* Every matching call site is reported, compliant or not, carrying HOW it is
  protected in ``exemption``.  A violations-only scanner would give the gate's
  anti-vacuity test nothing to measure: this tree is already green, so the
  ratchet passes trivially and a detector that silently stopped detecting would
  look identical.
* **What must be literal, and why.**  A wildcard site counts only when its
  ``disallowed_tools`` is a display holding a literal ``'*'``: a ``**kwargs``
  splat or a ``Name``-valued deny-list is silence, never a finding, because a
  false RED in a whole-tree gate blocks every merge until someone blesses a
  non-defect, which trains reviewers to bless rows unread and destroys the
  gate's value.  An ``output_schema`` need only be passed and not literally
  falsy — real callers pass it by name, so demanding a literal would blind the
  scanner.  An ``available_tools`` argument need only be passed and not the
  literal ``None``, for the same reason: real callers pass it by name, and
  EVERY non-None value emits ``--tools`` (``[]`` emits ``--tools ''``), so no
  non-None spelling can be a false RED.  A PROTECTION, by contrast, counts only when spelled literally at
  the call: an unresolvable argument must never be a blessing, and a name the
  scanner cannot see through may hold ``{}`` at runtime — the very footgun
  being guarded.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import NamedTuple

from silent_fallthrough_scan import (
    ParsedFile,
    _build_parent_map,
    _callee_name,
    _compute_qualname,
)

# --------------------------------------------------------------------------- #
# Public vocabulary
# --------------------------------------------------------------------------- #

#: The two entry points that reach ``build_claude_argv`` and its wildcard
#: substitution.  A call to anything else carrying the same kwargs is not a
#: site — the kwargs alone mean nothing.
TARGET_CALLEES: frozenset[str] = frozenset({
    'invoke_with_cap_retry',
    'invoke_claude_agent',
})

#: ``shared/src/shared/cli_invoke.py::no_mcp_servers_config`` — matched on the
#: bare name, so the ``cli_invoke.no_mcp_servers_config()`` spelling counts too.
NO_MCP_SERVERS_CALLEE = 'no_mcp_servers_config'

#: How a site is protected.  Named constants rather than bare literals at the
#: comparison sites, so a typo is an ImportError instead of a silent miss.
EXEMPT_STRICT_MCP = 'strict_mcp_config'
NOT_EXEMPT = ''

_VIOLATION_MESSAGE = (
    "a registry filter (available_tools, or a wildcard deny + output_schema, which "
    "becomes --tools '') does not filter MCP, so the ambient .mcp.json at cwd and "
    'account-scoped connectors stay reachable'
)
_STRICT_MCP_MESSAGE = (
    'MCP closed explicitly: a truthy mcp_config with strict_mcp_config=True '
    'scopes the run to only those servers'
)


class WildcardMcpScopingSite(NamedTuple):
    """One call that reaches the CLI through a registry filter that does not
    filter MCP: the wildcard-deny + ``output_schema`` substitution, or an
    ``available_tools`` argument.

    Reported whether or not it is protected: ``exemption`` says how it is, and
    is :data:`NOT_EXEMPT` when it is not.  Use :func:`is_violation` to ask.
    """

    filename: str        # repo-relative posix path, the caller's spelling
    qualname: str        # enclosing def, e.g. 'TaskCurator._call_llm'
    callee: str          # the target name as written
    lineno: int          # display only -- drifts on any edit above the site
    exemption: str       # how it is protected, or NOT_EXEMPT when it is not
    message: str         # one-line human explanation


def is_violation(site: WildcardMcpScopingSite) -> bool:
    """True when *site* reaches the CLI with ambient MCP still exposed."""
    return site.exemption == NOT_EXEMPT


# --------------------------------------------------------------------------- #
# Literal readers -- each answers one question about one keyword
# --------------------------------------------------------------------------- #


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    """Return the value node of keyword *name*, or None if not passed.

    A ``**kwargs`` splat carries ``arg is None`` and is skipped: its contents
    are unknowable, and guessing produces false REDs.
    """
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _denies_everything(node: ast.expr | None) -> bool:
    """True when *node* is a list/tuple DISPLAY containing the ``'*'`` literal.

    Only a display is read.  A ``Name``, a ``Starred`` element or a call-valued
    argument is unresolvable, and so is not a site at all.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    return any(
        isinstance(elt, ast.Constant) and elt.value == '*'
        for elt in node.elts
    )


def _sets_registry_filter(node: ast.expr | None) -> bool:
    """True when an ``available_tools`` argument is passed and is not the
    literal ``None``.

    Any other value counts — a ``Name``, a call, even an empty display, since
    ``[]`` still emits ``--tools ''``.  Every non-None value emits ``--tools``.
    """
    if node is None:
        return False
    return not (isinstance(node, ast.Constant) and node.value is None)


def _is_falsy_literal(node: ast.expr | None) -> bool:
    """True when *node* is absent, or a literal ``build_claude_argv`` would skip.

    That is: no node at all, ``None``/``False``/``0``/``''`` as constants, or
    an empty ``{}`` / ``[]`` display.  Anything else — a ``Name`` such as a
    module-level schema constant, a call, a non-empty display — is not known
    to be falsy, because the builder's own test is a plain ``if <value>:`` and
    only these spellings are statically known to fail it.
    """
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return not node.value
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    return False


def _is_known_truthy_config(node: ast.expr | None) -> bool:
    """True only for an ``mcp_config`` spelling statically known to be truthy.

    Two spellings qualify: a ``no_mcp_servers_config()`` call, and a dict
    display that spells at least one key of its own.  Nothing else is read
    through — a ``Name`` or an attribute may hold ``{}`` at runtime, and
    ``{**base}`` is only as truthy as ``base`` — so nothing else exempts.
    """
    if isinstance(node, ast.Dict):
        return any(key is not None for key in node.keys)
    return node is not None and _callee_name(node) == NO_MCP_SERVERS_CALLEE


def _is_true_literal(node: ast.expr | None) -> bool:
    """True only for the literal ``True``. A truthy non-bool does not count:
    ``strict_mcp_config`` is a ``bool`` parameter and every real caller spells
    it ``True``, so anything else is unresolvable rather than blessed.
    """
    return isinstance(node, ast.Constant) and node.value is True


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #


def records_worth_scanning(records: Iterable[ParsedFile]) -> list[ParsedFile]:
    """Keep only the parsed records whose source spells a target callee.

    Sound by construction: a call to a name cannot appear in a file whose
    source never contains that name, so the filter can drop files but never
    sites. ``test_wildcard_mcp_scoping_gate.TestPrefilterParity`` is what keeps
    that claim honest — it asserts the filtered and unfiltered scans agree over
    the real tree, and that the filter drops something rather than passing
    everything through.

    Skipping those files makes the walk several times faster; re-measure
    rather than trust any figure for that.
    """
    return [
        record for record in records
        if record.tree is not None
        and any(callee in record.source for callee in TARGET_CALLEES)
    ]


def find_wildcard_mcp_scoping_sites(
    tree: ast.Module,
    filename: str,
) -> list[WildcardMcpScopingSite]:
    """Report every call in *tree* that reaches the CLI through a registry
    filter that does not filter MCP.

    A call is a site when its callee's bare name is in :data:`TARGET_CALLEES`
    AND either trigger holds:

    * WILDCARD — ``disallowed_tools`` is a list/tuple display containing
      ``'*'`` AND ``output_schema`` is passed and is not a falsy literal; or
    * REGISTRY — ``available_tools`` is passed and is not the literal ``None``.

    Anything else contributes nothing — not a compliant site, nothing.

    Args:
        tree: Parsed module, walked READ-ONLY. Under the session-scoped
            ``first_party_tree`` fixture this object is shared with every other
            gate in ``shared/tests``.
        filename: Path string recorded on each site, for display. The caller's
            spelling is honoured verbatim; the gate keys on a repo-relative
            path, not on whatever the parser was told.
    """
    _parent_map: dict[int, ast.AST] | None = None

    def parent_map() -> dict[int, ast.AST]:
        """Built LAZILY, at most once, and only for a file that has a site.

        It exists only to name a site's enclosing scope, and all but a handful
        of the tree's files have none -- the same reason
        ``silent_fallthrough_scan.find_violations_in_tree`` defers it.
        """
        nonlocal _parent_map
        if _parent_map is None:
            _parent_map = _build_parent_map(tree)
        return _parent_map

    sites: list[WildcardMcpScopingSite] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node)
        if callee not in TARGET_CALLEES:
            continue
        wildcard = _denies_everything(
            _keyword(node, 'disallowed_tools')
        ) and not _is_falsy_literal(_keyword(node, 'output_schema'))
        if not (wildcard or _sets_registry_filter(_keyword(node, 'available_tools'))):
            continue

        pmap = parent_map()
        exemption, message = _classify(node)
        sites.append(WildcardMcpScopingSite(
            filename=filename,
            qualname=_compute_qualname(node, pmap),
            callee=callee,
            lineno=node.lineno,
            exemption=exemption,
            message=message,
        ))

    return sites


def _classify(call: ast.Call) -> tuple[str, str]:
    """Return the ``(exemption, message)`` pair describing how *call* is (un)safe."""
    strict = _is_true_literal(_keyword(call, 'strict_mcp_config'))
    if strict and _is_known_truthy_config(_keyword(call, 'mcp_config')):
        return EXEMPT_STRICT_MCP, _STRICT_MCP_MESSAGE
    return NOT_EXEMPT, _VIOLATION_MESSAGE

"""AST scanner for wildcard-tool-deny call sites that leave MCP reachable.

Why this exists
---------------
``shared/src/shared/cli_invoke.py::build_claude_argv`` does NOT forward a
``disallowed_tools=['*']`` wildcard verbatim when an ``output_schema`` is also
present.  The schema is delivered through a synthetic ``StructuredOutput``
tool that a ``'*'`` deny would block, failing every structured-output call, so
the builder substitutes ``_REAL_BUILTIN_TOOLS_DENYLIST`` instead.

That list is **built-ins only**.  It carries no MCP tool pattern.  So at a call
that reads as "deny every tool", MCP tools are still reachable — and the CLI
ambient-merges whatever ``.mcp.json`` sits at ``cwd``.  This repo's root holds a
live one (servers ``escalation``, ``fused-memory``), and every one of these
callers runs at ``permission_mode='bypassPermissions'``, so the result is
unreviewed MCP **write** access: ``halt_scheduler`` and ``delete_memory`` are
in the blast radius.

A caller closes the hole in one of exactly two ways, and this scanner
recognises both:

(a) ``mcp_config=no_mcp_servers_config()`` together with
    ``strict_mcp_config=True``.  See
    ``shared/src/shared/cli_invoke.py::no_mcp_servers_config``: it returns
    ``{'mcpServers': {}}``, a TRUTHY zero-server config, and the truthiness is
    load-bearing — ``--strict-mcp-config`` is emitted inside
    ``build_claude_argv``'s ``if mcp_config:`` block, so a bare ``{}`` silently
    emits NEITHER flag and reinstates the hole while looking correct.
(b) run at ``shared/src/shared/neutral_cwd.py::neutral_cli_cwd()`` — an empty
    scratch directory, so there is no ambient ``.mcp.json`` to merge.

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
* **An unresolvable argument is silence, never a finding.**  A ``**kwargs``
  splat, a ``Name``-valued ``disallowed_tools``, a computed schema — none is a
  site.  A false RED in a whole-tree gate blocks every merge until someone
  blesses a non-defect, which trains reviewers to bless rows unread and
  destroys the gate's value.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import NamedTuple

from loop_blocking_scan import _shallow_nodes
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

#: ``shared/src/shared/neutral_cwd.py::neutral_cli_cwd`` — matched on the bare
#: name, so the ``neutral_cwd.neutral_cli_cwd()`` attribute spelling counts too.
NEUTRAL_CWD_CALLEE = 'neutral_cli_cwd'

#: How a site is protected.  Named constants rather than bare literals at the
#: comparison sites, so a typo is an ImportError instead of a silent miss.
EXEMPT_STRICT_MCP = 'strict_mcp_config'
EXEMPT_NEUTRAL_CWD = 'neutral_cwd'
NOT_EXEMPT = ''

_VIOLATION_MESSAGE = (
    'wildcard deny + output_schema expands to a BUILT-INS-ONLY deny-list, so MCP '
    'tools stay reachable and the ambient .mcp.json at cwd is merged'
)
_STRICT_MCP_MESSAGE = (
    'MCP closed explicitly: a truthy mcp_config with strict_mcp_config=True '
    'scopes the run to only those servers'
)
_NEUTRAL_CWD_MESSAGE = (
    f'runs at {NEUTRAL_CWD_CALLEE}(), an empty scratch dir, so there is no '
    f'ambient .mcp.json for the CLI to merge'
)


class WildcardMcpScopingSite(NamedTuple):
    """One call that hits the wildcard-deny + ``output_schema`` substitution.

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


def _is_falsy_literal(node: ast.expr | None) -> bool:
    """True when *node* is a literal that ``build_claude_argv`` would skip.

    That is: ``None``/``False``/``0``/``''`` as constants, or an empty ``{}`` /
    ``[]`` display.  Anything else — a ``Name``, a call such as
    ``no_mcp_servers_config()``, a non-empty display — counts as truthy,
    because the builder's own test is a plain ``if <value>:`` and only these
    spellings are statically known to fail it.
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

    Why it is worth having: ``shared/tests`` is the FIRST segment of the repo
    ``test_command``, so this gate's runtime is charged to every subsequent
    task. Measured on this worktree at HEAD 97e3a4d097 (2026-09-22), over 524
    parsed records of which 28 survive, taking the best of three runs::

        unfiltered walk   1.21s
        prefiltered walk  0.21s   (the filter itself costs 0.019s of that)

    The parse is NOT part of either number — it is already paid once by the
    session-scoped ``first_party_tree`` fixture. Task 4891's plan recorded
    2.92s / 0.29s for the same two walks over the same 524/28 records, so the
    RATIO a reader should expect ranges from ~6x to ~10x depending on machine
    and cache state. Re-measure rather than re-judge: the record counts have
    reproduced exactly, the wall-clock has not.
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
    """Report every wildcard-deny + ``output_schema`` call in *tree*.

    A call is a site when ALL THREE hold, in order:

    1. the callee's bare name is in :data:`TARGET_CALLEES`;
    2. ``disallowed_tools`` is a list/tuple display containing ``'*'``;
    3. ``output_schema`` is passed and is not a falsy literal.

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
        if not _denies_everything(_keyword(node, 'disallowed_tools')):
            continue
        schema = _keyword(node, 'output_schema')
        if schema is None or _is_falsy_literal(schema):
            continue

        pmap = parent_map()
        exemption, message = _classify(node, pmap)
        sites.append(WildcardMcpScopingSite(
            filename=filename,
            qualname=_compute_qualname(node, pmap),
            callee=callee,
            lineno=node.lineno,
            exemption=exemption,
            message=message,
        ))

    return sites


def _enclosing_function(
    node: ast.AST,
    parent_map: dict[int, ast.AST],
) -> ast.AST | None:
    """Return the NEAREST enclosing ``def``/``async def``, or None at module scope."""
    current = parent_map.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parent_map.get(id(current))
    return None


def _neutral_cwd_locals(func: ast.AST) -> set[str]:
    """Local names *func*'s OWN body binds to a ``neutral_cli_cwd()`` call.

    ``_shallow_nodes`` (borrowed from ``loop_blocking_scan``, which in turn
    mirrors ``silent_fallthrough_scan``) stops descent at nested ``def`` /
    ``class`` / ``lambda``, so an assignment made inside a nested helper cannot
    launder the exemption up to its enclosing function.

    Both assignment spellings are read: ``ast.Assign`` — including the chained
    ``a = b = neutral_cli_cwd()`` form, which binds every ``Name`` target — and
    ``ast.AnnAssign`` carrying a value. A tuple-unpacking target binds no single
    name to the call's result and is skipped.
    """
    bound: set[str] = set()
    for node in _shallow_nodes(func):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if _callee_name(value) == NEUTRAL_CWD_CALLEE:
            bound.update(t.id for t in targets if isinstance(t, ast.Name))
    return bound


def _resolves_to_neutral_cwd(
    call: ast.Call,
    parent_map: dict[int, ast.AST],
) -> bool:
    """True when *call*'s ``cwd=`` argument is a ``neutral_cli_cwd()`` result.

    Two spellings resolve, and they are the two that actually occur:

    (i)  DIRECT — ``cwd=neutral_cli_cwd()``, bare or attribute-qualified.
    (ii) VIA A LOCAL NAME — ``cwd=<name>`` where the enclosing function's own
         body assigns ``<name> = neutral_cli_cwd()``. This is the spelling ALL
         THREE real neutral-cwd sites use.

    The scope limits below are the guard's PRECISION BOUNDARY, each deliberate:

    * ONE level of indirection, no transitive chains. No real site needs a
      second hop, and each hop widens the ways a non-neutral value could be
      laundered past the guard.
    * FUNCTION-LOCAL ONLY — never a module global, never an attribute such as
      ``self._cwd``. A guard that accepted ``cwd=self._cwd`` would pass ANY
      instance attribute whatsoever, which is precisely the silent-exposure
      shape being guarded against.
    * LAST-ASSIGNMENT-WINS IS NOT MODELLED: any matching assignment anywhere in
      the function exempts. A function that assigns ``cwd`` from
      ``neutral_cli_cwd()`` and then reassigns it to a project root is not a
      shape that occurs, and a false RED costs more here than this miss.

    This resolution is deliberately used INSTEAD of listing the three real
    sites in an allowlist. They are COMPLIANT, not exempt; an allowlist entry
    would freeze that judgement, so a later edit changing
    ``cwd = neutral_cli_cwd()`` to ``cwd = self._project_root`` would leave the
    blessing in place and the guard silent exactly where it matters. Tied to
    the property that makes the call safe, the exemption evaporates the moment
    the property does.
    """
    cwd = _keyword(call, 'cwd')
    if cwd is None:
        return False
    if _callee_name(cwd) == NEUTRAL_CWD_CALLEE:
        return True
    if not isinstance(cwd, ast.Name):
        return False
    func = _enclosing_function(call, parent_map)
    return func is not None and cwd.id in _neutral_cwd_locals(func)


def _classify(call: ast.Call, parent_map: dict[int, ast.AST]) -> tuple[str, str]:
    """Return the ``(exemption, message)`` pair describing how *call* is (un)safe.

    Explicit scoping is reported ahead of neutral-cwd when a call carries both,
    so the exemption named is the one the call states rather than the one it
    inherits from where it happens to run.
    """
    strict = _is_true_literal(_keyword(call, 'strict_mcp_config'))
    if strict and not _is_falsy_literal(_keyword(call, 'mcp_config')):
        return EXEMPT_STRICT_MCP, _STRICT_MCP_MESSAGE
    if _resolves_to_neutral_cwd(call, parent_map):
        return EXEMPT_NEUTRAL_CWD, _NEUTRAL_CWD_MESSAGE
    return NOT_EXEMPT, _VIOLATION_MESSAGE

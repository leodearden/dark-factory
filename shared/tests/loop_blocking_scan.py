"""Caller-side AST scanner for loop-blocking call sites (INV-8).

Why this exists
---------------
Task 3778 censused INV-8 ("no blocking call on the event loop") by enumerating
the sites where the blocking PRIMITIVE is written::

    "only 7 sync `subprocess.run` sites exist in the whole fused-memory server,
     across 3 modules -- live_workflow_detector.py (3, the culprit),
     recon_claim_verification_guard.py (3, already offloaded at its call sites),
     models/scope.py (1, resolve_main_checkout, cached but a cold miss runs on
     the loop thread). shared/ and escalation/ have zero."
                          -- task 3778's description, "SCOPE OF THE DEFECT CLASS"

That is a DEFINITION-side census, and it makes a per-MODULE offload claim.  One
caller that wraps a helper in ``asyncio.to_thread`` makes the whole module read
as "already offloaded at its call sites"; every OTHER caller of that helper is
invisible.  The shape predicts the observed misses exactly: tasks 4091 and 4201
each later found live sites in modules the census had passed over, and task
4484 found two more that nobody had filed at all --
``task_curator.py::TaskCurator._maybe_blocklist_drop`` -> ``load_blocklist``
and ``task_curator.py::TaskCurator._maybe_route_deterministic`` ->
``load_operational_registry``, siblings of the two ``_maybe_premise_refuted_drop``
sites task 4201 owns.  Four same-shape sites in one file, adjacent to each
other, all four of them ``async def _maybe_*`` guards lazily loading a YAML
registry off disk on the loop thread.

So this scanner asks the caller-side question instead: **which coroutines reach
a blocking primitive without an offload hop?**  Findings are per CALL SITE.  A
helper with five callers, one of which offloads, yields four findings -- which
is the number 3778's methodology could not produce.

Design
------
* Pure stdlib (``ast`` only), no filesystem I/O -- ``find_loop_blocking_sites``
  takes a ``{relpath: source}`` mapping so the unit tests can feed synthetic
  modules directly, exactly as ``silent_fallthrough_scan.find_violations`` does.
* ``SyntaxError`` in one module contributes nothing and never raises, so a
  mid-edit file cannot turn the whole-tree gate red.
* Callee names resolve through the calling module's ``from X import Y``
  bindings and the ``def``s actually VISIBLE at the call -- module-level defs
  plus the call's own enclosing function scopes.  A name a parameter or a local
  assignment shadows resolves to nothing.  **An unresolvable name is silence,
  never a finding.**  A false RED in a whole-tree gate is worse than a miss: it
  blocks every merge until someone blesses a non-defect, which trains reviewers
  to bless rows unread and destroys the gate's value.
* Findings key on ``(relpath, qualname, content_hash)`` -- never ``lineno``,
  which drifts on every unrelated edit above the site.  The key shape and its
  helpers are reused verbatim from ``silent_fallthrough_scan``.

Deliberate limits (misses, not false alarms)
--------------------------------------------
* Only ``Name(...)``, ``self.attr(...)``/``cls.attr(...)`` and dotted-primitive
  calls resolve.  An arbitrary ``obj.method(...)`` does not: the receiver's type
  is unknowable without inference, and guessing produces false REDs.  The
  same holds for dispatch through a runtime name, ``getattr(self, name)()``:
  ``MemoryConsolidator._render_required_sections`` is a live instance on a
  designed extension point, so a blocking renderer registered in its
  ``REQUIRED_SECTIONS`` is invisible here.
* Reachability walks each function shallowly, stopping at nested ``def`` /
  ``lambda`` / ``class`` boundaries (the ``silent_fallthrough_scan._shallow_nodes``
  idiom).  A nested helper is reached only when it is actually called.
* ``fused-memory/tests/_ast_guard.py::imported_names_from`` is the house
  import-binding resolver, but ``fused-memory/tests`` is not on ``sys.path``
  for a ``cd shared && pytest tests/`` run (only ``shared/tests``, ``shared/src``
  and the repo root are -- see ``shared/tests/conftest.py``).  The ~20-line
  binding map is therefore re-derived here rather than importing across
  packages, per task 4484's plan.

Triad
-----
``loop_blocking_scan`` (this file, the scanner) + ``loop_blocking_allowlist``
(pure-data dispositioned baseline) + ``test_loop_blocking_gate`` (assertions).
Same shape as ``silent_fallthrough_*`` and ``config_dir_archival_*`` in this
directory.
"""

from __future__ import annotations

import ast
from typing import NamedTuple

from silent_fallthrough_scan import (
    _build_parent_map,
    _compute_qualname,
    _content_hash,
    violation_key,
)

# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


class LoopBlockingSite(NamedTuple):
    """One coroutine call site that reaches a blocking primitive without a hop.

    ``filename`` (not ``relpath``) is deliberate: it makes this tuple
    structurally compatible with ``silent_fallthrough_scan.violation_key`` and
    therefore with ``reconcile_against_allowlist``, so the multiset ratchet is
    reused rather than re-implemented.
    """

    filename: str        # repo-relative posix path of the CALLING module
    qualname: str        # dotted qualname of the enclosing async def
    callee: str          # the name invoked at the site (bare, as written)
    primitive: str       # the blocking primitive it reaches
    lineno: int          # display only -- NEVER part of the identity key
    content_hash: str    # sha256(ast.unparse(call_node))[:12] -- drift-resistant
    message: str         # one-line human explanation


def site_key(site: LoopBlockingSite) -> tuple[str, str, str]:
    """Return the drift-resistant identity ``(relpath, qualname, content_hash)``.

    Delegates to ``silent_fallthrough_scan.violation_key`` so the two gates in
    this directory cannot drift apart on what "the same site" means.
    """
    return violation_key(site)


# --------------------------------------------------------------------------- #
# Blocking primitive table
# --------------------------------------------------------------------------- #
#
# Every entry pairs the primitive with WHY it belongs here.  That is not
# decoration: task 3778's census enumerated `subprocess.run` and nothing else,
# while INV-8's own Rule text names "subprocess, network, filesystem, lock,
# sleep" -- and both of task 4201's missed sites and one of task 4091's are
# FILESYSTEM, not subprocess.  A table whose entries carry no reason is one a
# future census narrows back to `subprocess.run` unchallenged, because nothing
# on the page argues against it.
#
# The five INV-8 limbs, all present: subprocess, network, filesystem, lock,
# sleep.

# Dotted paths, matched after `import X as Y` alias substitution
# (`subprocess.run(...)`) and after `from X import Y` binding resolution
# (a bare `run(...)` bound by `from subprocess import run`).
DOTTED_PRIMITIVES: dict[str, str] = {
    # -- subprocess: fork+exec parks the loop thread for the child's lifetime
    'subprocess.run': (
        'subprocess: the primitive task 3778 censused; fork+exec parks the loop '
        'thread until the child exits'
    ),
    'subprocess.check_output': (
        'subprocess: same fork+exec cost as subprocess.run, plus a blocking read '
        'of the child stdout pipe'
    ),
    'subprocess.check_call': 'subprocess: same fork+exec cost as subprocess.run',
    'subprocess.call': (
        'subprocess: subprocess.run\'s older spelling, same fork+exec and same '
        'wait -- absent from the first cut of this table purely because nothing '
        'in the scanned scope happens to use it yet'
    ),
    'subprocess.Popen': (
        'subprocess: the fork+exec itself blocks even when the caller never '
        'waits; .communicate()/.wait() then block again'
    ),
    'os.system': (
        'subprocess: spawns a shell AND blocks until it exits -- strictly worse '
        'than subprocess.run, never correct on a loop thread'
    ),
    'os.popen': (
        'subprocess: a shell spawn whose pipe is then read synchronously; the '
        'read blocks for as long as the child takes to produce output'
    ),

    # -- filesystem: the limb task 3778's vocabulary omitted entirely
    'yaml.safe_load': (
        'filesystem/CPU: measured by task 4201 at 8.15 ms for an 11 KB document '
        '-- the same order as a subprocess spawn, which is why it does not get '
        'read as "just parsing" and dropped from the vocabulary again'
    ),
    'yaml.safe_dump': (
        'filesystem/CPU: the serialising half of the task 4201 measurement '
        '(8.15 ms / 11 KB); a stamping coroutine pays it on the loop thread'
    ),
    'json.load': (
        'filesystem: reads a whole file handle to EOF and parses it; the same '
        'read-off-disk cost as yaml.safe_load, and the spelling a coroutine '
        'reaches for when the registry happens to be JSON rather than YAML'
    ),
    'json.dump': (
        'filesystem: writes a whole document through a file handle on the '
        'calling thread -- the serialising half of json.load'
    ),
    'os.listdir': (
        'filesystem: a directory read, whose cost scales with the entry count '
        'and is unbounded from the coroutine\'s point of view'
    ),
    'os.walk': (
        'filesystem: os.listdir applied recursively -- the same cost per level '
        'with no bound on the depth'
    ),
    'shutil.rmtree': (
        'filesystem: a recursive delete, one unlink syscall per file, all of '
        'them on the calling thread; this is the primitive behind '
        'reconciliation/cli_stage_runner.py::gc_run_config_dir, which three '
        'harness.py coroutines call inline'
    ),

    # -- lock: an flock waits on ANOTHER process, with no bound
    'fcntl.flock': (
        'lock: blocks the loop thread for as long as another process holds the '
        'lock -- an unbounded wait on a party this process does not control'
    ),

    # -- sleep: the deliberate one, and the easiest to write by reflex
    'time.sleep': (
        'sleep: parks the loop thread outright; asyncio.sleep is the sibling '
        'that yields instead, and is excluded below'
    ),

    # -- network: NOMINAL in the scanned scope, and said plainly rather than
    # left to read as coverage.  Measured at the amendment pass: no sync HTTP
    # client is imported anywhere under `fused-memory/src` (no requests, no
    # httpx, no urllib.request -- only urllib.parse, which is string work), and
    # `socket.create_connection` matches nothing.  Both entries are here so the
    # FIRST sync client to land is a finding rather than a fourth missed batch;
    # neither is currently enforcing anything.
    'socket.create_connection': (
        'network: a DNS lookup plus a TCP handshake, either of which can park '
        'the loop thread for the full connect timeout'
    ),
    'urllib.request.urlopen': (
        'network: the stdlib sync HTTP client -- blocks for connect, request '
        'and the whole response body, bounded only by the server'
    ),
}

# Builtins, matched as a bare `Name` that NOTHING has rebound: neither a
# `from X import Y` binding nor a parameter or local assignment (a module that
# writes its own `open` helper is then talking about that helper, not this).
BUILTIN_PRIMITIVES: dict[str, str] = {
    'open': (
        'filesystem: the plainest spelling of a blocking file read or write, '
        'and the one a coroutine reaches for without importing anything -- '
        'invisible to a table that only knows dotted paths and methods, which '
        'is how `open(p).read()` could have landed past this gate'
    ),
}

# Bare method names, matched on any receiver: `path.read_text()`.  Receiver
# types are not inferred, so these match by attribute name alone -- the
# deliberate trade for finding `self._path.read_text()` without type inference.
#
# DELIBERATELY ABSENT, with the counts that decided it: the directory-walk and
# metadata methods.  Measured over `fused-memory/src` at this commit, adding
# them would produce mkdir +18, exists +15, unlink +5, stat +4, iterdir +4,
# glob +2, open (as a method) +7 -- ~55 new rows, every one of which needs a
# hand-written disposition or the ledger becomes a page of "existing" waivers,
# which is precisely the silent waiver this gate exists to prevent.  They are
# also the names most likely to collide on an unrelated receiver, since match
# is by attribute name alone.  Widening here is a triage exercise of its own,
# filed as a follow-up rather than smuggled into an amendment pass; the
# `open` BUILTIN below is included because it is unambiguous and costs zero
# rows in the current tree.
METHOD_PRIMITIVES: dict[str, str] = {
    'read_text': (
        'filesystem: the primitive behind task 4091\'s and task 4201\'s missed '
        'sites -- a registry read off disk on the loop thread; page-cache-warm '
        'is fast and cold is not, and the coroutine cannot tell which it got'
    ),
    'write_text': (
        'filesystem: task 4091/4201 vocabulary; a write additionally waits on '
        'the filesystem, not merely the page cache'
    ),
    'read_bytes': (
        'filesystem: task 4091/4201 vocabulary; identical cost to read_text '
        'without the decode'
    ),
    'write_bytes': (
        'filesystem: task 4091/4201 vocabulary; identical cost to write_text '
        'without the encode'
    ),
}

# The non-blocking siblings.  Never a finding, whatever else matches.
_NON_BLOCKING_PREFIXES = ('asyncio.', 'anyio.')

# The offload hops, matched by RESOLVED dotted path (after `import X as Y` and
# `from X import Y` substitution).  ``asyncio.to_thread(fn, ...)`` and
# ``anyio.to_thread.run_sync(fn, ...)`` both exempt arg 0.
#
# Matching these by bare attribute name would be the one place in this scanner
# where a name collision produces a MISS rather than the documented silence:
# any `helper.to_thread(load_registry, p)` would exempt `load_registry` from
# the sweep on a whole-tree gate.  Hence the receiver is checked.
_OFFLOAD_DOTTED_ARG: dict[str, int] = {
    'asyncio.to_thread': 0,
    'anyio.to_thread.run_sync': 0,
}

# ``<loop>.run_in_executor(executor, fn, ...)`` exempts arg 1.  The receiver is
# an arbitrary event-loop object, so this one cannot be pinned to a dotted
# path -- but it IS pinned to the attribute shape: a bare `run_in_executor(...)`
# name never counts.
_OFFLOAD_ATTR_ARG: dict[str, int] = {
    'run_in_executor': 1,
}

_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


# --------------------------------------------------------------------------- #
# Module context
# --------------------------------------------------------------------------- #


class _ModuleCtx:
    """Everything the resolver needs to know about one parsed module."""

    def __init__(self, relpath: str, tree: ast.Module) -> None:
        self.relpath = relpath
        self.tree = tree
        self.name = module_name_for(relpath)
        self.parent_map = _build_parent_map(tree)

        # Bare name -> the MODULE-LEVEL function def(s) carrying it.  Class
        # methods and nested defs are deliberately absent: a bare ``foo(...)``
        # can never name them from outside their scope, and indexing them here
        # would resolve a call to a def that is not in the caller's scope at
        # all -- a false finding on a merge-blocking gate.  Nested defs are
        # reachable through the caller's own enclosing-function chain instead
        # (see :meth:`lookup_visible_def`), which is where a closure like
        # ``server/tools.py::create_mcp_server._normalize_project_root``
        # genuinely IS visible.  Ambiguity (two defs sharing a name in one
        # scope) resolves to silence: see :meth:`lookup_def`.
        self.defs_by_name: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        # Class name -> {method name -> def}, for `self.foo(...)` resolution.
        self.methods_by_class: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        # Local binding -> (source module dotted name, source name).
        self.bindings: dict[str, tuple[str, str]] = {}
        # Local alias -> real module dotted name, from `import X as Y`.
        self.module_aliases: dict[str, str] = {}
        # Per-scope caches, keyed by id() of the scope node.
        self._defs_in_scope: dict[int, dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]] = {}
        self._rebinds_in_scope: dict[int, frozenset[str]] = {}

        self._index()

    def _index(self) -> None:
        self.defs_by_name = {
            name: list(defs)
            for name, defs in _defs_bound_in_scope(self.tree).items()
        }
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    child.name: child
                    for child in node.body
                    if isinstance(child, _FUNC_TYPES)
                }
                self.methods_by_class[node.name] = methods
            elif isinstance(node, ast.ImportFrom):
                target = self._resolve_import_from_target(node)
                if target is None:
                    continue
                for alias in node.names:
                    if alias.name == '*':
                        continue
                    self.bindings[alias.asname or alias.name] = (target, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.module_aliases[alias.asname] = alias.name

    def _resolve_import_from_target(self, node: ast.ImportFrom) -> str | None:
        """Return the absolute dotted module a ``from ... import`` names."""
        if not node.level:
            return node.module
        # Relative import: drop `level` trailing components off this module's
        # own dotted name.  Imprecise for package `__init__` modules, which is
        # a MISS (the target module simply will not be found), never a false
        # positive.
        parts = self.name.split('.')[: -node.level]
        if node.module:
            parts = [*parts, node.module]
        return '.'.join(parts) or None

    def enclosing_class(self, node: ast.AST) -> str | None:
        """Name of the innermost ``class`` enclosing *node*, if any."""
        current = self.parent_map.get(id(node))
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return current.name
            current = self.parent_map.get(id(current))
        return None

    def lookup_def(self, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Return the unique MODULE-LEVEL def named *name*, else ``None``.

        Ambiguity is silence: two defs sharing a bare name (an overload in a
        branch, a test double) make the call unresolvable, and an unresolvable
        call is never a finding.  This is also the shape ``from X import Y``
        needs: ``Y`` is by construction a module-level name in ``X``.
        """
        candidates = self.defs_by_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None

    def enclosing_function_scopes(self, node: ast.AST):
        """Yield the function scopes enclosing *node*, innermost first."""
        current = self.parent_map.get(id(node))
        while current is not None:
            if isinstance(current, _FUNC_TYPES):
                yield current
            current = self.parent_map.get(id(current))

    def name_is_rebound(self, name: str, node: ast.AST) -> bool:
        """Is *name* bound to a non-``def`` in any function scope around *node*?

        A parameter, a local assignment, a ``with ... as`` or a ``for`` target
        shadows both this module's own defs and its ``from X import Y``
        bindings, so a call through that name reaches whatever the local
        binding holds -- which this scanner cannot know.  Silence, therefore,
        never a finding: ``async def h(load, p): return load(p)`` must not
        resolve to a module-level ``def load`` the parameter hides.
        """
        for scope in self.enclosing_function_scopes(node):
            if name in self._rebinds_for(scope):
                return True
        return False

    def lookup_visible_def(
        self, name: str, node: ast.AST
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Return the def a bare *name* names AT *node*, honouring scope.

        Nearest enclosing function scope first (closures -- the
        ``create_mcp_server._normalize_project_root`` shape), then module
        level.  A shadowed or ambiguous name is ``None``.
        """
        if self.name_is_rebound(name, node):
            return None
        for scope in self.enclosing_function_scopes(node):
            candidates = self._defs_for(scope).get(name, [])
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
        return self.lookup_def(name)

    def _defs_for(
        self, scope: ast.AST
    ) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
        cached = self._defs_in_scope.get(id(scope))
        if cached is None:
            cached = _defs_bound_in_scope(scope)
            self._defs_in_scope[id(scope)] = cached
        return cached

    def _rebinds_for(self, scope: ast.AST) -> frozenset[str]:
        cached = self._rebinds_in_scope.get(id(scope))
        if cached is None:
            cached = _names_rebound_in_scope(scope)
            self._rebinds_in_scope[id(scope)] = cached
        return cached


def _defs_bound_in_scope(
    scope: ast.AST,
) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """``def``s bound directly in *scope*'s own namespace, not in a nested one.

    ``scope`` is a ``Module`` or a function node.  Descent stops at every
    ``def`` / ``async def`` / ``class`` / ``lambda``: a def nested one level
    down binds a name in ITS body's namespace, not here, so a bare call in
    this scope cannot reach it.  Defs inside ``if`` / ``try`` / ``with``
    blocks DO bind here and are collected.
    """
    out: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}

    def _walk(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, _FUNC_TYPES):
                out.setdefault(child.name, []).append(child)
                continue  # its body is a deeper namespace
            if isinstance(child, (ast.ClassDef, ast.Lambda)):
                continue
            _walk(child)

    _walk(scope)
    return out


def _names_rebound_in_scope(scope: ast.AST) -> frozenset[str]:
    """Names *scope* binds to something OTHER than a ``def``.

    Parameters plus every ``Store``-context ``Name``: assignment (including
    unpacking and the walrus), ``for`` targets, ``with ... as``, ``except ...
    as``.  ``def``s are excluded on purpose -- they are what
    :func:`_defs_bound_in_scope` resolves, and treating one as a shadow would
    silence every legitimate closure call.

    Attribute and subscript targets (``self.x = ...``, ``d[k] = ...``) bind no
    bare name and are therefore not collected: only the ``Name`` node's own
    ``ctx`` decides.
    """
    names: set[str] = set()

    args = getattr(scope, 'args', None)
    if isinstance(args, ast.arguments):
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            names.add(arg.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)

    def _walk(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, _SCOPE_BOUNDARIES):
                continue  # a nested scope binds its own names
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            _walk(child)

    _walk(scope)
    return frozenset(names)


def module_name_for(relpath: str) -> str:
    """Derive a dotted module name from a repo-relative path.

    ``fused-memory/src/fused_memory/middleware/task_curator.py`` ->
    ``fused_memory.middleware.task_curator``.  Everything up to and including
    the first ``src`` component is dropped, matching the repo's
    ``<pkg>/src/<pkg>/`` layout; paths with no ``src`` component keep all of
    their components, which is harmless because nothing imports them that way.
    """
    stem = relpath[:-3] if relpath.endswith('.py') else relpath
    parts = [p for p in stem.split('/') if p]
    if 'src' in parts:
        parts = parts[parts.index('src') + 1:]
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def dotted_path(node: ast.expr) -> str | None:
    """Return ``'a.b.c'`` for a pure Name/Attribute chain, else ``None``."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return '.'.join(reversed(parts))


def _shallow_nodes(node: ast.AST, skip: frozenset[int] = frozenset()) -> list[ast.AST]:
    """All nodes under *node*'s body, stopping at nested scope boundaries.

    Mirrors ``silent_fallthrough_scan._shallow_nodes``: descent stops at
    ``def`` / ``async def`` / ``class`` / ``lambda`` so a nested helper is
    attributed to itself rather than to its enclosing function.  Node ids in
    *skip* (and their subtrees) are excluded -- that is how the callable
    argument of an offload hop is exempted.
    """
    out: list[ast.AST] = []

    def _walk(current: ast.AST) -> None:
        if id(current) in skip:
            return
        out.append(current)
        for child in ast.iter_child_nodes(current):
            if isinstance(child, _SCOPE_BOUNDARIES):
                continue
            _walk(child)

    body = getattr(node, 'body', [])
    for stmt in body:
        if not isinstance(stmt, _SCOPE_BOUNDARIES):
            _walk(stmt)
    return out


def _resolve_head(path: str, ctx: _ModuleCtx) -> str:
    """Substitute a dotted path's head through this module's import bindings.

    ``import asyncio as aio`` makes ``aio.to_thread`` -> ``asyncio.to_thread``;
    ``from anyio import to_thread`` makes ``to_thread.run_sync`` ->
    ``anyio.to_thread.run_sync``.  An unknown head is returned unchanged.
    """
    head, _, rest = path.partition('.')
    if head in ctx.module_aliases:
        resolved_head = ctx.module_aliases[head]
    elif head in ctx.bindings:
        source_module, source_name = ctx.bindings[head]
        resolved_head = f'{source_module}.{source_name}'
    else:
        resolved_head = head
    return f'{resolved_head}.{rest}' if rest else resolved_head


def _offload_callable_index(call: ast.Call, ctx: _ModuleCtx) -> int | None:
    """Argument index this offload hop moves onto a worker thread, or ``None``.

    The receiver is checked, not just the attribute name (see
    :data:`_OFFLOAD_DOTTED_ARG`): ``asyncio.to_thread`` / a ``from asyncio
    import to_thread`` binding / ``anyio.to_thread.run_sync`` count, and an
    unrelated ``helper.to_thread(...)`` does not.
    """
    func = call.func

    if isinstance(func, ast.Name):
        bound = ctx.bindings.get(func.id)
        if bound is None:
            return None
        return _OFFLOAD_DOTTED_ARG.get(f'{bound[0]}.{bound[1]}')

    if isinstance(func, ast.Attribute):
        path = dotted_path(func)
        if path is not None:
            index = _OFFLOAD_DOTTED_ARG.get(_resolve_head(path, ctx))
            if index is not None:
                return index
        return _OFFLOAD_ATTR_ARG.get(func.attr)

    return None


def _offloaded_callable_ids(func_node: ast.AST, ctx: _ModuleCtx) -> frozenset[int]:
    """Ids of the callable arguments handed to an offload hop inside *func_node*.

    ``asyncio.to_thread(load_registry, p)`` and
    ``loop.run_in_executor(None, load_registry, p)`` both move ``load_registry``
    onto a worker thread, so neither that name nor anything nested inside that
    argument expression (``to_thread(lambda: load_registry(p))``) is a finding.
    Other arguments are still walked: they DO evaluate on the loop thread.
    """
    skip: set[int] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        index = _offload_callable_index(node, ctx)
        if index is None or len(node.args) <= index:
            continue
        for descendant in ast.walk(node.args[index]):
            skip.add(id(descendant))
    return frozenset(skip)


# --------------------------------------------------------------------------- #
# Primitive matching
# --------------------------------------------------------------------------- #


def _primitive_for_call(call: ast.Call, ctx: _ModuleCtx) -> str | None:
    """Return the blocking-primitive justification key, or ``None``.

    Four match shapes, in order:
      1. a bare ``Name`` bound by ``from <mod> import <name>`` whose resolved
         dotted path is in :data:`DOTTED_PRIMITIVES`;
      2. an unbound, unshadowed bare ``Name`` in :data:`BUILTIN_PRIMITIVES`;
      3. a dotted ``Attribute`` chain (after ``import X as Y`` substitution)
         in :data:`DOTTED_PRIMITIVES`;
      4. a bare method name in :data:`METHOD_PRIMITIVES`, on any receiver.
    """
    func = call.func

    if isinstance(func, ast.Name):
        if ctx.name_is_rebound(func.id, call):
            # A local binding holds this name, so neither the module's import
            # bindings nor the builtins describe what it calls -- including a
            # rebound `open`, which is then somebody's own helper.
            return None
        bound = ctx.bindings.get(func.id)
        if bound is not None:
            candidate = f'{bound[0]}.{bound[1]}'
            if _is_non_blocking(candidate):
                return None
            if candidate in DOTTED_PRIMITIVES:
                return candidate
            return None
        if func.id in BUILTIN_PRIMITIVES:
            return func.id
        return None

    if isinstance(func, ast.Attribute):
        path = dotted_path(func)
        if path is not None:
            head, _, rest = path.partition('.')
            real_head = ctx.module_aliases.get(head, head)
            resolved = f'{real_head}.{rest}' if rest else real_head
            if _is_non_blocking(resolved):
                return None
            if resolved in DOTTED_PRIMITIVES:
                return resolved
        if func.attr in METHOD_PRIMITIVES:
            return func.attr

    return None


def _is_non_blocking(dotted: str) -> bool:
    """``asyncio.sleep`` is not ``time.sleep``; never flag the async siblings."""
    return dotted.startswith(_NON_BLOCKING_PREFIXES)


# --------------------------------------------------------------------------- #
# Reachability
# --------------------------------------------------------------------------- #


def _resolve_callee(
    call: ast.Call,
    ctx: _ModuleCtx,
    modules: dict[str, _ModuleCtx],
) -> tuple[_ModuleCtx, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Resolve a call to the ``def`` it invokes, or ``None`` if unknowable.

    Resolution is deliberately narrow (see the module docstring's "Deliberate
    limits"): a bare ``Name`` via import bindings or a def actually VISIBLE at
    the call (this module's module-level defs, plus the call's own enclosing
    function scopes), and ``self.attr`` / ``cls.attr`` via the enclosing
    class's methods.  Anything else -- an arbitrary ``obj.method()``, a call
    through a variable, a name a parameter or local assignment shadows -- is
    silence.
    """
    func = call.func

    if isinstance(func, ast.Name):
        if ctx.name_is_rebound(func.id, call):
            # A parameter or local binding holds this name; whatever it calls
            # is not something the module's own defs or imports can tell us.
            return None
        bound = ctx.bindings.get(func.id)
        if bound is not None:
            target_ctx = modules.get(bound[0])
            if target_ctx is None:
                return None
            target = target_ctx.lookup_def(bound[1])
            return (target_ctx, target) if target is not None else None
        own = ctx.lookup_visible_def(func.id, call)
        return (ctx, own) if own is not None else None

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id not in ('self', 'cls'):
            return None
        class_name = ctx.enclosing_class(call)
        if class_name is None:
            return None
        target = ctx.methods_by_class.get(class_name, {}).get(func.attr)
        return (ctx, target) if target is not None else None

    return None


def _reaches_blocking(
    func_node: ast.FunctionDef,
    ctx: _ModuleCtx,
    modules: dict[str, _ModuleCtx],
    memo: dict[int, str | None],
    seen: frozenset[int] = frozenset(),
) -> str | None:
    """Return the primitive a SYNC function transitively reaches, else ``None``.

    Memoised and cycle-safe: a function already on the current path
    contributes ``None`` rather than recursing, so mutual recursion terminates
    without hiding a primitive that any member of the cycle actually reaches.
    """
    key = id(func_node)
    if key in memo:
        return memo[key]
    if key in seen:
        return None  # cycle: this arm adds nothing, do not recurse

    inner_seen = seen | {key}
    skip = _offloaded_callable_ids(func_node, ctx)
    result: str | None = None

    for node in _shallow_nodes(func_node, skip):
        if not isinstance(node, ast.Call):
            continue
        primitive = _primitive_for_call(node, ctx)
        if primitive is not None:
            result = primitive
            break
        resolved = _resolve_callee(node, ctx, modules)
        if resolved is None:
            continue
        target_ctx, target = resolved
        if isinstance(target, ast.AsyncFunctionDef):
            # Awaiting a coroutine yields to the loop; it is not a sync hop.
            continue
        deeper = _reaches_blocking(target, target_ctx, modules, memo, inner_seen)
        if deeper is not None:
            result = deeper
            break

    # Only cache a verdict computed without a cycle cut-off, so a `None` that
    # merely means "already on the path" is never memoised as "clean".
    if not seen or result is not None:
        memo[key] = result
    return result


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def find_loop_blocking_sites(sources: dict[str, str]) -> list[LoopBlockingSite]:
    """Return every coroutine call site that reaches a blocking primitive.

    Args:
        sources: ``{repo-relative path: source text}``.  Cross-module callee
            resolution only sees the modules present in this mapping, so a
            helper defined outside it is unresolvable -- a miss, never a false
            positive.

    Returns:
        One :class:`LoopBlockingSite` per CALL SITE, ordered by
        ``(filename, lineno)``.  A module that fails to parse contributes
        nothing and does not suppress its siblings.
    """
    modules: dict[str, _ModuleCtx] = {}
    ordered: list[_ModuleCtx] = []
    for relpath, source in sources.items():
        try:
            tree = ast.parse(source, filename=relpath)
        except SyntaxError:
            continue
        ctx = _ModuleCtx(relpath, tree)
        modules[ctx.name] = ctx
        ordered.append(ctx)

    memo: dict[int, str | None] = {}
    findings: list[LoopBlockingSite] = []

    for ctx in ordered:
        for func_node in ast.walk(ctx.tree):
            if not isinstance(func_node, ast.AsyncFunctionDef):
                continue
            skip = _offloaded_callable_ids(func_node, ctx)
            for node in _shallow_nodes(func_node, skip):
                if not isinstance(node, ast.Call):
                    continue
                # (1) The primitive written DIRECTLY in the coroutine body --
                # no helper anywhere for a definition-side census to point at.
                # This is the manifest_stamping::_stamp_capability_manifests_impl
                # shape (read_text + yaml.safe_load + write_text + yaml.safe_dump
                # inline in one coroutine).
                primitive = _primitive_for_call(node, ctx)
                if primitive is None:
                    # (2) The primitive reached THROUGH a sync helper -- the
                    # shape task 3778's per-module offload claim made invisible.
                    resolved = _resolve_callee(node, ctx, modules)
                    if resolved is None:
                        continue
                    target_ctx, target = resolved
                    if isinstance(target, ast.AsyncFunctionDef):
                        continue
                    primitive = _reaches_blocking(target, target_ctx, modules, memo)
                    if primitive is None:
                        continue
                callee = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr  # type: ignore[union-attr]
                )
                findings.append(LoopBlockingSite(
                    filename=ctx.relpath,
                    qualname=_compute_qualname(node, ctx.parent_map),
                    callee=callee,
                    primitive=primitive,
                    lineno=node.lineno,
                    content_hash=_content_hash(node),
                    message=(
                        f'{callee}(...) runs on the event loop and reaches '
                        f'{primitive} -- wrap it in asyncio.to_thread(...) or '
                        f'record why the cost is acceptable in '
                        f'loop_blocking_allowlist.py'
                    ),
                ))

    findings.sort(key=lambda f: (f.filename, f.lineno, f.qualname, f.callee))
    return findings

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
  bindings and its own ``def``s.  **An unresolvable name is silence, never a
  finding.**  A false RED in a whole-tree gate is worse than a miss: it blocks
  every merge until someone blesses a non-defect, which trains reviewers to
  bless rows unread and destroys the gate's value.
* Findings key on ``(relpath, qualname, content_hash)`` -- never ``lineno``,
  which drifts on every unrelated edit above the site.  The key shape and its
  helpers are reused verbatim from ``silent_fallthrough_scan``.

Deliberate limits (misses, not false alarms)
--------------------------------------------
* Only ``Name(...)``, ``self.attr(...)``/``cls.attr(...)`` and dotted-primitive
  calls resolve.  An arbitrary ``obj.method(...)`` does not: the receiver's type
  is unknowable without inference, and guessing produces false REDs.
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
    'subprocess.Popen': (
        'subprocess: the fork+exec itself blocks even when the caller never '
        'waits; .communicate()/.wait() then block again'
    ),
    'os.system': (
        'subprocess: spawns a shell AND blocks until it exits -- strictly worse '
        'than subprocess.run, never correct on a loop thread'
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

    # -- network: a connect() can hang for the full TCP timeout
    'socket.create_connection': (
        'network: a DNS lookup plus a TCP handshake, either of which can park '
        'the loop thread for the full connect timeout'
    ),
}

# Bare method names, matched on any receiver: `path.read_text()`.  Receiver
# types are not inferred, so these match by attribute name alone -- the
# deliberate trade for finding `self._path.read_text()` without type inference.
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

# The offload hops.  ``asyncio.to_thread(fn, ...)`` exempts arg 0;
# ``<loop>.run_in_executor(executor, fn, ...)`` exempts arg 1.
_OFFLOAD_CALLABLE_ARG: dict[str, int] = {
    'to_thread': 0,
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

        # Bare name -> the function def(s) carrying it, at ANY nesting depth.
        # Ambiguity (two defs sharing a name in one module) resolves to
        # silence: see _lookup_def.
        self.defs_by_name: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        # Class name -> {method name -> def}, for `self.foo(...)` resolution.
        self.methods_by_class: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        # Local binding -> (source module dotted name, source name).
        self.bindings: dict[str, tuple[str, str]] = {}
        # Local alias -> real module dotted name, from `import X as Y`.
        self.module_aliases: dict[str, str] = {}

        self._index()

    def _index(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, _FUNC_TYPES):
                self.defs_by_name.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.ClassDef):
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
        """Return the unique def named *name* in this module, else ``None``.

        Ambiguity is silence: two defs sharing a bare name (an overload in a
        branch, a test double) make the call unresolvable, and an unresolvable
        call is never a finding.
        """
        candidates = self.defs_by_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None


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


def _offloaded_callable_ids(func_node: ast.AST) -> frozenset[int]:
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
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        index = _OFFLOAD_CALLABLE_ARG.get(name or '')
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

    Three match shapes, in order:
      1. a bare ``Name`` bound by ``from <mod> import <name>`` whose resolved
         dotted path is in :data:`DOTTED_PRIMITIVES`;
      2. a dotted ``Attribute`` chain (after ``import X as Y`` substitution)
         in :data:`DOTTED_PRIMITIVES`;
      3. a bare method name in :data:`METHOD_PRIMITIVES`, on any receiver.
    """
    func = call.func

    if isinstance(func, ast.Name):
        bound = ctx.bindings.get(func.id)
        if bound is not None:
            candidate = f'{bound[0]}.{bound[1]}'
            if _is_non_blocking(candidate):
                return None
            if candidate in DOTTED_PRIMITIVES:
                return candidate
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
    limits"): a bare ``Name`` via import bindings or this module's own defs, and
    ``self.attr`` / ``cls.attr`` via the enclosing class's methods.  Anything
    else -- an arbitrary ``obj.method()``, a call through a variable -- is
    silence.
    """
    func = call.func

    if isinstance(func, ast.Name):
        bound = ctx.bindings.get(func.id)
        if bound is not None:
            target_ctx = modules.get(bound[0])
            if target_ctx is None:
                return None
            target = target_ctx.lookup_def(bound[1])
            return (target_ctx, target) if target is not None else None
        own = ctx.lookup_def(func.id)
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
    skip = _offloaded_callable_ids(func_node)
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
            skip = _offloaded_callable_ids(func_node)
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

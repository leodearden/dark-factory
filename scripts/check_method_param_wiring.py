#!/usr/bin/env python3
"""Assert that a specific FUNCTION declares — and forwards — a parameter.

WHY THIS EXISTS (task 3364).

``orchestrator/src/orchestrator/delivered_checks.py::_run_grep_check`` shells to
``git grep -E -e <pattern> <ref>``: single-line POSIX ERE, no multiline mode. This
repo's house style spreads every non-trivial ``def`` across one parameter per line,
so *no* single-line pattern can bind a match to one signature — a ``grep`` check is
inherently FILE-scoped. That is not a hypothetical: capability
``qdrant-vector-access-for-ann`` (task 3210) was gated by
``{kind: grep, pattern: 'with_vectors: bool'}`` over
``fused-memory/src/fused_memory/backends/mem0_client.py``, which would have reported
DELIVERED had the parameter landed on ``get_point_by_id`` instead of the intended
``scroll_by_metadata``. An AST walk of the resolved function is *semantically*
method-scoped rather than regex-approximated.

DECLARATION IS NOT THE CAPABILITY. ``--forwards-to`` is the load-bearing conjunct: a
stub ``with_vectors: bool = False`` that is accepted and then dropped satisfies a
signature check while delivering nothing — the same hollow-DELIVERED failure mode as
the file-scoped grep, one level in. Requiring the keyword's value to be an
``ast.Name`` (not an ``ast.Constant``) is exactly what separates
``scroll_by_metadata``'s ``with_vectors=with_vectors`` from ``get_point_by_id``'s
hardcoded ``with_vectors=False``.

DELIVERED_CHECK INVOCATION (mirrored verbatim in
``docs/prds/memory-eval-program.capability-manifest.yaml``, capability
``qdrant-vector-access-for-ann``; keep the two in sync)::

    scripts/check_method_param_wiring.py \\
        --file fused-memory/src/fused_memory/backends/mem0_client.py \\
        --function scroll_by_metadata \\
        --param with_vectors \\
        --annotation bool \\
        --forwards-to scroll

EXIT-CODE CONTRACT. ``_run_script_check`` maps rc==0 -> DELIVERED and *every*
non-zero rc -> FAILED; only a raised exception (OSError, timeout) reaches ERRORED.
There is no exit code meaning "I could not evaluate this". FAILED is a definitive
absence claim that ``Scheduler._compute_delivered_check_cache`` renders as a
``DEP_CAPABILITY_NOT_DELIVERED`` escalation whose blocked dependents are explicitly
NOT auto-re-pended. So every non-zero return here must be one this script actually
understands, and must name which conjunct failed — an operator reading that
escalation has to be able to tell "the capability regressed" from "the check's own
arguments are stale". For the same reason this reads the WORKING CHECKOUT only: no
``--ref`` / ``git show`` mode, whose ref-absent failure (a worktree, a CI sandbox)
would exit non-zero and be misread as a regression. That working-checkout
approximation is already DECIDED in ``_run_script_check``'s docstring (PRD Open-Q 2).

Must stay directly executable (mode 100755): ``_run_script_check`` builds
``argv = [str(project_root / meta.script), *meta.args]`` and execs it, so a
non-executable target raises OSError -> ERRORED forever — a check that can never
pass and never fail, i.e. a silently un-gated capability.
"""
import ast


class AmbiguousFunction(Exception):
    """Raised when a name resolves to more than one function in a module.

    Deliberately NOT a silent first-match: a check that picks arbitrarily between
    two same-named candidates is not method-scoped, which is the entire property
    this script exists to provide.
    """


def parse_module(source: str) -> ast.Module:
    """Parse *source* into a module AST."""
    return ast.parse(source)


def resolve_function(
    tree: ast.Module, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the single ``def``/``async def`` named *name*, or None if absent.

    Walks the whole module (``ast.walk``), so a method nested in a class body
    resolves the same as a module-level function. Raises :class:`AmbiguousFunction`
    when more than one candidate matches.
    """
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    ]
    if len(matches) > 1:
        raise AmbiguousFunction(
            f'{name!r} resolves to {len(matches)} functions; '
            'cannot make a method-scoped assertion'
        )
    return matches[0] if matches else None


def declares_param(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    param: str,
    annotation: str | None = None,
) -> bool:
    """True when *fn*'s own signature declares *param*.

    Covers positional-only, positional-or-keyword and keyword-only parameters, so
    the assertion does not silently depend on where a ``*`` happens to sit today.

    When *annotation* is given, the declared annotation must match it exactly
    (compared via ``ast.unparse``); an UNannotated parameter never satisfies a
    required annotation. That preserves precisely what the superseded grep pattern
    ``'with_vectors: bool'`` asserted — now scoped to the right function.
    """
    args = fn.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if arg.arg != param:
            continue
        if annotation is None:
            return True
        if arg.annotation is None:
            return False
        return ast.unparse(arg.annotation) == annotation
    return False


def _callee_name(func: ast.expr) -> str | None:
    """Resolve a call's callee to a bare name, or None when it has none.

    ``client.scroll(...)`` and ``scroll(...)`` both resolve to ``'scroll'``.
    Anything else — a subscripted or called-expression func — resolves to None
    and simply does not match. The RECEIVER is deliberately not resolved:
    binding ``client`` to a type needs inference the AST cannot give, so the
    assertion is on the method name.
    """
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def forwards_param_to(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, param: str, callee: str
) -> bool:
    """True when *fn* passes its own *param* through to a call named *callee*.

    The keyword's value must be an ``ast.Name`` bound to *param* — a literal
    (``with_vectors=False``) is a hardcoded constant, not a forward, and that is
    exactly what separates ``scroll_by_metadata``'s ``with_vectors=with_vectors``
    from ``get_point_by_id``'s ``with_vectors=False``.

    ``ast.walk(fn)`` covers calls nested in any expression — on main the real
    call sits inside ``await asyncio.wait_for(client.scroll(...), timeout=...)``,
    which a top-level-statement scan would miss. Walking *fn* rather than the
    module is what keeps the assertion method-scoped.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or _callee_name(node.func) != callee:
            continue
        for kw in node.keywords:
            if kw.arg == param and isinstance(kw.value, ast.Name) and kw.value.id == param:
                return True
    return False

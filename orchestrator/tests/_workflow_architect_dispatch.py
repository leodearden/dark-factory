"""Static scan of workflow.py: which briefing builders reach the ARCHITECT role.

``TaskWorkflow`` dispatches the architect through ``self._invoke(ARCHITECT,
build_prompt, ...)``, either directly or through a *runner*: a method such as
``_run_architect_for_plan`` that receives a prompt builder as a parameter and
forwards it to ARCHITECT, bare or wrapped in ``partial(...)``, itself or through
another runner. This module finds
those dispatch sites and follows every argument they hand to ARCHITECT back
through the method's own bindings, nested ``def``s and assignments, to the
``self.briefing.build_*`` builders it names. A builder counts whether it is
called or passed by reference, as in ``partial(self.briefing.build_..., ...)``;
a builder the same method uses for another role does not count.

A workflow.py refactor that dispatches the architect in a shape this scan does
not recognise, such as passing ARCHITECT by keyword or through ``_invoke_writer``,
makes ``architect_briefing_builders`` come back short. Extend the scan here.
"""

from __future__ import annotations

import ast
import inspect
from collections import defaultdict

from orchestrator import workflow

_Method = ast.FunctionDef | ast.AsyncFunctionDef


def architect_briefing_builders() -> set[str]:
    """The ``self.briefing.build_*`` builders whose output workflow.py hands to ARCHITECT."""
    methods = _workflow_methods()
    runners = _architect_runners(methods)
    return {
        builder
        for method in methods.values()
        for builder in _builders_reachable(method, _dispatch_arguments(method, runners))
    }


def _workflow_methods() -> dict[str, _Method]:
    tree = ast.parse(inspect.getsource(workflow))
    return {
        method.name: method
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for method in cls.body if isinstance(method, _Method)
    }


def _calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _invokes_architect(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == '_invoke'
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == 'ARCHITECT'
    )


def _self_method_name(call: ast.Call) -> str | None:
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == 'self'
    ):
        return func.attr
    return None


def _dispatch_arguments(method: _Method, runners: set[str]) -> list[ast.expr]:
    """Every argument *method* hands to ARCHITECT, directly or through a runner.

    For ``self._invoke(ARCHITECT, ...)`` that is everything but the role. For a
    runner it is everything, since a runner's prompt parameter need not come first.
    Keyword arguments count in both, so ``build_prompt=`` is seen.
    """
    arguments: list[ast.expr] = []
    for call in _calls(method):
        if _invokes_architect(call):
            positional = call.args[1:]
        elif _self_method_name(call) in runners:
            positional = call.args
        else:
            continue
        arguments += [*positional, *(keyword.value for keyword in call.keywords)]
    return arguments


def _forwarded_names(argument: ast.expr) -> set[str]:
    """The names *argument* hands on as a prompt builder: itself, or what a ``partial(...)`` wraps."""
    if isinstance(argument, ast.Name):
        return {argument.id}
    if isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name) and argument.func.id == 'partial':
        return {name for wrapped in argument.args for name in _forwarded_names(wrapped)}
    return set()


def _is_runner(method: _Method, runners: set[str]) -> bool:
    parameters = {arg.arg for arg in (*method.args.args, *method.args.kwonlyargs)}
    return any(
        _forwarded_names(argument) & parameters
        for argument in _dispatch_arguments(method, runners)
    )


def _architect_runners(methods: dict[str, _Method]) -> set[str]:
    """Methods that hand a prompt builder they were given on to ARCHITECT."""
    runners: set[str] = set()
    while found := {
        name for name, method in methods.items()
        if name not in runners and _is_runner(method, runners)
    }:
        runners |= found
    return runners


def _local_bindings(method: _Method) -> dict[str, list[ast.AST]]:
    """What each name in *method* is bound to: a nested ``def`` or an assigned value."""
    bindings: dict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(method):
        if isinstance(node, _Method) and node is not method:
            bindings[node.name].append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id].append(node.value)
        elif (
            isinstance(node, ast.AnnAssign | ast.NamedExpr)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bindings[node.target.id].append(node.value)
    return bindings


def _builders_reachable(method: _Method, arguments: list[ast.expr]) -> set[str]:
    """The briefing builders *arguments* name, directly or through *method*'s bindings."""
    bindings = _local_bindings(method)
    builders: set[str] = set()
    followed: set[str] = set()
    pending: list[ast.AST] = list(arguments)
    while pending:
        node = pending.pop()
        builders |= _briefing_builders_named(node)
        for name in _names_read(node) - followed:
            followed.add(name)
            pending += bindings.get(name, [])
    return builders


def _names_read(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _briefing_builders_named(node: ast.AST) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr.startswith('build_')
        and isinstance(child.value, ast.Attribute)
        and child.value.attr == 'briefing'
    }

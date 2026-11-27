"""What the narrow merge gate runs, and the decision that carries it (rationale: docs/narrow-merge-gate.md)."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from orchestrator.config import ModuleConfig
from orchestrator.narrow_gate_class import GateClassification, GateMiss, MergeGateResolution
from orchestrator.verify_artifact_consumers import (
    BEHAVIOURAL_ARTIFACTS,
    ConsumerDemand,
    consumer_demand,
)
from orchestrator.verify_cmd import parse_config_command, render
from orchestrator.verify_plan import (
    FileKind,
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    classify_file,
    deselecting_expression_for_command,
    scope_prefix_to_keyword,
    whole_tree_guard_family,
)

WorktreeReader = Callable[[str], str | None]
PkgTestLister = Callable[[str], list[str]]


@dataclass(frozen=True)
class NarrowGateDecision:
    """A merge verify's gate: its classification, and the plan a narrow one runs."""

    classification: GateClassification
    plan: VerifyPlan | None
    changed_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.resolution.is_narrow != (self.plan is not None):
            raise ValueError(f'a {self.resolution.value} decision carries a plan iff it is narrow; got plan={self.plan!r}')

    @property
    def resolution(self) -> MergeGateResolution:
        return self.classification.resolution

    @property
    def miss(self) -> GateMiss | None:
        return self.classification.miss

    @classmethod
    def full(cls, miss: GateMiss, changed_paths: Sequence[str] = ()) -> NarrowGateDecision:
        """The full-gate decision for *miss*."""
        return cls(GateClassification(MergeGateResolution.FULL, miss), None, tuple(changed_paths))


def narrow_gate_plan(
    resolution: MergeGateResolution,
    changed_paths: Sequence[str],
    module_configs: Sequence[ModuleConfig],
    *,
    list_pkg_tests: PkgTestLister,
    worktree_reader: WorktreeReader,
) -> VerifyPlan | None:
    """The narrow *resolution*'s command set over *changed_paths*; None when no narrow set can vouch for them.

    Every module runs its whole-tree guard family plus the consumers of any touched behavioural
    artifact. The comments gate adds file-scoped ruff and pyright on each owning module's changed
    ``.py`` files. A ``.py`` no module owns, or a set in which nothing would run, yields None.
    """
    if not resolution.is_narrow:
        raise ValueError(f'narrow_gate_plan builds a narrow command set; {resolution.value} has none')
    py_by_owner = _changed_py_by_owner(changed_paths, module_configs) if resolution is MergeGateResolution.COMMENTS else {}
    if py_by_owner is None:
        return None
    demand = consumer_demand(
        changed_paths, [mc.prefix for mc in module_configs if mc.test_command], BEHAVIOURAL_ARTIFACTS,
    )
    runs = tuple(
        run
        for mc in module_configs
        for run in (
            _check_run(mc.prefix, mc.lint_command, 'ruff check', 'lint', resolution, py_by_owner.get(mc.prefix, ())),
            _check_run(mc.prefix, mc.type_check_command, 'pyright', 'pyright', resolution, py_by_owner.get(mc.prefix, ())),
            _pytest_run(mc, resolution, demand.get(mc.prefix), list_pkg_tests, worktree_reader),
        )
    )
    if not any(run.cmd is not None for run in runs):
        return None
    return VerifyPlan(runs)


def planned_commands(plan: VerifyPlan) -> tuple[str, ...]:
    """The rendered command of every run *plan* prescribes, in plan order; the verify cache may replay one rather than run it."""
    return tuple(render(run.cmd) for run in plan.runs if run.cmd is not None)


def module_test_lister(tree_paths: Sequence[str], prefixes: Sequence[str]) -> PkgTestLister:
    """A ``list_pkg_tests`` seam over a tree's paths: each module's collectable test files, wherever under it they sit.

    A path counts under its longest owning prefix only, so a nested module's tests run once, under its own command.
    """
    owned: dict[str, list[str]] = {}
    for path in tree_paths:
        owner = _owner(path, prefixes)
        if owner is not None and classify_file(path, None) is FileKind.COLLECTABLE_TEST:
            owned.setdefault(owner, []).append(path)
    return lambda prefix: list(owned.get(prefix, ()))


def _owner(path: str, prefixes: Sequence[str]) -> str | None:
    return max((p for p in prefixes if path.startswith(p + '/')), key=len, default=None)


def _changed_py_by_owner(
    changed_paths: Sequence[str], module_configs: Sequence[ModuleConfig],
) -> Mapping[str, tuple[str, ...]] | None:
    prefixes = [mc.prefix for mc in module_configs]
    owned: dict[str, list[str]] = {}
    for path in sorted(p for p in changed_paths if p.endswith('.py')):
        owner = _owner(path, prefixes)
        if owner is None:
            return None
        owned.setdefault(owner, []).append(path)
    return {owner: tuple(paths) for owner, paths in owned.items()}


def _check_run(
    prefix: str,
    command: str | None,
    keyword: str,
    tool: str,
    resolution: MergeGateResolution,
    py_files: tuple[str, ...],
) -> PlannedRun:
    label = f'{tool}: {resolution.value}:'
    if resolution is MergeGateResolution.DOCS:
        return PlannedRun(prefix, None, ScopeKind.SKIPPED, f'{label} prose changes only, nothing for {tool} to check')
    if not py_files:
        return PlannedRun(prefix, None, ScopeKind.SKIPPED, f'{label} no changed .py file under {prefix}')
    if not command:
        return PlannedRun(prefix, None, ScopeKind.SKIPPED, f'{label} no {tool} command configured')
    return PlannedRun(
        prefix,
        scope_prefix_to_keyword(command, keyword, list(py_files)),
        ScopeKind.FILE_SCOPED,
        f'{label} file-scoped to {len(py_files)} changed .py file(s) whose change is comments only',
        scoped_targets=py_files,
    )


def _pytest_run(
    mc: ModuleConfig,
    resolution: MergeGateResolution,
    demand: ConsumerDemand | None,
    list_pkg_tests: PkgTestLister,
    worktree_reader: WorktreeReader,
) -> PlannedRun:
    label = f'pytest: {resolution.value}:'
    if not mc.test_command:
        return PlannedRun(mc.prefix, None, ScopeKind.SKIPPED, f'{label} no test_command configured')
    artifacts = ', '.join(demand.artifacts) if demand else ''
    if demand is not None and demand.full_suite:
        return PlannedRun(
            mc.prefix, parse_config_command(mc.test_command), ScopeKind.FULL_SUITE,
            f'{label} behavioural artifact(s) {artifacts} touched — {mc.prefix} full suite consumes them',
        )
    family = whole_tree_guard_family(mc.prefix, list_pkg_tests, worktree_reader)
    consumers = demand.test_files if demand else ()
    targets = tuple(sorted({*family, *consumers}))
    if not targets:
        return PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED,
            f'{label} no whole-tree guard family and no behavioural-artifact consumer under {mc.prefix}',
        )
    deselecting = deselecting_expression_for_command(mc.test_command, targets, worktree_reader)
    if deselecting is not None:
        return PlannedRun(
            mc.prefix, None, ScopeKind.SKIPPED,
            f"{label} the narrowed selection is entirely deselected by the module's -m {deselecting!r}",
        )
    consumed = f' plus consumer test(s) of behavioural artifact(s) {artifacts}' if consumers else ''
    return PlannedRun(
        mc.prefix,
        scope_prefix_to_keyword(mc.test_command, 'pytest', list(targets)),
        ScopeKind.FILE_SCOPED,
        f'{label} the {len(family)}-module whole-tree guard family{consumed}',
        scoped_targets=targets,
    )

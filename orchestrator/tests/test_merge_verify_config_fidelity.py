"""Which host config fields can decide a merge verdict — a census-guarded table.

WHAT IS MEASURED.  Both merge-verify legs run one bundle,
``orchestrator/src/orchestrator/verify_runner.py::LocalRunner.run_merge_verify``.
The local leg runs it on the dispatcher's own ``OrchestratorConfig``.  The remote
leg runs it on the REMOTE host's config, after
``verify_runner.py::run_merge_verify_on_worktree`` has applied the dispatcher's
``MergeVerifySpec`` onto it.  Every config name the bundle reads is a place where
the two legs can disagree.  The census below walks the bundle's static call
graph and lists every ``OrchestratorConfig`` field, public property and public
method it names.  It over-approximates on purpose: a ``ModuleConfig`` attribute
with the same name counts too.

THE THREE CLASSES.

- ``SPEC_CARRIED``: the dispatcher's value rides the spec and wins on the
  remote leg.  Each such row is checked by a behavioural probe below, which goes
  through ``run_merge_verify_on_worktree`` and the wire codec.
- ``MERGE_NEUTRAL``: the host supplies the value, but no value of it can turn a
  failing verify into a passing one.  Scheduling, fan-out caps and timeouts that
  fail closed are in this class.
- ``HOST_DECIDING``: the host's value can decide the verdict.  This is a latent
  false green.  Each row names the follow-up work that owns closing it.

CLASSIFYING A NEW READ.  When the census reports a name the table lacks, add a
row to ``MERGE_PATH_CONFIG_READS``.  If the host's value can decide the verdict,
classify it ``HOST_DECIDING``, file a narrow follow-up task for it, and put the
filing's ``escalation_id`` key (or the owning task id) in ``follow_ups``.  Do not
fix the hole in the same change that classifies it.  When a read disappears,
delete its row: a stale row fails as loudly as a missing one.
"""
from __future__ import annotations

import ast
import functools
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import FunctionType, MappingProxyType

from orchestrator import verify, verify_runner
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane import gates


class Fidelity(Enum):
    SPEC_CARRIED = 'spec_carried'
    MERGE_NEUTRAL = 'merge_neutral'
    HOST_DECIDING = 'host_deciding'


@dataclass(frozen=True)
class Classified:
    fidelity: Fidelity
    why: str
    follow_ups: tuple[str, ...] = ()


_PROFILE_RIDES_SPEC = (
    'Task 2822 fix a: the merge-gate profile rides MergeVerifySpec, and '
    "run_merge_verify_on_worktree applies it through apply_spec_to_host_config's field_overrides."
)
_GLOBAL_COMMANDS_FROM_HOST = (
    'Carried as spec.global_verify_command only when the spec has no verify_commands and the '
    'dispatcher has a global command. The force_workspace route with a non-empty module set ends '
    "in run_verification(module_config=None), which runs the host's own globals."
)
_FAN_OUT_CAP = 'A fan-out cap: it bounds how many module verifies run at once, never which ones run.'
_ADMISSION = (
    'Verify admission scheduling: it decides when a run starts and at what priority, never what it selects.'
)
_CGROUP_SHAPING = 'cgroup CPU shaping: it weights the run against other work and leaves its commands alone.'
_FAILS_CLOSED_TIMEOUT = (
    'A timeout only ends a run timed_out, which the merge path fails closed: the scoped verify does '
    'not pass, and the unscoped gate runs with block_on_timeout=True.'
)

MERGE_PATH_CONFIG_READS: Mapping[str, Classified] = MappingProxyType({
    # SPEC_CARRIED
    'merge_verify_workspace': Classified(
        Fidelity.SPEC_CARRIED,
        f'{_PROFILE_RIDES_SPEC} LocalRunner.run_merge_verify reads it as force_workspace.',
    ),
    'merge_verify_breadth': Classified(Fidelity.SPEC_CARRIED, _PROFILE_RIDES_SPEC),
    'module_configs_or_empty': Classified(
        Fidelity.SPEC_CARRIED,
        'Task 4536: apply_spec_to_host_config installs the module registry from spec.verify_commands, '
        "so the host's own discovery walk never decides.",
    ),
    'verify_env': Classified(
        Fidelity.SPEC_CARRIED,
        'Task 5496: {**host effective_verify_env, **spec.verify_env}. The spec wins on every key it '
        'ships; host-only keys survive by design.',
    ),
    'verify_cold_preprovision_command': Classified(
        Fidelity.SPEC_CARRIED,
        "Task 6127: a non-empty spec value replaces the host's, and '' keeps the host's own by design "
        '(plans/remote-task-leg-verify-prd.md §4.1).',
    ),
    # HOST_DECIDING
    'scope_cargo': Classified(
        Fidelity.HOST_DECIDING,
        "Gates verify.py::_apply_cargo_scope's --workspace to -p rewrite, including the zero-module "
        'cargo-scoped route. MergeVerifySpec does not carry it.',
        follow_ups=('agent-followup-5295-scope-cargo',),
    ),
    'git': Classified(
        Fidelity.HOST_DECIDING,
        'git.merge_config_only_full_gate_globs forces the full gate for a config-only diff and is never '
        "carried. git.main_branch drives the remote's _derive_task_files_from_git whenever "
        'spec.task_files is None: merge_lane/drift.py ships None, and so does the serial route when '
        "the dispatcher's own derivation comes back empty.",
        follow_ups=(
            'agent-followup-5295-config-only-globs',
            'agent-followup-5295-remote-task-files-derivation',
        ),
    ),
    'test_command': Classified(
        Fidelity.HOST_DECIDING, _GLOBAL_COMMANDS_FROM_HOST, follow_ups=('agent-followup-5295-global-commands',)
    ),
    'lint_command': Classified(
        Fidelity.HOST_DECIDING, _GLOBAL_COMMANDS_FROM_HOST, follow_ups=('agent-followup-5295-global-commands',)
    ),
    'type_check_command': Classified(
        Fidelity.HOST_DECIDING, _GLOBAL_COMMANDS_FROM_HOST, follow_ups=('agent-followup-5295-global-commands',)
    ),
    # MERGE_NEUTRAL
    'concurrent_verify': Classified(Fidelity.MERGE_NEUTRAL, 'Leg order only: every leg still runs.'),
    'sequential_lint_first': Classified(
        Fidelity.MERGE_NEUTRAL, 'Lint-first skips test and type check only after lint is already red.'
    ),
    'cpu_governance': Classified(Fidelity.MERGE_NEUTRAL, 'An exec wrapper around the same commands.'),
    'max_concurrent_module_verifies': Classified(Fidelity.MERGE_NEUTRAL, _FAN_OUT_CAP),
    'merge_verify_max_concurrent_modules': Classified(Fidelity.MERGE_NEUTRAL, _FAN_OUT_CAP),
    'project_root': Classified(
        Fidelity.MERGE_NEUTRAL, "The host's own checkout, read only to tag the cgroup scope."
    ),
    'verify_admission_background_promote_secs': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_enabled': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_nice_background': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_nice_merge': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_nice_task': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_order': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_pytest_n': Classified(
        Fidelity.MERGE_NEUTRAL, f'{_ADMISSION} It changes xdist parallelism, never test selection.'
    ),
    'verify_admission_slots_dir': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_admission_task_slots': Classified(Fidelity.MERGE_NEUTRAL, _ADMISSION),
    'verify_cgroup_cpu_weight_background': Classified(Fidelity.MERGE_NEUTRAL, _CGROUP_SHAPING),
    'verify_cgroup_cpu_weight_merge': Classified(Fidelity.MERGE_NEUTRAL, _CGROUP_SHAPING),
    'verify_cgroup_cpu_weight_task': Classified(Fidelity.MERGE_NEUTRAL, _CGROUP_SHAPING),
    'verify_use_cgroup_scope': Classified(Fidelity.MERGE_NEUTRAL, _CGROUP_SHAPING),
    'verify_clock_stop_enabled': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_clock_stop_heartbeat_idle_max': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_clock_stop_marker_heartbeat': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_clock_stop_marker_start': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_clock_stop_marker_stop': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_clock_stop_max_total_secs': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_command_timeout_secs': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_cold_command_timeout_secs': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'merge_verify_cold_command_timeout_secs': Classified(Fidelity.MERGE_NEUTRAL, _FAILS_CLOSED_TIMEOUT),
    'verify_timeout_retries': Classified(
        Fidelity.MERGE_NEUTRAL,
        'Every run_verification call reachable from LocalRunner.run_merge_verify passes max_retries=0.',
    ),
})

UNREAD_CANDIDATES: Mapping[str, str] = MappingProxyType({
    'lock_depth': 'Module prefixes on the merge path come from spec.verify_commands.',
    'for_module': (
        'No host-side prefix resolution happens on the merge path; '
        'config._discover_module_configs is unreachable from the entry points.'
    ),
})


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
_PACKAGE = 'orchestrator'

# The bundle both legs run, plus the callables run_merge_verify_on_worktree wires into it.
_ENTRY_POINTS = (
    verify_runner.LocalRunner.run_merge_verify,
    verify.run_scoped_verification,
    verify.run_verification,
    gates._run_unscoped_typechecks,
)

CENSUS_NAMES: frozenset[str] = frozenset(OrchestratorConfig.model_fields) | frozenset(
    name
    for name, member in vars(OrchestratorConfig).items()
    if isinstance(member, property | FunctionType) and not name.startswith(('_', 'model_'))
)

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_FunctionKey = tuple[str, str]
_ImportTarget = tuple[str, str | None]


@dataclass(frozen=True)
class _ModuleIndex:
    defs: Mapping[str, _FunctionNode]
    imports: Mapping[str, frozenset[_ImportTarget]]


@dataclass(frozen=True)
class _Census:
    reads: Mapping[str, tuple[str, ...]]
    visited_modules: frozenset[str]


def _module_path(module: str) -> Path | None:
    base = _SRC_DIR.joinpath(*module.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _in_package(module: str) -> bool:
    return module == _PACKAGE or module.startswith(f'{_PACKAGE}.')


def _import_source(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ''
    anchor = package.split('.')[: len(package.split('.')) - (node.level - 1)]
    return '.'.join([*anchor, *([node.module] if node.module else [])])


def _import_targets(tree: ast.Module, package: str) -> Iterator[tuple[str, _ImportTarget]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and _in_package(alias.name):
                    yield alias.asname, (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            source = _import_source(node, package)
            if not _in_package(source):
                continue
            for alias in node.names:
                submodule = f'{source}.{alias.name}'
                target = (submodule, None) if _module_path(submodule) else (source, alias.name)
                yield alias.asname or alias.name, target


@functools.cache
def _index(module: str) -> _ModuleIndex:
    path = _module_path(module)
    if path is None:
        return _ModuleIndex(defs=MappingProxyType({}), imports=MappingProxyType({}))
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    defs: dict[str, _FunctionNode] = {}
    for node in tree.body:
        if isinstance(node, _FunctionNode):
            defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, _FunctionNode):
                    defs[f'{node.name}.{member.name}'] = member
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    imports: defaultdict[str, set[_ImportTarget]] = defaultdict(set)
    for local_name, target in _import_targets(tree, package):
        imports[local_name].add(target)
    return _ModuleIndex(
        defs=MappingProxyType(defs),
        imports=MappingProxyType({name: frozenset(t) for name, t in imports.items()}),
    )


def _resolve(module: str, attr: str, seen: frozenset[_FunctionKey] = frozenset()) -> Iterator[_FunctionKey]:
    index = _index(module)
    if attr in index.defs:
        yield module, attr
        return
    for target_module, target_attr in index.imports.get(attr, ()):
        if target_attr is not None and (target_module, target_attr) not in seen:
            yield from _resolve(target_module, target_attr, seen | {(module, attr)})


def _callees(module: str, qualname: str, node: _FunctionNode) -> Iterator[_FunctionKey]:
    index = _index(module)
    enclosing_class = qualname.rpartition('.')[0]
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id in index.defs:
                yield module, sub.id
            for target_module, attr in index.imports.get(sub.id, ()):
                if attr is not None:
                    yield from _resolve(target_module, attr)
        elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            owner = sub.value.id
            method = f'{enclosing_class}.{sub.attr}'
            if owner in ('self', 'cls') and enclosing_class and method in index.defs:
                yield module, method
            for target_module, attr in index.imports.get(owner, ()):
                if attr is None:
                    yield from _resolve(target_module, sub.attr)


def _config_names_read(node: _FunctionNode) -> Iterator[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in CENSUS_NAMES:
            yield sub.attr
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == 'getattr'
            and len(sub.args) >= 2
            and isinstance(sub.args[1], ast.Constant)
            and sub.args[1].value in CENSUS_NAMES
        ):
            yield sub.args[1].value


@functools.cache
def _merge_path_census() -> _Census:
    entries = [(fn.__module__, fn.__qualname__) for fn in _ENTRY_POINTS]
    for module, qualname in entries:
        assert qualname in _index(module).defs, f'entry point {module}::{qualname} not found under {_SRC_DIR}'
    visited: set[_FunctionKey] = set()
    queue = deque(entries)
    sites: defaultdict[str, set[str]] = defaultdict(set)
    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)
        module, qualname = key
        node = _index(module).defs[qualname]
        for name in _config_names_read(node):
            sites[name].add(f'{module}::{qualname}')
        queue.extend(callee for callee in _callees(module, qualname, node) if callee not in visited)
    return _Census(
        reads=MappingProxyType({name: tuple(sorted(where)) for name, where in sites.items()}),
        visited_modules=frozenset(module for module, _ in visited),
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_every_merge_path_config_read_is_classified() -> None:
    reads = _merge_path_census().reads
    unclassified = sorted(set(reads) - set(MERGE_PATH_CONFIG_READS))
    assert not unclassified, (
        'The merge path reads OrchestratorConfig names this file does not classify. '
        'Add a MERGE_PATH_CONFIG_READS row for each one. If the host value can decide '
        'the verdict, classify it HOST_DECIDING, file a narrow follow-up, and cite its '
        'key in follow_ups:\n'
        + '\n'.join(f'  {name}: read at {", ".join(reads[name])}' for name in unclassified)
    )


def test_every_classified_row_is_still_read() -> None:
    stale = sorted(set(MERGE_PATH_CONFIG_READS) - set(_merge_path_census().reads))
    assert not stale, (
        f'These MERGE_PATH_CONFIG_READS rows are no longer read on the merge path; delete them: {stale}'
    )


def test_census_resolves_reads_across_modules() -> None:
    census = _merge_path_census()
    assert len(census.reads) >= 30, sorted(census.reads)
    assert {'orchestrator.verify_plan', 'orchestrator.merge_lane.gates'} <= census.visited_modules, sorted(
        census.visited_modules
    )

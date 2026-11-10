"""Unit keys: content hashes of a dispatched tree, one per verifiable unit.

A key is the sha256 of its unit's sorted ``(name, value)`` inputs; every input
is recorded beside the key, so an audit can say which input a stale replay
missed.
"""

from __future__ import annotations

import asyncio
import hashlib
import posixpath
import shlex
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from orchestrator import verify_plan
from orchestrator.verify_cache import tree_objects
from orchestrator.verify_cache.toolchain import ToolchainFingerprint
from orchestrator.verify_cache.workspace import WorkspaceGraph, read_workspace
from orchestrator.verify_cmd import render
from orchestrator.verify_evidence import UnitTier, module_leg_name, run_spec_hash
from orchestrator.verify_plan import PlannedRun, VerifyPlan
from orchestrator.verify_spec_shared import canonical_json

Inputs = tuple[tuple[str, str], ...]

#: The value of a path input the dispatched tree does not hold.
ABSENT = 'absent'
#: The tree-relative path naming the whole tree.
WHOLE_TREE = '.'

_TREE_UNIT_ID = 'tree'
_UNIT_ID_PREFIX = {UnitTier.MODULE: 'module:', UnitTier.TEST: 'test:'}
_SHELL_PUNCTUATION = frozenset('();<>|&')


def digest(inputs: Iterable[tuple[str, str]]) -> str:
    """sha256 of *inputs*, sorted, as canonical JSON."""
    return hashlib.sha256(canonical_json(sorted(inputs)).encode()).hexdigest()


@dataclass(frozen=True)
class UnitKey:
    """One unit's key and the inputs it digests; construction checks every invariant."""

    unit_id: str
    tier: UnitTier
    key: str
    spec_hash: str
    inputs: Inputs

    def __post_init__(self) -> None:
        if not _unit_id_fits(self.tier, self.unit_id):
            raise ValueError(f'unit id {self.unit_id!r} does not name a {self.tier} unit')
        names = [name for name, _value in self.inputs]
        if list(self.inputs) != sorted(self.inputs) or len(set(names)) != len(names):
            raise ValueError(f'unit {self.unit_id!r}: inputs are not sorted with unique names: {names}')
        if ('spec', self.spec_hash) not in self.inputs:
            raise ValueError(f"unit {self.unit_id!r}: spec_hash {self.spec_hash!r} is not its 'spec' input")
        expected = digest(self.inputs)
        if self.key != expected:
            raise ValueError(f'unit {self.unit_id!r}: key {self.key!r} is not the digest {expected!r} of its inputs')

    @classmethod
    def of(cls, unit_id: str, tier: UnitTier, spec_hash: str, inputs: Iterable[tuple[str, str]]) -> UnitKey:
        ordered = tuple(sorted(inputs))
        return cls(unit_id, tier, digest(ordered), spec_hash, ordered)


@dataclass(frozen=True)
class TreeInputs:
    """What one dispatched tree holds at every path a plan's unit keys read."""

    tree_oid: str
    root_blobs: tuple[tuple[str, str], ...]
    object_ids: Mapping[str, str | None]
    workspace: WorkspaceGraph

    def oid(self, path: str) -> str | None:
        if path == WHOLE_TREE:
            return self.tree_oid
        try:
            return self.object_ids[path]
        except KeyError:
            raise KeyError(f'path {path!r} was not read for this plan') from None


class CommandPath(NamedTuple):
    """A tree path a command token may name; ``path_like`` when the token reads as a path."""

    path: str
    path_like: bool


def command_path_candidates(run: PlannedRun) -> frozenset[CommandPath]:
    """Every tree path *run*'s rendered command may name, root-relative and, when
    the command has a cwd, cwd-relative too. Absolute tokens name the host, the
    toolchain's domain; a token escaping the tree keeps its in-tree remainder."""
    if run.cmd is None:
        return frozenset()
    bases = ('',) if run.cmd.cwd_rel is None else ('', run.cmd.cwd_rel)
    return frozenset(
        CommandPath(path, _is_path_like(token))
        for token in _command_tokens(render(run.cmd))
        for base in bases
        if (path := _tree_path(base, token)) is not None
    )


def unit_input_paths(run: PlannedRun, inputs: TreeInputs) -> Inputs:
    """``('tree:<path>', oid)`` for *run*'s module, cwd, uv project, the paths its
    command names, and the workspace members those depend on, transitively."""
    keyed = {path: inputs.oid(path) for path in _anchor_paths(run)}
    for candidate in command_path_candidates(run):
        oid = inputs.oid(candidate.path)
        if oid is not None or candidate.path_like:
            keyed[candidate.path] = oid
    for member in inputs.workspace.closure(inputs.workspace.members_containing(keyed)):
        keyed[member] = inputs.oid(member)
    return tuple(sorted((f'tree:{path}', oid or ABSENT) for path, oid in keyed.items()))


async def read_tree_inputs(repo: Path, tree: str, plan: VerifyPlan) -> TreeInputs:
    """Read every path *plan*'s module keys need from tree oid *tree*: its root
    files, its workspace manifests, then one batch of object ids."""
    root_blobs, workspace = await asyncio.gather(
        tree_objects.root_blobs(repo, tree), read_workspace(repo, tree),
    )
    paths = set(workspace.members)
    for run in plan.runs:
        if _module_unit_id(run) is not None:
            paths |= _anchor_paths(run)
            paths |= {candidate.path for candidate in command_path_candidates(run)}
    paths.discard(WHOLE_TREE)
    object_ids = await tree_objects.object_ids(repo, tree, sorted(paths))
    return TreeInputs(tree, root_blobs, MappingProxyType(object_ids), workspace)


def tree_unit_key(tree_oid: str, plan_hash: str, toolchain: ToolchainFingerprint) -> UnitKey:
    """The whole-tree unit: this tree, verified by this plan, under this toolchain."""
    return UnitKey.of(
        _TREE_UNIT_ID, UnitTier.TREE, plan_hash,
        (('tree', tree_oid), ('spec', plan_hash), *toolchain.as_inputs()),
    )


def module_unit_keys(inputs: TreeInputs, plan: VerifyPlan, toolchain: ToolchainFingerprint) -> tuple[UnitKey, ...]:
    """One key per run of *plan* that is a registered module's leg with a command."""
    root = tuple((f'root:{name}', oid) for name, oid in inputs.root_blobs)
    units: dict[str, UnitKey] = {}
    for run in plan.runs:
        unit_id = _module_unit_id(run)
        if unit_id is None:
            continue
        if unit_id in units:
            raise ValueError(f'plan has two runs for unit {unit_id!r}')
        spec_hash = run_spec_hash(run.to_dict())
        units[unit_id] = UnitKey.of(
            unit_id, UnitTier.MODULE, spec_hash,
            (*root, *unit_input_paths(run, inputs), ('spec', spec_hash), *toolchain.as_inputs()),
        )
    return tuple(units.values())


@dataclass(frozen=True)
class RecordedInputs:
    """What one test read when it last ran green: its file and conftest chain as
    ``(path, oid)``, and every other path it read."""

    test_file: tuple[str, str]
    conftest_chain: tuple[tuple[str, str], ...]
    reads: tuple[str, ...]


def test_unit_key(
    nodeid: str,
    recorded: RecordedInputs,
    current_oids: Mapping[str, str | None],
    run: PlannedRun,
    toolchain: ToolchainFingerprint,
) -> UnitKey | None:
    """The per-test key in a tree whose oids are *current_oids*; None when the
    test file or a conftest moved since the recording (A4: re-record, never key).

    Shape only: the recordings come from the per-test tier's recorder.
    """
    pinned = (recorded.test_file, *recorded.conftest_chain)
    if any(current_oids[path] != oid for path, oid in pinned):
        return None
    spec_hash = run_spec_hash(run.to_dict())
    return UnitKey.of(
        f'test:{nodeid}', UnitTier.TEST, spec_hash,
        (
            *((f'tree:{path}', oid) for path, oid in pinned),
            *((f'read:{path}', current_oids[path] or ABSENT) for path in dict.fromkeys(recorded.reads)),
            ('spec', spec_hash),
            *toolchain.as_inputs(),
        ),
    )


def _unit_id_fits(tier: UnitTier, unit_id: str) -> bool:
    if tier is UnitTier.TREE:
        return unit_id == _TREE_UNIT_ID
    return unit_id.startswith(_UNIT_ID_PREFIX[tier])


def _module_unit_id(run: PlannedRun) -> str | None:
    leg = verify_plan.planned_run_leg(run)
    module = verify_plan.module_of(run)
    if run.cmd is None or leg is None or module is None:
        return None
    return module_leg_name(module, leg)


def _anchor_paths(run: PlannedRun) -> set[str]:
    anchors = [run.module_prefix]
    if run.cmd is not None:
        anchors += [run.cmd.cwd_rel or '', run.cmd.uv_project or '']
    return {path for anchor in anchors if anchor and (path := _tree_path('', anchor)) is not None}


def _command_tokens(command: str) -> Iterator[str]:
    """The words of *command* that may name a path: shell operators and bare
    flags dropped, ``name=value`` reduced to its value, and a quoted shell
    payload (a ``bash -c`` argument) split again."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        if any(char.isspace() for char in token):
            yield from _command_tokens(token)
        if not token or set(token) <= _SHELL_PUNCTUATION:
            continue
        _name, assigns, value = token.partition('=')
        if assigns:
            if value:
                yield value
        elif not token.startswith('-'):
            yield token


def _tree_path(base: str, token: str) -> str | None:
    if token.startswith(('/', '~')) or '$' in token:
        return None
    parts = posixpath.normpath(posixpath.join(base, token)).split('/')
    while parts and parts[0] == '..':
        parts.pop(0)
    return '/'.join(parts) or None


def _is_path_like(token: str) -> bool:
    return '/' in token or '.' in token

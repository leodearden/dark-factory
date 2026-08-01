"""The ``warm_lane_bash`` bucket runs OFF the verify hot path and ON the offline lane.

Task 3349 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` §11 q4, re-decided).

§11 q4 originally put reify's ported warm-lane bash tests in the default
``orchestrator`` suite.  Task 3349 re-decided it and took the escape the PRD
itself documented: the bucket is deselected from the default suite in
``orchestrator/pyproject.toml``'s ``addopts`` and re-selected post-merge by the
``warm-lane-bash`` ``LaneCommand`` in ``dark-factory-orchestrator.yaml``.  That
is a MOVE of coverage, not a loss of it — but only for as long as BOTH halves
are in place.

**This module is that guarantee.**  The failure mode the escape introduces is
landing the deselection and forgetting (or later breaking) the lane entry:
silent ZERO coverage, which is precisely the "appearance of coverage without the
fact of it" that q4 originally refused, and which leaf **κ** — which deletes
reify's originals on the strength of this suite being green — cannot tolerate.

Three properties are pinned, and the third is why the module exists:

* the bucket is deselected from the verify hot path (``pyproject.toml``);
* the offline lane carries it (``dark-factory-orchestrator.yaml``);
* those two states are **equal**.  Asserting the biconditional, rather than each
  side separately, is what makes "neither lane carries it" unreachable.  The two
  config edits are COUPLED and must never move independently.

A fourth test is the non-vacuity guard: it takes the lane entry's ``command``
and ``cwd`` VERBATIM from the loaded config and actually executes a
``--collect-only``.  A ``LaneCommand`` whose command or cwd does not resolve
inside the reset ``_offline-deep`` worktree would file green forever while
running nothing, and no source-level assertion could see that.

This module is deliberately **NOT** marked ``warm_lane_bash``.  It must keep
running on the verify hot path after the deselection lands, or it would deselect
itself along with the thing it guards.

See ``orchestrator/tests/warm-lane/README.md`` for the full re-decision record.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_warm_lane_bash_suite import PORTED_TESTS

from orchestrator.config import LaneCommand, OrchestratorConfig

# Resolved from THIS FILE, never from the process CWD: the merge-verify harness
# runs pytest from the ``orchestrator/`` cwd (orchestrator/orchestrator.yaml:5)
# while a plain ``pytest orchestrator/tests`` runs from the repo root, and this
# contract must hold identically under both.  Same idiom as
# test_warm_lane_bash_suite.py.
ORCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_PYPROJECT = ORCH_DIR / 'pyproject.toml'
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

#: The marker whose placement this module governs.
BUCKET = 'warm_lane_bash'

#: Items in ``test_warm_lane_bash_suite.py`` that are NOT one-per-ported-script:
#: ``test_manifest_matches_disk``, ``test_required_host_tools_are_present``,
#: ``test_script_dir_resolution_ignores_the_env_override`` and
#: ``test_every_invocable_script_has_ported_coverage``.  Held separately so the
#: expected collection count tracks ``PORTED_TESTS`` automatically when PRD
#: leaves δ/δ2/ε add ports, rather than going stale as a hard-coded total.
STRUCTURAL_GUARD_ITEMS = 4

#: Environment keys stripped from the collection probe's subprocess.
#: ``GIT_DIR``/``GIT_WORK_TREE`` follow ``test_warm_lane_bash_suite``'s
#: ``_sanitized_env`` discipline (an inherited git pointer would aim the child at
#: the wrong tree).  ``PYTEST_ADDOPTS`` is stripped because an outer value
#: carrying its own ``-m`` would override the lane command's marker selection in
#: the child and make this whole pin vacuous.
_HOSTILE_ENV_KEYS = frozenset({
    'GIT_DIR',
    'GIT_WORK_TREE',
    'GIT_INDEX_FILE',
    'PYTEST_ADDOPTS',
    'PYTEST_CURRENT_TEST',
})


def _sanitized_env() -> dict[str, str]:
    """``os.environ`` minus the keys that would contaminate the probe."""
    return {k: v for k, v in os.environ.items() if k not in _HOSTILE_ENV_KEYS}


def _marker_expression(argv: Sequence[str]) -> str | None:
    """The effective ``-m`` expression in ``argv``, or ``None`` if there is none.

    Last-wins, matching pytest's own precedence (which is exactly how a lane
    command's CLI ``-m`` overrides the ``addopts`` ``-m`` it inherits).  Handles
    both the separated ``-m EXPR`` and the joined ``-mEXPR`` spellings.
    """
    expr: str | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == '-m' and index + 1 < len(argv):
            expr = argv[index + 1]
            index += 2
            continue
        if token.startswith('-m') and not token.startswith('--') and len(token) > 2:
            expr = token[2:]
        index += 1
    return expr


def _deselects_bucket(expr: str | None) -> bool:
    """Does ``expr`` exclude the bucket (``not warm_lane_bash``)?"""
    if expr is None:
        return False
    return f'not {BUCKET}' in ' '.join(expr.split())


def _selects_bucket(expr: str | None) -> bool:
    """Does ``expr`` positively select the bucket?"""
    if expr is None:
        return False
    return BUCKET in expr and not _deselects_bucket(expr)


def _orchestrator_addopts() -> list[str]:
    """``[tool.pytest.ini_options] addopts`` from the REAL orchestrator pyproject."""
    with ORCH_PYPROJECT.open('rb') as handle:
        data = tomllib.load(handle)
    addopts = data['tool']['pytest']['ini_options'].get('addopts', '')
    if isinstance(addopts, list):
        return [str(token) for token in addopts]
    return shlex.split(addopts)


def _load_committed_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the COMMITTED dark-factory-orchestrator.yaml through the real model.

    Deliberately not a hand-parse of the YAML: config layering means the LOADED
    ATTRIBUTE — not the file text — is what the daemon acts on, so a lane entry
    that fails validation or is shadowed by a later layer must FAIL this test
    rather than pass a naive text match.  Same rationale as
    ``test_config_merge_deep.TestDefaultsYamlMergeDeepBlock``.
    """
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig()


def _lane_commands_carrying_bucket(config: OrchestratorConfig) -> list[LaneCommand]:
    """Enabled ``offline_lane_commands`` entries that positively select the bucket."""
    return [
        entry for entry in config.git.offline_lane_commands
        if entry.enabled and _selects_bucket(_marker_expression(shlex.split(entry.command)))
    ]


def _deselected_from_hot_path() -> bool:
    return _deselects_bucket(_marker_expression(_orchestrator_addopts()))


def test_the_bucket_is_deselected_from_the_verify_hot_path() -> None:
    """``addopts`` must exclude the bucket from the default orchestrator suite.

    Deselecting here rather than in ``orchestrator/orchestrator.yaml``'s
    ``test_command`` is the shipped ``integration`` precedent
    (``shared/pyproject.toml``, ``fused-memory/pyproject.toml``): the
    ``test_command`` is a bare directory-arg pytest with no ``-m`` of its own, so
    one edit point covers the merge-verify invocation AND a bare local ``pytest``
    with no risk of the two spellings drifting apart.
    """
    addopts = _orchestrator_addopts()
    expression = _marker_expression(addopts)
    assert _deselects_bucket(expression), (
        f'{ORCH_PYPROJECT} [tool.pytest.ini_options] addopts does not deselect '
        f"the {BUCKET!r} bucket: marker expression is {expression!r} in "
        f'{addopts!r}. PRD §11 q4 was re-decided (task 3349) to take the '
        f"documented escape — append -m 'not {BUCKET}' here."
    )


def test_the_offline_lane_carries_the_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """``git.offline_lane_commands`` must re-select the bucket post-merge."""
    config = _load_committed_config(monkeypatch)
    carriers = _lane_commands_carrying_bucket(config)
    assert carriers, (
        f'{DF_CONFIG_PATH} git.offline_lane_commands has no enabled entry '
        f'selecting -m {BUCKET}; it holds '
        f'{[(e.name, e.command, e.enabled) for e in config.git.offline_lane_commands]!r}. '
        f'Without one, deselecting the bucket from addopts drops its coverage '
        f'entirely instead of relocating it.'
    )


def test_bucket_coverage_is_relocated_never_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two config edits are COUPLED — deselected iff carried by the lane.

    This is the standing guard and the reason this module exists.  Asserting the
    two states are EQUAL, rather than asserting each separately, is what makes
    "neither lane carries it" impossible to reach: deselecting in ``addopts``
    while forgetting the lane entry is silent ZERO coverage, exactly the
    "appearance of coverage without the fact of it" that PRD §11 q4 refused, and
    leaf κ deletes reify's originals on the strength of this suite being green.
    """
    deselected = _deselected_from_hot_path()
    carried = bool(_lane_commands_carrying_bucket(_load_committed_config(monkeypatch)))
    assert deselected == carried, (
        f'The {BUCKET!r} bucket is deselected from the verify hot path='
        f'{deselected} but carried by the offline lane={carried}. These two '
        f'edits are coupled and must never move independently:\n'
        f'  (1) {ORCH_PYPROJECT}\n'
        f"      [tool.pytest.ini_options] addopts -> -m 'not {BUCKET}'\n"
        f'  (2) {DF_CONFIG_PATH}\n'
        f'      git.offline_lane_commands -> an enabled entry running '
        f'-m {BUCKET}\n'
        + (
            'Deselected but NOT carried: the bucket now runs NOWHERE. Restore '
            'the lane entry or revert the deselection.'
            if deselected else
            'Carried but NOT deselected: the bucket runs twice, paying its '
            'full cost on the verify hot path for nothing.'
        )
    )


@pytest.mark.timeout(120)
def test_the_configured_lane_command_actually_collects_the_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the configured command/cwd — a source assertion cannot see this.

    ``command`` and ``cwd`` are taken VERBATIM from the loaded config, never
    from a hard-coded copy, and actually run.  A ``LaneCommand`` whose command
    does not resolve (wrong venv, wrong cwd) inside the reset ``_offline-deep``
    worktree would exit non-zero on every advance and file its red as a fix task
    forever, or — worse for a collect-only shape — appear configured while
    running nothing.  This worktree is a structurally identical stand-in for the
    lane's worktree, so resolving here is the meaningful pin.

    ``REPO_ROOT`` (not ``config.project_root``) anchors the cwd because the
    autouse ``_isolate_orch_config`` fixture pins ``project_root`` to a
    ``tmp_path``; the lane resolves ``cwd`` against ITS worktree root, which is
    what ``REPO_ROOT`` stands in for here.
    """
    config = _load_committed_config(monkeypatch)
    carriers = _lane_commands_carrying_bucket(config)
    assert carriers, (
        f'no enabled {BUCKET} entry in {DF_CONFIG_PATH} git.offline_lane_commands '
        f'to probe — see test_the_offline_lane_carries_the_bucket'
    )
    entry = carriers[0]

    workdir = (REPO_ROOT / entry.cwd).resolve()
    assert workdir.is_dir(), (
        f'lane entry {entry.name!r} has cwd={entry.cwd!r}, which does not '
        f'resolve to a directory ({workdir}) — the sub-run would fail to start'
    )

    # ``-p no:randomly``-style minimalism: ``-n0`` only disables xdist for the
    # probe (collection is what is being pinned, not distribution), keeping this
    # deterministic and fast under load.  Everything else — crucially the ``-m``
    # selection — comes from the configured command.
    argv = [*shlex.split(entry.command), '-n0', '--collect-only', '-q',
            'tests/test_warm_lane_bash_suite.py']
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=90,
        cwd=workdir,
        env=_sanitized_env(),
        check=False,
    )
    context = (
        f'command={entry.command!r} cwd={entry.cwd!r} argv={argv!r}\n'
        f'--- stdout ---\n{proc.stdout[-3000:]}\n'
        f'--- stderr ---\n{proc.stderr[-3000:]}'
    )
    assert proc.returncode == 0, (
        f'lane entry {entry.name!r} did not resolve — rc={proc.returncode}. '
        f'It would run nothing on every offline-lane advance.\n{context}'
    )

    collected = [
        line for line in proc.stdout.splitlines()
        if 'test_warm_lane_bash_suite.py::' in line
    ]
    expected = len(PORTED_TESTS) + STRUCTURAL_GUARD_ITEMS
    assert len(collected) == expected, (
        f'lane entry {entry.name!r} collected {len(collected)} items from the '
        f'warm-lane driver, expected {expected} '
        f'({len(PORTED_TESTS)} ported scripts + {STRUCTURAL_GUARD_ITEMS} '
        f'structural guards). A zero here is the silent-no-coverage failure '
        f'this module exists to catch; a mismatch means the driver grew or lost '
        f'items and STRUCTURAL_GUARD_ITEMS needs a deliberate update.\n{context}'
    )

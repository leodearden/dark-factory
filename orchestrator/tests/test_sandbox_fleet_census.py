"""Census gate: the sandbox block of every orchestrator config in this repo.

It pins CONFIG STATE: what the orchestrator reads at dispatch time to decide
whether to confine a sandboxed role. It is the in-repo half of the γ7 fleet
census in ``plans/os-sandbox-worktree-containment-prd.md``.

The sibling-repo half is deliberately absent. Its absolute host paths are not
portable, and each sibling's flip is held by that project's own registry (D6),
so it is re-measured by hand; ``docs/sandbox-fleet-status.md`` carries the
dated census and the recipe.

The eval runner's exception (D12) is pinned by
``orchestrator/tests/test_eval_profile.py::test_build_eval_orch_config_applies_profile_and_inherits_base``
and is not re-asserted here (INV-5).

The forward check compares the raw committed mapping, never a ``SandboxConfig``
built from it, because the model defaults ``enabled`` to True and ignores
unknown keys while the loader merges the block over ``defaults.yaml``'s
``enabled: false``, so a misspelled or omitted key reads as enabled to the
model and resolves to disabled at dispatch.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

from orchestrator.config import SandboxConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

FACTORY_TARGETS: tuple[str, ...] = ('dark-factory-orchestrator.yaml', 'dashboard/orchestrator.yaml')
LANDLOCK_PINNED = {'enabled': True, 'backend': 'landlock'}
SHIPPED_DEFAULTS = 'orchestrator/src/orchestrator/defaults.yaml'
# The canonical `--config` name, plus the module-config name that
# orchestrator/src/orchestrator/config.py::_discover_module_configs walks for.
CONFIG_BASENAMES = frozenset({'dark-factory-orchestrator.yaml', 'orchestrator.yaml'})


def _load_config(rel: str) -> object:
    """Parse the committed YAML at *rel*, failing loudly rather than skipping.

    The error stance follows
    ``orchestrator/tests/_verify_config_corpus.py::load_config_scalar``.
    """
    try:
        raw = (REPO_ROOT / rel).read_text(encoding='utf-8')
    except OSError as exc:
        raise AssertionError(
            f'cannot read {rel} while taking the sandbox census: '
            f'{exc.__class__.__name__}: {exc}.\n'
            f'FIX: if the config was deleted or renamed, update FACTORY_TARGETS / '
            f'SHIPPED_DEFAULTS in this file and the census in docs/sandbox-fleet-status.md.'
        ) from exc
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise AssertionError(
            f'{rel} is not parseable YAML: {exc}.\n'
            f'FIX: repair the config. An unparseable config is a failure, not an absent '
            f'sandbox block, so the census must not skip past it.'
        ) from exc


def _sandbox_block(rel: str) -> object:
    """The top-level ``sandbox`` value of the config at *rel*; None when it has none."""
    config = _load_config(rel)
    return config.get('sandbox') if isinstance(config, dict) else None


def _declared_sandbox_configs() -> set[str]:
    """The orchestrator configs, in git's view of this checkout, that declare ``sandbox``.

    Enumeration mirrors ``tests/scripts/test_nonmember_ruff_config.py::_git_ls_files``:
    ``--others`` censuses a new config before it is committed, ``--exclude-standard``
    keeps ignored trees (``.worktrees/``, ``.pytest-tmp/``, ``.task/``, ``.venv/``) out,
    and ``GIT_*`` is scrubbed so an exported ``GIT_INDEX_FILE`` (the pre-commit hook
    sets one) cannot point ls-files at another index. A bare ``sandbox:`` whose
    children are commented out (value None) still counts: presence is what makes it
    a census row.
    """
    command = [
        'git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '*.yaml',
    ]
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    try:
        proc = subprocess.run(
            command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise AssertionError(
            f'cannot run `{" ".join(command)}`: {exc!r}.\n'
            f'FIX: make git available. The completeness census takes its candidates from '
            f'git and fails rather than skipping without it.'
        ) from exc
    assert proc.returncode == 0, (
        f'`{" ".join(command)}` exited {proc.returncode} in {REPO_ROOT}: '
        f'{proc.stderr.strip()}\n'
        f'FIX: run the suite from a git checkout of this repo. The completeness census '
        f'fails rather than skipping when it cannot enumerate the configs.'
    )
    declared: set[str] = set()
    for rel in proc.stdout.split('\0'):
        if PurePosixPath(rel).name not in CONFIG_BASENAMES:
            continue
        config = _load_config(rel)
        if isinstance(config, dict) and 'sandbox' in config:
            declared.add(rel)
    return declared


@pytest.mark.parametrize('rel', FACTORY_TARGETS)
def test_factory_target_pins_landlock(rel: str) -> None:
    block = _sandbox_block(rel)
    assert block == LANDLOCK_PINNED, (
        f'{rel} is a factory target, but its sandbox block is {block!r}, not '
        f'{LANDLOCK_PINNED!r}.\n'
        f'FIX: a factory target must declare BOTH keys explicitly under `sandbox:`. An '
        f'omitted or misspelled `enabled` is merged over defaults.yaml\'s `enabled: false` '
        f'and leaves the project unconfined; why the backend is pinned rather than `auto` '
        f'is in docs/sandbox-fleet-status.md.'
    )


@pytest.mark.parametrize('rel', FACTORY_TARGETS)
def test_factory_target_keys_are_all_consumed_by_the_model(rel: str) -> None:
    block = _sandbox_block(rel)
    assert isinstance(block, dict), f'{rel} declares no sandbox mapping; it holds {block!r}.'
    consumed = SandboxConfig.model_validate(block).model_fields_set
    assert consumed == set(block), (
        f'SandboxConfig does not consume every key {rel} declares under `sandbox:`: '
        f'declared {sorted(block)}, consumed {sorted(consumed)}. The model sets no `extra`, '
        f'so an unconsumed key is silently ignored.\n'
        f'FIX: if the config misspells the key, correct it; if a SandboxConfig field was '
        f'renamed, rename the key in every factory-target config across the fleet.'
    )


def test_every_config_declaring_a_sandbox_is_a_census_row() -> None:
    declared = _declared_sandbox_configs()
    unclassified = declared - set(FACTORY_TARGETS)
    assert not unclassified, (
        f'These configs declare a top-level `sandbox` block but are not census rows: '
        f'{sorted(unclassified)}.\n'
        f'FIX: each is a new census row, so classify it. A factory target goes into '
        f'FACTORY_TARGETS, pinned to landlock, and gets a row in docs/sandbox-fleet-status.md. '
        f'A config that no unit runs as a project config should not declare a sandbox block '
        f'at all. Do not loosen this check.'
    )
    missing = set(FACTORY_TARGETS) - declared
    assert not missing, (
        f'These factory targets no longer declare a top-level `sandbox` block: '
        f'{sorted(missing)}.\n'
        f'FIX: restore the pinned block, since defaults.yaml ships `enabled: false` and the '
        f'project would run its sandboxed roles unconfined. If it is no longer a factory '
        f'target, drop its row here and in docs/sandbox-fleet-status.md.'
    )


def test_shipped_default_leaves_the_sandbox_off() -> None:
    block = _sandbox_block(SHIPPED_DEFAULTS)
    assert isinstance(block, dict) and block.get('enabled') is False, (
        f'{SHIPPED_DEFAULTS} must declare `sandbox.enabled: false` explicitly; its sandbox '
        f'block is {block!r}.\n'
        f'FIX: restore the explicit key. Enablement is explicit per-project config (PRD D7), '
        f'and SandboxConfig\'s model default is `enabled=True`, so deleting the key from '
        f'defaults.yaml silently turns sandboxing ON for every adopter.'
    )

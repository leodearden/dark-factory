"""Behavioral regression guard: pyright is wired into the merge gate for all subprojects.

The merge gate's scoped verification (orchestrator/src/orchestrator/verify_plan.py:
derive_verify_plan, executed via verify.py:_executed_module_configs_from_plan — task κ,
verify-scope-inversion-prd.md) builds the pyright command from
ModuleConfig.type_check_command, loaded from each subproject's orchestrator.yaml via the
_OVERRIDABLE_FIELDS loader (orchestrator/src/orchestrator/config.py:_discover_module_configs).

This test drives the REAL loader (_discover_module_configs) and the REAL gate mechanism
(derive_verify_plan + _executed_module_configs_from_plan) to assert the resolved/scoped
type_check_command actually routes to pyright for each subproject.  It catches:

  - A subproject dropping type_check_command from its orchestrator.yaml.
  - type_check_command being dropped from _OVERRIDABLE_FIELDS (config.py:403-410),
    which leaves the YAML text intact but stops the field from being loaded into
    ModuleConfig, silently disabling pyright for that subproject.
  - the plan-driven execution bridge stopping its emission of the pyright command
    (verify.py:_executed_module_configs_from_plan, verify_plan.py:_derive_module_runs).

The previous escalation/tests/test_pyright_merge_gate.py and
shared/tests/test_pyright_merge_gate.py read orchestrator.yaml as TEXT and regex-matched
its contents via _extract_type_check_command; they could NOT detect the _OVERRIDABLE_FIELDS
or plan/execution-bridge regressions above.  This test supersedes them.

See task 1477 for full context (scope_module_config, task 1477's original gate helper, was
deleted by task κ once derive_verify_plan became the sole scope decision tree).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, _discover_module_configs

# Repo root (dark-factory/) — two levels above orchestrator/tests/.
_REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    'prefix',
    [
        'escalation',   # required: fixed by task 1477
        'shared',       # required: fixed by task 1477
        'orchestrator', # positive control: already wired before task 1477
        'fused-memory', # positive control: already wired before task 1477
    ],
)
def test_pyright_wired_via_real_loader_and_gate(prefix: str) -> None:
    """pyright must be wired into the merge gate for each subproject.

    Two regression sites are exercised:

    (1) LOADER: _discover_module_configs(repo_root) — the exact function
        load_config calls at config.py:1126 to populate the merge gate's
        _module_configs — must yield a ModuleConfig for *prefix* whose
        type_check_command is set, invokes pyright, and covers tests/.
        A regression that drops type_check_command from _OVERRIDABLE_FIELDS
        (config.py:403-410) makes this assertion fail even when the YAML key
        is still present, because the loader would skip the field.

    (2) GATE: derive_verify_plan([changed_test_file], [mc], ...) executed via
        _executed_module_configs_from_plan([mc], plan) — the exact plan +
        execution-bridge run_scoped_verification calls to build the scoped
        pyright command — must yield a ModuleConfig whose type_check_command
        is non-None, contains 'pyright', and references the changed file.
        A regression in classify_file, derive_verify_plan, or
        _executed_module_configs_from_plan would make this assertion fail
        while (1) remains green, catching a class of regression the
        YAML-text-pin approach cannot detect.

    If this test fails: pyright will SILENTLY stop running on changed test
    files in {prefix}/ during the merge gate, and type errors (like those
    fixed in task 1476) will slip through undetected.  See task 1477.
    """
    # -------------------------------------------------------------------
    # (1) LOADER assertion
    # -------------------------------------------------------------------
    configs: dict[str, ModuleConfig] = _discover_module_configs(_REPO_ROOT)

    assert prefix in configs, (
        f"[task 1477] {prefix} has no ModuleConfig after _discover_module_configs(repo_root). "
        f"Either {prefix}/orchestrator.yaml is missing, or all its fields are non-overridable. "
        "Without a ModuleConfig the merge gate falls back to the global config and "
        "pyright may not run on this subproject's changed files."
    )

    mc = configs[prefix]
    cmd = mc.type_check_command

    assert cmd is not None, (
        f"[task 1477] configs['{prefix}'].type_check_command is None after loading via "
        "_discover_module_configs. This means either (a) {prefix}/orchestrator.yaml is "
        "missing its type_check_command key, or (b) 'type_check_command' was dropped from "
        "_OVERRIDABLE_FIELDS in config.py. Either way, pyright will SILENTLY stop running "
        "on {prefix}'s changed files during the merge gate. Add the key or restore the field."
    )

    assert 'pyright' in cmd, (
        f"[task 1477] configs['{prefix}'].type_check_command = {cmd!r} does not invoke pyright. "
        "The merge gate uses type_check_command specifically for pyright (not ruff or pytest). "
        "Without 'pyright' in the command, type errors in {prefix} test files are not caught. "
        "See task 1477."
    )

    assert 'tests' in cmd, (
        f"[task 1477] configs['{prefix}'].type_check_command = {cmd!r} does not cover tests/. "
        "scope_module_config narrows the command to changed files, but the files must be in "
        "the command's covered paths. Without 'tests' in the type_check_command, changed test "
        "files in {prefix}/tests/ are excluded from pyright during the merge gate. "
        "See task 1477."
    )

    # -------------------------------------------------------------------
    # (2) GATE assertion
    # -------------------------------------------------------------------
    # No real worktree/content is available for this synthetic probe path —
    # a stub reader returning None is equivalent to the pre-task-κ gate
    # helper's own worktree=None default: the probe file is COLLECTABLE_TEST
    # by path pattern alone (classify_file's precedence checks that before
    # STRUCTURAL, which is the only branch content ever affects), so no
    # structural-content read is needed to resolve this test's scope_kind.
    probe_file = f'{prefix}/tests/test_pyright_gate_probe.py'
    plan = verify_plan.derive_verify_plan([probe_file], [mc], None, lambda _path: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)

    assert len(executed) == 1, (
        f"[task 1477] derive_verify_plan(['{probe_file}'], [configs['{prefix}']], ...) executed "
        f"via _executed_module_configs_from_plan dropped '{prefix}' entirely. This means no .py "
        f"files under '{prefix}/' were matched, which should not happen since the probe file is "
        "directly under the prefix. Check classify_file's prefix-matching logic in "
        "verify_plan.py. See task 1477."
    )

    scoped_cmd = executed[0].type_check_command

    assert scoped_cmd is not None, (
        f"[task 1477] the plan-driven executed ModuleConfig for '{prefix}' has "
        "type_check_command=None. The plan/execution bridge (verify_plan.derive_verify_plan + "
        "verify._executed_module_configs_from_plan) did not build a pyright command for the "
        f"scoped changed file. This means pyright would SILENTLY be skipped for {prefix}'s "
        "changed test files in the merge gate. See task 1477."
    )

    assert 'pyright' in scoped_cmd, (
        f"[task 1477] scoped type_check_command = {scoped_cmd!r} does not invoke pyright. "
        "See task 1477."
    )

    assert probe_file in scoped_cmd or probe_file.split('/', 1)[1] in scoped_cmd, (
        f"[task 1477] scoped type_check_command = {scoped_cmd!r} does not reference the "
        f"changed file '{probe_file}'. The plan/execution bridge must narrow the command to "
        "the specific changed files; if the probe file path is absent, the gate is not "
        "actually scoping pyright to the changed test file. See task 1477."
    )

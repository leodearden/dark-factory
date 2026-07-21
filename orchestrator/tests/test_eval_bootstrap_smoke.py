"""Assertion-logic tests for scripts/eval_bootstrap_smoke.sh (task 2847 SMOKE).

Drives the go/no-go smoke script in its DRY-RUN mode (``SMOKE_SKIP_EVAL=1``)
over SYNTHETIC result JSONs the test writes, so the BUG-1 / BUG-2 assertion
logic is deterministically covered without a live eval-ofat LLM run (which
stays the operator go/no-go gate, not a CI test).

Hermetic: the whole module is skipped unless ``bash`` AND the repo's own
``.venv/bin/python`` 3.13 (the script's Phase-0 fail-fast tie) are both
available — the test exercises the assertion logic only, never a paid run.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'eval_bootstrap_smoke.sh'
PYBIN = REPO_ROOT / '.venv' / 'bin' / 'python'


def _repo_python_is_313() -> bool:
    if not PYBIN.exists():
        return False
    try:
        out = subprocess.run(
            [str(PYBIN), '--version'], capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    return (out.stdout + out.stderr).startswith('Python 3.13.')


pytestmark = pytest.mark.skipif(
    shutil.which('bash') is None or not _repo_python_is_313(),
    reason='needs bash + repo .venv python 3.13 (the script Phase-0 fail-fast tie)',
)


def _write_result(results_dir: Path, name: str, payload: dict) -> None:
    (results_dir / name).write_text(json.dumps(payload))


def _venv_python_stub(
    worktree: Path,
    version_line: str,
    subdir: str | None = None,
    has_aiosqlite: bool = True,
) -> None:
    """Create ``<worktree>[/<subdir>]/.venv/bin/python`` as an executable stub
    standing in for a real worktree interpreter.

    For a ``--version`` invocation (and any args NOT containing an
    ``import aiosqlite`` probe) the stub echoes *version_line* and exits 0. When
    *has_aiosqlite* is False, an invocation whose args contain
    ``import aiosqlite`` (the BUG-2 hardening probe the smoke gate runs) exits
    non-zero with a ModuleNotFoundError message — simulating the exact
    old-baseline venv that is Python 3.13 yet cannot import aiosqlite, so eval
    verify would die at pytest collection. Default True preserves the existing
    version-only tests (the probe exits 0).

    With *subdir* set, plants the venv one level down at
    ``<worktree>/<subdir>/.venv`` — exactly where uv creates it when a fixture's
    ``setup_commands`` do ``cd <subdir> && uv sync`` with
    ``UV_PROJECT_ENVIRONMENT`` scrubbed (e.g. df_task_12 →
    ``<wt>/orchestrator/.venv``), NOT the top-level ``<wt>/.venv``."""
    base = worktree if subdir is None else worktree / subdir
    bin_dir = base / '.venv' / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / 'python'
    if has_aiosqlite:
        aiosqlite_branch = 'exit 0'
    else:
        aiosqlite_branch = (
            "echo \"ModuleNotFoundError: No module named 'aiosqlite'\" >&2; exit 1"
        )
    stub.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$*" == *"import aiosqlite"* ]]; then\n'
        f'  {aiosqlite_branch}\n'
        'fi\n'
        f'echo "{version_line}"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_smoke(results_dir: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        'SMOKE_SKIP_EVAL': '1',
        'SMOKE_RESULTS_DIR': str(results_dir),
        'SMOKE_FIXTURE': 'df_task_12',
    }
    return subprocess.run(
        ['bash', str(SCRIPT)],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _arch_result(plan_steps: int = 8, outcome: str = 'done') -> dict:
    return {
        'task_id': 'df_task_12',
        'config_name': 'architect-x',
        'outcome': outcome,
        'metrics': {
            'role_under_test': 'architect',
            'plan_steps': plan_steps,
            'plan_quality': 0.8,
        },
        'worktree_path': '/tmp/architect-worktree-not-checked',
        'run_id': 'r1',
        'trial': 1,
    }


def _impl_result(
    worktree_path: str,
    outcome: str = 'done',
    tests_pass: bool | None = True,
    run_id: str = 'r2',
    config_name: str = 'opus-high',
) -> dict:
    return {
        'task_id': 'df_task_12',
        'config_name': config_name,
        'outcome': outcome,
        'metrics': {'composite_score': 1.0, 'tests_pass': tests_pass},
        'worktree_path': worktree_path,
        'run_id': run_id,
        'trial': 2,
    }


def test_good_results_pass(tmp_path):
    # (a) GOOD: architect plan_steps>0 & done, implementer worktree venv is 3.13.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    assert proc.returncode == 0, (
        f'expected PASS, got rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\n'
        f'STDERR:\n{proc.stderr}'
    )


def test_bug1_plan_steps_zero_fails(tmp_path):
    # (b) BUG 1: an architect result with plan_steps=0 → the plan-tools MCP was
    # not wired into the eval architect session. Must fail, naming BUG 1.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result(plan_steps=0))
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 1' in combined
    assert 'plan-tools' in combined


def test_bug2_wrong_python_fails(tmp_path):
    # (c) BUG 2: the implementer worktree venv is 3.12, not the pinned 3.13.
    # Must fail, naming BUG 2 and the expected 3.13.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.12.9')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 2' in combined
    assert '3.13' in combined


def test_no_architect_result_fails(tmp_path):
    # (d) No architect result at all → must NOT silently pass (a BUG-1-class
    # failure: the architect eval never produced a scorable result).
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5')
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 1' in combined


# ---------------------------------------------------------------------------
# task 2875 BUG 2 (venv path) — a fixture's setup_commands `cd orchestrator &&
# uv sync` with UV_PROJECT_ENVIRONMENT scrubbed (verify._target_subprocess_env)
# create the implementer venv at <wt>/orchestrator/.venv, NOT <wt>/.venv. The
# smoke gate must resolve the ACTUAL interpreter location (direct <wt>/.venv,
# else the one-level subproject <wt>/*/.venv) so this live subproject-venv layout
# — never exercised by the top-level-<wt>/.venv synthetic tests above — is
# version-checked instead of reported "missing".
# ---------------------------------------------------------------------------

def test_subproject_venv_313_passes(tmp_path):
    # (e) GOOD subproject layout: the venv exists ONLY at <wt>/orchestrator/.venv
    # and is 3.13 → smoke must resolve it and PASS. Currently RED: the script
    # checks only <wt>/.venv/bin/python, so this venv reads as "missing" and BUG 2
    # fails before ever reaching SMOKE PASS.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5', subdir='orchestrator')
    # Only the subproject venv exists — NO top-level <wt>/.venv.
    assert not (wt / '.venv').exists()
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    assert proc.returncode == 0, (
        f'expected PASS, got rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\n'
        f'STDERR:\n{proc.stderr}'
    )


def test_subproject_venv_wrong_python_fails(tmp_path):
    # (f) Same subproject layout, but the interpreter is 3.12 → must FAIL naming
    # BUG 2 and the expected 3.13 (proving the resolved subproject venv is
    # actually version-checked, not merely found).
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.12.9', subdir='orchestrator')
    assert not (wt / '.venv').exists()
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 2' in combined
    assert '3.13' in combined


# ---------------------------------------------------------------------------
# task 2875 amendment — a worktree can legitimately carry MORE THAN ONE
# subproject venv (e.g. a fixture whose setup builds both fused-memory/.venv and
# orchestrator/.venv). The BUG-2 gate must version-check EVERY subproject venv,
# not just the alphabetically-first, or a wrong interpreter in a non-first
# subproject would pass undetected (reviewer robustness finding).
# ---------------------------------------------------------------------------

def test_multiple_subproject_venvs_all_checked(tmp_path):
    # (g) Two subproject venvs: the alphabetically-FIRST ('aaa') is a good 3.13,
    # the later one ('zzz') is a bad 3.12. A single-pick gate would inspect only
    # 'aaa' and PASS; checking every venv must catch 'zzz' → FAIL naming BUG 2 and
    # the expected 3.13.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5', subdir='aaa')
    _venv_python_stub(wt, 'Python 3.12.9', subdir='zzz')
    # No top-level <wt>/.venv → the subproject-glob arm resolves BOTH venvs.
    assert not (wt / '.venv').exists()
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 2' in combined
    assert '3.13' in combined


def test_multiple_subproject_venvs_all_313_pass(tmp_path):
    # (h) Two subproject venvs, BOTH 3.13 → checking every one must still PASS
    # (multi-venv coverage does not spuriously fail a clean worktree).
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5', subdir='fused-memory')
    _venv_python_stub(wt, 'Python 3.13.5', subdir='orchestrator')
    assert not (wt / '.venv').exists()
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    assert proc.returncode == 0, (
        f'expected PASS, got rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\n'
        f'STDERR:\n{proc.stderr}'
    )


# ---------------------------------------------------------------------------
# task 2876 DEFECT 4 — hardening BUG 2. The current gate asserts ONLY the venv
# interpreter version, so a 3.13 venv with a broken verify (aiosqlite absent)
# still emits SMOKE PASS — the exact false-negative that let 2875 merge on a
# state its own acceptance would reject. Harden with a per-venv `import aiosqlite`
# probe (this test) plus a >=1-done/tests_pass implementer assertion (step-7/8).
# ---------------------------------------------------------------------------

def test_bug2_aiosqlite_missing_fails(tmp_path):
    # (i) BUG 2 hardening: the implementer worktree venv IS Python 3.13 but
    # aiosqlite is not importable → eval verify would die at pytest collection
    # (orchestrator.config → shared → import aiosqlite). The smoke must FAIL
    # naming BUG 2 and aiosqlite, not emit a hollow SMOKE PASS on version alone.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5', has_aiosqlite=False)
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(results, 'df_task_12__opus-high__r2.json', _impl_result(str(wt)))

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'BUG 2' in combined
    assert 'aiosqlite' in combined


# ---------------------------------------------------------------------------
# task 2876 DEFECT 4 (cont.) — the second BUG-2 hardening arm: after the
# implementer loop, require >=1 implementer cell with outcome=='done' AND
# metrics.tests_pass is True. This directly encodes the task acceptance so ANY
# future verify breakage (not just aiosqlite) is caught, instead of a hollow
# SMOKE PASS on a 3.13 venv whose verify never actually succeeded.
# ---------------------------------------------------------------------------

def test_implementer_no_tests_pass_fails(tmp_path):
    # (j) The ONLY implementer cell reached outcome='done' but its verify did
    # NOT pass (metrics.tests_pass=False) — a hollow pass. Venv is 3.13 + has
    # aiosqlite, so BUG-2 per-venv checks are clean; the smoke must still FAIL on
    # the "no verified implementer cell" acceptance.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(
        results, 'df_task_12__opus-high__r2.json',
        _impl_result(str(wt), tests_pass=False),
    )

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'SMOKE FAIL' in combined
    assert 'tests_pass' in combined


def test_implementer_cancelled_outcome_fails(tmp_path):
    # (k) Same acceptance from the outcome axis: the only implementer cell was
    # cancelled (never reached done) though tests_pass is nominally True → still
    # no verified cell → FAIL.
    results = tmp_path / 'results'
    results.mkdir()
    wt = tmp_path / 'impl-wt'
    _venv_python_stub(wt, 'Python 3.13.5')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(
        results, 'df_task_12__opus-high__r2.json',
        _impl_result(str(wt), outcome='cancelled'),
    )

    proc = _run_smoke(results)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f'expected FAIL:\n{combined}'
    assert 'SMOKE FAIL' in combined


def test_at_least_one_verified_implementer_passes(tmp_path):
    # (l) Positive companion: a MIXED set — one implementer cell failed verify
    # (tests_pass=False) but ANOTHER reached done+tests_pass=True. The ">=1
    # verified cell" acceptance is satisfied → smoke PASSES. Both worktree venvs
    # are 3.13 + aiosqlite so the BUG-2 per-venv checks stay green.
    results = tmp_path / 'results'
    results.mkdir()
    wt_bad = tmp_path / 'impl-wt-bad'
    wt_good = tmp_path / 'impl-wt-good'
    _venv_python_stub(wt_bad, 'Python 3.13.5')
    _venv_python_stub(wt_good, 'Python 3.13.5')
    _write_result(results, 'df_task_12__architect-x__r1.json', _arch_result())
    _write_result(
        results, 'df_task_12__opus-high__r2.json',
        _impl_result(str(wt_bad), tests_pass=False),
    )
    _write_result(
        results, 'df_task_12__sonnet-high__r3.json',
        _impl_result(str(wt_good), outcome='done', tests_pass=True,
                     run_id='r3', config_name='sonnet-high'),
    )

    proc = _run_smoke(results)
    assert proc.returncode == 0, (
        f'expected PASS, got rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\n'
        f'STDERR:\n{proc.stderr}'
    )

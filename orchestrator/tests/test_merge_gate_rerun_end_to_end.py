"""The merge gate's isolated re-run, end to end, against a REAL pytest.

The ONE file in this area whose tests let ``confirm_isolated_rerun_verdict`` reach a
real ``run_verification`` and a real pytest subprocess. Every sibling suite —
test_flake_discriminator.py, test_verify_merge_flake_suppression.py,
test_verify_preexisting_main_break.py — patches ``run_verification`` out, which
is precisely the layer this file exists to cover: the merge gate rewrites its
scoped command as a STRING twice, once in the discriminator
(``_with_pytest_timeout_str``) and once inside ``run_verification``
(``_with_junitxml_str``), and the second rewrite RE-PARSES the first one's
output. Nothing below that seam and nothing above it can observe a splice
between them.

That gap was not hypothetical. On main 8f088c1455 the merge_gate call site had
350 ``fails_in_isolation`` observations, 4 ``unconfirmable`` and ZERO
``passes_in_isolation`` over its whole recorded history (2026-08-30 ->
2026-09-17), because the re-run argv came out as::

    pytest -p no:xdist -o addopts= --timeout --junitxml <path> 300 <node>
    pytest: error: argument --timeout: expected one argument      (rc=4)

— pytest rejected the command without running a test, and the gate recorded
that as "the test really is red" (task 5580). The whole fleet's flake
suppression was off, and every unit test in the area was green.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.flake_ledger import FlakeVerdict
from orchestrator.verify import VerifyResult, confirm_isolated_rerun_verdict

#: The one probe test the isolated re-run is pointed at. It PASSES, so the
#: only way the gate can answer anything but ``passes_in_isolation`` is if the
#: command it built never ran it.
PROBE_REL = 'pkg/tests/test_probe.py'
PROBE_NODE_ID = f'{PROBE_REL}::test_ok'

#: A probe that requests xdist's ``worker_id`` fixture.
WORKER_ID_PROBE_REL = 'pkg/tests/test_probe_worker_id.py'
WORKER_ID_PROBE_NODE_ID = f'{WORKER_ID_PROBE_REL}::test_uses_worker_id'


def _make_worktree(tmp_path: Path) -> Path:
    """A minimal on-disk worktree carrying one PASSING test.

    The layout is what ``_group_node_ids_by_subproject`` probes: the node-id's
    file component is already worktree-root-relative and starts with the
    module prefix, so ``<worktree>/pkg/tests/test_probe.py`` existing is what
    maps it to the ``pkg`` group.
    """
    probe = tmp_path / PROBE_REL
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text('def test_ok():\n    assert True\n', encoding='utf-8')
    return tmp_path


def _make_config(worktree: Path) -> OrchestratorConfig:
    """A real OrchestratorConfig calibrated to the MERGE gate's own path.

    ``merge_verify_breadth='full'`` is load-bearing: it is the condition under
    which ``run_verification`` computes a junit path for ``role='merge'`` and
    performs the SECOND string rewrite. Without it there is one rewrite, no
    re-parse, and no splice to find.

    ``verify_env``'s PATH entry is equally load-bearing, and is a real config
    field rather than a patch of verify internals:
    ``verify._target_subprocess_env`` deliberately scrubs ``VIRTUAL_ENV`` and
    the venv's bin off PATH so a target resolves its OWN environment, then
    overlays ``verify_env`` LAST — so putting this interpreter's bin back is
    the supported way to make the rendered bare ``pytest`` resolve.
    """
    return OrchestratorConfig(
        project_root=worktree,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        merge_verify_breadth='full',
        verify_admission_enabled=False,
        verify_cold_preprovision_command='',
        verify_env={
            'PATH': f'{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}',
        },
    )


def _module_config() -> ModuleConfig:
    """One subproject whose test_command is a bare ``pytest``.

    Bare ``pytest`` parses to ``ToolKind.PYTEST`` and renders back as
    ``pytest``, which is what keeps the assertion about argv readable.
    ``python -m pytest`` and an absolute interpreter path both parse OPAQUE,
    where every structured mutator is a no-op (P1) — so neither would exercise
    the rewrite seam at all.
    """
    return ModuleConfig(
        prefix='pkg',
        test_command='pytest',
        lint_command=None,
        type_check_command=None,
    )


def _failing_result(*, node_id: str = PROBE_NODE_ID, kind: str = 'FAILED') -> VerifyResult:
    """The load-shed failure the merge gate is asking the discriminator about."""
    return VerifyResult(
        passed=False,
        test_output=f'{kind} {node_id}\n',
        lint_output='',
        type_output='',
        summary='fail',
        category='test_failure',
        cause_hint=f'{kind} {node_id}',
        failing_leg_categories=['test_failure'],
    )


def _write_worker_id_probe(worktree: Path) -> None:
    probe = worktree / WORKER_ID_PROBE_REL
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text('def test_uses_worker_id(worker_id):\n    assert worker_id\n', encoding='utf-8')


# Grouped so this file's real-subprocess probe lands on ONE xdist worker
# rather than competing with its siblings, and marked with the VERIFY CLI
# BUDGET itself rather than a number chosen to sit just above the run's
# expected wall clock — a timeout marker is a two-way override, so any value
# under that budget would TIGHTEN the merge-gating run instead of loosening
# this slow probe (the inversion test_timeout_marker_inversion_guard.py
# ratchets). Mirrors test_verify_cmd.py::TestSerialPytest's real-pytest probe.
@pytest.mark.xdist_group('merge_gate_rerun_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_merge_gate_isolated_rerun_actually_runs_the_test(tmp_path):
    """A passing test, re-run in isolation by the merge gate, must pass.

    The assertion an operator cares about is the VERDICT; the argv assertion
    below it is there so a regression names the splice instead of leaving a
    bare ``fails_in_isolation`` to be re-diagnosed by hand — which is what it
    cost the first time.

    MEASURED RED before task 5580: ``fails_in_isolation``, with the captured
    argv ``pytest -p no:xdist -o addopts= --timeout --junitxml
    <wt>/.df-verify-junit/report.pkg.xml 300 pkg/tests/test_probe.py::test_ok``.
    """
    worktree = _make_worktree(tmp_path)
    config = _make_config(worktree)
    real_run_cmd = verify_module._run_cmd
    seen: list[str] = []

    async def spy(cmd, *args, **kwargs):
        seen.append(cmd)
        return await real_run_cmd(cmd, *args, **kwargs)

    with patch.object(verify_module, '_run_cmd', spy):
        suppression = asyncio.run(confirm_isolated_rerun_verdict(
            worktree, config, [_module_config()], _failing_result(),
            call_site='merge_gate',
        ))

    pytest_argvs = [c for c in seen if 'pytest' in c]
    assert pytest_argvs, f'no pytest command was ever run; commands seen: {seen}'
    rendered = pytest_argvs[-1]
    tokens = shlex.split(rendered)
    assert suppression.verdict is FlakeVerdict.passes_in_isolation, (
        f'the merge gate could not confirm a test that passes when run alone; '
        f'verdict={suppression.verdict} reason={suppression.unconfirmable_reason!r}\n'
        f'the command it actually ran was: {rendered!r}'
    )
    assert tokens[tokens.index('--timeout') + 1].isdigit(), (
        f'--timeout was severed from its value by the junitxml rewrite: {rendered!r}'
    )


@pytest.mark.xdist_group('merge_gate_rerun_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_merge_gate_rerun_that_only_errors_at_setup_is_unconfirmable(tmp_path):
    """Under xdist this body runs and ``worker_id`` is ``'gw<N>'``; the re-run's
    ``-p no:xdist`` unregisters the fixture, so it ERRORs at setup with
    "fixture 'worker_id' not found".  MEASURED before task 3880:
    ``fails_in_isolation``.
    """
    _write_worker_id_probe(tmp_path)

    suppression = asyncio.run(confirm_isolated_rerun_verdict(
        tmp_path, _make_config(tmp_path), [_module_config()],
        _failing_result(node_id=WORKER_ID_PROBE_NODE_ID),
        call_site='merge_gate',
    ))

    rendered = f'verdict={suppression.verdict} reason={suppression.unconfirmable_reason!r}'
    assert suppression.verdict is FlakeVerdict.unconfirmable, rendered
    assert (suppression.unconfirmable_reason or '').startswith('rerun_setup_errors_only:'), rendered
    assert suppression.test_ids == (WORKER_ID_PROBE_NODE_ID,), suppression.test_ids


@pytest.mark.xdist_group('merge_gate_rerun_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_merge_gate_rerun_setup_error_reproduces_an_original_error(tmp_path, caplog):
    """The control: an original that ERRORed too is reproduced, although it
    names the test relative to the module and the re-run relative to the root.
    The fail-closed exception path also answers ``fails_in_isolation``, so it
    is ruled out by its WARNING."""
    _write_worker_id_probe(tmp_path)
    module_relative_id = WORKER_ID_PROBE_NODE_ID.removeprefix('pkg/')

    with caplog.at_level(logging.WARNING, logger=verify_module.logger.name):
        suppression = asyncio.run(confirm_isolated_rerun_verdict(
            tmp_path, _make_config(tmp_path), [_module_config()],
            _failing_result(node_id=module_relative_id, kind='ERROR'),
            call_site='merge_gate',
        ))

    rendered = f'verdict={suppression.verdict} reason={suppression.unconfirmable_reason!r}'
    assert suppression.verdict is FlakeVerdict.fails_in_isolation, rendered
    unexpected = [r.getMessage() for r in caplog.records if 'unexpected error' in r.getMessage()]
    assert not unexpected, unexpected


GROUPED_PROBE_REL = 'pkg/tests/test_grouped_probe.py'
_GROUPED_PROBE = '''\
import os

import pytest

pytestmark = pytest.mark.xdist_group('probe_group')


@pytest.mark.parametrize('p', ['1', 'a@b'])
def test_ok(p):
    assert os.environ.get('PROBE_FAIL') != '1'
'''


def _loadgroup_detect_output(worktree: Path) -> str:
    """Real ``-n 2 --dist loadgroup`` output of the grouped probe, failing because PROBE_FAIL is set."""
    env = {key: value for key, value in os.environ.items() if not key.startswith('PYTEST_')}
    env['PROBE_FAIL'] = '1'
    completed = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '-n', '2', '--dist', 'loadgroup', '-o', 'addopts=',
            '-p', 'no:cacheprovider', '--rootdir', str(worktree), '-q', GROUPED_PROBE_REL,
        ],
        cwd=worktree, env=env, capture_output=True, text=True, check=False,
    )
    return completed.stdout + completed.stderr


@pytest.mark.xdist_group('merge_gate_rerun_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_merge_gate_reruns_xdist_loadgroup_failures_by_their_plain_node_ids(tmp_path):
    """Task 7091: a loadgroup run names each failure ``<id>@<group>``, which plain pytest reports 'not found'.

    MEASURED RED before task 7091: ``fails_in_isolation``, the shape of all 52
    suffixed merge_gate rows in the flake ledger.
    """
    worktree = _make_worktree(tmp_path)
    (worktree / GROUPED_PROBE_REL).write_text(_GROUPED_PROBE, encoding='utf-8')
    detect_output = _loadgroup_detect_output(worktree)
    assert '@probe_group' in detect_output, (
        f'xdist no longer suffixes loadgroup node ids as this test assumes:\n{detect_output}'
    )
    failing = replace(_failing_result(), test_output=detect_output)

    suppression = asyncio.run(confirm_isolated_rerun_verdict(
        worktree, _make_config(worktree), [_module_config()], failing, call_site='merge_gate',
    ))

    assert suppression.verdict is FlakeVerdict.passes_in_isolation, (
        f'verdict={suppression.verdict} reason={suppression.unconfirmable_reason!r}'
    )
    assert suppression.test_ids == (
        f'{GROUPED_PROBE_REL}::test_ok[1]', f'{GROUPED_PROBE_REL}::test_ok[a@b]',
    ), suppression.test_ids

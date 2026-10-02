"""The merge gate's isolated re-run, end to end, against a REAL pytest.

The ONE test in this area that lets ``confirm_isolated_rerun_verdict`` reach a
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
import os
import shlex
import sys
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


def _failing_result() -> VerifyResult:
    """The load-shed failure the merge gate is asking the discriminator about."""
    return VerifyResult(
        passed=False,
        test_output=f'FAILED {PROBE_NODE_ID}\n',
        lint_output='',
        type_output='',
        summary='fail',
        category='test_failure',
        cause_hint=f'FAILED {PROBE_NODE_ID}',
    )


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

"""Tests for the predicate-contradiction filing convention (task 2779).

Gives reconciliation stages a reusable, durable way to file a one-off
``task_kind='deterministic'`` task with ``before_done.kind='predicate'`` when a
recon-stage contradiction can only be settled by live command execution (a
specific pytest test, a git grep/log check) — instead of ad-hoc Mem0
investigation notes/suppression records that silently age across cycles
(motivating precedent: task 2643).

Assertions are pinned to runtime return values (the builder's returned
dataclass/dict, the reference runner's exit codes) and stable load-bearing
substrings within the rendered prompt section — NOT verbatim prompt-text
equality — mirroring the test_recon_self_model.py convention.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fused_memory.middleware.deterministic_task_guard import deterministic_task_error
from fused_memory.middleware.execution_class_guard import execution_class_error
from fused_memory.reconciliation.predicate_contradiction import (
    PredicateContradictionTask,
    build_predicate_contradiction_task,
    render_predicate_contradiction_section,
)
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

# --- Shared fixtures for the builder tests ---------------------------------- #
_CONTRADICTION = (
    "Memory claims tests/test_x.py::test_y passes on main, but finding 9cfd4de9 "
    "asserts it fails — only a live pytest run can settle this."
)
_FINDING_ID = '9cfd4de9-1111-4000-8000-000000000001'
_RUN_ID = '23632bc9-2222-4000-8000-000000000002'
_CITED = ['03a2b6d7-3333-4000-8000-000000000003', '8ae43c70-4444-4000-8000-000000000004']
_PYTEST_ARGS = ['--pytest', 'tests/test_x.py::test_y']


def _base_build_kwargs(**overrides: Any) -> dict[str, Any]:
    """Valid baseline kwargs for build_predicate_contradiction_task; override
    individual fields per test.

    Return type is ``dict[str, Any]`` (not the inferred heterogeneous
    ``str | int | list[str]`` union) so the values splat cleanly into the
    strongly-typed keyword-only ``build_predicate_contradiction_task`` params.
    """
    kwargs: dict[str, Any] = dict(
        contradiction=_CONTRADICTION,
        script='check.sh',
        timeout_secs=120,
        args=_PYTEST_ARGS,
        finding_id=_FINDING_ID,
        run_id=_RUN_ID,
        cited_memory_ids=_CITED,
    )
    kwargs.update(overrides)
    return kwargs

# Repo root: this file is <root>/fused-memory/tests/test_predicate_contradiction.py
_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / 'scripts' / 'recon_predicate_check.sh'

# Committed username/email so the hermetic tmp-repo commit does not depend on a
# global git identity being configured in the test environment.
_GIT_ENV_ARGS = [
    '-c',
    'user.email=recon-test@example.com',
    '-c',
    'user.name=recon-test',
    '-c',
    'commit.gpgsign=false',
]


def _init_tmp_git_repo(tmp_path: Path, *, sentinel: str) -> Path:
    """Create a hermetic git repo under *tmp_path* containing a committed file
    whose contents include *sentinel*. Returns the repo path."""
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / 'tracked.txt').write_text(f'preamble\n{sentinel}\ntrailer\n')
    subprocess.run(['git', 'add', 'tracked.txt'], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ['git', *_GIT_ENV_ARGS, 'commit', '-q', '-m', 'seed'],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


class TestReconPredicateCheckScript:
    """scripts/recon_predicate_check.sh is a committed, executable reference
    runner whose exit codes honour the predicate contract (0 = settled as
    expected -> task done; non-zero = mismatch -> milestone_check_failed)."""

    def test_script_exists_and_is_executable(self):
        assert _SCRIPT.exists(), f'reference runner missing at {_SCRIPT}'
        assert os.access(_SCRIPT, os.X_OK), f'reference runner not executable: {_SCRIPT}'

    def test_no_args_exits_2(self):
        proc = subprocess.run([str(_SCRIPT)], capture_output=True, text=True)
        assert proc.returncode == 2, proc.stderr

    def test_unknown_mode_exits_2(self):
        proc = subprocess.run([str(_SCRIPT), '--bogus-mode'], capture_output=True, text=True)
        assert proc.returncode == 2, proc.stderr

    def test_git_grep_found_exits_0(self, tmp_path):
        sentinel = 'RECON_SENTINEL_PRESENT'
        repo = _init_tmp_git_repo(tmp_path, sentinel=sentinel)
        proc = subprocess.run(
            [str(_SCRIPT), '--git-grep', sentinel],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_git_grep_absent_exits_1(self, tmp_path):
        repo = _init_tmp_git_repo(tmp_path, sentinel='RECON_SENTINEL_PRESENT')
        proc = subprocess.run(
            [str(_SCRIPT), '--git-grep', 'absent-6f1e2d3c-0000-4000-8000-000000000000'],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, proc.stderr


class TestBuildPredicateContradictionTask:
    """build_predicate_contradiction_task assembles a well-formed
    task_kind='deterministic' + before_done.kind='predicate' submission whose
    payload passes BOTH submit-boundary guards."""

    def _build(self, **overrides):
        return build_predicate_contradiction_task(**_base_build_kwargs(**overrides))

    def test_returns_predicate_contradiction_task(self):
        task = self._build()
        assert isinstance(task, PredicateContradictionTask)
        assert task.task_kind == 'deterministic'

    def test_before_done_is_predicate_without_target_unit(self):
        bd = self._build().metadata['before_done']
        assert bd['kind'] == 'predicate'
        assert bd['script'] == 'check.sh'
        assert bd['args'] == _PYTEST_ARGS
        assert bd['timeout_secs'] == 120
        # A predicate has no unit to verify — target_unit must be absent entirely.
        assert 'target_unit' not in bd

    def test_execution_class_is_code_tdd_and_no_always_escalates(self):
        meta = self._build().metadata
        assert meta['execution_class'] == 'code_tdd'
        # always_escalates=True is forbidden for a predicate; the builder never sets it.
        assert 'always_escalates' not in meta

    def test_provenance_recorded_under_x_namespace(self):
        prov = self._build().metadata['x_recon_predicate_contradiction']
        assert prov['contradiction'] == _CONTRADICTION
        assert prov['finding_id'] == _FINDING_ID
        assert prov['run_id'] == _RUN_ID
        assert prov['cited_memory_ids'] == _CITED

    def test_title_and_description_non_empty(self):
        task = self._build()
        assert isinstance(task.title, str) and task.title
        assert isinstance(task.description, str) and task.description

    def test_payload_passes_both_submit_guards(self, tmp_path):
        script = tmp_path / 'check.sh'
        script.write_text('#!/usr/bin/env bash\nexit 0\n')
        os.chmod(script, 0o755)
        task = self._build(script='check.sh')
        assert (
            deterministic_task_error('deterministic', task.metadata, str(tmp_path)) is None
        )
        assert (
            execution_class_error(
                task.metadata, 'recon-stage-task_knowledge_sync', str(tmp_path)
            )
            is None
        )

    def test_as_submit_task_kwargs_is_splattable(self):
        kwargs = self._build().as_submit_task_kwargs()
        assert isinstance(kwargs, dict)
        allowed = {'title', 'description', 'details', 'priority', 'task_kind', 'metadata'}
        assert set(kwargs).issubset(allowed)
        # Load-bearing keys a submit_task call needs.
        assert kwargs['task_kind'] == 'deterministic'
        assert kwargs['metadata']['execution_class'] == 'code_tdd'
        assert kwargs['title']


class TestBuildPredicateContradictionRejections:
    """The builder FAILS LOUD (ValueError) on an execution_class other than
    'code_tdd', and propagates BeforeDone shape-validation failures."""

    def test_execution_class_operational_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            build_predicate_contradiction_task(
                **_base_build_kwargs(execution_class='operational')
            )
        msg = str(excinfo.value).lower()
        assert 'code_tdd' in msg
        # The rationale must name why: operational/decision coerce to
        # always_escalates=True, illegal for a predicate.
        assert 'always_escalates' in msg or 'predicate' in msg

    def test_execution_class_decision_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            build_predicate_contradiction_task(
                **_base_build_kwargs(execution_class='decision')
            )
        msg = str(excinfo.value).lower()
        assert 'code_tdd' in msg
        assert 'always_escalates' in msg or 'predicate' in msg

    def test_execution_class_unknown_rejected(self):
        # A value not even in EXECUTION_CLASSES is rejected too.
        assert 'banana' not in EXECUTION_CLASSES
        with pytest.raises(ValueError):
            build_predicate_contradiction_task(**_base_build_kwargs(execution_class='banana'))

    def test_empty_script_raises(self):
        # BeforeDone(script='') fails min_length=1 — propagated as an error.
        with pytest.raises(ValueError):
            build_predicate_contradiction_task(**_base_build_kwargs(script=''))

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError):
            build_predicate_contradiction_task(**_base_build_kwargs(timeout_secs=0))

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError):
            build_predicate_contradiction_task(**_base_build_kwargs(timeout_secs=-5))

    def test_happy_path_does_not_raise(self):
        # Regression guard: the valid baseline must NOT raise.
        task = build_predicate_contradiction_task(**_base_build_kwargs())
        assert task.metadata['execution_class'] == 'code_tdd'


class TestRenderPredicateContradictionSection:
    """render_predicate_contradiction_section() single-sources the prompt
    guidance: WHEN to use this path, the exit-code contract, the code_tdd
    rationale, and that it replaces ad-hoc Mem0 notes/suppression records.
    Pins stable load-bearing substrings only (not verbatim prose), per the
    test_recon_self_model.py convention."""

    def test_returns_non_empty_str(self):
        assert isinstance(render_predicate_contradiction_section(), str)
        assert render_predicate_contradiction_section()

    def test_has_section_header(self):
        assert 'Predicate' in render_predicate_contradiction_section()

    def test_names_before_done_predicate(self):
        text = render_predicate_contradiction_section()
        assert 'before_done' in text
        assert 'predicate' in text

    def test_states_exit_code_contract(self):
        text = render_predicate_contradiction_section()
        # exit 0 -> done via done_provenance.kind='deterministic-milestone'
        assert 'exit 0' in text
        assert 'deterministic-milestone' in text
        # non-zero -> milestone_check_failed born-at-L2 escalation
        assert 'non-zero' in text
        assert 'milestone_check_failed' in text

    def test_states_code_tdd_and_forbidden_classes(self):
        text = render_predicate_contradiction_section()
        assert 'code_tdd' in text
        # operational/decision are called out as forbidden for this path.
        assert 'operational' in text
        assert 'decision' in text

    def test_points_away_from_adhoc_mem0_notes(self):
        text = render_predicate_contradiction_section()
        lowered = text.lower()
        assert 'instead of' in lowered
        assert 'suppression' in lowered or 'mem0' in lowered


class TestStage2PromptWiring:
    """The rendered section is interpolated into STAGE2_SYSTEM_PROMPT so recon
    stages that file tasks receive the guidance — mirroring how
    render_execution_class_section() content is present in the prompt."""

    def test_render_section_present_verbatim(self):
        # Full-section faithfulness: the exact rendered text is a substring of
        # the prompt (the drift invariant — the f-string interpolates it verbatim).
        assert render_predicate_contradiction_section() in STAGE2_SYSTEM_PROMPT

    def test_section_header_present(self):
        assert '## Predicate Contradiction Checks' in STAGE2_SYSTEM_PROMPT

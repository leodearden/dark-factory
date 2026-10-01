"""Unit tests for fused_memory.middleware.gitignored_deliverable_guard (task 3611)."""

from __future__ import annotations

import shutil
import subprocess

import pytest
from _fm_helpers import _init_git_repo

from fused_memory.middleware.gitignored_deliverable_guard import (
    GitignoredDeliverableFinding,
    gitignored_deliverable_enforced,
    gitignored_deliverable_finding,
    gitignored_deliverable_reject,
    gitignored_deliverable_warning,
    make_gitignore_probe,
)


def _require_git() -> None:
    if shutil.which('git') is None:
        pytest.skip('git is not available')


@pytest.fixture
def gitignore_repo(tmp_path):
    """A real repo ignoring tasks.db, *.log and .taskmaster/, plus a TRACKED tracked.log."""
    _require_git()
    _init_git_repo(tmp_path)
    (tmp_path / '.gitignore').write_text('tasks.db\n*.log\n.taskmaster/\n')
    (tmp_path / 'tracked.log').write_text('tracked despite the *.log rule\n')
    subprocess.run(
        ['git', '-C', str(tmp_path), 'add', '-f', '.gitignore', 'tracked.log'],
        check=True,
    )
    subprocess.run(
        [
            'git', '-C', str(tmp_path),
            '-c', 'user.email=t@e.example', '-c', 'user.name=T',
            'commit', '-q', '-m', 'ignore rules and a force-tracked log',
        ],
        check=True,
    )
    return tmp_path


class TestMakeGitignoreProbe:
    """The one impure adapter, pinned against real ``git check-ignore``."""

    def test_ignored_absent_path_is_reported(self, gitignore_repo):
        probe = make_gitignore_probe(gitignore_repo)
        assert probe(['tasks.db']) == frozenset({'tasks.db'})

    def test_unignored_path_yields_empty_set_not_none(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['src/foo.py'])
        assert result is not None
        assert result == frozenset()

    def test_mixed_declaration_reports_only_the_ignored_subset(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tasks.db', 'src/foo.py'])
        assert result == frozenset({'tasks.db'})

    def test_tracked_path_matching_a_rule_counts_as_committable(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tracked.log'])
        assert result == frozenset()

    def test_nested_and_absolute_spellings_echo_verbatim(self, gitignore_repo):
        absolute = str(gitignore_repo / 'tasks.db')
        result = make_gitignore_probe(gitignore_repo)(
            ['.taskmaster/tasks/tasks.db', absolute],
        )
        assert result == frozenset({'.taskmaster/tasks/tasks.db', absolute})

    def test_non_git_directory_fails_open_with_none(self, tmp_path):
        _require_git()
        not_a_repo = tmp_path / 'plain'
        not_a_repo.mkdir()
        assert make_gitignore_probe(not_a_repo)(['tasks.db']) is None

    def test_missing_directory_fails_open_with_none(self, tmp_path):
        _require_git()
        assert make_gitignore_probe(tmp_path / 'does-not-exist')(['tasks.db']) is None

    def test_path_outside_repo_fails_open_despite_partial_stdout(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tasks.db', '/etc/hosts'])
        assert result is None

    def test_empty_declaration_yields_empty_set(self, gitignore_repo):
        assert make_gitignore_probe(gitignore_repo)([]) == frozenset()


class _FakeProbe:
    """Answers with a preset ignored set (or ``None``) and records every call."""

    def __init__(self, answer: frozenset[str] | None):
        self.answer = answer
        self.calls: list[list[str]] = []

    def __call__(self, paths):
        self.calls.append(list(paths))
        return self.answer


class TestGitignoredDeliverableFindingMatrix:
    """Exemption/detection matrix, hermetic via an injected fake probe."""

    def test_all_declared_files_ignored_is_a_finding(self):
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'files': ['tasks.db']},
            probe=_FakeProbe(frozenset({'tasks.db'})),
        )
        assert finding is not None
        assert finding.ignored_paths == ('tasks.db',)

    def test_one_committable_entry_suppresses_the_finding(self):
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'files': ['tasks.db', 'src/foo.py']},
            probe=_FakeProbe(frozenset({'tasks.db'})),
        )
        assert finding is None

    def test_deterministic_kind_is_exempt_without_probing(self):
        probe = _FakeProbe(frozenset({'tasks.db'}))
        finding = gitignored_deliverable_finding(
            task_kind='deterministic',
            metadata={'files': ['tasks.db']},
            probe=probe,
        )
        assert finding is None
        assert probe.calls == []

    @pytest.mark.parametrize('execution_class', ['operational', 'decision'])
    def test_non_code_execution_class_is_exempt(self, execution_class):
        probe = _FakeProbe(frozenset({'tasks.db'}))
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'execution_class': execution_class, 'files': ['tasks.db']},
            probe=probe,
        )
        assert finding is None
        assert probe.calls == []

    @pytest.mark.parametrize('metadata', [{'files': []}, None])
    def test_no_declared_files_is_exempt_without_probing(self, metadata):
        probe = _FakeProbe(frozenset({'tasks.db'}))
        finding = gitignored_deliverable_finding(
            task_kind='normal', metadata=metadata, probe=probe,
        )
        assert finding is None
        assert probe.calls == []

    def test_hand_set_cross_repo_marker_is_exempt(self):
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'cross_repo': True, 'files': ['tasks.db']},
            probe=_FakeProbe(frozenset({'tasks.db'})),
        )
        assert finding is None

    def test_probe_unable_to_answer_fails_open(self):
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'files': ['tasks.db']},
            probe=_FakeProbe(None),
        )
        assert finding is None

    def test_json_string_metadata_is_parsed(self):
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata='{"files": ["tasks.db"]}',
            probe=_FakeProbe(frozenset({'tasks.db'})),
        )
        assert finding is not None
        assert finding.ignored_paths == ('tasks.db',)

    def test_blank_entries_are_not_probed(self):
        probe = _FakeProbe(frozenset({'tasks.db'}))
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'files': ['', 'tasks.db', '   ']},
            probe=probe,
        )
        assert finding is not None
        assert probe.calls == [['tasks.db']]

    def test_only_blank_entries_is_exempt_without_probing(self):
        probe = _FakeProbe(frozenset())
        finding = gitignored_deliverable_finding(
            task_kind='normal',
            metadata={'files': ['', '  ']},
            probe=probe,
        )
        assert finding is None
        assert probe.calls == []


_FINDING = GitignoredDeliverableFinding(
    ignored_paths=('tasks.db', '.taskmaster/tasks/tasks.db'),
)
_DETERMINISTIC_HINT = "task_kind='deterministic'"


def _payload_text(payload) -> str:
    if isinstance(payload, dict):
        return ' '.join(_payload_text(v) for v in payload.values())
    if isinstance(payload, list):
        return ' '.join(_payload_text(v) for v in payload)
    return str(payload)


class TestGitignoredDeliverablePayloads:
    """Reject and warning payloads carry an accurate, actionable message."""

    def test_reject_is_a_validation_error_naming_every_path(self):
        payload = gitignored_deliverable_reject(_FINDING)
        assert payload['error_type'] == 'ValidationError'
        for path in _FINDING.ignored_paths:
            assert path in payload['error']

    def test_reject_names_the_commit_requirement(self):
        error = gitignored_deliverable_reject(_FINDING)['error']
        assert 'commit' in error
        assert 'gitignored' in error

    def test_reject_hint_suggests_deterministic(self):
        assert _DETERMINISTIC_HINT in gitignored_deliverable_reject(_FINDING)['hint']

    @pytest.mark.parametrize(
        'build', [gitignored_deliverable_reject, gitignored_deliverable_warning],
    )
    def test_confirm_plan_is_only_described_as_a_declaration_check(self, build):
        text = _payload_text(build(_FINDING))
        if 'confirm_plan' in text:
            assert 'declar' in text

    def test_warning_is_a_single_non_error_key(self):
        payload = gitignored_deliverable_warning(_FINDING)
        assert set(payload) == {'gitignored_deliverable_warning'}
        assert 'error' not in payload
        assert 'error_type' not in payload

    def test_warning_exposes_paths_and_deterministic_hint(self):
        nested = gitignored_deliverable_warning(_FINDING)['gitignored_deliverable_warning']
        assert nested['ignored_paths'] == list(_FINDING.ignored_paths)
        assert _DETERMINISTIC_HINT in nested['hint']

    def test_warning_emits_the_flagged_census_line(self, caplog):
        with caplog.at_level('WARNING'):
            gitignored_deliverable_warning(_FINDING)
        census = [
            r.getMessage() for r in caplog.records
            if 'gitignored_deliverable_lint.flagged' in r.getMessage()
        ]
        assert census
        assert 'tasks.db' in census[0]


class TestGitignoredDeliverableEnforced:
    """FUSED_GITIGNORED_DELIVERABLE_ENFORCE parsing: warn by default."""

    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on', ' 1 '])
    def test_truthy_values_enforce(self, monkeypatch, value):
        monkeypatch.setenv('FUSED_GITIGNORED_DELIVERABLE_ENFORCE', value)
        assert gitignored_deliverable_enforced() is True

    @pytest.mark.parametrize('value', ['', '0', 'maybe'])
    def test_other_values_warn(self, monkeypatch, value):
        monkeypatch.setenv('FUSED_GITIGNORED_DELIVERABLE_ENFORCE', value)
        assert gitignored_deliverable_enforced() is False

    def test_unset_warns(self, monkeypatch):
        monkeypatch.delenv('FUSED_GITIGNORED_DELIVERABLE_ENFORCE', raising=False)
        assert gitignored_deliverable_enforced() is False

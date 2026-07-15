"""Tests for audit_found_on_main_provenance.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py /
test_audit_duplicate_memories.py.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_found_on_main_provenance.py'


def _load_module() -> types.ModuleType:
    """Load audit_found_on_main_provenance.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_found_on_main_provenance'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
TaskProvenanceAudit = _mod.TaskProvenanceAudit
CITATION_PATTERN = _mod.CITATION_PATTERN
select_found_on_main_tasks = _mod.select_found_on_main_tasks
extract_cited_task_ids = _mod.extract_cited_task_ids
classify = _mod.classify
GitFacts = _mod.GitFacts
build_audit_report = _mod.build_audit_report
apply_audit_annotations = _mod.apply_audit_annotations
_has_flagged_findings = _mod._has_flagged_findings
_git_show_files = _mod._git_show_files
_git_is_ancestor = _mod._git_is_ancestor
_git_find_revert = _mod._git_find_revert
_git_files_missing_on_ref = _mod._git_files_missing_on_ref
_git_commit_message = _mod._git_commit_message


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _task(
    id: str,
    title: str = 'Some task',
    *,
    done_provenance: dict | None = None,
    files: list[str] | None = None,
    metadata: dict | str | None = None,
    status: str = 'done',
) -> dict:
    """Build a synthetic task dict carrying a found_on_main-shaped metadata blob.

    Either pass ``metadata`` directly (a dict OR a JSON string — exercises
    parse_metadata's dual-shape handling), or let this helper assemble one
    from ``done_provenance``/``files`` for the common case.
    """
    if metadata is not None:
        meta: dict | str = metadata
    else:
        built: dict[str, object] = {}
        if done_provenance is not None:
            built['done_provenance'] = done_provenance
        if files is not None:
            built['files'] = files
        meta = built
    return {'id': id, 'title': title, 'status': status, 'metadata': meta}


def _audit(
    task_id: str = '50',
    title: str = 'Some task',
    commit: str = 'a' * 40,
    note: str | None = 'found on main',
    declared_files: list[str] | None = None,
    *,
    is_ancestor: bool = True,
    commit_subject: str = '',
    commit_message: str = '',
    commit_files: list[str] | None = None,
    revert_commit: str | None = None,
    declared_files_missing_on_main: list[str] | None = None,
):
    """Build a TaskProvenanceAudit with sensible defaults for classify() tests."""
    return TaskProvenanceAudit(
        task_id=task_id,
        title=title,
        commit=commit,
        note=note,
        declared_files=list(declared_files) if declared_files is not None else [],
        is_ancestor=is_ancestor,
        commit_subject=commit_subject,
        commit_message=commit_message,
        commit_files=list(commit_files) if commit_files is not None else [],
        revert_commit=revert_commit,
        declared_files_missing_on_main=(
            list(declared_files_missing_on_main)
            if declared_files_missing_on_main is not None else []
        ),
    )


# ===========================================================================
# Step-1: select_found_on_main_tasks
# ===========================================================================

class TestSelectFoundOnMainTasksBasics:
    """Core filtering: only kind=='found_on_main' tasks are selected."""

    def test_selects_found_on_main_task_and_extracts_facts(self):
        """A found_on_main task's commit/note/files are pulled onto the audit."""
        task = _task(
            '50', 'Fix the thing',
            done_provenance={
                'kind': 'found_on_main', 'commit': 'a' * 40, 'note': 'already on main',
            },
            files=['src/thing.py'],
        )
        result = select_found_on_main_tasks([task])
        assert len(result) == 1
        audit = result[0]
        assert audit.task_id == '50'
        assert audit.title == 'Fix the thing'
        assert audit.commit == 'a' * 40
        assert audit.note == 'already on main'
        assert audit.declared_files == ['src/thing.py']

    def test_skips_merged_kind(self):
        """kind=='merged' tasks are not found_on_main — excluded."""
        task = _task('51', 'Merged task', done_provenance={'kind': 'merged', 'commit': 'b' * 40})
        assert select_found_on_main_tasks([task]) == []

    def test_skips_deterministic_deploy_kind(self):
        """kind=='deterministic-deploy' tasks are excluded."""
        task = _task(
            '52', 'Deploy task',
            done_provenance={'kind': 'deterministic-deploy'},
        )
        assert select_found_on_main_tasks([task]) == []

    def test_skips_task_with_absent_done_provenance(self):
        """A task with a metadata blob but no done_provenance key is skipped."""
        task = _task('53', 'No provenance', files=['x.py'])
        assert select_found_on_main_tasks([task]) == []

    def test_skips_task_with_no_metadata_key_at_all(self):
        """A task dict with no 'metadata' key at all is skipped, not an error."""
        task = {'id': '54', 'title': 'No metadata at all', 'status': 'done'}
        assert select_found_on_main_tasks([task]) == []

    def test_mixed_list_selects_only_found_on_main(self):
        """Over a mixed list, only the found_on_main member is selected."""
        tasks = [
            _task('60', 'Found on main', done_provenance={
                'kind': 'found_on_main', 'commit': 'e' * 40, 'note': 'n',
            }),
            _task('61', 'Merged', done_provenance={'kind': 'merged', 'commit': 'f' * 40}),
            _task('62', 'No provenance'),
        ]
        result = select_found_on_main_tasks(tasks)
        assert [a.task_id for a in result] == ['60']


class TestSelectFoundOnMainTasksMetadataShapes:
    """parse_metadata tolerates both dict and JSON-string metadata blobs."""

    def test_tolerates_metadata_as_json_string(self):
        """A JSON-string metadata blob is parsed the same as a dict."""
        blob = json.dumps({
            'done_provenance': {
                'kind': 'found_on_main', 'commit': 'c' * 40, 'note': 'string blob',
            },
            'files': ['a.py', 'b.py'],
        })
        task = _task('55', 'String metadata', metadata=blob)
        result = select_found_on_main_tasks([task])
        assert len(result) == 1
        assert result[0].commit == 'c' * 40
        assert result[0].declared_files == ['a.py', 'b.py']

    def test_tolerates_metadata_as_dict(self):
        """A dict metadata blob (the common in-process shape) works directly."""
        task = _task(
            '56', 'Dict metadata',
            done_provenance={'kind': 'found_on_main', 'commit': 'd' * 40, 'note': 'dict blob'},
        )
        result = select_found_on_main_tasks([task])
        assert len(result) == 1
        assert result[0].commit == 'd' * 40

    def test_malformed_metadata_blob_does_not_raise_and_is_omitted(self):
        """Unparseable-JSON metadata hits parse_metadata's warn path — no raise, just omitted."""
        task = _task('57', 'Malformed metadata', metadata='{not valid json')
        result = select_found_on_main_tasks([task])  # must not raise
        assert result == []


class TestSelectFoundOnMainTasksOrdering:
    """select_found_on_main_tasks returns results in deterministic int(task_id) order."""

    def test_deterministic_ordering_by_int_task_id(self):
        """Results are sorted by numeric task id, regardless of input order."""
        def _prov(c: str) -> dict:
            return {'kind': 'found_on_main', 'commit': c * 40, 'note': 'n'}

        tasks = [
            _task('300', 'C', done_provenance=_prov('a')),
            _task('100', 'A', done_provenance=_prov('b')),
            _task('200', 'B', done_provenance=_prov('c')),
        ]
        result = select_found_on_main_tasks(tasks)
        assert [a.task_id for a in result] == ['100', '200', '300']


# ===========================================================================
# Step-3: extract_cited_task_ids / commit_cites_task
# ===========================================================================

class TestExtractCitedTaskIdsConventions:
    """The three citation conventions mirrored from DEFAULT_COMMIT_CITATION_PATTERN."""

    def test_merge_subject_convention(self):
        """The canonical no-ff merge subject cites the task id."""
        assert extract_cited_task_ids('Merge task/50 into main') == {'50'}

    def test_conventional_commit_convention(self):
        """Conventional-commit `type(id):` subjects cite the task id."""
        assert extract_cited_task_ids('impl(50): add X') == {'50'}

    def test_task_branch_mention_convention(self):
        """A bare `task/{id}` mention anywhere in the message cites the id."""
        assert extract_cited_task_ids('fix: touch task/50 handler') == {'50'}

    def test_no_citation_returns_empty_set(self):
        """A message with no citation of any kind yields an empty set."""
        assert extract_cited_task_ids('chore: general cleanup, no ticket') == set()

    def test_empty_message_returns_empty_set(self):
        """An empty message string does not raise and yields an empty set."""
        assert extract_cited_task_ids('') == set()


class TestExtractCitedTaskIdsWordBoundary:
    r"""`\b` word-boundary guard: no substring overlap between numeric ids."""

    def test_task_339_does_not_match_as_3399(self):
        """Citing task/339 must not also register as a citation of '3399'."""
        ids = extract_cited_task_ids('Merge task/339 into main')
        assert '3399' not in ids
        assert ids == {'339'}

    def test_task_3399_does_not_match_as_339(self):
        """Citing task/3399 must not also register as a citation of '339' (reverse case)."""
        ids = extract_cited_task_ids('Merge task/3399 into main')
        assert '339' not in ids
        assert ids == {'3399'}


class TestExtractCitedTaskIdsMultipleCitations:
    """A message can cite more than one task id — all are captured."""

    def test_multiple_ids_all_captured(self):
        """Both a merge-subject id and a body-mention id are captured together."""
        message = 'Merge task/50 into main\n\nAlso relates to task/77.'
        assert extract_cited_task_ids(message) == {'50', '77'}


# ===========================================================================
# Step-5/6: classify — commit_not_on_main (highest precedence)
# ===========================================================================

class TestClassifyCommitNotOnMain:
    """commit_not_on_main wins whenever is_ancestor is False, regardless of other facts."""

    def test_not_ancestor_yields_commit_not_on_main(self):
        """A cited commit unreachable from the audited ref is the strongest bogus signal."""
        audit = _audit(is_ancestor=False)
        verdict, reasons = classify(audit)
        assert verdict == 'commit_not_on_main'
        assert reasons
        assert any('ancestor' in r or 'not reachable' in r for r in reasons)

    def test_wins_over_every_other_signal(self):
        """Even with misattribution/revert/deliverable-absent facts also present,
        commit_not_on_main takes precedence over all of them.
        """
        audit = _audit(
            is_ancestor=False,
            commit_message='Merge task/999 into main',  # would-be misattribution
            revert_commit='f' * 40,  # would-be reverted
            declared_files=['a.py'],
            commit_files=[],  # would-be deliverable_absent
        )
        verdict, _reasons = classify(audit)
        assert verdict == 'commit_not_on_main'


# ===========================================================================
# Step-7/8: classify — misattributed
# ===========================================================================
#
# The negative cases below intentionally assert `verdict != 'misattributed'`
# rather than a concrete fallback verdict: at step 8's GREEN, everything past
# the misattributed check is still an interim placeholder (the
# deliverable_absent/unverifiable distinction isn't implemented until step
# 12), so pinning an exact fallback value here would make this test flip
# once step 12 lands. "Does not flag misattributed" is the actual claim.

class TestClassifyMisattributed:
    """misattributed: cited commit is on-ref, but its message cites a different task."""

    def test_message_cites_only_a_different_task(self):
        """Task 50's provenance points at a commit whose subject cites task 77."""
        audit = _audit(task_id='50', is_ancestor=True, commit_message='Merge task/77 into main')
        verdict, reasons = classify(audit)
        assert verdict == 'misattributed'
        assert reasons
        assert any('77' in r for r in reasons)

    def test_mixed_citation_does_not_flag_misattributed(self):
        """A message citing BOTH this task and another task is not misattribution."""
        audit = _audit(
            task_id='50', is_ancestor=True,
            commit_message='Merge task/50 into main\n\nAlso relates to task/77.',
        )
        verdict, _reasons = classify(audit)
        assert verdict != 'misattributed'

    def test_no_citation_at_all_does_not_flag_misattributed(self):
        """No citation at all means nothing to compare — not misattribution
        (falls through to a later verdict in the ladder).
        """
        audit = _audit(task_id='50', is_ancestor=True, commit_message='chore: general cleanup')
        verdict, _reasons = classify(audit)
        assert verdict != 'misattributed'


# ===========================================================================
# Step-9/10: classify — reverted (post-hoc-revert blind spot)
# ===========================================================================

class TestClassifyReverted:
    """reverted: a correctly-attributed, on-ref commit was later undone."""

    def test_revert_commit_set_yields_reverted(self):
        """A `This reverts commit <sha>` marker found on the ref flags reverted."""
        audit = _audit(
            is_ancestor=True,
            commit_message='Merge task/50 into main',
            revert_commit='f' * 40,
        )
        verdict, reasons = classify(audit)
        assert verdict == 'reverted'
        assert reasons
        assert any('revert' in r.lower() for r in reasons)

    def test_declared_files_missing_on_main_yields_reverted(self):
        """A declared deliverable file absent from the ref HEAD flags reverted."""
        audit = _audit(
            is_ancestor=True,
            commit_message='Merge task/50 into main',
            declared_files=['src/thing.py'],
            declared_files_missing_on_main=['src/thing.py'],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'reverted'
        assert reasons
        assert any('missing' in r.lower() for r in reasons)

    def test_reason_distinguishes_the_two_subsignals(self):
        """When both signals are present, the reasons mention both distinctly."""
        audit = _audit(
            is_ancestor=True,
            commit_message='Merge task/50 into main',
            revert_commit='f' * 40,
            declared_files=['src/thing.py'],
            declared_files_missing_on_main=['src/thing.py'],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'reverted'
        joined = ' '.join(reasons).lower()
        assert 'revert' in joined
        assert 'missing' in joined


# ===========================================================================
# Step-11/12: classify — deliverable_absent / unverifiable / ok + precedence
# ===========================================================================

class TestClassifyDeliverableAbsent:
    """deliverable_absent: declared files exist but none appear in the commit diff."""

    def test_no_declared_file_in_commit_diff_yields_deliverable_absent(self):
        audit = _audit(
            is_ancestor=True,
            commit_message='Merge task/50 into main',
            declared_files=['src/thing.py'],
            commit_files=['README.md'],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'deliverable_absent'
        assert reasons


class TestClassifyUnverifiable:
    """unverifiable: nothing to check against — no declared files and no self-citation."""

    def test_no_declared_files_and_no_self_citation_yields_unverifiable(self):
        audit = _audit(
            is_ancestor=True,
            commit_message='chore: general cleanup',
            declared_files=[],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'unverifiable'
        assert reasons


class TestClassifyOk:
    """ok: every check passes — self-cited (or declared files present), no revert/missing."""

    def test_self_cited_with_no_declared_files_is_ok(self):
        """Self-citation alone (no declared files to check) is sufficient for ok."""
        audit = _audit(
            task_id='50', is_ancestor=True,
            commit_message='Merge task/50 into main',
            declared_files=[],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'ok'
        assert reasons == []

    def test_declared_files_present_in_commit_is_ok(self):
        """Declared files present in the commit diff are sufficient, even with no citation."""
        audit = _audit(
            is_ancestor=True,
            commit_message='chore: unrelated subject',
            declared_files=['src/thing.py'],
            commit_files=['src/thing.py', 'README.md'],
        )
        verdict, reasons = classify(audit)
        assert verdict == 'ok'
        assert reasons == []


class TestClassifyPrecedence:
    """Earlier rungs of the ladder win over later ones when multiple facts apply."""

    def test_not_on_main_wins_over_deliverable_absent(self):
        """A not-on-main + deliverable-absent audit resolves to commit_not_on_main,
        not deliverable_absent.
        """
        audit = _audit(
            is_ancestor=False,
            commit_message='Merge task/50 into main',
            declared_files=['src/thing.py'],
            commit_files=[],
        )
        verdict, _reasons = classify(audit)
        assert verdict == 'commit_not_on_main'

    def test_reverted_wins_over_deliverable_absent(self):
        """A revert + deliverable-absent audit resolves to reverted, not deliverable_absent."""
        audit = _audit(
            is_ancestor=True,
            commit_message='Merge task/50 into main',
            revert_commit='f' * 40,
            declared_files=['src/thing.py'],
            commit_files=[],
        )
        verdict, _reasons = classify(audit)
        assert verdict == 'reverted'


# ===========================================================================
# Step-13/14: build_audit_report (async orchestrator)
# ===========================================================================

class FakeGitFacts:
    """Test double for GitFacts — returns canned facts per commit sha, and
    records every ``gather`` call so tests can assert which commits were
    actually inspected (i.e. that non-found_on_main tasks are never touched).
    """

    def __init__(self, facts_by_commit: dict[str, dict] | None = None):
        self.facts_by_commit = facts_by_commit or {}
        self.calls: list[tuple[str, str, list[str]]] = []

    async def gather(self, commit, ref, declared_files):
        self.calls.append((commit, ref, list(declared_files)))
        return self.facts_by_commit.get(commit, {})


def _facts(
    *,
    is_ancestor: bool = True,
    commit_subject: str = '',
    commit_message: str = '',
    commit_files: list[str] | None = None,
    revert_commit: str | None = None,
    declared_files_missing_on_main: list[str] | None = None,
) -> dict:
    """Build a canned git-facts dict, shaped like GitFacts.gather()'s return value."""
    return {
        'is_ancestor': is_ancestor,
        'commit_subject': commit_subject,
        'commit_message': commit_message,
        'commit_files': commit_files or [],
        'revert_commit': revert_commit,
        'declared_files_missing_on_main': declared_files_missing_on_main or [],
    }


def _fom_task(task_id: str, commit: str, files: list[str] | None = None) -> dict:
    """A found_on_main task dict, terse for build_audit_report fixtures."""
    return _task(
        task_id, f'Task {task_id}',
        done_provenance={'kind': 'found_on_main', 'commit': commit, 'note': 'n'},
        files=files,
    )


@pytest.mark.asyncio
class TestBuildAuditReportSelection:
    """Only found_on_main tasks are audited; git facts are gathered per audited task."""

    @staticmethod
    def _mixed_tasks():
        return [
            _fom_task('10', 'a' * 40, files=['src/a.py']),
            _fom_task('20', 'b' * 40, files=['src/b.py']),
            _task('99', 'Merged, not found_on_main',
                  done_provenance={'kind': 'merged', 'commit': 'z' * 40}),
        ]

    async def test_only_found_on_main_tasks_are_audited(self):
        tasks = self._mixed_tasks()
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/10 into main', commit_files=['src/a.py']),
            'b' * 40: _facts(commit_message='Merge task/20 into main', commit_files=['src/b.py']),
        })
        report = await build_audit_report(tasks, git, ref='main')
        assert report['total'] == 2
        assert [t['task_id'] for t in report['tasks']] == ['10', '20']
        # task 99 (kind='merged') was never gathered — only found_on_main commits are inspected.
        gathered_commits = {c for c, _ref, _files in git.calls}
        assert gathered_commits == {'a' * 40, 'b' * 40}
        assert all(_ref == 'main' for _c, _ref, _files in git.calls)

    async def test_declared_files_threaded_into_gather_call(self):
        """Each task's declared_files are passed through to git.gather()."""
        tasks = [_fom_task('10', 'a' * 40, files=['src/a.py', 'src/other.py'])]
        git = FakeGitFacts({'a' * 40: _facts(commit_message='Merge task/10 into main')})
        await build_audit_report(tasks, git, ref='main')
        [(_commit, _ref, declared_files)] = git.calls
        assert declared_files == ['src/a.py', 'src/other.py']

    async def test_git_facts_gathered_and_classify_run_per_task(self):
        """Each audited task's canned facts flow through classify() into its verdict."""
        tasks = self._mixed_tasks()
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/10 into main', commit_files=['src/a.py']),
            'b' * 40: _facts(commit_message='Merge task/20 into main', commit_files=['src/b.py']),
        })
        report = await build_audit_report(tasks, git, ref='main')
        by_id = {t['task_id']: t for t in report['tasks']}
        assert by_id['10']['verdict'] == 'ok'
        assert by_id['20']['verdict'] == 'ok'
        assert by_id['10']['commit'] == 'a' * 40
        assert by_id['10']['reasons'] == []


@pytest.mark.asyncio
class TestBuildAuditReportShape:
    """The report dict's structural fields: verdict_counts, tasks, total, ref, dry_run."""

    async def test_clean_set_yields_all_ok_and_zeroed_other_counts(self):
        tasks = [
            _fom_task('10', 'a' * 40, files=['src/a.py']),
            _fom_task('20', 'b' * 40, files=['src/b.py']),
        ]
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/10 into main', commit_files=['src/a.py']),
            'b' * 40: _facts(commit_message='Merge task/20 into main', commit_files=['src/b.py']),
        })
        report = await build_audit_report(tasks, git, ref='main')
        assert report['verdict_counts'] == {
            'commit_not_on_main': 0, 'misattributed': 0, 'reverted': 0,
            'deliverable_absent': 0, 'unverifiable': 0, 'ok': 2,
        }
        assert report['total'] == 2
        assert report['ref'] == 'main'
        assert report['dry_run'] is True

    async def test_per_task_detail_shape(self):
        tasks = [_fom_task('10', 'a' * 40, files=['src/a.py'])]
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/10 into main', commit_files=['src/a.py']),
        })
        report = await build_audit_report(tasks, git, ref='main')
        assert report['tasks'] == [{
            'task_id': '10', 'verdict': 'ok', 'commit': 'a' * 40,
            'commit_subject': '', 'reasons': [],
        }]

    async def test_per_task_detail_carries_commit_subject_for_triage(self):
        """commit_subject is surfaced on the per-task detail (e.g. for a
        misattributed verdict, a human can see the offending subject line
        without a separate git lookup)."""
        tasks = [_fom_task('30', 'c' * 40)]
        git = FakeGitFacts({
            'c' * 40: _facts(
                commit_subject='Merge task/999 into main',
                commit_message='Merge task/999 into main',
            ),
        })
        report = await build_audit_report(tasks, git, ref='main')
        assert report['tasks'][0]['verdict'] == 'misattributed'
        assert report['tasks'][0]['commit_subject'] == 'Merge task/999 into main'

    async def test_deterministic_ordering_by_int_task_id(self):
        """Per-task detail is ordered by numeric task id regardless of input order."""
        tasks = [
            _fom_task('300', 'c' * 40),
            _fom_task('100', 'a' * 40),
            _fom_task('200', 'b' * 40),
        ]
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/100 into main'),
            'b' * 40: _facts(commit_message='Merge task/200 into main'),
            'c' * 40: _facts(commit_message='Merge task/300 into main'),
        })
        report = await build_audit_report(tasks, git, ref='main')
        assert [t['task_id'] for t in report['tasks']] == ['100', '200', '300']

    async def test_custom_ref_is_echoed_back(self):
        tasks = [_fom_task('10', 'a' * 40)]
        git = FakeGitFacts({'a' * 40: _facts(commit_message='Merge task/10 into main')})
        report = await build_audit_report(tasks, git, ref='origin/main')
        assert report['ref'] == 'origin/main'

    async def test_planted_misattributed_and_reverted_tasks_get_right_verdicts(self):
        """A clean task plus a planted misattributed task and a planted reverted task."""
        tasks = [
            _fom_task('10', 'a' * 40, files=['src/a.py']),
            _fom_task('30', 'c' * 40),  # misattributed: cites a different task
            _fom_task('40', 'd' * 40, files=['src/d.py']),  # reverted: revert marker present
        ]
        git = FakeGitFacts({
            'a' * 40: _facts(commit_message='Merge task/10 into main', commit_files=['src/a.py']),
            'c' * 40: _facts(commit_message='Merge task/999 into main'),
            'd' * 40: _facts(
                commit_message='Merge task/40 into main', commit_files=['src/d.py'],
                revert_commit='e' * 40,
            ),
        })
        report = await build_audit_report(tasks, git, ref='main')
        by_id = {t['task_id']: t for t in report['tasks']}
        assert by_id['10']['verdict'] == 'ok'
        assert by_id['30']['verdict'] == 'misattributed'
        assert by_id['40']['verdict'] == 'reverted'
        assert report['verdict_counts'] == {
            'commit_not_on_main': 0, 'misattributed': 1, 'reverted': 1,
            'deliverable_absent': 0, 'unverifiable': 0, 'ok': 1,
        }
        assert report['total'] == 3


@pytest.mark.asyncio
class TestBuildAuditReportMissingFactsDefaultSafe:
    """A facts provider that omits keys entirely (unlike GitFacts, which
    always supplies them) must fail safe: is_ancestor defaults to False —
    the loud commit_not_on_main outcome — never a silent True."""

    async def test_empty_facts_dict_yields_commit_not_on_main(self):
        tasks = [_fom_task('10', 'a' * 40, files=['src/a.py'])]
        git = FakeGitFacts({'a' * 40: {}})  # omits every key, including is_ancestor
        report = await build_audit_report(tasks, git, ref='main')
        assert report['tasks'][0]['verdict'] == 'commit_not_on_main'
        assert report['verdict_counts']['commit_not_on_main'] == 1


# ===========================================================================
# Step-15/16: real git I/O wrappers (_git_show_files, _git_is_ancestor,
# _git_find_revert, _git_files_missing_on_ref) against a real tmp_path repo.
# ===========================================================================

def _git(root: Path, *args: str) -> str:
    """Run a git command synchronously in *root*; return stripped stdout.

    Raises ``CalledProcessError`` on failure — fixture setup should fail
    loudly, unlike the script's own wrappers under test (which degrade to
    safe defaults).
    """
    result = subprocess.run(
        ['git', *args], cwd=root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _write(root: Path, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_test_repo(root: Path) -> dict[str, str]:
    """Build a small git repo exercising every wrapper under test.

    Mirrors the ``_setup_repo`` convention in
    ``orchestrator/tests/test_git_ops.py`` (``git init -b main`` + local
    user.email/user.name config). Commit graph on ``main``, oldest first:

      1. ``c_init``        — adds README.md.
      2. ``c_keep_drop``   — adds src/keep.py + src/drop.py together (the
         "commit under test" for _git_show_files / _git_files_missing_on_ref).
      3. ``c_drop_removed``— deletes src/drop.py (now missing on main HEAD).
      4. ``c_revert_target`` — adds src/revert_target.py.
      5. ``c_revert``      — ``git revert`` of c_revert_target; its message
         carries the canonical "This reverts commit <full-sha>" trailer.

    Plus a sibling branch (``sidebranch``, off ``c_init``) holding one commit
    (``c_side``) that is never merged into ``main`` — not an ancestor of it.
    """
    _git(root, 'init', '-b', 'main')
    _git(root, 'config', 'user.email', 'test@test.com')
    _git(root, 'config', 'user.name', 'Test')
    # Defensive: don't let a global gpgsign=true / commit hooks make a
    # throwaway test repo's commits hang or fail (fused-memory precedent —
    # see test_main_checkout_resolver.py's _init_repo).
    _git(root, 'config', 'commit.gpgsign', 'false')

    _write(root, 'README.md', '# Test\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'chore: init')
    c_init = _git(root, 'rev-parse', 'HEAD')

    _write(root, 'src/keep.py', 'keep = 1\n')
    _write(root, 'src/drop.py', 'drop = 1\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'feat: add keep and drop')
    c_keep_drop = _git(root, 'rev-parse', 'HEAD')

    (root / 'src' / 'drop.py').unlink()
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'chore: remove drop')
    c_drop_removed = _git(root, 'rev-parse', 'HEAD')

    _write(root, 'src/revert_target.py', 'target = 1\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'feat: add revert target')
    c_revert_target = _git(root, 'rev-parse', 'HEAD')

    _git(root, 'revert', '--no-edit', c_revert_target)
    c_revert = _git(root, 'rev-parse', 'HEAD')

    # Side branch off the initial commit — never merged into main.
    _git(root, 'checkout', '-q', '-b', 'sidebranch', c_init)
    _write(root, 'side.py', 'side = 1\n')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'feat: side-only work')
    c_side = _git(root, 'rev-parse', 'HEAD')
    _git(root, 'checkout', '-q', 'main')

    return {
        'c_init': c_init,
        'c_keep_drop': c_keep_drop,
        'c_drop_removed': c_drop_removed,
        'c_revert_target': c_revert_target,
        'c_revert': c_revert,
        'c_side': c_side,
    }


@pytest.fixture
def repo_facts(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A real git repo (in tmp_path) plus a dict of named commit shas.

    See :func:`_build_test_repo` for the commit-graph shape.
    """
    root = tmp_path / 'repo'
    root.mkdir()
    shas = _build_test_repo(root)
    return root, shas


@pytest.mark.asyncio
class TestGitShowFiles:
    """_git_show_files returns the changed paths for a real commit."""

    async def test_returns_changed_paths(self, repo_facts):
        root, shas = repo_facts
        files = await _git_show_files(str(root), shas['c_keep_drop'])
        assert sorted(files) == ['src/drop.py', 'src/keep.py']


@pytest.mark.asyncio
class TestGitIsAncestor:
    """_git_is_ancestor reflects real reachability from the audited ref."""

    async def test_true_for_ancestor(self, repo_facts):
        root, shas = repo_facts
        assert await _git_is_ancestor(str(root), shas['c_keep_drop'], 'main') is True

    async def test_false_for_branch_only_commit(self, repo_facts):
        """A commit that only exists on a sibling branch is not an ancestor of main."""
        root, shas = repo_facts
        assert await _git_is_ancestor(str(root), shas['c_side'], 'main') is False


@pytest.mark.asyncio
class TestGitFindRevert:
    """_git_find_revert locates the `git revert` commit that undid a commit."""

    async def test_returns_reverting_commit_sha(self, repo_facts):
        root, shas = repo_facts
        result = await _git_find_revert(str(root), shas['c_revert_target'], 'main')
        assert result == shas['c_revert']

    async def test_none_when_never_reverted(self, repo_facts):
        root, shas = repo_facts
        result = await _git_find_revert(str(root), shas['c_keep_drop'], 'main')
        assert result is None


@pytest.mark.asyncio
class TestGitFilesMissingOnRef:
    """_git_files_missing_on_ref distinguishes present vs. deleted declared files."""

    async def test_deleted_file_reported_missing_present_file_is_not(self, repo_facts):
        root, _shas = repo_facts
        missing = await _git_files_missing_on_ref(
            str(root), ['src/keep.py', 'src/drop.py'], 'main',
        )
        assert missing == ['src/drop.py']


@pytest.mark.asyncio
class TestGitFactsGather:
    """GitFacts.gather() aggregates the four _git_* wrappers per commit, and
    short-circuits all of them when the commit is not on the ref at all."""

    async def test_ancestor_commit_aggregates_all_facts(self, repo_facts):
        root, shas = repo_facts
        git = GitFacts(str(root))
        facts = await git.gather(shas['c_keep_drop'], 'main', ['src/keep.py', 'src/drop.py'])
        assert facts['is_ancestor'] is True
        assert sorted(facts['commit_files']) == ['src/drop.py', 'src/keep.py']
        assert facts['revert_commit'] is None
        assert facts['declared_files_missing_on_main'] == ['src/drop.py']
        assert facts['commit_subject']  # non-empty: first line of the commit message
        assert facts['commit_message']

    async def test_non_ancestor_short_circuits_remaining_git_calls(self, repo_facts, monkeypatch):
        """When is_ancestor is False, none of the other four subprocesses run
        — classify() would ignore their facts anyway (commit_not_on_main wins
        at the very first precedence check), so gathering them is pure waste.
        """
        root, shas = repo_facts
        calls: list[str] = []

        async def _spy_commit_message(project_root, commit):
            calls.append('commit_message')
            return 'should not be called'

        async def _spy_show_files(project_root, commit):
            calls.append('show_files')
            return ['should-not-be-called.py']

        async def _spy_find_revert(project_root, commit, ref):
            calls.append('find_revert')
            return 'f' * 40

        async def _spy_files_missing(project_root, files, ref):
            calls.append('files_missing')
            return list(files)

        monkeypatch.setattr(_mod, '_git_commit_message', _spy_commit_message)
        monkeypatch.setattr(_mod, '_git_show_files', _spy_show_files)
        monkeypatch.setattr(_mod, '_git_find_revert', _spy_find_revert)
        monkeypatch.setattr(_mod, '_git_files_missing_on_ref', _spy_files_missing)

        git = GitFacts(str(root))
        facts = await git.gather(shas['c_side'], 'main', ['src/keep.py'])

        assert facts == {
            'is_ancestor': False,
            'commit_subject': '',
            'commit_message': '',
            'commit_files': [],
            'revert_commit': None,
            'declared_files_missing_on_main': [],
        }
        assert calls == []  # none of the other four wrappers ran


# ===========================================================================
# Step-17/18: apply_audit_annotations (non-destructive apply layer)
# ===========================================================================

def _flagged_report(extra_ok: bool = True) -> dict:
    """A report with one task per flagged verdict, plus (by default) an
    ``ok`` and an ``unverifiable`` task that must be left untouched.
    """
    tasks = [
        {'task_id': '10', 'verdict': 'commit_not_on_main', 'commit': 'a' * 40,
         'reasons': ['cited commit is not an ancestor of the audited ref']},
        {'task_id': '20', 'verdict': 'misattributed', 'commit': 'b' * 40,
         'reasons': ['commit message cites task(s) 77, not task 20']},
        {'task_id': '30', 'verdict': 'reverted', 'commit': 'c' * 40,
         'reasons': [f'commit was later reverted by {"d" * 40}']},
        {'task_id': '40', 'verdict': 'deliverable_absent', 'commit': 'e' * 40,
         'reasons': ["none of the declared file(s) ['x.py'] appear in the diff"]},
    ]
    if extra_ok:
        tasks.append({'task_id': '50', 'verdict': 'ok', 'commit': 'f' * 40, 'reasons': []})
        tasks.append({
            'task_id': '60', 'verdict': 'unverifiable', 'commit': 'g' * 40,
            'reasons': ['no declared files and the commit message does not cite this task'],
        })
    counts = dict.fromkeys(
        ('commit_not_on_main', 'misattributed', 'reverted', 'deliverable_absent',
         'unverifiable', 'ok'), 0,
    )
    for t in tasks:
        counts[t['verdict']] += 1
    return {
        'ref': 'main', 'dry_run': True, 'total': len(tasks),
        'verdict_counts': counts, 'tasks': tasks,
    }


class FakeAuditBackend:
    """Test double recording ``update_task``/``set_task_status`` calls.

    ``fail_for`` names task ids whose ``update_task`` call raises, to
    exercise :func:`apply_audit_annotations`' per-task error isolation.
    """

    def __init__(self, fail_for: set[str] | None = None):
        self.fail_for = fail_for or set()
        self.update_calls: list[dict] = []
        self.status_calls: list[tuple] = []

    async def update_task(self, task_id, project_root, metadata=None, tag=None, **kwargs):
        if task_id in self.fail_for:
            raise RuntimeError(f'simulated update_task failure for {task_id}')
        self.update_calls.append({
            'task_id': task_id, 'project_root': project_root,
            'metadata': metadata, 'tag': tag,
        })
        return {'success': True}

    async def set_task_status(self, *args, **kwargs):
        self.status_calls.append((args, kwargs))


@pytest.mark.asyncio
class TestApplyAuditAnnotationsFlaggedVerdicts:
    """Only flagged verdicts get an update_task call annotating x_provenance_audit."""

    async def test_flagged_verdicts_are_annotated(self):
        report = _flagged_report()
        backend = FakeAuditBackend()
        await apply_audit_annotations(backend, '/proj', report)
        annotated_ids = {c['task_id'] for c in backend.update_calls}
        assert annotated_ids == {'10', '20', '30', '40'}

    async def test_annotation_contains_verdict_reasons_and_ref(self):
        report = _flagged_report(extra_ok=False)
        backend = FakeAuditBackend()
        await apply_audit_annotations(backend, '/proj', report)
        by_id = {c['task_id']: c for c in backend.update_calls}
        call = by_id['20']
        metadata = call['metadata']
        parsed = json.loads(metadata) if isinstance(metadata, str) else metadata
        annotation = parsed['x_provenance_audit']
        assert annotation['verdict'] == 'misattributed'
        assert annotation['reasons'] == report['tasks'][1]['reasons']
        assert annotation['ref'] == 'main'

    async def test_ok_and_unverifiable_left_untouched(self):
        report = _flagged_report()
        backend = FakeAuditBackend()
        await apply_audit_annotations(backend, '/proj', report)
        annotated_ids = {c['task_id'] for c in backend.update_calls}
        assert '50' not in annotated_ids
        assert '60' not in annotated_ids

    async def test_all_ok_report_makes_no_update_calls(self):
        report = _flagged_report(extra_ok=False)
        report['tasks'] = [{'task_id': '99', 'verdict': 'ok', 'commit': 'a' * 40, 'reasons': []}]
        backend = FakeAuditBackend()
        result = await apply_audit_annotations(backend, '/proj', report)
        assert backend.update_calls == []
        assert result['annotated'] == 0


@pytest.mark.asyncio
class TestApplyAuditAnnotationsSummaryAndErrors:
    """The returned summary counts successes/errors and never aborts the batch."""

    async def test_returned_summary_counts_annotated_and_errors(self):
        report = _flagged_report()
        backend = FakeAuditBackend(fail_for={'20'})
        result = await apply_audit_annotations(backend, '/proj', report)
        assert result['annotated'] == 3
        assert result['errors'] == 1

    async def test_per_task_failure_does_not_abort_batch(self):
        """A failure on the FIRST flagged task must not prevent later ones
        from being processed."""
        report = _flagged_report()
        backend = FakeAuditBackend(fail_for={'10'})
        await apply_audit_annotations(backend, '/proj', report)
        annotated_ids = {c['task_id'] for c in backend.update_calls}
        assert annotated_ids == {'20', '30', '40'}


@pytest.mark.asyncio
class TestApplyAuditAnnotationsNeverReopens:
    """apply_audit_annotations never calls set_task_status — reopening a
    done task back to pending is a human decision (see design_decisions)."""

    async def test_no_set_task_status_call_is_made(self):
        report = _flagged_report()
        backend = FakeAuditBackend()
        await apply_audit_annotations(backend, '/proj', report)
        assert backend.status_calls == []


@pytest.mark.asyncio
class TestApplyAuditAnnotationsNeedsHumanReview:
    """needs_human_review lists every flagged task id, regardless of whether
    its annotation write succeeded — a human still needs to disposition it."""

    async def test_needs_human_review_lists_all_flagged_ids(self):
        report = _flagged_report()
        backend = FakeAuditBackend(fail_for={'20'})
        result = await apply_audit_annotations(backend, '/proj', report)
        assert set(result['needs_human_review']) == {'10', '20', '30', '40'}


# ===========================================================================
# _has_flagged_findings — pure helper backing the CLI's --fail-on-findings
# ===========================================================================

def _counts(**overrides: int) -> dict:
    """A verdict_counts dict, all-zero except for *overrides*."""
    counts: dict[str, int] = dict.fromkeys(
        ('commit_not_on_main', 'misattributed', 'reverted', 'deliverable_absent',
         'unverifiable', 'ok'), 0,
    )
    counts.update(overrides)
    return counts


class TestHasFlaggedFindings:
    """True iff verdict_counts shows any flagged verdict; ok/unverifiable
    counts never trigger it on their own."""

    def test_false_when_all_ok(self):
        report = {'verdict_counts': _counts(ok=5)}
        assert _has_flagged_findings(report) is False

    def test_false_when_only_unverifiable(self):
        report = {'verdict_counts': _counts(unverifiable=3, ok=2)}
        assert _has_flagged_findings(report) is False

    def test_true_when_misattributed_present(self):
        report = {'verdict_counts': _counts(misattributed=1, ok=4)}
        assert _has_flagged_findings(report) is True

    def test_true_when_reverted_present(self):
        report = {'verdict_counts': _counts(reverted=1)}
        assert _has_flagged_findings(report) is True

    def test_true_when_commit_not_on_main_present(self):
        report = {'verdict_counts': _counts(commit_not_on_main=1)}
        assert _has_flagged_findings(report) is True

    def test_true_when_deliverable_absent_present(self):
        report = {'verdict_counts': _counts(deliverable_absent=1)}
        assert _has_flagged_findings(report) is True

    def test_missing_verdict_counts_key_defaults_false(self):
        """A malformed/absent verdict_counts key is treated as 'nothing
        flagged' rather than raising — this is a defensive CLI-exit-code
        helper, not a data-integrity check."""
        assert _has_flagged_findings({}) is False

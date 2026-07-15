"""Tests for audit_found_on_main_provenance.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py /
test_audit_duplicate_memories.py.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

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
commit_cites_task = _mod.commit_cites_task
classify = _mod.classify
GitFacts = _mod.GitFacts
build_audit_report = _mod.build_audit_report
apply_audit_annotations = _mod.apply_audit_annotations
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


class TestCommitCitesTask:
    """commit_cites_task(message, task_id) == task_id in extract_cited_task_ids(message)."""

    def test_true_when_cited(self):
        assert commit_cites_task('impl(50): add X', '50') is True

    def test_false_when_not_cited(self):
        assert commit_cites_task('chore: general cleanup', '50') is False

    def test_false_for_different_task_in_merge_commit(self):
        """A merge commit citing task/77 does not cite task 50."""
        message = 'Merge task/77 into main'
        assert commit_cites_task(message, '50') is False

    def test_false_when_word_boundary_would_be_violated(self):
        """task/3399 must not satisfy a commit_cites_task('...', '339') check."""
        assert commit_cites_task('Merge task/3399 into main', '339') is False


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

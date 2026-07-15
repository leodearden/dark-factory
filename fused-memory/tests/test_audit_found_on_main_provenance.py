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

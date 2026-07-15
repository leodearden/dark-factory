"""Tests for audit_found_on_main_provenance.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py /
test_audit_duplicate_memories.py.
"""
from __future__ import annotations

import importlib.util
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

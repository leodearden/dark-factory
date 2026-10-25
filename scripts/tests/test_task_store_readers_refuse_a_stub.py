"""Every tasks.db reader in scripts/ opens through ``connect_ro`` (task 5335).

One adoption check parametrised over the readers, rather than a copy in each
script's own suite. What a refusal looks like is
``scripts/tests/test_task_db_scan.py``'s business; all this pins is that each
reader routes its open through ``_task_db_scan.py::connect_ro`` rather than a
raw ``mode=ro`` connect, which would answer a stub with ``no such table``.

``audit_delivered_checks.load_open_dependents`` is pinned in its own suite,
because there the stub proves more: its ``no such table`` swallow must not read
a store with no tables at all as "nobody is blocked".
"""
from __future__ import annotations

import audit_combine_gate_marker_loss
import audit_delivered_checks
import audit_manifest_descriptor_drift
import audit_wiped_metadata_files
import census_tagger_debris
import pytest
import scan_provenance_note_log_leaks
import scan_task_toolcall_leaks
from _task_db_scan import TaskDbProblem, TaskDbUnreadable, tasks_db_path


def _reads_the_store_path(reader):
    return pytest.param(
        lambda root: reader(str(tasks_db_path(root))),
        id=f"{reader.__module__}.{reader.__name__}",
    )


def _reads_the_project_root(reader):
    return pytest.param(
        lambda root: reader(str(root)),
        id=f"{reader.__module__}.{reader.__name__}",
    )


@pytest.mark.parametrize("read_store", [
    _reads_the_store_path(audit_combine_gate_marker_loss.load_combine_targets),
    _reads_the_store_path(audit_delivered_checks.load_task_index),
    _reads_the_store_path(audit_manifest_descriptor_drift.load_task_store_scan),
    _reads_the_store_path(audit_wiped_metadata_files.load_task_records),
    _reads_the_store_path(census_tagger_debris.load_stamped_records),
    _reads_the_store_path(scan_provenance_note_log_leaks.scan_db),
    _reads_the_store_path(scan_task_toolcall_leaks.scan_db),
    _reads_the_project_root(census_tagger_debris.census_project),
])
def test_a_zero_byte_stub_is_refused_as_an_empty_stub(tmp_path, read_store):
    store = tasks_db_path(tmp_path)
    store.parent.mkdir(parents=True)
    store.write_bytes(b"")

    with pytest.raises(TaskDbUnreadable) as refused:
        read_store(tmp_path)

    assert refused.value.reason is TaskDbProblem.EMPTY_STUB

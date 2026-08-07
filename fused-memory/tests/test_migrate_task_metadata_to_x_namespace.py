"""Tests for scripts/migrate_task_metadata_to_x_namespace.py (task 3697).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_strip_leaked_control_keys.py, the precedent this migration follows.

This suite covers the script's PURE transform (``plan_x_namespace_migration``),
its write-payload builder (``build_update_payload``), its read-back proof
(``_verify_read_back``), its ``--keys`` safety guard
(``validate_migration_keys``), its pre-write snapshot (``write_backup``) and
its CLI contract. No network, no live DB: the live httpx/JSON-RPC plumbing is
reused wholesale from strip_leaked_control_keys.py and is exercised
operationally.

The payload-builder tests are load-bearing safety, not ceremony. Task 3083
carries a one-of-a-kind ~6 KB hand-curated evidence log in its `description`
and `details` columns, and ``update_task`` rewrites exactly the columns it is
handed — so a payload that grew a `description` key would silently destroy it.
``test_update_payload_carries_only_metadata_fields`` is what stops a future
edit from doing that.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from shared.task_metadata import parse_metadata

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'migrate_task_metadata_to_x_namespace.py'
)


def _load_module() -> types.ModuleType:
    """Load migrate_task_metadata_to_x_namespace.py from its file path."""
    mod_name = 'migrate_task_metadata_to_x_namespace'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
plan_x_namespace_migration = _mod.plan_x_namespace_migration
build_update_payload = _mod.build_update_payload
build_parser = _mod.build_parser
DEFAULT_KEYS = _mod.DEFAULT_KEYS
BACKEND_STRIPPED_KEYS = _mod.BACKEND_STRIPPED_KEYS
MigrationCollisionError = _mod.MigrationCollisionError
assert_write_accepted = _mod.assert_write_accepted
WriteRejectedError = _mod.WriteRejectedError
validate_migration_keys = _mod.validate_migration_keys
UnsafeMigrationKeyError = _mod.UnsafeMigrationKeyError
write_backup = _mod.write_backup
default_backup_path = _mod.default_backup_path
_verify_read_back = _mod._verify_read_back


# The six ad-hoc keys measured on task 3083's live blob (2026-08-06) that are
# x_-renameable: none has a code reader anywhere in orchestrator/,
# fused-memory/src, shared/src, escalation/, dashboard/, scripts/ or docs/.
# The seventh unknown_key on that blob, `last_blocked_at`, is deliberately
# absent here — it IS machine-read, and was promoted to Tier-A instead.
_TASK_3083_SHAPED_METADATA = {
    # --- legit keys that must survive untouched ---
    'source': 'reconciliation',
    'task_kind': 'normal',
    '_causation_id': '8d3af293-0000-0000-0000-000000000000',
    'branch_base_sha': 'deadbeef',
    'files': ['fused-memory/src/fused_memory/server/markup_tripwire.py'],
    'memory_hints': {'entities': ['A'], 'queries': ['q1']},
    'last_blocked_at': '2026-08-01T07:31:13.914220+00:00',
    # --- the six migration targets ---
    # Nested-object values with lists inside, matching the real blob's shape:
    # a value-preserving rename must carry these through byte-identically.
    'markup_tripwire_rejections_20260730': {
        'rejections': [{'tool': 'submit_task', 'count': 2}],
    },
    'markup_tripwire_rejections_20260730_burst3': {
        'rejections': [{'tool': 'add_memory', 'count': 1}, {'tool': 'submit_task', 'count': 3}],
    },
    'related_reify_memories': ['mem-1'],
    'related_reify_tasks': ['3141', '3083'],
    'origin_escalation': 'esc-markup-tripwire-3',
    'origin_reify_task': '2954',
}


# --- case 1: the rename itself, value preserved -----------------------------

def test_renames_target_keys_to_x_namespace_preserving_values():
    """Each target `k` becomes `x_k`; the value carries over by identity."""
    out, applied = plan_x_namespace_migration(_TASK_3083_SHAPED_METADATA, DEFAULT_KEYS)

    assert applied == sorted(DEFAULT_KEYS)
    for key in DEFAULT_KEYS:
        assert key not in out, f'old spelling {key!r} survived the migration'
        assert f'x_{key}' in out, f'x_{key} missing from migrated blob'
        # Identity, not merely equality: the transform must not deep-copy or
        # re-serialize a nested evidence record.
        assert out[f'x_{key}'] is _TASK_3083_SHAPED_METADATA[key]

    # The nested list-of-dicts value survives intact.
    assert out['x_markup_tripwire_rejections_20260730_burst3'] == {
        'rejections': [{'tool': 'add_memory', 'count': 1}, {'tool': 'submit_task', 'count': 3}],
    }


# --- case 2: sibling preservation (the DO-NOT-LOSE contract) ----------------

def test_every_non_target_key_survives_untouched():
    """Non-target keys are preserved with identical values.

    This is the sibling-preservation contract that the task description's
    `metadata_mode='merge'` instruction was protecting. We get it by
    construction (the full blob is written back), and this asserts it.
    """
    out, _ = plan_x_namespace_migration(_TASK_3083_SHAPED_METADATA, DEFAULT_KEYS)

    survivors = {k: v for k, v in out.items() if not k.startswith('x_')}
    expected = {
        k: v for k, v in _TASK_3083_SHAPED_METADATA.items() if k not in DEFAULT_KEYS
    }
    assert survivors == expected
    # No key is invented or lost: 18-key blob in, 18-key blob out.
    assert len(out) == len(_TASK_3083_SHAPED_METADATA)


def test_transform_does_not_mutate_the_caller_blob():
    """The input dict is left intact — the caller keeps a pre-write snapshot
    to diff the read-back against, so mutating it would destroy the evidence."""
    before = json.loads(json.dumps(_TASK_3083_SHAPED_METADATA))
    plan_x_namespace_migration(_TASK_3083_SHAPED_METADATA, DEFAULT_KEYS)
    assert before == _TASK_3083_SHAPED_METADATA


# --- case 3: idempotency ----------------------------------------------------

def test_migration_is_idempotent():
    """Feeding the output back in is a no-op with an EMPTY applied list.

    Safe to re-run after a partial failure — the operator never has to
    reason about whether a retry will double-apply.
    """
    once, applied_once = plan_x_namespace_migration(_TASK_3083_SHAPED_METADATA, DEFAULT_KEYS)
    twice, applied_twice = plan_x_namespace_migration(once, DEFAULT_KEYS)

    assert applied_twice == []
    assert twice == once
    assert applied_once != []


# --- case 4: absent target key is a silent no-op ----------------------------

def test_absent_target_key_is_a_no_op_not_an_error():
    """A partially-migrated row is recoverable: missing targets are skipped."""
    partial = {'source': 'x', 'origin_escalation': 'esc-1'}
    out, applied = plan_x_namespace_migration(partial, DEFAULT_KEYS)

    assert applied == ['origin_escalation']
    assert out == {'source': 'x', 'x_origin_escalation': 'esc-1'}


def test_empty_blob_is_a_no_op():
    out, applied = plan_x_namespace_migration({}, DEFAULT_KEYS)
    assert out == {}
    assert applied == []


# --- case 5: collision guard — fail loud, never silently merge --------------

def test_collision_with_differing_value_is_refused():
    """If `x_k` already exists with a DIFFERENT value, refuse rather than
    clobber either side. Two evidence records must never be silently merged
    into one, and neither is safe to discard without a human deciding."""
    colliding = {
        'origin_escalation': 'esc-markup-tripwire-3',
        'x_origin_escalation': 'esc-some-other-thing',
    }
    with pytest.raises(MigrationCollisionError) as excinfo:
        plan_x_namespace_migration(colliding, DEFAULT_KEYS)

    # The refusal names the offending key so the operator can act on it.
    assert 'origin_escalation' in str(excinfo.value)


def test_collision_with_identical_value_collapses_safely():
    """An `x_k` that already holds the SAME value is not a conflict: the
    stale old spelling is simply dropped. Nothing is lost, so refusing here
    would strand a half-migrated row with no recovery path."""
    already = {
        'origin_escalation': 'esc-markup-tripwire-3',
        'x_origin_escalation': 'esc-markup-tripwire-3',
    }
    out, applied = plan_x_namespace_migration(already, DEFAULT_KEYS)

    assert out == {'x_origin_escalation': 'esc-markup-tripwire-3'}
    assert applied == ['origin_escalation']


# --- case 6: the warning oracle ---------------------------------------------

def test_migrated_blob_emits_no_unknown_key_warning_for_any_target():
    """The binding user-observable signal, asserted per-key.

    `parse_metadata` is the deterministic oracle this task measures against
    (replacing the PRD's slow journalctl grep). After the transform, neither
    the six old spellings NOR their x_ forms may emit code=unknown_key.

    Asserted per-key rather than as a global zero: a caller's blob may
    legitimately carry other unrelated unknown keys, and this transform makes
    no promise about those.
    """
    out, _ = plan_x_namespace_migration(_TASK_3083_SHAPED_METADATA, DEFAULT_KEYS)

    _, warnings = parse_metadata(json.dumps(out), direction='write')
    unknown_key_fields = {w.field for w in warnings if w.code == 'unknown_key'}

    for key in DEFAULT_KEYS:
        assert key not in unknown_key_fields, (
            f'old spelling {key!r} still warns after migration'
        )
        assert f'x_{key}' not in unknown_key_fields, (
            f'x_{key} warns — the x_ prefix should suppress unknown_key'
        )


# --- case 7: the PAYLOAD GUARD (DO-NOT-LOSE enforcement) --------------------

def test_update_payload_carries_only_metadata_fields():
    """The update_task arguments carry ONLY {id, project_root, metadata,
    metadata_mode} — and metadata_mode is 'replace'.

    THIS IS THE TEST THAT PROTECTS TASK 3083'S ~6 KB EVIDENCE LOG.
    `update_task` rewrites exactly the columns it is given, so any extra key
    here (`description`, `details`, ...) would overwrite a one-of-a-kind
    hand-curated record with whatever this script happened to pass.
    """
    payload = build_update_payload('3083', '/home/leo/src/dark-factory', {'source': 'x'})

    assert set(payload) == {'id', 'project_root', 'metadata', 'metadata_mode'}
    assert payload['metadata_mode'] == 'replace'
    assert payload['id'] == '3083'
    assert payload['project_root'] == '/home/leo/src/dark-factory'


@pytest.mark.parametrize(
    'forbidden', ['description', 'details', 'title', 'prompt', 'status', 'append']
)
def test_update_payload_never_carries_a_content_column(forbidden):
    """Each column that would destroy content if rewritten, asserted absent
    by name — so a future edit adding one fails loudly and specifically."""
    payload = build_update_payload('3083', '/home/leo/src/dark-factory', {'source': 'x'})
    assert forbidden not in payload


def test_update_payload_metadata_round_trips_to_the_given_blob():
    """The metadata argument deserializes back to exactly the blob handed in
    — the script serializes with json.dumps, matching the
    strip_leaked_control_keys.py precedent's replace-mode write."""
    blob = {'source': 'x', 'x_origin_escalation': 'esc-1', 'files': ['a.py']}
    payload = build_update_payload('3083', '/home/leo/src/dark-factory', blob)

    serialized = payload['metadata']
    parsed = json.loads(serialized) if isinstance(serialized, str) else serialized
    assert parsed == blob


# --- case 8: default target list + CLI refuses a corpus-wide sweep ----------

def test_default_keys_are_exactly_the_six_measured_targets():
    """Pinned so a later edit cannot quietly widen the blast radius."""
    assert sorted(DEFAULT_KEYS) == sorted([
        'markup_tripwire_rejections_20260730',
        'markup_tripwire_rejections_20260730_burst3',
        'related_reify_memories',
        'related_reify_tasks',
        'origin_escalation',
        'origin_reify_task',
    ])
    # `last_blocked_at` is NOT a migration target — it was promoted to Tier-A
    # (task 3697 step-2) because the orchestrator writes and reads it.
    assert 'last_blocked_at' not in DEFAULT_KEYS


# --- tool-level rejection must not read as an accepted write ---------------

def test_done_provenance_rejection_is_detected():
    """The VERBATIM rejection observed against the live server on 2026-08-06.

    `update_task` refuses any metadata payload containing `done_provenance`
    (sqlite_task_backend.py, presence-only write-authority floor, checked
    BEFORE metadata_mode is resolved) — and returns that refusal inside a
    normal JSON-RPC success envelope. The shared client only raises on an
    envelope-level error, so without this assertion the script printed
    'write submitted' for a write that never happened.
    """
    observed = {
        'success': False,
        'error': 'done_provenance_via_update_task',
        'task_id': '3083',
        'hint': (
            'update_task cannot write metadata.done_provenance. Use '
            'set_task_status(status="done", done_provenance={...}) instead'
        ),
    }
    with pytest.raises(WriteRejectedError) as excinfo:
        assert_write_accepted(observed)

    message = str(excinfo.value)
    # The true cause and the remedy must both reach the operator.
    assert 'done_provenance_via_update_task' in message
    assert 'set_task_status' in message


def test_generic_error_shaped_reply_is_detected():
    """@mcp_tool_errors converts any escaping exception into
    {'error': ..., 'error_type': ...} — also a rejection, also invisible."""
    with pytest.raises(WriteRejectedError):
        assert_write_accepted({'error': 'boom', 'error_type': 'ValueError'})


def test_successful_write_shape_is_accepted():
    """The real success shape must NOT trip the guard."""
    assert_write_accepted({
        'id': '3083', 'message': 'Task 3083 updated',
        'updated': True, 'updated_task': {'id': '3083'},
    })


def test_non_dict_reply_is_not_treated_as_a_rejection():
    """Fail-safe: an unrecognized reply shape is left to the read-back diff
    rather than being misreported as a rejection."""
    assert_write_accepted('unexpected')
    assert_write_accepted(None)


# --- case 9: the READ-BACK PROOF, one case per failure branch ---------------
#
# `_verify_read_back` is the script's central safety claim — the CHANGELOG, the
# module docstring and docs/task-authoring.md §8 all advertise it as the thing
# that proves the write did exactly what was intended. If one of its checks
# were wrong (comparing against the wrong snapshot, an unreachable branch), a
# corrupted or partial write to task 3083's one-of-a-kind evidence blob would
# print "read-back verified" and this suite would stay green. It is a pure
# function over plain dicts, so every branch is cheap to pin.

_BEFORE_TASK = {
    'id': '3083',
    'status': 'done',
    'description': 'EVIDENCE LOG — 6269 bytes of hand-curated record',
    'details': 'THE ~14 KB EVIDENCE LOG — irreplaceable',
    'metadata': _TASK_3083_SHAPED_METADATA,
}


def _run_verify(mutate=None, extra_before=None, **after_task_overrides):
    """Drive `_verify_read_back` over a simulated write + read-back.

    `mutate` corrupts the read-back metadata (the write went wrong);
    `after_task_overrides` corrupts the read-back's other columns.
    Everything is deep-copied so cases cannot leak into one another.
    """
    before_task = json.loads(json.dumps(_BEFORE_TASK))
    if extra_before:
        before_task['metadata'].update(extra_before)
    before_meta = before_task['metadata']
    migrated, applied = plan_x_namespace_migration(before_meta, DEFAULT_KEYS)

    after_meta = json.loads(json.dumps(migrated))
    if mutate is not None:
        mutate(after_meta)
    after_task = json.loads(json.dumps(before_task))
    after_task['metadata'] = after_meta
    after_task.update(after_task_overrides)

    return _verify_read_back(
        before_meta=before_meta,
        expected_meta=migrated,
        applied=applied,
        before_task=before_task,
        after_task=after_task,
    )


def test_verify_read_back_passes_on_a_faithful_write():
    """The clean case: nothing to report. Without this the failure cases below
    could all be passing for the wrong reason (e.g. a check that always fires)."""
    problems, notes = _run_verify()
    assert problems == []
    assert notes == []


@pytest.mark.parametrize(
    'label, mutate, after_overrides, fragment',
    [
        # (a) the rename did not land
        ('a-missing', lambda m: m.pop('x_origin_escalation'), {},
         '(a) x_origin_escalation missing'),
        # (a) it landed but carrying the wrong value — the check must compare
        # against the PRE-WRITE snapshot, not against itself
        ('a-value-changed', lambda m: m.__setitem__('x_origin_escalation', 'esc-WRONG'), {},
         '(a) x_origin_escalation value changed'),
        # (b) both spellings coexist — precisely the outcome metadata_mode
        # 'merge' would have produced, and the reason replace-mode is used
        ('b-old-spelling', lambda m: m.__setitem__('origin_escalation', 'esc-markup-tripwire-3'), {},
         '(b) old spelling origin_escalation still present'),
        # (c) an untouched sibling was dropped by the write
        ('c-missing-sibling', lambda m: m.pop('branch_base_sha'), {},
         '(c) key-set drift'),
        # (c) the write invented a key
        ('c-extra-key', lambda m: m.__setitem__('surprise', 1), {},
         '(c) key-set drift'),
        # (c) an untouched sibling's VALUE changed — key sets still match, so
        # only the per-key value pass can catch this
        ('c-value-drift', lambda m: m.__setitem__('files', ['SOMETHING ELSE.py']), {},
         '(c) value drift on files'),
        # (d) the DO-NOT-LOSE columns
        ('d-details', None, {'details': 'clobbered'}, '(d) details CHANGED'),
        ('d-description', None, {'description': 'clobbered'}, '(d) description CHANGED'),
        # (e) a metadata-only write must never move status
        ('e-status', None, {'status': 'pending'}, '(e) status changed'),
    ],
)
def test_verify_read_back_catches_each_corruption(label, mutate, after_overrides, fragment):
    """Every branch fires on its own corruption, with a specific message."""
    problems, _ = _run_verify(mutate=mutate, **after_overrides)
    assert any(fragment in p for p in problems), (
        f'{label}: expected a problem containing {fragment!r}; got {problems}'
    )


def test_verify_read_back_reports_backend_stripped_control_keys_as_info():
    """A stored blob carrying a leaked `append` must not read as corruption.

    `SqliteTaskBackend.update_task` runs `_strip_reserved_control_keys` over the
    incoming payload for EVERY mode, replace included — so a target task that
    already carried a leaked control key (the task-2682/2735 defect class, in
    this same corpus) comes back one key lighter than what we sent. That is the
    backend doing the right thing. Reporting it as `(c) key-set drift` would
    tell the operator the write corrupted the blob, after the write already
    committed, on an irreplaceable evidence-log task.
    """
    problems, notes = _run_verify(
        extra_before={'append': True},
        mutate=lambda m: m.pop('append'),  # the backend's strip
    )

    assert problems == [], problems
    assert len(notes) == 1
    assert 'append' in notes[0]
    assert notes[0].startswith('(i)')


def test_backend_stripped_keys_match_the_backend_source_of_truth():
    """The literal in the script must not drift from the backend's frozenset.

    The script hard-codes the set to keep the heavy backend package off its
    import path; this is the anti-drift guard that buys that back.
    """
    from fused_memory.backends.sqlite_task_backend import (
        _RESERVED_METADATA_CONTROL_KEYS,
    )
    assert BACKEND_STRIPPED_KEYS == _RESERVED_METADATA_CONTROL_KEYS


# --- case 10: --keys safety validation --------------------------------------
#
# The module docstring's "verified by grep, no reader anywhere" argument covers
# DEFAULT_KEYS ONLY, but docs/task-authoring.md §8 points the future
# corpus-wide sweep at a per-task re-run with other --keys. `_verify_read_back`
# cannot cover this gap: it verifies INTENT (did the rename land), not SAFETY
# (should it have), so a mangle would be reported as "verified".

def test_default_keys_pass_validation():
    """The shipped defaults must not trip their own guard."""
    validate_migration_keys(DEFAULT_KEYS)


@pytest.mark.parametrize(
    'key, why',
    [
        # An easy slip when re-running against a partially-migrated task:
        # plan_x_namespace_migration prefixes unconditionally, so this would
        # mint x_x_origin_escalation AND drop the correct spelling.
        ('x_origin_escalation', 'already x_-namespaced'),
        # Typed TaskMetadata fields: renaming one blinds every live reader.
        ('files', 'a typed TaskMetadata field'),
        ('task_kind', 'a typed TaskMetadata field'),
        # Tier-A blessed — including the one this very task promoted.
        ('last_blocked_at', 'a Tier-A blessed key'),
    ],
)
def test_unsafe_keys_are_refused_before_any_write(key, why):
    with pytest.raises(UnsafeMigrationKeyError) as excinfo:
        validate_migration_keys([key])

    message = str(excinfo.value)
    assert key in message, f'the refusal must name the offending key: {message}'
    assert why in message


def test_validation_names_every_offender_not_just_the_first():
    """One run, one full list — so the operator fixes the invocation once."""
    with pytest.raises(UnsafeMigrationKeyError) as excinfo:
        validate_migration_keys(['files', 'origin_escalation', 'x_related_reify_tasks'])

    message = str(excinfo.value)
    assert 'files' in message
    assert 'x_related_reify_tasks' in message
    # ...and a legitimate ad-hoc target is NOT flagged.
    assert "'origin_escalation'" not in message


def test_cli_exposes_force_to_override_key_validation():
    """The escape hatch exists but must be opted into."""
    assert build_parser().parse_args(['--task-id', '3083']).force is False
    assert build_parser().parse_args(['--task-id', '3083', '--force']).force is True


# --- case 11: the durable pre-write snapshot --------------------------------

def test_write_backup_captures_the_whole_row_including_content_columns():
    """Metadata alone is not enough to recover from.

    If the write lands and the read-back then reports drift, this file is the
    only surviving copy of the original row — and what is actually irreplaceable
    on task 3083 is the evidence log in `description`/`details`, not the
    metadata blob. Both must be in the artifact.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'nested' / 'before.json'
        returned = write_backup(path, _BEFORE_TASK)

        assert returned == path
        assert path.exists(), 'the backup must be on disk BEFORE the write is attempted'
        restored = json.loads(path.read_text())
        assert restored['description'] == _BEFORE_TASK['description']
        assert restored['details'] == _BEFORE_TASK['details']
        assert restored['metadata'] == _BEFORE_TASK['metadata']
        assert restored['status'] == 'done'


def test_default_backup_path_is_task_scoped():
    """Two tasks must not overwrite each other's only recovery artifact."""
    assert default_backup_path('3083') != default_backup_path('3141')
    assert '3083' in str(default_backup_path('3083'))


def test_cli_backup_path_defaults_to_none_and_is_overridable():
    """Resolution to the task-scoped default happens in main_async, so the
    parser default stays None and an explicit path wins."""
    assert build_parser().parse_args(['--task-id', '3083']).backup_path is None
    args = build_parser().parse_args(['--task-id', '3083', '--backup-path', '/tmp/x.json'])
    assert args.backup_path == '/tmp/x.json'


def test_cli_requires_an_explicit_task_id():
    """No accidental corpus-wide sweep. These spellings live on other tasks
    too (origin_escalation on 19, related_reify_tasks on 8, origin_reify_task
    on 4, as of 2026-08-06), so a defaulted --task-id would be a foot-gun."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_defaults_to_dry_run_and_requires_apply_to_write():
    """A real write must be opted into explicitly."""
    args = build_parser().parse_args(['--task-id', '3083'])
    assert args.apply is False
    assert args.task_id == '3083'
    assert list(args.keys) == list(DEFAULT_KEYS)

    applied_args = build_parser().parse_args(['--task-id', '3083', '--apply'])
    assert applied_args.apply is True

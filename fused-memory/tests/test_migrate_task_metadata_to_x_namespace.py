"""Tests for scripts/migrate_task_metadata_to_x_namespace.py (task 3697).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_strip_leaked_control_keys.py, the precedent this migration follows.

This suite covers the script's PURE transform (``plan_x_namespace_migration``),
its write-payload builder (``build_update_payload``) and its CLI contract.
No network, no live DB: the live httpx/JSON-RPC plumbing is reused wholesale
from strip_leaked_control_keys.py and is exercised operationally.

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
MigrationCollisionError = _mod.MigrationCollisionError


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
    assert _TASK_3083_SHAPED_METADATA == before


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

"""Unit tests for fused_memory.middleware.deterministic_task_guard.

Step 1 (RED → step-2 GREEN): invariant matrix for deterministic_task_error.
Step 3 (RED → step-4 GREEN): before_done structural + filesystem checks.
Step 5 (RED → step-6 GREEN): inject_task_kind normalization.
"""

from __future__ import annotations

import json
import logging
import os

from fused_memory.middleware.deterministic_task_guard import (
    deterministic_task_error,
)

_DTG_LOGGER = 'fused_memory.middleware.deterministic_task_guard'

# ---------------------------------------------------------------------------
# Step-1: invariant matrix tests (no filesystem)
# ---------------------------------------------------------------------------


class TestDeterministicTaskErrorInvariantMatrix:
    """Validation matrix for kind × before_done × always_escalates invariants."""

    def test_deterministic_no_before_done_no_always_escalates_rejects(self):
        """task_kind='deterministic', no before_done, always_escalates absent → reject (no-op)."""
        result = deterministic_task_error('deterministic', {}, '/proj')
        assert result is not None
        assert 'error' in result
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        # Must name the no-op invariant
        assert 'ill-formed no-op' in msg or 'must run an action' in msg or 'always escalate' in msg

    def test_deterministic_always_escalates_false_rejects(self):
        """task_kind='deterministic', always_escalates=False, no before_done → reject."""
        result = deterministic_task_error('deterministic', {'always_escalates': False}, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_normal_with_before_done_rejects(self):
        """task_kind='normal', metadata has before_done → reject (before_done is only valid on deterministic)."""
        result = deterministic_task_error('normal', {'before_done': {'script': 'run.sh', 'timeout_secs': 60}}, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'before_done' in msg
        assert 'deterministic' in msg

    def test_deterministic_always_escalates_true_accepts(self):
        """task_kind='deterministic', always_escalates=True, no before_done → accept (pure gate)."""
        result = deterministic_task_error('deterministic', {'always_escalates': True}, '/proj')
        assert result is None

    def test_normal_no_before_done_accepts(self):
        """task_kind='normal', metadata=None → accept."""
        result = deterministic_task_error('normal', None, '/proj')
        assert result is None

    def test_normal_empty_metadata_accepts(self):
        """task_kind='normal', metadata={} → accept."""
        result = deterministic_task_error('normal', {}, '/proj')
        assert result is None

    def test_bogus_task_kind_rejects_with_enum_error(self):
        """task_kind='bogus' → reject (enum validation)."""
        result = deterministic_task_error('bogus', {}, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'bogus' in msg or 'normal' in msg or 'deterministic' in msg

    def test_metadata_as_json_string_accepted(self):
        """metadata can be a JSON string; None is extracted when 'before_done' absent."""
        meta_str = json.dumps({'foo': 'bar'})
        result = deterministic_task_error('normal', meta_str, '/proj')
        assert result is None

    def test_metadata_as_json_string_with_before_done_on_normal_rejects(self):
        """metadata as JSON string with before_done on normal task → reject."""
        meta_str = json.dumps({'before_done': {'script': 'x.sh', 'timeout_secs': 10}})
        result = deterministic_task_error('normal', meta_str, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_deterministic_always_escalates_non_bool_truthy_rejects(self):
        """always_escalates=1 (truthy int, not bool True) + no before_done → reject.

        Only the exact boolean True satisfies the gate; non-bool truthy values
        (int 1, string 'true', etc.) do NOT, preventing 'false' → True coercion bugs.
        """
        result = deterministic_task_error('deterministic', {'always_escalates': 1}, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_normal_with_always_escalates_true_rejects(self):
        """task_kind='normal', always_escalates=True → reject (mirrors the before_done rule)."""
        result = deterministic_task_error('normal', {'always_escalates': True}, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'always_escalates' in msg
        assert 'deterministic' in msg

    def test_non_dict_json_metadata_silently_coerced_to_empty(self):
        """metadata that parses to a non-dict JSON value (list, number) is silently coerced
        to an empty dict — the payload is lost rather than raising an error.

        This mirrors TaskInterceptor._inject_routing_override (an accepted project convention).
        Documented here so the behavior is explicit and observable in the test suite.
        """
        # A JSON array parses fine but is not a dict → coerced to {} → normal task
        # with no before_done → accepts (no validation errors)
        result = deterministic_task_error('normal', '["a", "b"]', '/proj')
        assert result is None

    def test_returns_hint_key_on_rejection(self):
        """Rejected results always include a 'hint' key with an actionable string."""
        result = deterministic_task_error('bogus', {}, '/proj')
        assert result is not None
        assert 'hint' in result
        assert isinstance(result['hint'], str)


# ---------------------------------------------------------------------------
# Step-3: before_done structural + filesystem checks
# ---------------------------------------------------------------------------


class TestDeterministicTaskErrorBeforeDoneStructural:
    """before_done structural and filesystem validation (requires tmp_path)."""

    def test_before_done_not_a_dict_rejects(self, tmp_path):
        """before_done is a string (not dict) → reject."""
        result = deterministic_task_error(
            'deterministic',
            {'before_done': 'not-a-dict'},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'before_done' in msg

    def test_before_done_list_rejects(self, tmp_path):
        """before_done is a list (not dict) → reject."""
        result = deterministic_task_error(
            'deterministic',
            {'before_done': ['a', 'b']},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_before_done_missing_script_rejects(self, tmp_path):
        """before_done={'timeout_secs': 60} with no 'script' → reject."""
        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'timeout_secs': 60}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'script' in msg

    def test_before_done_script_not_exist_rejects(self, tmp_path):
        """before_done.script points to a non-existent file under project_root → reject."""
        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'does_not_exist.sh', 'timeout_secs': 60}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'script' in msg or 'does_not_exist' in msg

    def test_before_done_script_exists_but_not_executable_rejects(self, tmp_path):
        """before_done.script exists but is NOT executable → reject."""
        script = tmp_path / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        # chmod 644 — not executable
        os.chmod(script, 0o644)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'run.sh', 'timeout_secs': 60}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'execut' in msg.lower() or 'script' in msg

    def test_before_done_script_escapes_project_root_rejects(self, tmp_path):
        """before_done.script='../escape.sh' resolving outside project_root → reject."""
        # Create the escape script outside tmp_path so it "exists" but is outside
        parent = tmp_path.parent
        escape_script = parent / 'escape.sh'
        escape_script.write_text('#!/bin/sh\n')
        os.chmod(escape_script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': '../escape.sh', 'timeout_secs': 60}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'script' in msg or 'project_root' in msg or 'contain' in msg.lower()

    def test_before_done_script_executable_missing_timeout_rejects(self, tmp_path):
        """before_done.script is executable under project_root but no timeout_secs → reject."""
        script = tmp_path / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'run.sh'}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'timeout_secs' in msg or 'timeout' in msg

    def test_before_done_script_executable_zero_timeout_rejects(self, tmp_path):
        """timeout_secs=0 (not positive) → reject."""
        script = tmp_path / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'run.sh', 'timeout_secs': 0}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'timeout_secs' in msg or 'positive' in msg

    def test_before_done_script_executable_negative_timeout_rejects(self, tmp_path):
        """timeout_secs=-1 → reject."""
        script = tmp_path / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'run.sh', 'timeout_secs': -1}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_before_done_script_executable_float_timeout_rejects(self, tmp_path):
        """timeout_secs=1.5 (float, not int) → reject."""
        script = tmp_path / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'run.sh', 'timeout_secs': 1.5}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        msg = result['error']
        assert 'timeout_secs' in msg or 'int' in msg

    def test_valid_before_done_executable_under_project_root_accepts(self, tmp_path):
        """before_done with real chmod+x script under project_root and timeout_secs=60 → None."""
        script = tmp_path / 'deploy.sh'
        script.write_text('#!/bin/sh\necho deploy\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'deploy.sh', 'timeout_secs': 60}},
            str(tmp_path),
        )
        assert result is None

    def test_valid_before_done_in_subdir_accepts(self, tmp_path):
        """Script in a subdirectory under project_root → accept."""
        subdir = tmp_path / 'scripts'
        subdir.mkdir()
        script = subdir / 'run.sh'
        script.write_text('#!/bin/sh\necho hi\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'script': 'scripts/run.sh', 'timeout_secs': 30}},
            str(tmp_path),
        )
        assert result is None

    def test_deterministic_always_escalates_and_before_done_both_valid(self, tmp_path):
        """both always_escalates=True and valid before_done → accept."""
        script = tmp_path / 'deploy.sh'
        script.write_text('#!/bin/sh\necho deploy\n')
        os.chmod(script, 0o755)

        result = deterministic_task_error(
            'deterministic',
            {
                'always_escalates': True,
                'before_done': {'script': 'deploy.sh', 'timeout_secs': 60},
            },
            str(tmp_path),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Step-5: inject_task_kind tests
# ---------------------------------------------------------------------------


class TestInjectTaskKind:
    """inject_task_kind normalization: None/JSON-str/dict → dict with task_kind set."""

    def _import_inject(self):
        from fused_memory.middleware.deterministic_task_guard import inject_task_kind
        return inject_task_kind

    def test_none_metadata_returns_dict_with_task_kind(self):
        """metadata=None, task_kind='normal' → {'task_kind': 'normal'}."""
        inject_task_kind = self._import_inject()
        result = inject_task_kind(None, 'normal')
        assert result == {'task_kind': 'normal'}
        assert isinstance(result, dict)

    def test_dict_metadata_preserves_keys_and_sets_task_kind(self):
        """metadata with before_done & always_escalates → preserved + task_kind set."""
        inject_task_kind = self._import_inject()
        meta = {'before_done': {'script': 'x.sh', 'timeout_secs': 60}, 'always_escalates': True}
        result = inject_task_kind(meta, 'deterministic')
        assert result['task_kind'] == 'deterministic'
        assert result['before_done'] == meta['before_done']
        assert result['always_escalates'] is True

    def test_json_string_metadata_parsed_and_task_kind_set(self):
        """metadata as JSON string → parsed dict + task_kind set."""
        inject_task_kind = self._import_inject()
        meta_str = json.dumps({'foo': 'bar', 'baz': 42})
        result = inject_task_kind(meta_str, 'normal')
        assert result['task_kind'] == 'normal'
        assert result['foo'] == 'bar'
        assert result['baz'] == 42

    def test_returns_dict_not_string(self):
        """Result is always a dict (not JSON string)."""
        inject_task_kind = self._import_inject()
        result = inject_task_kind(None, 'normal')
        assert isinstance(result, dict)

    def test_caller_dict_not_mutated(self):
        """Caller's dict is not mutated (shallow copy)."""
        inject_task_kind = self._import_inject()
        meta = {'existing': 'value'}
        original_meta = dict(meta)
        result = inject_task_kind(meta, 'normal')
        # Original not mutated
        assert meta == original_meta
        # Result has task_kind
        assert result['task_kind'] == 'normal'
        assert result is not meta

    def test_unparseable_string_returns_dict_with_task_kind(self):
        """Unparseable metadata string → fresh dict with task_kind."""
        inject_task_kind = self._import_inject()
        result = inject_task_kind('not valid json!!!', 'normal')
        assert result == {'task_kind': 'normal'}

    def test_deterministic_kind_set_correctly(self):
        """task_kind='deterministic' is set in the result dict."""
        inject_task_kind = self._import_inject()
        result = inject_task_kind({}, 'deterministic')
        assert result['task_kind'] == 'deterministic'

    def test_unparseable_string_warns_schema_warning_census_token(self, caplog):
        """Unparseable metadata string emits a task_metadata.schema_warning WARNING.

        I4: the whole-metadata discard must be observable, mirroring
        TaskInterceptor._inject_routing_override's discard WARNING. The
        return value is unchanged — {'task_kind': ...} remains the only
        sensible fallback for an unparseable string.
        """
        inject_task_kind = self._import_inject()
        with caplog.at_level(logging.WARNING, logger=_DTG_LOGGER):
            result = inject_task_kind('not valid json!!!', 'normal')
        assert result == {'task_kind': 'normal'}
        warns = [
            r for r in caplog.records
            if r.name == _DTG_LOGGER and r.levelno >= logging.WARNING
        ]
        assert any('task_metadata.schema_warning' in r.message for r in warns), (
            f'expected a task_metadata.schema_warning WARNING; got '
            f'{[r.message for r in warns]!r}'
        )

    def test_non_dict_json_value_warns_schema_warning_census_token(self, caplog):
        """A syntactically valid JSON value that isn't an object (a bare list)
        is also a whole-metadata discard and must warn identically."""
        inject_task_kind = self._import_inject()
        with caplog.at_level(logging.WARNING, logger=_DTG_LOGGER):
            result = inject_task_kind('[1,2,3]', 'normal')
        assert result == {'task_kind': 'normal'}
        warns = [
            r for r in caplog.records
            if r.name == _DTG_LOGGER and r.levelno >= logging.WARNING
        ]
        assert any('task_metadata.schema_warning' in r.message for r in warns), (
            f'expected a task_metadata.schema_warning WARNING; got '
            f'{[r.message for r in warns]!r}'
        )

    def test_valid_dict_metadata_emits_no_warning(self, caplog):
        """A well-formed dict is not a discard: no schema_warning WARNING, and
        curator-internal extras (spawned_from) survive alongside task_kind."""
        inject_task_kind = self._import_inject()
        with caplog.at_level(logging.WARNING, logger=_DTG_LOGGER):
            result = inject_task_kind({'foo': 'bar', 'spawned_from': 'x'}, 'deterministic')
        assert result == {'foo': 'bar', 'spawned_from': 'x', 'task_kind': 'deterministic'}
        warns = [
            r for r in caplog.records
            if r.name == _DTG_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, (
            f'valid dict metadata must not emit a WARNING; got '
            f'{[r.message for r in warns]!r}'
        )

    def test_empty_string_metadata_no_warning(self, caplog):
        """Amendment: an empty-string metadata is benign-absent, not a discard.

        Mirrors the pre-collapse ``isinstance(raw, str) and raw`` guard —
        '' must fall back to a fresh dict exactly like None, without
        emitting a task_metadata.schema_warning census line.
        """
        inject_task_kind = self._import_inject()
        with caplog.at_level(logging.WARNING, logger=_DTG_LOGGER):
            result = inject_task_kind('', 'normal')
        assert result == {'task_kind': 'normal'}
        warns = [
            r for r in caplog.records
            if r.name == _DTG_LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, (
            f'empty-string metadata must not emit a WARNING; got '
            f'{[r.message for r in warns]!r}'
        )


# ---------------------------------------------------------------------------
# δ (task 2337, step-1): metadata.milestone validation (deterministic_task_error)
# ---------------------------------------------------------------------------


class TestDeterministicTaskErrorMilestone:
    """metadata.milestone validation via deterministic_task_error (PRD §6.1).

    milestone is orthogonal to task_kind — allowed (and validated) on BOTH
    'normal' and 'deterministic' tasks. mode='dated' requires a
    datetime.fromisoformat-parseable 'at' (and no after_secs); mode='delayed'
    requires a positive int 'after_secs' (and no 'at'). Delegates to the
    shared.task_metadata.Milestone pydantic model.
    """

    # -- rejections ----------------------------------------------------

    def test_delayed_without_after_secs_rejects(self):
        """mode='delayed' with no after_secs → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'delayed'}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_dated_with_unparseable_at_rejects(self):
        """mode='dated' with an unparseable at ('not-a-date') → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'dated', 'at': 'not-a-date'}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_dated_without_at_rejects(self):
        """mode='dated' with no at → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'dated'}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_bogus_mode_rejects(self):
        """mode='bogus' (invalid enum) → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'bogus'}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_after_secs_zero_rejects(self):
        """after_secs=0 (non-positive) → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'delayed', 'after_secs': 0}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_after_secs_negative_rejects(self):
        """after_secs=-1 (non-positive) → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'delayed', 'after_secs': -1}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_after_secs_non_int_rejects(self):
        """after_secs='7d' (non-int) → reject."""
        result = deterministic_task_error(
            'normal', {'milestone': {'mode': 'delayed', 'after_secs': '7d'}}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_milestone_as_bare_string_rejects(self):
        """milestone as a non-dict (bare string) → TypeError-wrapped rejection."""
        result = deterministic_task_error(
            'normal', {'milestone': 'not-a-dict'}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_milestone_as_list_rejects(self):
        """milestone as a non-dict (list) → TypeError-wrapped rejection."""
        result = deterministic_task_error(
            'normal', {'milestone': ['a', 'b']}, '/proj',
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_malformed_milestone_via_json_string_metadata_rejects(self):
        """Malformed milestone supplied via a JSON-string metadata is also rejected."""
        meta_str = json.dumps({'milestone': {'mode': 'delayed'}})
        result = deterministic_task_error('normal', meta_str, '/proj')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    # -- acceptance / orthogonality guards ------------------------------

    def test_dated_valid_at_on_normal_task_accepts(self):
        """mode='dated' with a valid ISO at, on a NORMAL task → accept."""
        result = deterministic_task_error(
            'normal',
            {'milestone': {'mode': 'dated', 'at': '2026-08-01T00:00:00+00:00'}},
            '/proj',
        )
        assert result is None

    def test_delayed_valid_after_secs_on_deterministic_task_accepts(self):
        """mode='delayed' with after_secs=604800 on a DETERMINISTIC task carrying
        always_escalates=True → accept (milestone validated on both task_kinds,
        without breaking the deterministic accept path)."""
        result = deterministic_task_error(
            'deterministic',
            {
                'always_escalates': True,
                'milestone': {'mode': 'delayed', 'after_secs': 604800},
            },
            '/proj',
        )
        assert result is None


# ---------------------------------------------------------------------------
# δ (task 2337, step-3): predicate before_done validation (_validate_before_done)
# ---------------------------------------------------------------------------


class TestDeterministicTaskErrorPredicateBeforeDone:
    """before_done.kind='predicate' validation (PRD §5 dec 7,8).

    A predicate before_done has no unit to verify, so it FORBIDS a set
    target_unit and FORBIDS always_escalates=True (predicate escalation is
    conditional on rc != 0, not unconditional). Predicate structural checks
    (script exists + executable, positive timeout_secs) still apply.
    kind='deploy' (the default) is left byte-identical.
    """

    @staticmethod
    def _make_script(tmp_path, name='check.sh'):
        """tmp_path executable-script fixture, mirroring the before_done-structural tests."""
        script = tmp_path / name
        script.write_text('#!/bin/sh\nexit 0\n')
        os.chmod(script, 0o755)
        return script.name

    # -- rejections ----------------------------------------------------

    def test_predicate_with_target_unit_rejects(self, tmp_path):
        """kind='predicate' with a set target_unit → reject, msg mentions target_unit."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'kind': 'predicate',
                    'script': script_name,
                    'timeout_secs': 60,
                    'target_unit': 'merge.service',
                },
            },
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'target_unit' in result['error']

    def test_predicate_with_always_escalates_true_rejects(self, tmp_path):
        """kind='predicate' with top-level always_escalates=True → reject, msg mentions always_escalates."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'always_escalates': True,
                'before_done': {
                    'kind': 'predicate',
                    'script': script_name,
                    'timeout_secs': 60,
                },
            },
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'always_escalates' in result['error']

    def test_predicate_nonexistent_script_rejects(self, tmp_path):
        """Predicate structural checks still apply: non-existent script → reject."""
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'kind': 'predicate',
                    'script': 'does_not_exist.sh',
                    'timeout_secs': 60,
                },
            },
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_predicate_missing_timeout_rejects(self, tmp_path):
        """Predicate structural checks still apply: no timeout_secs → reject."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {'before_done': {'kind': 'predicate', 'script': script_name}},
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    # -- acceptance / deploy regression ---------------------------------

    def test_well_formed_predicate_accepts(self, tmp_path):
        """kind='predicate' with a valid script, timeout_secs, no target_unit,
        no always_escalates → accept."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'kind': 'predicate',
                    'script': script_name,
                    'timeout_secs': 120,
                },
            },
            str(tmp_path),
        )
        assert result is None

    def test_deploy_with_target_unit_still_accepts(self, tmp_path):
        """DEPLOY regression: default-kind before_done WITH target_unit still accepted
        (deploy allows target_unit — kind='deploy' behaviour stays byte-identical)."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'script': script_name,
                    'timeout_secs': 60,
                    'target_unit': 'foo.service',
                },
            },
            str(tmp_path),
        )
        assert result is None

    def test_deploy_with_always_escalates_true_still_accepts(self, tmp_path):
        """DEPLOY regression: deploy before_done + always_escalates=True
        (existing act-then-ask preset) still accepted."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'always_escalates': True,
                'before_done': {'script': script_name, 'timeout_secs': 60},
            },
            str(tmp_path),
        )
        assert result is None

    # -- kind enum validation --------------------------------------------

    def test_unknown_kind_rejects(self, tmp_path):
        """before_done.kind must be one of {'deploy', 'predicate'} — an
        unrecognized value (e.g. a typo) is rejected at the guard boundary
        rather than silently falling through as if it were 'deploy'."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'kind': 'gate',
                    'script': script_name,
                    'timeout_secs': 60,
                },
            },
            str(tmp_path),
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'kind' in result['error']

    def test_explicit_deploy_kind_accepts(self, tmp_path):
        """kind='deploy' set explicitly (not just omitted) is still accepted,
        including alongside target_unit (deploy allows it)."""
        script_name = self._make_script(tmp_path)
        result = deterministic_task_error(
            'deterministic',
            {
                'before_done': {
                    'kind': 'deploy',
                    'script': script_name,
                    'timeout_secs': 60,
                    'target_unit': 'foo.service',
                },
            },
            str(tmp_path),
        )
        assert result is None

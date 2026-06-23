"""Unit tests for fused_memory.middleware.deterministic_task_guard.

Step 1 (RED → step-2 GREEN): invariant matrix for deterministic_task_error.
Step 3 (RED → step-4 GREEN): before_done structural + filesystem checks.
Step 5 (RED → step-6 GREEN): inject_task_kind normalization.
"""

from __future__ import annotations

import json
import os

from fused_memory.middleware.deterministic_task_guard import (
    deterministic_task_error,
)

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

    def test_deterministic_before_done_present_accepts_without_filesystem_check(self):
        """task_kind='deterministic', before_done={any dict} → accepted (before_done presence alone valid at step-2).

        NOTE: After step-4 this case will be further validated for filesystem constraints.
        This specific subtest is only valid before filesystem checks are added.
        """
        # Not testing filesystem here — that's covered in step-3/step-4
        # Just confirm before_done dict presence is recognized (invariant-level only)
        pass  # covered by step-3 tests

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

    def test_deterministic_always_escalates_truthy_int_accepts(self):
        """always_escalates=1 (truthy) + no before_done → accept."""
        result = deterministic_task_error('deterministic', {'always_escalates': 1}, '/proj')
        assert result is None

    def test_returns_hint_key_on_rejection(self):
        """Rejected results may include a 'hint' key (optional but encouraged)."""
        result = deterministic_task_error('bogus', {}, '/proj')
        assert result is not None
        # hint is optional but if present must be a string
        if 'hint' in result:
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

"""Tests for shared.capability_manifest — the capability-manifest sidecar schema.

Built bottom-up in TDD order, following shared/tests/test_task_metadata.py's
convention (see plans/capability-delivered-checks-prd.md §Contract):
  - TestDeliveredCheck: the kind-conditional check descriptor in isolation.
  - TestManifestCapability / TestManifestTask: the capability + task containers.
  - TestCapabilityManifestDoc: the doc-level label uniqueness invariant.
  - TestLoader: load_capability_manifest / parse_capability_manifest, including
    the committed exemplar sidecar as a CI fixture.
  - TestDeliveredCheckMeta / TestMetadataRegistration: the
    metadata.delivered_checks registered sub-model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.capability_manifest import DeliveredCheck


class TestDeliveredCheck:
    def test_grep_check_constructs(self):
        check = DeliveredCheck(kind='grep', pattern='foo', expect='present')
        assert check.kind == 'grep'
        assert check.pattern == 'foo'
        assert check.expect == 'present'
        assert check.paths == []

    def test_grep_check_with_paths_constructs(self):
        check = DeliveredCheck(
            kind='grep', pattern='foo', expect='absent', paths=['shared/src/shared/']
        )
        assert check.paths == ['shared/src/shared/']

    def test_script_check_constructs(self):
        check = DeliveredCheck(kind='script', script='scripts/check_x.sh', timeout_secs=30)
        assert check.kind == 'script'
        assert check.script == 'scripts/check_x.sh'
        assert check.timeout_secs == 30
        assert check.args == []

    def test_script_check_with_args_constructs(self):
        check = DeliveredCheck(
            kind='script', script='scripts/check_x.sh', timeout_secs=30, args=['--flag']
        )
        assert check.args == ['--flag']

    def test_manual_check_constructs(self):
        check = DeliveredCheck(kind='manual')
        assert check.kind == 'manual'
        assert check.reason is None

    def test_manual_check_with_reason_constructs(self):
        check = DeliveredCheck(kind='manual', reason='judged by test fixtures')
        assert check.reason == 'judged by test fixtures'

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({'kind': 'grep', 'expect': 'present'}, id='grep_missing_pattern'),
            pytest.param(
                {'kind': 'grep', 'pattern': '', 'expect': 'present'}, id='grep_empty_pattern'
            ),
            pytest.param({'kind': 'grep', 'pattern': 'foo'}, id='grep_missing_expect'),
            pytest.param(
                {'kind': 'grep', 'pattern': 'foo', 'expect': 'present', 'script': 'x.sh'},
                id='grep_with_script',
            ),
            pytest.param(
                {'kind': 'grep', 'pattern': 'foo', 'expect': 'present', 'args': ['--flag']},
                id='grep_with_args',
            ),
            pytest.param(
                {'kind': 'grep', 'pattern': 'foo', 'expect': 'present', 'timeout_secs': 10},
                id='grep_with_timeout_secs',
            ),
            pytest.param(
                {'kind': 'grep', 'pattern': 'foo', 'expect': 'sideways'},
                id='grep_expect_not_in_vocab',
            ),
            pytest.param({'kind': 'script', 'timeout_secs': 30}, id='script_missing_script'),
            pytest.param(
                {'kind': 'script', 'script': 'scripts/x.sh'}, id='script_missing_timeout_secs'
            ),
            pytest.param(
                {'kind': 'script', 'script': 'scripts/x.sh', 'timeout_secs': 0},
                id='script_timeout_secs_zero',
            ),
            pytest.param(
                {'kind': 'script', 'script': 'scripts/x.sh', 'timeout_secs': -5},
                id='script_timeout_secs_negative',
            ),
            pytest.param(
                {
                    'kind': 'script',
                    'script': 'scripts/x.sh',
                    'timeout_secs': 30,
                    'pattern': 'foo',
                },
                id='script_with_pattern',
            ),
            pytest.param(
                {
                    'kind': 'script',
                    'script': 'scripts/x.sh',
                    'timeout_secs': 30,
                    'expect': 'present',
                },
                id='script_with_expect',
            ),
            pytest.param(
                {
                    'kind': 'script',
                    'script': 'scripts/x.sh',
                    'timeout_secs': 30,
                    'paths': ['a/'],
                },
                id='script_with_paths',
            ),
            pytest.param({'kind': 'manual', 'pattern': 'foo'}, id='manual_with_pattern'),
            pytest.param({'kind': 'manual', 'expect': 'present'}, id='manual_with_expect'),
            pytest.param({'kind': 'manual', 'paths': ['a/']}, id='manual_with_paths'),
            pytest.param({'kind': 'manual', 'script': 'scripts/x.sh'}, id='manual_with_script'),
            pytest.param({'kind': 'manual', 'args': ['--flag']}, id='manual_with_args'),
            pytest.param({'kind': 'manual', 'timeout_secs': 30}, id='manual_with_timeout_secs'),
            pytest.param({'kind': 'bogus'}, id='kind_not_in_vocab'),
        ],
    )
    def test_invalid_specs_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            DeliveredCheck(**kwargs)  # type: ignore[arg-type]

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheck(kind='manual', typo_field='oops')  # type: ignore[call-arg]

    def test_error_names_kind_and_field_for_missing_pattern(self):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='grep', expect='present')
        message = str(exc_info.value)
        assert 'grep' in message
        assert 'pattern' in message

    def test_error_names_kind_and_field_for_script_with_pattern(self):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='script', script='scripts/x.sh', timeout_secs=30, pattern='foo')
        message = str(exc_info.value)
        assert 'script' in message
        assert 'pattern' in message

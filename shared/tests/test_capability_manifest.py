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

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

import shared.task_metadata as task_metadata_module
from shared.capability_manifest import (
    CapabilityManifestDoc,
    DeliveredCheck,
    DeliveredCheckMeta,
    ManifestCapability,
    ManifestTask,
    load_capability_manifest,
    parse_capability_manifest,
)
from shared.task_metadata import parse_metadata


def _task_dict(label='α', **overrides):
    payload = {
        'label': label,
        'capabilities': [{'name': 'c', 'binding': 'b', 'verdict': 'PASS'}],
    }
    payload.update(overrides)
    return payload


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


class TestManifestCapability:
    def test_constructs_with_required_fields_only(self):
        cap = ManifestCapability(
            name='foo', binding='capability→producer (wired)', verdict='PASS'
        )
        assert cap.name == 'foo'
        assert cap.binding == 'capability→producer (wired)'
        assert cap.verdict == 'PASS'
        assert cap.delivered_check is None

    def test_constructs_with_delivered_check_instance(self):
        check = DeliveredCheck(kind='grep', pattern='foo', expect='present')
        cap = ManifestCapability(name='foo', binding='b', verdict='PASS', delivered_check=check)
        assert cap.delivered_check is check

    def test_delivered_check_coerces_from_dict(self):
        cap = ManifestCapability(
            name='foo',
            binding='b',
            verdict='PASS',
            delivered_check={  # type: ignore[arg-type]
                'kind': 'grep',
                'pattern': 'foo',
                'expect': 'present',
            },
        )
        assert isinstance(cap.delivered_check, DeliveredCheck)
        assert cap.delivered_check.pattern == 'foo'

    def test_verdict_fail_accepted(self):
        cap = ManifestCapability(name='foo', binding='b', verdict='FAIL')
        assert cap.verdict == 'FAIL'

    def test_verdict_outside_vocab_rejected(self):
        with pytest.raises(ValidationError):
            ManifestCapability(name='foo', binding='b', verdict='MAYBE')  # type: ignore[arg-type]

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ManifestCapability(name='', binding='b', verdict='PASS')

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            ManifestCapability(name='foo', binding='b', verdict='PASS', typo='x')  # type: ignore[call-arg]


class TestManifestTask:
    def test_constructs_with_all_fields(self):
        task = ManifestTask(
            label='α',
            task_id=2574,
            title='Shared capability-manifest sidecar schema',
            capabilities=[ManifestCapability(name='foo', binding='b', verdict='PASS')],
        )
        assert task.label == 'α'
        assert task.task_id == 2574
        assert task.title == 'Shared capability-manifest sidecar schema'
        assert len(task.capabilities) == 1
        assert isinstance(task.capabilities[0], ManifestCapability)

    def test_task_id_none_ok(self):
        task = ManifestTask(label='α', capabilities=[])
        assert task.task_id is None

    def test_title_omitted_ok(self):
        task = ManifestTask(label='α', capabilities=[])
        assert task.title is None

    def test_task_id_non_int_string_rejected(self):
        with pytest.raises(ValidationError):
            ManifestTask(label='α', task_id='not-an-int', capabilities=[])  # type: ignore[arg-type]

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ManifestTask(label='', capabilities=[])
        assert 'label' in str(exc_info.value)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            ManifestTask(label='α', capabilities=[], typo='x')  # type: ignore[call-arg]


class TestCapabilityManifestDoc:
    def test_constructs_with_valid_tasks(self):
        doc = CapabilityManifestDoc(
            prd='plans/example-prd.md',
            schema_version=1,
            tasks=[_task_dict('α'), _task_dict('β')],  # type: ignore[arg-type]
        )
        assert doc.prd == 'plans/example-prd.md'
        assert doc.schema_version == 1
        assert len(doc.tasks) == 2
        assert isinstance(doc.tasks[0], ManifestTask)
        assert isinstance(doc.tasks[0].capabilities[0], ManifestCapability)

    def test_empty_tasks_list_ok(self):
        doc = CapabilityManifestDoc(prd='plans/example-prd.md', schema_version=1, tasks=[])
        assert doc.tasks == []

    def test_duplicate_label_rejected_names_the_label(self):
        with pytest.raises(ValidationError) as exc_info:
            CapabilityManifestDoc(
                prd='plans/example-prd.md',
                schema_version=1,
                tasks=[_task_dict('α'), _task_dict('α')],  # type: ignore[arg-type]
            )
        assert 'α' in str(exc_info.value)

    def test_empty_label_rejected(self):
        with pytest.raises(ValidationError):
            CapabilityManifestDoc(
                prd='plans/example-prd.md',
                schema_version=1,
                tasks=[_task_dict('')],  # type: ignore[arg-type]
            )

    def test_schema_version_not_one_rejected(self):
        with pytest.raises(ValidationError):
            CapabilityManifestDoc(
                prd='plans/example-prd.md',
                schema_version=2,  # type: ignore[arg-type]
                tasks=[_task_dict('α')],  # type: ignore[arg-type]
            )

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError):
            CapabilityManifestDoc(
                prd='plans/example-prd.md',
                schema_version=1,
                tasks=[_task_dict('α')],
                typo='x',  # type: ignore[call-arg]
            )


class TestLoader:
    _VALID_YAML = """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "α"
    task_id: 1
    title: "Example task"
    capabilities:
      - name: "cap-one"
        binding: "capability→producer (wired)"
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: "foo"
          expect: present
"""

    _MALFORMED_YAML_DUP_LABEL = """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "α"
    capabilities: []
  - label: "α"
    capabilities: []
"""

    def test_parse_capability_manifest_from_dict(self):
        data = {
            'prd': 'plans/example-prd.md',
            'schema_version': 1,
            'tasks': [_task_dict('α')],
        }
        doc = parse_capability_manifest(data)
        assert isinstance(doc, CapabilityManifestDoc)
        assert doc.tasks[0].label == 'α'

    def test_load_valid_sidecar_round_trips(self, tmp_path):
        sidecar = tmp_path / 'example-prd.capability-manifest.yaml'
        sidecar.write_text(self._VALID_YAML)
        doc = load_capability_manifest(sidecar)
        assert isinstance(doc, CapabilityManifestDoc)
        assert doc.prd == 'plans/example-prd.md'
        assert doc.schema_version == 1
        assert len(doc.tasks) == 1
        task = doc.tasks[0]
        assert task.label == 'α'
        assert task.task_id == 1
        cap = task.capabilities[0]
        assert cap.name == 'cap-one'
        assert isinstance(cap.delivered_check, DeliveredCheck)
        assert cap.delivered_check.kind == 'grep'
        assert cap.delivered_check.pattern == 'foo'

    def test_load_capability_manifest_accepts_str_path(self, tmp_path):
        sidecar = tmp_path / 'example-prd.capability-manifest.yaml'
        sidecar.write_text(self._VALID_YAML)
        doc = load_capability_manifest(str(sidecar))
        assert doc.prd == 'plans/example-prd.md'

    def test_malformed_duplicate_label_raises_naming_the_entry(self, tmp_path):
        sidecar = tmp_path / 'bad.capability-manifest.yaml'
        sidecar.write_text(self._MALFORMED_YAML_DUP_LABEL)
        with pytest.raises(ValidationError) as exc_info:
            load_capability_manifest(sidecar)
        assert 'α' in str(exc_info.value)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_capability_manifest(tmp_path / 'does-not-exist.yaml')

    def test_malformed_yaml_syntax_raises(self, tmp_path):
        sidecar = tmp_path / 'bad-syntax.capability-manifest.yaml'
        sidecar.write_text('prd: [unterminated')
        with pytest.raises(yaml.YAMLError):
            load_capability_manifest(sidecar)

    def test_committed_exemplar_sidecar_validates(self):
        # CI fixture: this PRD's own committed exemplar sidecar (PRD
        # §Contract signal for task α — "this PRD's own committed exemplar
        # sidecar parses and validates in a CI test").
        repo_root = Path(__file__).resolve().parents[2]
        sidecar_path = (
            repo_root / 'plans' / 'capability-delivered-checks-prd.capability-manifest.yaml'
        )
        doc = load_capability_manifest(sidecar_path)
        assert doc.prd == 'plans/capability-delivered-checks-prd.md'
        assert doc.schema_version == 1
        labels = [task.label for task in doc.tasks]
        assert labels == ['α', 'β', 'γ', 'δ', 'ε', 'ζ']
        alpha = doc.tasks[0]
        assert alpha.task_id == 2574
        first_cap = alpha.capabilities[0]
        assert first_cap.delivered_check is not None
        assert first_cap.delivered_check.kind == 'grep'
        assert first_cap.delivered_check.expect == 'present'
        manual_caps = [
            cap
            for cap in alpha.capabilities
            if cap.delivered_check is not None and cap.delivered_check.kind == 'manual'
        ]
        assert len(manual_caps) == 1
        manual_check = manual_caps[0].delivered_check
        assert manual_check is not None
        assert manual_check.reason


class TestDeliveredCheckMeta:
    def test_grep_entry_with_name_constructs(self):
        check = DeliveredCheckMeta(name='cap-one', kind='grep', pattern='foo', expect='present')
        assert check.name == 'cap-one'
        assert check.kind == 'grep'
        assert check.pattern == 'foo'
        assert check.expect == 'present'

    def test_script_entry_with_name_constructs(self):
        check = DeliveredCheckMeta(
            name='cap-two', kind='script', script='scripts/x.sh', timeout_secs=30
        )
        assert check.name == 'cap-two'
        assert check.kind == 'script'
        assert check.script == 'scripts/x.sh'
        assert check.timeout_secs == 30

    def test_manual_kind_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='manual')  # type: ignore[arg-type]

    def test_missing_name_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(kind='grep', pattern='foo', expect='present')  # type: ignore[call-arg]

    def test_grep_missing_pattern_rejected(self):
        # Parity with DeliveredCheck's grep/script cross-field rules.
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='grep', expect='present')

    def test_grep_with_script_field_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(
                name='cap-one',
                kind='grep',
                pattern='foo',
                expect='present',
                script='scripts/x.sh',
            )

    def test_script_missing_timeout_secs_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='script', script='scripts/x.sh')

    def test_script_with_pattern_field_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(
                name='cap-one',
                kind='script',
                script='scripts/x.sh',
                timeout_secs=30,
                pattern='foo',
            )

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='manual', typo='x')  # type: ignore[call-arg]


class TestMetadataRegistration:
    def test_registered_under_delivered_checks_key(self):
        assert task_metadata_module._SUBMODEL_REGISTRY['delivered_checks'] is DeliveredCheckMeta

    def test_parse_metadata_write_enforce_accepts_typed_list_and_round_trips(self):
        grep_entry = {'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}
        script_entry = {
            'name': 'cap-two',
            'kind': 'script',
            'script': 'scripts/x.sh',
            'timeout_secs': 30,
        }
        model, warnings = parse_metadata(
            {'delivered_checks': [grep_entry, script_entry]},
            direction='write',
            enforce=True,
        )
        assert warnings == []
        slice_value = model.delivered_checks  # type: ignore[attr-defined]
        assert isinstance(slice_value, list)
        assert all(isinstance(item, DeliveredCheckMeta) for item in slice_value)
        assert [item.name for item in slice_value] == ['cap-one', 'cap-two']
        dumped = model.model_dump()['delivered_checks']
        assert all(not isinstance(item, BaseModel) for item in dumped)
        # Full dump includes every field (defaults included), not just the
        # sparse input — pins the round-tripped shape.
        assert dumped == [
            {
                'name': 'cap-one',
                'kind': 'grep',
                'pattern': 'foo',
                'expect': 'present',
                'paths': [],
                'script': None,
                'args': [],
                'timeout_secs': None,
            },
            {
                'name': 'cap-two',
                'kind': 'script',
                'pattern': None,
                'expect': None,
                'paths': [],
                'script': 'scripts/x.sh',
                'args': [],
                'timeout_secs': 30,
            },
        ]

    def test_no_unknown_key_warning_for_delivered_checks(self):
        grep_entry = {'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}
        _model, warnings = parse_metadata(
            {'delivered_checks': [grep_entry]}, direction='write', enforce=True
        )
        assert not any(w.code == 'unknown_key' for w in warnings)

    def test_manual_kind_in_metadata_read_warns(self):
        model, warnings = parse_metadata(
            {'delivered_checks': [{'name': 'cap-one', 'kind': 'manual'}]},
            direction='read',
        )
        assert len(warnings) == 1
        assert warnings[0].field == 'delivered_checks'
        assert warnings[0].code == 'invalid_submodel'
        assert model.model_dump()['delivered_checks'] == [{'name': 'cap-one', 'kind': 'manual'}]

    def test_manual_kind_in_metadata_write_enforce_raises(self):
        with pytest.raises(ValidationError):
            parse_metadata(
                {'delivered_checks': [{'name': 'cap-one', 'kind': 'manual'}]},
                direction='write',
                enforce=True,
            )

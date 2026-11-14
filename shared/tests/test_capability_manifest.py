"""Tests for shared.capability_manifest — the capability-manifest sidecar schema.

Built bottom-up in TDD order, following shared/tests/test_task_metadata.py's
convention (see plans/capability-delivered-checks-prd.md §Contract):
  - TestDeliveredCheck: the kind-conditional check descriptor in isolation.
  - TestManifestCapability / TestManifestTask: the capability + task containers.
  - TestCapabilityManifestDoc: the doc-level label uniqueness invariant.
  - TestLoader: load_capability_manifest / parse_capability_manifest.
  - TestManifestCorpusDiscovery / TestCheckManifest / TestIterScriptChecks /
    TestCheckScriptTarget / TestCommittedFileMode / TestMissingFromWorktreeMessage:
    the test-support module shared/tests/cross_suite_helpers/capability_manifest_corpus.py
    (task 3362) against planted and failing inputs.
  - TestSidecarPath: the PRD key → sidecar path rule, over examples (task 6582).
  - TestDeliveredCheckMeta / TestMechanicalCheckKinds / TestMetadataRegistration:
    the metadata.delivered_checks registered sub-model and the derived
    MECHANICAL_CHECK_KINDS vocabulary its consumers import.

Every test that reads the committed tree or the git index — the exemplar
sidecar, the checked-in corpus sweeps, the real-index modes — lives in the
checked-in corpus guard in tests/repo_guards/test_checked_in_capability_manifests.py,
because no shared cache key covers what it reads.
"""

from __future__ import annotations

import subprocess
from typing import get_args

import pytest
import yaml
from capability_manifest_corpus import (
    MANIFEST_SUFFIX,
    REPO_ROOT,
    AmbiguousIndexEntry,
    GitUnavailable,
    ScriptCheckRef,
    check_manifest,
    check_script_target,
    committed_file_mode,
    discover_manifests,
    iter_script_checks,
    missing_from_worktree_message,
)
from pydantic import BaseModel, ValidationError

import shared.capability_manifest as capability_manifest_module
import shared.task_metadata as task_metadata_module
from shared.capability_manifest import (
    CHECK_SUBJECT_FIELD,
    MECHANICAL_CHECK_KINDS,
    CapabilityManifestDoc,
    DeliveredCheck,
    DeliveredCheckMeta,
    ManifestCapability,
    ManifestTask,
    load_capability_manifest,
    parse_capability_manifest,
    sidecar_path,
)
from shared.task_metadata import parse_metadata
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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

    def test_path_check_present_constructs(self):
        check = DeliveredCheck(
            kind='path', expect='present', paths=['orchestrator/tests/test_x.py']
        )
        assert check.kind == 'path'
        assert check.expect == 'present'
        assert check.paths == ['orchestrator/tests/test_x.py']
        assert check.pattern is None
        assert check.script is None

    def test_path_check_absent_constructs(self):
        check = DeliveredCheck(kind='path', expect='absent', paths=['legacy/dead_module.py'])
        assert check.expect == 'absent'
        assert check.paths == ['legacy/dead_module.py']

    def test_path_check_with_several_paths_constructs(self):
        check = DeliveredCheck(
            kind='path', expect='present', paths=['a/one.py', 'b/two.py', 'c/three.py']
        )
        assert check.paths == ['a/one.py', 'b/two.py', 'c/three.py']

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
            pytest.param(
                {
                    'kind': 'grep',
                    'pattern': 'foo',
                    'expect': 'present',
                    'reason': 'nope',
                },
                id='grep_with_reason',
            ),
            pytest.param(
                {
                    'kind': 'script',
                    'script': 'scripts/x.sh',
                    'timeout_secs': 30,
                    'reason': 'nope',
                },
                id='script_with_reason',
            ),
            pytest.param({'kind': 'path', 'paths': ['a/one.py']}, id='path_missing_expect'),
            pytest.param({'kind': 'path', 'expect': 'present'}, id='path_missing_paths'),
            pytest.param(
                {'kind': 'path', 'expect': 'present', 'paths': []}, id='path_empty_paths'
            ),
            pytest.param(
                {'kind': 'path', 'expect': 'sideways', 'paths': ['a/one.py']},
                id='path_expect_not_in_vocab',
            ),
            pytest.param(
                {'kind': 'path', 'expect': 'present', 'paths': ['a/one.py'], 'pattern': 'foo'},
                id='path_with_pattern',
            ),
            pytest.param(
                {
                    'kind': 'path',
                    'expect': 'present',
                    'paths': ['a/one.py'],
                    'script': 'scripts/x.sh',
                },
                id='path_with_script',
            ),
            pytest.param(
                {
                    'kind': 'path',
                    'expect': 'present',
                    'paths': ['a/one.py'],
                    'args': ['--flag'],
                },
                id='path_with_args',
            ),
            pytest.param(
                {
                    'kind': 'path',
                    'expect': 'present',
                    'paths': ['a/one.py'],
                    'timeout_secs': 30,
                },
                id='path_with_timeout_secs',
            ),
            pytest.param(
                {
                    'kind': 'path',
                    'expect': 'present',
                    'paths': ['a/one.py'],
                    'reason': 'nope',
                },
                id='path_with_reason',
            ),
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

    def test_error_names_kind_and_field_for_grep_with_reason(self):
        # reason is documented as manual-only free text; grep/script must
        # reject it explicitly rather than silently accepting it.
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='grep', pattern='foo', expect='present', reason='nope')
        message = str(exc_info.value)
        assert 'grep' in message
        assert 'reason' in message

    def test_error_names_kind_and_field_for_path_missing_paths(self):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='path', expect='present')
        message = str(exc_info.value)
        assert 'path' in message
        assert 'paths' in message

    def test_error_names_kind_and_field_for_path_with_pattern(self):
        # A path check asserts existence, never content: naming `pattern`
        # alongside kind='path' is the exact grep/path confusion this kind
        # exists to prevent, so the rejection must name both.
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='path', expect='present', paths=['a/one.py'], pattern='foo')
        message = str(exc_info.value)
        assert 'path' in message
        assert 'pattern' in message


class TestPathCheckHygiene:
    """kind='path' entries must be repo-relative, non-empty, and '..'-free.

    Stricter than kind='grep' on the SAME field, and deliberately so: for
    grep, `paths` merely NARROWS a search, so a bad entry degrades to a
    wider-or-empty scope. For kind='path' the entry IS the assertion, and
    a pathspec git cannot resolve inside the repository exits 128, which
    the runner maps to ERRORED — a fail-safe wait with no streak bump and
    no escalation, i.e. a SILENT INDEFINITE HOLD on every dependent. One
    typo'd leading slash would wedge a dependent forever while emitting
    nothing a human would ever see, so the descriptor is refused loudly at
    authoring time instead.
    """

    @pytest.mark.parametrize(
        'bad_path',
        [
            pytest.param('/etc/passwd', id='absolute'),
            pytest.param('/orchestrator/tests/test_x.py', id='absolute_repo_shaped'),
            pytest.param('../outside.py', id='parent_segment_leading'),
            pytest.param('orchestrator/../../outside.py', id='parent_segment_interior'),
            pytest.param('', id='empty'),
            pytest.param('   ', id='whitespace_only'),
        ],
    )
    def test_delivered_check_rejects_bad_path(self, bad_path):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(kind='path', expect='present', paths=[bad_path])
        assert repr(bad_path) in str(exc_info.value)

    @pytest.mark.parametrize(
        'bad_path',
        [
            pytest.param('/etc/passwd', id='absolute'),
            pytest.param('../outside.py', id='parent_segment_leading'),
            pytest.param('', id='empty'),
        ],
    )
    def test_delivered_check_meta_rejects_bad_path(self, bad_path):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheckMeta(name='cap-one', kind='path', expect='present', paths=[bad_path])
        assert repr(bad_path) in str(exc_info.value)

    def test_bad_entry_alongside_good_entries_is_still_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheck(
                kind='path', expect='present', paths=['a/one.py', '/etc/passwd', 'b/two.py']
            )
        assert repr('/etc/passwd') in str(exc_info.value)

    @pytest.mark.parametrize(
        'good_path',
        [
            pytest.param(
                'orchestrator/tests/test_workflow_merge_gating_strand.py', id='nested_file'
            ),
            pytest.param('README.md', id='repo_root_file'),
            pytest.param('shared/src/shared/', id='trailing_slash_directory'),
            pytest.param('scripts/audit_combine_gate_marker_loss.py', id='script_path'),
        ],
    )
    def test_ordinary_repo_relative_paths_are_still_accepted(self, good_path):
        # Positive control: the rule must not be satisfiable by rejecting
        # everything.
        check = DeliveredCheck(kind='path', expect='present', paths=[good_path])
        assert check.paths == [good_path]

    def test_grep_paths_are_not_subject_to_the_stricter_rule(self):
        # The asymmetry is the point: for grep, `paths` only narrows a
        # search, so it keeps its existing (unvalidated) latitude.
        check = DeliveredCheck(kind='grep', pattern='foo', expect='present', paths=['../x/'])
        assert check.paths == ['../x/']



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

    def test_verdict_open_accepted(self):
        # OPEN means the binding is deliberately deferred INTO the leaf that
        # owns it — it is neither a green G3 binding (PASS) nor a
        # queue-blocking FAIL.
        cap = ManifestCapability(name='foo', binding='b', verdict='OPEN')
        assert cap.verdict == 'OPEN'

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

    def test_note_accepted(self):
        note = (
            'SPLIT 2026-08-19. The original gamma row was one task across four '
            'servers; this leaf carries fused-memory.'
        )
        task = ManifestTask(label='α', capabilities=[], note=note)
        assert task.note == note

    def test_note_omitted_defaults_none(self):
        task = ManifestTask(label='α', capabilities=[])
        assert task.note is None

    def test_task_id_non_int_string_rejected(self):
        with pytest.raises(ValidationError):
            ManifestTask(label='α', task_id='not-an-int', capabilities=[])  # type: ignore[arg-type]

    def test_external_task_id_accepted_in_canonical_form(self):
        """A block whose producer lives in ANOTHER project's registry.

        The value is the repo's canonical qualified form
        (``"project_id:task_id"``, docs/task-authoring.md §3.2). An
        already-canonical value round-trips unchanged, which is what makes
        the normalisation in ``_check_producer_binding`` idempotent — see
        ``test_external_task_id_surrounding_whitespace_normalised`` for the
        non-canonical input it exists for.
        """
        task = ManifestTask(label='η', external_task_id='reify:5613', capabilities=[])
        assert task.external_task_id == 'reify:5613'
        assert task.task_id is None

    def test_external_task_id_omitted_defaults_none(self):
        task = ManifestTask(label='α', capabilities=[])
        assert task.external_task_id is None

    def test_external_task_id_loads_through_parse_capability_manifest(self):
        doc = parse_capability_manifest(
            {
                'prd': 'plans/example-prd.md',
                'schema_version': 1,
                'tasks': [_task_dict('η', external_task_id='reify:5613')],
            }
        )
        assert doc.tasks[0].external_task_id == 'reify:5613'
        assert doc.tasks[0].task_id is None

    def test_both_task_id_and_external_task_id_rejected(self):
        """A block binds exactly one producer, in exactly one registry.

        ``manifest_stamping`` step 4 skips a batch label whose block is
        bound to a foreign producer, and its re-validation of the stamped
        sidecar before writing it is the backstop: a stamp that slipped past
        the skip is refused rather than written alongside the
        ``external_task_id``. Failing loudly at load is the point — the
        corpus sweep also turns a hand-made violation into a red CI signal
        naming the file.
        """
        with pytest.raises(ValidationError) as exc_info:
            ManifestTask(label='η', task_id=5613, external_task_id='reify:5613', capabilities=[])
        message = str(exc_info.value)
        assert 'η' in message
        assert 'reify:5613' in message

    @pytest.mark.parametrize(
        'value',
        [
            pytest.param('reify', id='no-colon'),
            pytest.param('reify:', id='empty-task-id'),
            pytest.param(':5613', id='empty-project-id'),
            pytest.param('a:b:c', id='three-parts'),
            pytest.param('  ', id='blank'),
            pytest.param('', id='empty'),
        ],
    )
    def test_malformed_external_task_id_rejected(self, value):
        """Structural form is delegated to ``ExternalDep.parse``, not re-implemented."""
        with pytest.raises(ValidationError) as exc_info:
            ManifestTask(label='η', external_task_id=value, capabilities=[])
        message = str(exc_info.value)
        # The message names the offending value AND the label, so a
        # corpus-sweep failure is self-locating without opening the file.
        assert repr(value) in message
        assert 'η' in message

    def test_external_task_id_surrounding_whitespace_normalised(self):
        """Whitespace the shared parser tolerates is ACCEPTED and normalised away.

        ``ExternalDep.parse`` strips before splitting, so this is a
        well-formed value and validation must not reject it. But it is NOT
        stored verbatim: every consumer treats this field as an opaque key
        (the live-corpus test compares it with ``==``, and the docstring
        promises the same spelling as ``metadata.external_deps``), so a
        padded value would make any join against an ``external_deps``
        entry silently miss. The model stores ``ExternalDep.render()``'s
        canonical spelling instead, which is the one form callers may rely
        on.
        """
        task = ManifestTask(label='η', external_task_id=' reify:5613 ', capabilities=[])
        assert task.external_task_id == 'reify:5613'

    def test_external_task_id_normalisation_is_idempotent(self):
        """Re-validating a stored value is a no-op, so a round-trip is stable.

        ``manifest_stamping``'s write-back re-parses and validates the
        stamped sidecar and the corpus sweep re-loads it; if normalisation
        were not idempotent, the same value would validate differently on
        each pass.
        """
        once = ManifestTask(label='η', external_task_id=' reify:5613 ', capabilities=[])
        twice = ManifestTask(
            label='η', external_task_id=once.external_task_id, capabilities=[]
        )
        assert twice.external_task_id == once.external_task_id == 'reify:5613'

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

    def test_check_labels_rejects_empty_label_reached_directly(self):
        # ManifestTask.label already enforces min_length=1 at the field
        # level, so an empty label can never reach CapabilityManifestDoc's
        # own defense-in-depth `if not task.label` branch via normal (dict)
        # construction — model_construct() bypasses ManifestTask's field
        # validators to actually exercise that doc-level branch.
        bad_task = ManifestTask.model_construct(label='', capabilities=[])
        with pytest.raises(ValidationError) as exc_info:
            CapabilityManifestDoc(
                prd='plans/example-prd.md',
                schema_version=1,
                tasks=[bad_task],
            )
        assert 'task label must be non-empty' in str(exc_info.value)

    @staticmethod
    def _doc_with_signal_leaves(signal_leaves):
        return parse_capability_manifest(
            {
                'prd': 'plans/example-prd.md',
                'schema_version': 1,
                'tasks': [_task_dict('α'), _task_dict('β')],
                'signal_leaves': signal_leaves,
            }
        )

    def test_signal_leaves_naming_known_labels_load(self):
        assert self._doc_with_signal_leaves(['β']).signal_leaves == ['β']
        assert self._doc_with_signal_leaves(['β', 'α']).signal_leaves == ['β', 'α']

    def test_signal_leaves_defaults_to_none(self):
        doc = parse_capability_manifest(
            {
                'prd': 'plans/example-prd.md',
                'schema_version': 1,
                'tasks': [_task_dict('α')],
            }
        )
        assert doc.signal_leaves is None

    def test_signal_leaves_unknown_label_raises_naming_it_and_the_known_labels(self):
        with pytest.raises(ValidationError) as exc_info:
            self._doc_with_signal_leaves(['ω'])
        message = str(exc_info.value)
        assert 'ω' in message
        assert 'α' in message
        assert 'β' in message

    def test_signal_leaves_mixing_known_and_unknown_names_the_unknown_one(self):
        with pytest.raises(ValidationError) as exc_info:
            self._doc_with_signal_leaves(['α', 'ω'])
        assert "'ω'" in str(exc_info.value)

    def test_empty_signal_leaves_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            self._doc_with_signal_leaves([])
        message = str(exc_info.value)
        assert 'signal_leaves' in message
        assert 'non-empty' in message


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

    #: A task block carrying task-level ``note:`` provenance — the shape the
    #: real sidecars use, so this pins the YAML path and not just kwargs
    #: construction.
    _VALID_YAML_WITH_NOTE = """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "γ1"
    task_id: 1
    title: "Example task"
    note: "SPLIT 2026-08-19. The original γ row was one task across four servers; this leaf carries fused-memory."
    capabilities:
      - name: "cap-one"
        binding: "capability→producer (wired)"
        verdict: OPEN
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

    def test_load_sidecar_with_task_level_note(self, tmp_path):
        # A LOAD test, deliberately not named "round_trips": it proves the
        # YAML path decodes `note:` into the model, nothing more. The claim
        # ManifestTask's docstring actually rests on — that a declared field
        # survives manifest_stamping's write-back — can only be asserted on
        # the stamping side, in
        # fused-memory/tests/test_manifest_stamping.py, which is outside this
        # task's lock scope. Filed as tkt_0RSNVJT1ZNWKS5Y7BM5F7A2QAD
        # (follow-up from task 4471); the round-trip name is reserved for it.
        sidecar = tmp_path / 'example-prd.capability-manifest.yaml'
        sidecar.write_text(self._VALID_YAML_WITH_NOTE)
        doc = load_capability_manifest(sidecar)
        task = doc.tasks[0]
        assert task.label == 'γ1'
        assert task.note is not None
        assert task.note.startswith('SPLIT 2026-08-19.')
        assert 'this leaf carries fused-memory' in task.note
        assert task.capabilities[0].verdict == 'OPEN'

    #: The minimal sidecar a legacy, chain-only PRD gets (plans/dashboard-prd-lanes-prd.md
    #: §9 ε): one hand-bound block, no capabilities, and its signal leaf named.
    _MINIMAL_SIGNAL_LEAF_YAML = """\
prd: plans/legacy-prd.md
schema_version: 1
signal_leaves:
  - "ζ"
tasks:
  - label: "ζ"
    task_id: 1234
    capabilities: []
"""

    def test_minimal_sidecar_with_only_the_signal_leaf_block_loads(self, tmp_path):
        sidecar = tmp_path / 'legacy-prd.capability-manifest.yaml'
        sidecar.write_text(self._MINIMAL_SIGNAL_LEAF_YAML, encoding='utf-8')
        doc = load_capability_manifest(sidecar)
        assert doc.signal_leaves == ['ζ']
        assert doc.tasks[0].task_id == 1234
        assert doc.tasks[0].capabilities == []

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


class TestManifestCorpusDiscovery:
    """discover_manifests() — git-ls-files-based corpus discovery.

    Not Path.rglob: see capability_manifest_corpus.py's module docstring for
    the measurement (30 tracked sidecars via git ls-files, 0 of them under
    .worktrees/, vs 3793 extra sidecars an rglob from the main checkout would
    sweep in from other in-flight tasks' gitignored worktree checkouts).
    """

    def test_returns_none_for_a_non_checkout_directory(self, tmp_path):
        assert discover_manifests(tmp_path) is None

    def test_returns_none_when_git_binary_is_missing(self, tmp_path, monkeypatch):
        # subprocess.run raises OSError (FileNotFoundError, not a non-zero
        # exit) when the git binary itself can't be found. The corpus guard
        # runs discover_manifests() or [] at *module import time*, so an
        # uncaught exception here would take down collection of that entire
        # test module, not just its corpus-sweep classes.
        def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError('git')

        monkeypatch.setattr(subprocess, 'run', _raise)
        assert discover_manifests(tmp_path) is None

    def test_returns_none_when_git_invocation_times_out(self, tmp_path, monkeypatch):
        # A wedged git (e.g. index.lock contention) must fail fast via the
        # explicit timeout=, not hang the test run indefinitely.
        def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd='git', timeout=30)

        monkeypatch.setattr(subprocess, 'run', _raise)
        assert discover_manifests(tmp_path) is None


class TestCheckManifest:
    """check_manifest(path) — the file-attributed reporting wrapper.

    Each case writes a single inline sidecar to tmp_path (matching
    TestLoader._VALID_YAML / _MALFORMED_YAML_DUP_LABEL's convention — no
    committed bad-fixture files) and drives one of check_manifest's failure
    modes. This is the mutation proof the checked-in corpus guard in
    tests/repo_guards/test_checked_in_capability_manifests.py rests its
    corpus sweep on.
    """

    _VALID_YAML = """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-one"
        binding: "capability→producer (wired)"
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: "foo"
          expect: present
"""

    def test_valid_sidecar_returns_none(self, tmp_path):
        sidecar = tmp_path / 'ok.capability-manifest.yaml'
        sidecar.write_text(self._VALID_YAML)
        assert check_manifest(sidecar) is None

    def test_unknown_capability_key_names_file_and_key(self, tmp_path):
        sidecar = tmp_path / 'bad-cap-key.capability-manifest.yaml'
        sidecar.write_text(
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-one"
        binding: "b"
        verdict: PASS
        verdcit: PASS
"""
        )
        message = check_manifest(sidecar)
        assert message is not None
        assert 'bad-cap-key.capability-manifest.yaml' in message
        assert 'verdcit' in message

    def test_unknown_delivered_check_key_names_file_and_key(self, tmp_path):
        sidecar = tmp_path / 'bad-check-key.capability-manifest.yaml'
        sidecar.write_text(
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-one"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: "foo"
          expect: present
          bogus_field: x
"""
        )
        message = check_manifest(sidecar)
        assert message is not None
        assert 'bad-check-key.capability-manifest.yaml' in message
        assert 'bogus_field' in message

    def test_broken_yaml_syntax_names_file_and_is_a_yaml_error(self, tmp_path):
        sidecar = tmp_path / 'bad-syntax.capability-manifest.yaml'
        sidecar.write_text('prd: [unterminated')
        message = check_manifest(sidecar)
        assert message is not None
        assert 'bad-syntax.capability-manifest.yaml' in message
        assert 'YAML' in message

    def test_duplicate_task_label_names_file_and_label(self, tmp_path):
        sidecar = tmp_path / 'bad-dup-label.capability-manifest.yaml'
        sidecar.write_text(
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities: []
  - label: "alpha"
    capabilities: []
"""
        )
        message = check_manifest(sidecar)
        assert message is not None
        assert 'bad-dup-label.capability-manifest.yaml' in message
        assert 'alpha' in message

    def test_unsupported_schema_version_names_file(self, tmp_path):
        sidecar = tmp_path / 'bad-schema-version.capability-manifest.yaml'
        sidecar.write_text(
            """\
prd: plans/example-prd.md
schema_version: 2
tasks: []
"""
        )
        message = check_manifest(sidecar)
        assert message is not None
        assert 'bad-schema-version.capability-manifest.yaml' in message

    def test_missing_file_propagates_file_not_found(self, tmp_path):
        # A missing file is a caller bug, not manifest drift — check_manifest
        # must not silently absorb it into a reported "problem" string.
        with pytest.raises(FileNotFoundError):
            check_manifest(tmp_path / 'does-not-exist.capability-manifest.yaml')


class TestIterScriptChecks:
    """iter_script_checks(path) — extract one ref per `kind: script` check.

    The extraction layer the script-target sweep in
    tests/repo_guards/test_checked_in_capability_manifests.py is
    parametrized over. Follows TestCheckManifest's convention: each case
    writes one inline sidecar to tmp_path (no committed bad-fixture files)
    and drives one extraction property.
    """

    def _write(self, tmp_path, name, body):
        sidecar = tmp_path / f'{name}{MANIFEST_SUFFIX}'
        sidecar.write_text(body)
        return sidecar

    def test_grep_only_sidecar_yields_nothing(self, tmp_path):
        sidecar = self._write(
            tmp_path,
            'grep-only',
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-one"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: "foo"
          expect: present
""",
        )
        assert iter_script_checks(sidecar) == []

    def test_script_check_is_fully_attributed(self, tmp_path):
        sidecar = self._write(
            tmp_path,
            'one-script',
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "δ"
    capabilities:
      - name: "cap-scripted"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: script
          script: scripts/gc_agent_transcripts.py
          args:
            - --check
          timeout_secs: 60
""",
        )
        refs = iter_script_checks(sidecar)
        assert refs == [
            ScriptCheckRef(
                manifest=sidecar,
                task_label='δ',
                capability='cap-scripted',
                script='scripts/gc_agent_transcripts.py',
            )
        ]

    def test_only_script_kind_is_extracted(self, tmp_path):
        # grep / path / manual / a capability with NO delivered_check at all
        # must neither be mistaken for a script check nor raise on the missing
        # `delivered_check`.
        sidecar = self._write(
            tmp_path,
            'mixed',
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-grep"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: grep
          pattern: "foo"
          expect: present
      - name: "cap-script"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: script
          script: scripts/x.py
          timeout_secs: 30
      - name: "cap-manual"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: manual
          reason: "covered by E8"
      - name: "cap-path"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: path
          expect: present
          paths:
            - orchestrator/tests/test_x.py
      - name: "cap-unchecked"
        binding: "b"
        verdict: PASS
""",
        )
        refs = iter_script_checks(sidecar)
        assert [ref.capability for ref in refs] == ['cap-script']
        assert refs[0].script == 'scripts/x.py'

    def test_two_scripts_under_two_labels_are_each_attributed(self, tmp_path):
        sidecar = self._write(
            tmp_path,
            'two-labels',
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-a"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: script
          script: scripts/a.py
          timeout_secs: 30
  - label: "beta"
    capabilities:
      - name: "cap-b"
        binding: "b"
        verdict: PASS
        delivered_check:
          kind: script
          script: scripts/b.py
          timeout_secs: 30
""",
        )
        refs = iter_script_checks(sidecar)
        assert [(ref.task_label, ref.capability, ref.script) for ref in refs] == [
            ('alpha', 'cap-a', 'scripts/a.py'),
            ('beta', 'cap-b', 'scripts/b.py'),
        ]

    def test_unloadable_sidecar_yields_nothing_instead_of_raising(self, tmp_path):
        # The corpus guard computes its script checks at MODULE IMPORT time
        # to feed @pytest.mark.parametrize — the same hazard
        # discover_manifests' docstring names. A raise here would take down
        # collection of that entire module, not just its script-target
        # class. No signal is lost: its per-sidecar validation case already
        # reports an unloadable sidecar loudly and with a better,
        # field-path-attributed message.
        sidecar = self._write(
            tmp_path,
            'bad-schema',
            """\
prd: plans/example-prd.md
schema_version: 1
tasks:
  - label: "alpha"
    capabilities:
      - name: "cap-one"
        binding: "b"
        verdcit: PASS
""",
        )
        assert iter_script_checks(sidecar) == []

    def test_broken_yaml_sidecar_yields_nothing_instead_of_raising(self, tmp_path):
        sidecar = self._write(tmp_path, 'bad-syntax', 'prd: [unterminated')
        assert iter_script_checks(sidecar) == []


class TestCheckScriptTarget:
    """check_script_target(ref) — the working-tree exists+executable checker.

    Mirrors check_manifest's contract: None on success, else ONE grep-able
    line naming the sidecar and the capability. Each case builds a fake repo
    root under tmp_path plus a hand-built ScriptCheckRef, so no committed
    fixture files are needed (TestCheckManifest's convention).
    """

    def _ref(self, repo_root, script, capability='cap-one'):
        return ScriptCheckRef(
            manifest=repo_root / 'plans' / f'example-prd{MANIFEST_SUFFIX}',
            task_label='δ',
            capability=capability,
            script=script,
        )

    def _make_script(self, repo_root, rel, mode):
        target = repo_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#!/usr/bin/env python3\n')
        target.chmod(mode)
        return target

    def test_existing_executable_target_returns_none(self, tmp_path):
        self._make_script(tmp_path, 'scripts/ok.py', 0o755)
        ref = self._ref(tmp_path, 'scripts/ok.py')
        assert check_script_target(ref, repo_root=tmp_path) is None

    def test_missing_target_names_manifest_capability_and_script(self, tmp_path):
        # The whole point of the corpus sweep is that the failure tells you
        # WHICH capability in WHICH sidecar is broken — mirroring
        # check_manifest's file-attribution contract.
        (tmp_path / 'plans').mkdir()
        ref = self._ref(tmp_path, 'scripts/gone.py', capability='cap-missing')
        message = check_script_target(ref, repo_root=tmp_path)
        assert message is not None
        assert f'plans/example-prd{MANIFEST_SUFFIX}' in message
        assert 'cap-missing' in message
        assert 'scripts/gone.py' in message

    def test_non_executable_target_names_the_bit_and_says_it_errors(self, tmp_path):
        self._make_script(tmp_path, 'scripts/plain.py', 0o644)
        ref = self._ref(tmp_path, 'scripts/plain.py')
        message = check_script_target(ref, repo_root=tmp_path)
        assert message is not None
        assert 'executable' in message
        # ERRORed, not FAILed: _run_script_check execs the target as argv[0],
        # so the OSError maps to DeliveredCheckResult.ERRORED at dispatch —
        # a distinction the reader of a red corpus sweep needs.
        assert 'ERROR' in message
        assert 'scripts/plain.py' in message

    def test_directory_at_script_path_is_reported_as_missing(self, tmp_path):
        # A directory is +x and would pass a naive os.access check, but
        # exec'ing it raises — is_file() must gate first.
        (tmp_path / 'scripts' / 'adir.py').mkdir(parents=True)
        ref = self._ref(tmp_path, 'scripts/adir.py')
        message = check_script_target(ref, repo_root=tmp_path)
        assert message is not None
        assert 'does not exist' in message

    def test_message_is_a_single_grepable_line(self, tmp_path):
        (tmp_path / 'plans').mkdir()
        ref = self._ref(tmp_path, 'scripts/gone.py')
        message = check_script_target(ref, repo_root=tmp_path)
        assert message is not None
        assert '\n' not in message


class TestCommittedFileMode:
    """committed_file_mode(rel) — the INDEX mode, not the working-tree bit.

    os.access is greened by a local `chmod +x` that is never staged, which
    would leave main exactly as broken while the guard reported green. This
    checker is what proves a mode fix actually LANDS.

    Its three outcomes are asserted separately (code-review amendment, task
    3649), because collapsing them is what makes a red lie: a mode string, a
    None meaning genuinely-untracked, and a raised GitUnavailable meaning git
    never answered. The two answered from the real index live in the
    checked-in corpus guard in tests/repo_guards/test_checked_in_capability_manifests.py.
    """

    def test_non_checkout_root_raises_git_unavailable(self, tmp_path):
        # git ls-files exits non-zero outside a checkout. discover_manifests
        # swallows that into None because it runs at module import time; this
        # one runs at test time, so it keeps the distinction rather than
        # letting a non-checkout masquerade as an untracked path.
        with pytest.raises(GitUnavailable):
            committed_file_mode('scripts/anything.py', repo_root=tmp_path)

    def test_missing_git_binary_raises_git_unavailable(self, tmp_path, monkeypatch):
        def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError('git')

        monkeypatch.setattr(subprocess, 'run', _raise)
        with pytest.raises(GitUnavailable):
            committed_file_mode('scripts/anything.py', repo_root=tmp_path)

    def test_wedged_git_raises_git_unavailable(self, tmp_path, monkeypatch):
        # The index.lock-contention case: the merge worker operates on this
        # same checkout concurrently (CLAUDE.md "Working in the main
        # checkout"), so a wedge is a live possibility mid-run. It must be
        # distinguishable from a staged-mode defect, not rendered as one.
        def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd='git', timeout=30)

        monkeypatch.setattr(subprocess, 'run', _raise)
        with pytest.raises(GitUnavailable):
            committed_file_mode('scripts/anything.py', repo_root=tmp_path)

    def test_ambiguous_index_entry_is_catchable_as_git_unavailable(self):
        # The subclass relationship is load-bearing: a caller that only wants
        # "no single usable answer" writes one except clause.
        assert issubclass(AmbiguousIndexEntry, GitUnavailable)


class TestMissingFromWorktreeMessage:
    def test_names_the_repo_relative_path(self):
        target = REPO_ROOT / 'docs' / 'prds' / 'nonexistent.capability-manifest.yaml'
        assert missing_from_worktree_message(target) == (
            'docs/prds/nonexistent.capability-manifest.yaml: '
            'tracked in git but missing from the working tree'
        )


class TestSidecarPath:
    """sidecar_path — the single PRD key → sidecar path rule (task 6582)."""

    @pytest.mark.parametrize(
        ('prd_key', 'expected'),
        [
            pytest.param(
                'plans/x-prd.md', 'plans/x-prd.capability-manifest.yaml', id='plans'
            ),
            pytest.param(
                'docs/prds/memory-metadata-vocabulary.md',
                'docs/prds/memory-metadata-vocabulary.capability-manifest.yaml',
                id='docs_prds',
            ),
            pytest.param(
                'plans/x-prd', 'plans/x-prd.capability-manifest.yaml', id='no_md_suffix'
            ),
            pytest.param(
                'plans/a.md-notes.md',
                'plans/a.md-notes.capability-manifest.yaml',
                id='only_trailing_md_stripped',
            ),
        ],
    )
    def test_maps_a_prd_key_to_its_sidecar(self, prd_key, expected):
        assert sidecar_path(prd_key) == expected


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

    def test_path_entry_with_name_constructs(self):
        check = DeliveredCheckMeta(
            name='cap-three',
            kind='path',
            expect='present',
            paths=['orchestrator/tests/test_x.py'],
        )
        assert check.name == 'cap-three'
        assert check.kind == 'path'
        assert check.expect == 'present'
        assert check.paths == ['orchestrator/tests/test_x.py']

    def test_path_entry_absent_constructs(self):
        check = DeliveredCheckMeta(
            name='cap-four', kind='path', expect='absent', paths=['legacy/dead.py']
        )
        assert check.expect == 'absent'

    def test_manual_kind_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='manual')  # type: ignore[arg-type]

    def test_path_missing_expect_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='path', paths=['a/one.py'])

    def test_path_missing_paths_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='path', expect='present')

    def test_path_empty_paths_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(name='cap-one', kind='path', expect='present', paths=[])

    def test_path_with_script_field_rejected(self):
        with pytest.raises(ValidationError):
            DeliveredCheckMeta(
                name='cap-one',
                kind='path',
                expect='present',
                paths=['a/one.py'],
                script='scripts/x.sh',
            )

    def test_path_error_names_deliveredcheckmeta_and_the_field(self):
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheckMeta(name='cap-one', kind='path', expect='present')
        message = str(exc_info.value)
        assert 'DeliveredCheckMeta' in message
        assert 'paths' in message

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

    def test_error_names_deliveredcheckmeta_not_deliveredcheck(self):
        # A DeliveredCheckMeta failure must name itself, not the sidecar
        # DeliveredCheck model — they share validation via
        # _check_kind_conditional_fields, which threads a model_name.
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheckMeta(name='cap-one', kind='grep', expect='present')
        message = str(exc_info.value)
        assert 'DeliveredCheckMeta: pattern is required' in message


class TestMechanicalCheckKinds:
    """MECHANICAL_CHECK_KINDS — the one place "mechanical" is defined.

    Mechanical is not an independent concept that happens to coincide with
    DeliveredCheckMeta's vocabulary — it IS that vocabulary, definitionally:
    mechanical means "copied into metadata.delivered_checks", and
    DeliveredCheckMeta is precisely the model of a metadata entry. The
    derivation assertion below is the point of this class: it is what makes
    adding a future kind impossible to half-apply.
    """

    def test_value_is_the_three_mechanical_kinds(self):
        assert MECHANICAL_CHECK_KINDS == ('grep', 'script', 'path')

    def test_is_derived_from_the_delivered_check_meta_literal(self):
        assert (
            get_args(DeliveredCheckMeta.model_fields['kind'].annotation)
            == MECHANICAL_CHECK_KINDS
        )

    def test_does_not_contain_manual(self):
        # Derived from DeliveredCheckMeta, NOT DeliveredCheck: the latter
        # also carries 'manual', the one kind that must never be copied
        # into metadata.
        assert 'manual' not in MECHANICAL_CHECK_KINDS
        assert 'manual' in get_args(DeliveredCheck.model_fields['kind'].annotation)

    def test_is_exported_in_module_all(self):
        assert 'MECHANICAL_CHECK_KINDS' in capability_manifest_module.__all__


class TestCheckSubjectField:
    """CHECK_SUBJECT_FIELD — the per-kind field a failure is ABOUT."""

    def test_maps_each_mechanical_kind_to_its_subject(self):
        assert CHECK_SUBJECT_FIELD == {
            'grep': 'pattern',
            'script': 'script',
            'path': 'paths',
        }

    def test_covers_exactly_the_mechanical_kinds(self):
        # The coupling that keeps a future kind from being added to one and
        # missed in the other, leaving a renderer with no subject to name.
        assert set(CHECK_SUBJECT_FIELD) == set(MECHANICAL_CHECK_KINDS)

    def test_every_subject_is_a_real_descriptor_field(self):
        for field in CHECK_SUBJECT_FIELD.values():
            assert field in DeliveredCheckMeta.model_fields

    def test_is_exported_in_module_all(self):
        assert 'CHECK_SUBJECT_FIELD' in capability_manifest_module.__all__


class TestMetadataRegistration:
    def test_registered_under_delivered_checks_key(self):
        assert task_metadata_module._SUBMODEL_REGISTRY['delivered_checks'] is DeliveredCheckMeta

    def test_registered_with_list_cardinality(self):
        # delivered_checks is the ONE genuinely list-valued metadata slice
        # (orchestrator/delivered_checks.py reads it and
        # verify_delivered_checks_on_main iterates it). The declaration is
        # what keeps parse_metadata's enforced shape gate from rejecting a
        # well-formed list — 'dict' is the fail-closed default (task 4142).
        assert task_metadata_module._SUBMODEL_CARDINALITY['delivered_checks'] == 'list'

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

    def test_dict_shaped_slice_warns_wrong_cardinality(self):
        # The mirror direction (task 4142): a bare mapping for the
        # list-declared slice used to validate silently into a SINGLE
        # DeliveredCheckMeta. verify_delivered_checks_on_main ITERATES this
        # value, so a dict would iterate its string keys and yield garbage
        # checks against a mark-done gate.
        grep_entry = {'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}
        model, warnings = parse_metadata(
            {'delivered_checks': dict(grep_entry)}, direction='write', enforce=False
        )
        assert len(warnings) == 1
        assert warnings[0].field == 'delivered_checks'
        assert warnings[0].code == 'wrong_cardinality'
        assert model.model_dump()['delivered_checks'] == grep_entry

    def test_dict_shaped_slice_write_enforce_raises(self):
        # A perfectly VALID single entry — the shape is the only defect, so
        # this cannot pass for an element-validation reason.
        with pytest.raises(TypeError):
            parse_metadata(
                {
                    'delivered_checks': {
                        'name': 'cap-one',
                        'kind': 'grep',
                        'pattern': 'foo',
                        'expect': 'present',
                    }
                },
                direction='write',
                enforce=True,
            )

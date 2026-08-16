"""Tests for shared.capability_manifest — the capability-manifest sidecar schema.

Built bottom-up in TDD order, following shared/tests/test_task_metadata.py's
convention (see plans/capability-delivered-checks-prd.md §Contract):
  - TestDeliveredCheck: the kind-conditional check descriptor in isolation.
  - TestManifestCapability / TestManifestTask: the capability + task containers.
  - TestCapabilityManifestDoc: the doc-level label uniqueness invariant.
  - TestLoader: load_capability_manifest / parse_capability_manifest, including
    the committed exemplar sidecar as a CI fixture.
  - TestManifestCorpusDiscovery / TestCheckManifest / TestMissingFromWorktreeMessage /
    TestCheckedInManifestCorpus: a corpus-wide guard validating every
    checked-in sidecar (not just the one committed exemplar TestLoader
    covers), built on the sibling test-support module
    shared/tests/capability_manifest_corpus.py (task 3362).
  - TestDeliveredCheckMeta / TestMetadataRegistration: the
    metadata.delivered_checks registered sub-model.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from capability_manifest_corpus import (
    MANIFEST_SUFFIX,
    REPO_ROOT,
    ScriptCheckRef,
    check_manifest,
    check_script_target,
    committed_file_mode,
    discover_manifests,
    iter_script_checks,
)
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


#: The one committed exemplar sidecar TestLoader validates directly, and the
#: anchor TestManifestCorpusDiscovery asserts is present in the full corpus
#: sweep — hoisted once (code-review amendment, task 3362) so a rename of
#: this PRD or its sidecar breaks exactly one place instead of two.
_EXEMPLAR_SIDECAR_REL = 'plans/capability-delivered-checks-prd.capability-manifest.yaml'


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
        sidecar_path = REPO_ROOT / _EXEMPLAR_SIDECAR_REL
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


class TestManifestCorpusDiscovery:
    """discover_manifests() — git-ls-files-based corpus discovery.

    Not Path.rglob: see capability_manifest_corpus.py's module docstring for
    the measurement (30 tracked sidecars via git ls-files, 0 of them under
    .worktrees/, vs 3793 extra sidecars an rglob from the main checkout would
    sweep in from other in-flight tasks' gitignored worktree checkouts).
    """

    def test_returns_nonempty_list_for_a_git_checkout(self):
        paths = discover_manifests()
        assert paths is not None
        assert paths != []

    def test_returns_none_for_a_non_checkout_directory(self, tmp_path):
        assert discover_manifests(tmp_path) is None

    def test_returns_none_when_git_binary_is_missing(self, tmp_path, monkeypatch):
        # subprocess.run raises OSError (FileNotFoundError, not a non-zero
        # exit) when the git binary itself can't be found. _MANIFEST_PATHS =
        # discover_manifests() or [] runs at *module import time* below, so
        # an uncaught exception here would take down collection of this
        # entire test module, not just the corpus-guard classes.
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

    def test_every_path_is_a_valid_sidecar_not_nested_in_worktrees(self):
        # The three properties that are real assertions on discover_manifests()'s
        # behavior, folded into one test (code-review amendment, task 3362:
        # the individual is_absolute() and sorted-with-no-duplicates cases
        # dropped as guaranteed by construction — repo_root / rel with an
        # absolute repo_root; sorted(set(...)) — three lines inside
        # discover_manifests itself, so they can't fail without a rewrite
        # that would also break this test and added no regression signal of
        # their own).
        paths = discover_manifests()
        assert paths is not None
        for path in paths:
            assert path.name.endswith(MANIFEST_SUFFIX)
            assert path.exists()
            # Load-bearing anti-rglob assertion — see class docstring. Checked
            # relative to REPO_ROOT, not as an absolute-path substring: when
            # this suite itself runs from inside a worktree, REPO_ROOT (this
            # file's own parents[2]) legitimately resolves to
            # .../.worktrees/<n>, so the absolute path always contains that
            # segment. What must never happen is a *nested* .worktrees
            # segment relative to REPO_ROOT — that would mean discovery
            # walked into another task's checkout.
            assert '.worktrees' not in path.relative_to(REPO_ROOT).parts

    def test_known_stable_anchor_is_present(self):
        # Shares _EXEMPLAR_SIDECAR_REL with
        # TestLoader.test_committed_exemplar_sidecar_validates (code-review
        # amendment, task 3362) so a rename of that PRD/sidecar breaks
        # exactly one place instead of two independently-hardcoded ones.
        # Deliberately does NOT also assert that both docs/prds/ and plans/
        # are represented (a dropped sibling test) — that pinned a corpus
        # fact (today's directory layout), not a discover_manifests()
        # invariant, and would go red on a harmless doc reorganization.
        paths = discover_manifests()
        assert paths is not None
        rels = {str(path.relative_to(REPO_ROOT)) for path in paths}
        assert _EXEMPLAR_SIDECAR_REL in rels


class TestCheckManifest:
    """check_manifest(path) — the file-attributed reporting wrapper.

    Each case writes a single inline sidecar to tmp_path (matching
    TestLoader._VALID_YAML / _MALFORMED_YAML_DUP_LABEL's convention — no
    committed bad-fixture files) and drives one of check_manifest's failure
    modes. This is the mutation proof TestCheckedInManifestCorpus rests its
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

    The extraction layer TestCheckedInScriptCheckTargets' corpus sweep is
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
        # grep / manual / a capability with NO delivered_check at all must
        # neither be mistaken for a script check nor raise on the missing
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
        # _SCRIPT_CHECKS is computed at MODULE IMPORT time to feed
        # @pytest.mark.parametrize — the same hazard discover_manifests'
        # docstring names. A raise here would take down collection of this
        # entire module, not just the corpus-guard class. No signal is lost:
        # TestCheckedInManifestCorpus.test_checked_in_manifest_validates
        # already reports an unloadable sidecar loudly and with a better,
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
    """

    def test_known_executable_script_reports_100755(self):
        # scripts/check_method_param_wiring.py is committed 100755 (task 3364).
        assert committed_file_mode('scripts/check_method_param_wiring.py') == '100755'

    def test_untracked_path_returns_none(self):
        assert committed_file_mode('scripts/no-such-file-3649.py') is None

    def test_non_checkout_root_returns_none(self, tmp_path):
        # Mirrors discover_manifests' non-checkout contract: git ls-files
        # exits non-zero outside a checkout, and this must not raise.
        assert committed_file_mode('scripts/anything.py', repo_root=tmp_path) is None


_MANIFEST_PATHS = discover_manifests() or []
_MANIFEST_IDS = [str(p.relative_to(REPO_ROOT)) for p in _MANIFEST_PATHS]

#: Derived from _MANIFEST_PATHS, NOT a second corpus walk — script-check
#: discovery inherits discover_manifests' anti-rglob, worktree-excluding,
#: non-raising git ls-files behaviour verbatim. iter_script_checks is
#: likewise non-raising (see its docstring): this runs at module import
#: time, so anything escaping here takes down collection of the whole module.
_SCRIPT_CHECKS = [ref for path in _MANIFEST_PATHS for ref in iter_script_checks(path)]
_SCRIPT_CHECK_IDS = [
    f'{ref.manifest.relative_to(REPO_ROOT)}::{ref.capability}' for ref in _SCRIPT_CHECKS
]


def _missing_from_worktree_message(manifest_path: Path) -> str:
    """Message for a sidecar ``git ls-files`` tracks but the working tree lacks.

    ``git ls-files`` reports index entries, not working-tree entries: an
    unstaged deletion (``rm sidecar.yaml`` without ``git rm``) or a
    skip-worktree/sparse-checkout entry makes :func:`discover_manifests`
    return a path that doesn't exist on disk. ``check_manifest`` letting
    ``FileNotFoundError`` propagate is correct for a direct caller (see
    ``TestCheckManifest.test_missing_file_propagates_file_not_found`` — a
    missing file is a caller bug, not manifest drift) — but in the corpus
    sweep below, the "caller" is the guard itself, so a tracked-but-missing
    file must fail loudly and name the file, not surface as an uncaught
    traceback from this guard (code-review amendment, task 3362).
    """
    rel = manifest_path.relative_to(REPO_ROOT)
    return f'{rel}: tracked in git but missing from the working tree'


class TestMissingFromWorktreeMessage:
    def test_names_the_repo_relative_path(self):
        target = REPO_ROOT / 'plans' / 'nonexistent.capability-manifest.yaml'
        assert _missing_from_worktree_message(target) == (
            'plans/nonexistent.capability-manifest.yaml: '
            'tracked in git but missing from the working tree'
        )


class TestCheckedInManifestCorpus:
    """Every checked-in capability-manifest sidecar validates against the schema.

    This is the guard the task was filed for: TestLoader above covers exactly
    one committed exemplar; this class sweeps all of them (measured at
    planning time: 30 files under docs/prds/ and plans/, 447 capabilities,
    0 failures — see task 3362's plan.json for the full measurement). It is
    GREEN on arrival, which is inherent to a preventative guard, not a
    reason to trust it unverified — so the RED signal was proven by mutation
    instead of relying on a naturally-failing first run:

        1. Inserted an unknown key (`verdcit: PASS`) into one real capability
           entry of plans/capability-delivered-checks-prd.capability-manifest.yaml
           (the "sidecar-loader-models-exist" capability under task α).
        2. Ran this class: both parametrized cases keyed to that one file
           failed (test_checked_in_manifest_validates and
           _prd_field_matches_its_own_filename — each loads the same mutated
           file), while every other file's cases across both methods stayed
           green. The test_checked_in_manifest_validates failure carried the
           exact assertion message: "plans/capability-delivered-checks-prd.
           capability-manifest.yaml: schema validation failed: 1 validation
           error for CapabilityManifestDoc tasks.0.capabilities.1.verdcit
           Extra inputs are not permitted [...]" — naming both the mutated
           file and the offending `verdcit` key, exactly as check_manifest's
           docstring promises.
        3. Reverted the sidecar (`git checkout --`) and reran: full class
           green again, all 30 files (150 tests across the whole module —
           re-measured after the code-review amendments, task 3362, that
           dropped the third, tautological parametrization and trimmed
           TestManifestCorpusDiscovery).

    Does NOT assert delivered_check `script:` targets exist on disk — that
    property now has its own sweep, TestCheckedInScriptCheckTargets below.
    The exclusion this docstring used to record ("out of scope: n=1 in the
    corpus at planning time") was retired by task 3649: n reached 2, and the
    second one (plans/agent-transcript-archival-prd.capability-manifest.yaml
    -> scripts/gc_agent_transcripts.py) was committed at mode 100644 and had
    been ERRORing forever. This class stays scoped to manifest SHAPE; the
    sibling class covers script LIFECYCLE. Does NOT separately re-validate each capability via
    `ManifestCapability.model_validate(cap.model_dump()) == cap`
    (code-review amendment, task 3362: dropped as tautological) —
    `load_capability_manifest` already constructs every `ManifestCapability`
    through that same `extra='forbid'` model with plain str/list/int fields,
    so the round trip could only fail when `load_capability_manifest` itself
    already raised, which `test_checked_in_manifest_validates` already
    reports with a better, file-attributed message.
    """

    def test_corpus_discovery_is_not_vacuous(self):
        # An empty parametrize list below would collect ZERO test cases and
        # pass silently — exactly the fail-soft this guard exists to
        # prevent. Skip (not fail) when the tree genuinely isn't a git
        # checkout, mirroring test_lock_charter_guard.py's own split.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert _MANIFEST_PATHS, 'discover_manifests() returned an empty corpus'

    @pytest.mark.parametrize('manifest_path', _MANIFEST_PATHS, ids=_MANIFEST_IDS)
    def test_checked_in_manifest_validates(self, manifest_path):
        try:
            message = check_manifest(manifest_path)
        except FileNotFoundError:
            pytest.fail(_missing_from_worktree_message(manifest_path))
        assert message is None, message

    @pytest.mark.parametrize('manifest_path', _MANIFEST_PATHS, ids=_MANIFEST_IDS)
    def test_checked_in_manifest_prd_field_matches_its_own_filename(self, manifest_path):
        # manifest_stamping.py derives the sidecar path as
        # re.sub(r'\.md$', '', prd_path) + MANIFEST_SUFFIX, so a doc.prd that
        # disagrees with its own sidecar's filename means commit_planning
        # looks for a sidecar that isn't there and silently no-ops (its
        # documented "no sidecar on disk is a complete no-op" contract) —
        # the drift is invisible at exactly the moment it matters.
        try:
            doc = load_capability_manifest(manifest_path)
        except FileNotFoundError:
            pytest.fail(_missing_from_worktree_message(manifest_path))
        rel = str(manifest_path.relative_to(REPO_ROOT))
        expected_prd = rel[: -len(MANIFEST_SUFFIX)] + '.md'
        assert doc.prd == expected_prd
        assert (REPO_ROOT / doc.prd).is_file()


class TestCheckedInScriptCheckTargets:
    """Every checked-in `kind: script` delivered_check names a runnable target.

    The generalization of the per-capability assertion task 3364 left behind in
    scripts/tests/test_check_method_param_wiring.py::TestManifestScriptCoherence
    (its own NOTE earmarked this ticket, tkt_0RS3PJJZFSJRJERYB11FNNKF5W /
    esc-3364-2, for both the generalization and the retirement of that copy).

    Why it matters: `_run_script_check` builds
    argv = [str(project_root / meta.script), *args] and execs the target
    DIRECTLY, so a missing or non-executable script raises OSError, which
    run_delivered_check's catch-all maps to DeliveredCheckResult.ERRORED. The
    check never FAILs — it ERRORs forever, so the capability it was authored to
    gate is silently un-gated. That is not hypothetical: this guard arrived
    NATURALLY RED (task 3649), unlike TestCheckedInManifestCorpus above, which
    was green on arrival and needed a mutation to prove its RED signal.

    RED on arrival, measured across the corpus (2 script checks in 30 sidecars):

        FAILED ...::test_target_exists_and_is_executable[
            plans/agent-transcript-archival-prd.capability-manifest.yaml::
            gc-prunes-archive-by-cap]
        FAILED ...::test_target_is_committed_executable[  (same id)  ]

    — working tree -rw-rw-r--, committed mode 100644 — while BOTH
    docs/prds/memory-eval-program.capability-manifest.yaml::
    qdrant-vector-access-for-ann cases stayed green. A one-line `chmod +x` on
    scripts/gc_agent_transcripts.py greened both halves with no logic change.

    Both halves are asserted, as two separate methods rather than one combined
    assertion, because either alone is a hole: os.access is the exact property
    _run_script_check depends on (it execs against the WORKING checkout), but it
    is greened by a local `chmod +x` that is never staged — which would leave
    main exactly as broken while this guard reported green. The committed-mode
    assertion is what proves the fix LANDS. Two methods means the red tells you
    WHICH is wrong (patch the tree vs stage the mode).
    """

    def test_script_check_corpus_is_not_vacuous(self):
        # @pytest.mark.parametrize over an empty list collects ZERO cases and
        # reports success — the precise silent-pass this guard exists to
        # remove. Skip (not fail) on a non-checkout, mirroring
        # test_corpus_discovery_is_not_vacuous.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert _SCRIPT_CHECKS, 'no kind: script delivered_check found in any checked-in sidecar'

    def test_both_known_script_targets_are_present(self):
        # Acceptance criterion 3 as a machine-checked assertion: dropping
        # either real target from coverage goes red rather than quietly
        # shrinking the sweep. Mirrors test_known_stable_anchor_is_present.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert {ref.script for ref in _SCRIPT_CHECKS} >= {
            'scripts/gc_agent_transcripts.py',
            'scripts/check_method_param_wiring.py',
        }

    @pytest.mark.parametrize('ref', _SCRIPT_CHECKS, ids=_SCRIPT_CHECK_IDS)
    def test_target_exists_and_is_executable(self, ref):
        # The WORKING-tree property _run_script_check actually execs.
        message = check_script_target(ref)
        assert message is None, message

    @pytest.mark.parametrize('ref', _SCRIPT_CHECKS, ids=_SCRIPT_CHECK_IDS)
    def test_target_is_committed_executable(self, ref):
        # The INDEX mode — so a local-only `chmod +x` that is never staged
        # cannot green this guard while main stays broken.
        mode = committed_file_mode(ref.script)
        assert mode == '100755', (
            f'{ref.manifest.relative_to(REPO_ROOT)}: capability {ref.capability!r}: '
            f'{ref.script} is committed at mode {mode!r}, not 100755 — a working-tree '
            'chmod that is never staged leaves main ERRORing'
        )


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

    def test_error_names_deliveredcheckmeta_not_deliveredcheck(self):
        # A DeliveredCheckMeta failure must name itself, not the sidecar
        # DeliveredCheck model — they share validation via
        # _check_kind_conditional_fields, which threads a model_name.
        with pytest.raises(ValidationError) as exc_info:
            DeliveredCheckMeta(name='cap-one', kind='grep', expect='present')
        message = str(exc_info.value)
        assert 'DeliveredCheckMeta: pattern is required' in message


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

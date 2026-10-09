"""Tests for fused_memory.server.manifest_sidecar_text.stamp_task_ids.

A stamp rewrites ONLY task_id values in a capability-manifest sidecar's YAML
text; every other byte (comments, quoting, blank lines, BOM, line endings)
survives. Each case states its exact expected output as the input with one
targeted edit, never as a re-dump.
"""

import pytest
import yaml

from fused_memory.server.manifest_sidecar_text import SidecarStampRefused, stamp_task_ids

# The header the eval-framework-revival sidecar carried before a
# yaml.safe_dump write-back discarded it.
_MEASURED_HEADER = (
    '# Machine-readable sidecar for plans/eval-framework-revival-prd.md\n'
    '# Created 2026-07-20 for the paired-edit task π (architect-fable OFAT candidate,\n'
    "# added by plans/fable-architect-eval-admission-prd.md's authoring session —\n"
    '# the ο precedent). Earlier eval-revival tasks predate the sidecar convention;\n'
    '# their bindings live in the hand-named .md manifest only.\n'
    '# Schema: plans/capability-delivered-checks-prd.md §Contract\n'
    '# (shared/src/shared/capability_manifest.py). task_id stamped by commit_planning.\n'
)

_COMMENTED_SIDECAR = _MEASURED_HEADER + (
    'prd: plans/eval-framework-revival-prd.md\n'
    'schema_version: 1\n'
    'tasks:\n'
    '  - label: "π"  # the paired-edit task\n'
    '    task_id: null  # stamp me\n'
    '    title: "architect-fable OFAT candidate config"\n'
    '    capabilities: []\n'
    '\n'
    '  # ρ is authored but not yet decomposed.\n'
    '  - label: "ρ"\n'
    '    task_id: null\n'
    "    title: 'second task'\n"
    '    capabilities: []\n'
)


def test_only_the_bound_task_id_value_changes():
    stamp = stamp_task_ids(_COMMENTED_SIDECAR, {'π': 101})

    assert stamp.text == _COMMENTED_SIDECAR.replace(
        '    task_id: null  # stamp me\n', '    task_id: 101  # stamp me\n'
    )
    assert stamp.document == yaml.safe_load(stamp.text)


def test_absent_task_id_key_is_inserted_after_the_label_line():
    text = (
        '# header\n'
        'tasks:\n'
        '  - label: alpha  # first\n'
        '    title: A\n'
        '    capabilities: []\n'
        '  # between entries\n'
        '  - label: beta\n'
        '    task_id: null\n'
        '    capabilities: []\n'
    )

    stamped = stamp_task_ids(text, {'alpha': 7}).text

    assert stamped == text.replace(
        '  - label: alpha  # first\n', '  - label: alpha  # first\n    task_id: 7\n'
    )
    assert yaml.safe_load(stamped)['tasks'][0]['task_id'] == 7


def test_label_on_a_final_line_without_newline_gets_one():
    text = 'tasks:\n  - label: zed'

    assert stamp_task_ids(text, {'zed': 3}).text == 'tasks:\n  - label: zed\n    task_id: 3\n'


def test_empty_task_id_value_gets_a_separating_space():
    text = 'tasks:\n  - label: alpha\n    task_id:\n    capabilities: []\n'

    stamped = stamp_task_ids(text, {'alpha': 7}).text

    assert stamped == text.replace('    task_id:\n', '    task_id: 7\n')


def test_already_stamped_id_is_replaced_and_restamping_is_idempotent():
    text = 'tasks:\n  - label: alpha\n    task_id: 3\n    capabilities: []\n'

    stamped = stamp_task_ids(text, {'alpha': 5}).text

    assert stamped == text.replace('task_id: 3', 'task_id: 5')
    assert stamp_task_ids(stamped, {'alpha': 5}).text == stamped


def test_bom_and_crlf_survive_including_on_an_inserted_line():
    text = (
        '﻿prd: plans/x-prd.md\r\n'
        'tasks:\r\n'
        '  - label: a\r\n'
        '    task_id: null\r\n'
        '  - label: b\r\n'
        '    title: B\r\n'
    )

    stamped = stamp_task_ids(text, {'a': 1, 'b': 2}).text

    assert stamped == text.replace('task_id: null', 'task_id: 1').replace(
        '  - label: b\r\n', '  - label: b\r\n    task_id: 2\r\n'
    )


def test_cr_only_line_endings_insert_after_the_label_line():
    text = 'tasks:\r  - label: a\r    title: t\r  - label: b\r    title: u\r'

    stamped = stamp_task_ids(text, {'a': 7}).text

    assert stamped == text.replace('  - label: a\r', '  - label: a\r    task_id: 7\r')


def test_cr_only_final_label_line_without_terminator_gets_a_cr():
    text = 'tasks:\r  - label: zed'

    assert stamp_task_ids(text, {'zed': 3}).text == 'tasks:\r  - label: zed\r    task_id: 3\r'


def test_flow_style_entry_is_refused_naming_its_label():
    text = 'tasks:\n  - {label: flowy, task_id: null, capabilities: []}\n'

    with pytest.raises(SidecarStampRefused, match='flowy'):
        stamp_task_ids(text, {'flowy': 4})


def test_label_absent_from_the_sidecar_leaves_text_unchanged_with_no_document():
    stamp = stamp_task_ids(_COMMENTED_SIDECAR, {'nope': 9})

    assert stamp.text == _COMMENTED_SIDECAR
    assert stamp.document is None

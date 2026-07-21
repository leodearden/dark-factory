"""Tests for orchestrator.module_tagger_prompt — the shared module-tagger
prompt/schema/system-prompt extracted from harness._tag_task_modules (task 2540).

These pin the extracted constants + builder so the harness refactor stays
behavior-preserving and the offline replay trial
(scripts/trial_module_tagger_haiku.py) can reuse byte-identical production
inputs (single source of truth, faithful replay "at the call site").
"""
from __future__ import annotations

import json

from orchestrator import module_tagger_prompt as mtp


def test_tagger_schema_is_single_predictions_array_of_id_files():
    schema = mtp.TAGGER_SCHEMA
    assert schema['type'] == 'object'
    # A SINGLE top-level 'predictions' array (not nested).
    assert schema['required'] == ['predictions']
    assert set(schema['properties']) == {'predictions'}

    predictions = schema['properties']['predictions']
    assert predictions['type'] == 'array'

    item = predictions['items']
    assert item['type'] == 'object'
    assert set(item['required']) == {'id', 'files'}
    assert item['properties']['id']['type'] == 'string'

    files = item['properties']['files']
    assert files['type'] == 'array'
    assert files['items']['type'] == 'string'

    # TAGGER_SYSTEM_PROMPT is public API surface (imported by both harness and
    # the trial), so pin its existence — not its cosmetic wording.
    assert mtp.TAGGER_SYSTEM_PROMPT


def test_build_tagger_prompt_embeds_dirs_and_indented_summaries():
    entries = ['orchestrator', 'scripts', 'shared']
    task_summaries = [
        {'id': '12', 'title': 'Foo', 'description': 'do foo'},
        {'id': '13', 'title': 'Bar', 'description': ''},
    ]
    prompt = mtp.build_tagger_prompt(entries, task_summaries)
    assert isinstance(prompt, str)

    # The dir listing is embedded json-dumped under its header.
    assert '# Codebase top-level directories' in prompt
    assert json.dumps(entries) in prompt

    # The task summaries are embedded indent=2 under their header.
    assert '# Tasks to tag' in prompt
    assert json.dumps(task_summaries, indent=2) in prompt

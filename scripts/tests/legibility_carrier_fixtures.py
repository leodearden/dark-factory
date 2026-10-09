"""Carrier texts shared by the digest and sampler suites, so both pin
``digest.is_non_dialogue_carrier`` against the same payloads.

A plain module rather than a conftest fixture, the same convention as
``suite_census_fixtures``: conftest appends this directory to ``sys.path``, so
the test files import it by bare name.
"""
from __future__ import annotations

import json


def coder_judgment(note: str = "that's wrong: the watcher re-armed on a stale lease") -> str:
    """A trickle-coder answer as it lands in a transcript: the whole
    judgment on ONE line, shaped after session b203a05c record 22. *note*
    rides inside a match, where a real coder quotes the digest it codes."""
    return json.dumps({
        'matches': [{
            'entry_id': 'watcher-loop-harness-mismatch',
            'origin_phase': 'unknown',
            'manifested_phase': 'recon',
            'invariant_violated': None,
            'note': note,
        }],
        'candidates': [{
            'title': 'Recon reaper closes a filed task with its escalation',
            'cause': 'closure keys off escalation linkage, not task merit',
            'area': 'recon',
            'origin_phase': 'unknown',
            'manifested_phase': 'recon',
            'evidence_quote': 'Re-filing without an escalation_id.',
        }],
    })


def json_fenced(payload: str) -> str:
    """*payload* inside one outer ```json fence, trailing newline included."""
    return f'```json\n{payload}\n```\n'


PROSE_QUOTING_THE_SCHEMA = (
    'The coder answers with {"matches": [], "candidates": []} '
    "— that's wrong, it should be a single object per session."
)
"""Assistant prose that quotes the coder schema inline: dialogue, not a
coder judgment, so its self-correction still counts."""


def briefing_text(body_filler: str = 'Project overview and recent decisions go here.') -> str:
    """Build the text of an orchestrator-briefing-shaped user turn -- the
    literal shape ``BriefingAssembler`` injects as the harness preamble of
    every dispatched-agent session
    (orchestrator/src/orchestrator/agents/briefing.py): a '# Context'
    heading (``orchestrator/src/orchestrator/agents/memory_recall.py::render_context_block``,
    emits '# Context\\n\\n...'), a
    '## Agent Identity' heading with an agent_id bullet
    (``_agent_identity``), and a '# Task' heading with a task block (the
    per-role prompt templates, e.g. ``build_architect_prompt``)."""
    return (
        f'# Context\n\n{body_filler}\n\n'
        '## Agent Identity\n\n'
        '- **agent_id:** `claude-task-3278-implementer`\n'
        '- **project_id:** `dark_factory`\n\n'
        '# Task\n\n'
        '**Task:** Some task title\n'
        '**Description:** Do the thing.\n'
    )

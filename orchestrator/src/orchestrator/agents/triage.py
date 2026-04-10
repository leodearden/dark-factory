"""Pre-triage agent for large review suggestion sets.

Runs a cheap, read-only Sonnet agent with structured JSON output to classify
suggestions before the steward session processes them.  This avoids the steward
exhausting its budget reading 15+ code locations and classifying each one.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — classification only, no code edits
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are a review suggestion classifier. You receive a numbered list of code
review suggestions and classify each as ACCEPT or SKIP.

## Classification rules

**ACCEPT** if the suggestion has genuine merit:
- Real bugs or correctness issues
- Missing tests for important code paths (especially error paths, edge cases)
- Code duplication across 3+ sites with maintenance risk
- Violations of project conventions
- Stale comments that would mislead future readers

**SKIP** only if genuinely meritless:
- Duplicates work already tracked in another task
- Proposes deleting code an upcoming task depends on
- Refactors that would pessimize the design or impede planned work
- Renames that don't actually improve semantic transparency
- Pre-existing issues not introduced by the diff

When in doubt, ACCEPT. The cost of a small unnecessary task is low;
the cost of missing a real issue compounds.

## Deduplication check

Before classifying, call `get_tasks` with the project root to retrieve existing
tasks.  SKIP any suggestion that duplicates an existing pending or in-progress
task — same module and same fix intent counts as a duplicate even if the wording
differs.  Cite the existing task ID in your skip reason.

After classifying, group related accepted suggestions into logical task groups.
Each group should be a single coherent follow-up task.

You may read files to verify suggestions before classifying.
"""

# ---------------------------------------------------------------------------
# Output schema — enforced via --json-schema
# ---------------------------------------------------------------------------

TRIAGE_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'accepted': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer', 'description': 'Zero-based index in the original suggestion list'},
                    'suggestion': {'type': 'string', 'description': 'Brief description of the suggestion'},
                    'reason': {'type': 'string', 'description': 'Why this has merit'},
                    'files': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Affected file paths'},
                    'proposed_task_title': {'type': 'string', 'description': 'Concise follow-up task title'},
                },
                'required': ['index', 'suggestion', 'reason', 'files', 'proposed_task_title'],
            },
        },
        'skipped': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer', 'description': 'Zero-based index in the original suggestion list'},
                    'suggestion': {'type': 'string', 'description': 'Brief description'},
                    'reason': {'type': 'string', 'description': 'Why this is meritless'},
                },
                'required': ['index', 'suggestion', 'reason'],
            },
        },
        'proposed_task_groups': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string', 'description': 'Task title grouping related accepted items'},
                    'description': {'type': 'string', 'description': 'What needs to be done, with file paths and specifics'},
                    'accepted_indices': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'Indices into the accepted array',
                    },
                },
                'required': ['title', 'description', 'accepted_indices'],
            },
        },
    },
    'required': ['accepted', 'skipped', 'proposed_task_groups'],
}


def build_triage_prompt(suggestions: list[dict], task: dict) -> str:
    """Format suggestions + task context into a triage prompt."""
    task_ctx = (
        f'Task {task.get("id", "?")}: {task.get("title", "Unknown")}\n'
        f'Description: {task.get("description", "N/A")}'
    )

    numbered = []
    for i, s in enumerate(suggestions):
        location = s.get('location', 'unknown')
        reviewer = s.get('reviewer', 'unknown')
        category = s.get('category', '')
        desc = s.get('description', '')
        fix = s.get('suggested_fix', '')
        numbered.append(
            f'[{i}] ({reviewer}/{category}) {location}\n'
            f'    {desc}\n'
            f'    Suggested fix: {fix}'
        )

    suggestions_block = '\n\n'.join(numbered)

    return f"""\
## Task Context

{task_ctx}

## Review Suggestions to Triage ({len(suggestions)} items)

{suggestions_block}

Classify each suggestion as ACCEPT or SKIP, then group accepted items into
logical follow-up task groups. Read the code at referenced locations as needed.
"""


def parse_triage_result(result) -> dict | None:
    """Extract structured triage output from an AgentResult.

    Returns the parsed dict on success, None on failure.
    """
    if result.structured_output and isinstance(result.structured_output, dict):
        required = {'accepted', 'skipped', 'proposed_task_groups'}
        if required <= result.structured_output.keys():
            return result.structured_output
        logger.warning('Triage result missing required keys: %s', required - result.structured_output.keys())
    else:
        logger.warning('Triage agent returned no structured output (success=%s)', result.success)
    return None


def format_pretriaged_detail(triage_result: dict, original_suggestions: list[dict]) -> str:
    """Format triage results as markdown for the steward's escalation detail.

    The ``## Pre-Triaged Results`` header signals to the steward that
    classification is already done and it should act on the groups directly.
    """
    accepted = triage_result.get('accepted', [])
    skipped = triage_result.get('skipped', [])
    groups = triage_result.get('proposed_task_groups', [])

    lines = [
        '## Pre-Triaged Results',
        '',
        f'**{len(accepted)} accepted, {len(skipped)} skipped, '
        f'{len(groups)} task group(s) proposed.**',
        '',
    ]

    if groups:
        lines.append('### Task Groups')
        lines.append('')
        for g in groups:
            lines.append(f'#### {g["title"]}')
            lines.append(f'{g["description"]}')
            # Collect files from accepted items in this group
            group_files: list[str] = []
            for idx in g.get('accepted_indices', []):
                if 0 <= idx < len(accepted):
                    group_files.extend(accepted[idx].get('files', []))
            if group_files:
                lines.append(f'Files: {", ".join(sorted(set(group_files)))}')
            lines.append('')

    if skipped:
        lines.append('### Skipped')
        lines.append('')
        for s in skipped:
            lines.append(f'- [{s["index"]}] {s["suggestion"]}: {s["reason"]}')
        lines.append('')

    # Append raw suggestions as reference
    lines.append('### Original Suggestions (reference)')
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps(original_suggestions, indent=2))
    lines.append('```')

    return '\n'.join(lines)

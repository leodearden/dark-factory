"""Pre-triage agent for large review suggestion sets.

Runs a cheap, read-only Sonnet agent with structured JSON output to classify
suggestions before the steward session processes them.  This avoids the steward
exhausting its budget reading 15+ code locations and classifying each one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def sha256_16(data: str) -> str:
    """Return the first 16 hex chars of the SHA-256 digest of *data*.

    Canonical shape contract: every 16-char sha256-hex token in this module
    (``suggestion_hash``, ``_combine_suggestion_hashes``) is produced through
    this helper.  The ``cleanup_needed`` R4 snippet in
    ``skills/escalation-watcher/SKILL.md`` uses an equivalent inline
    ``hashlib.sha256(payload.encode()).hexdigest()[:16]`` expression (stdlib-only,
    so the skill works without the orchestrator package installed); any change to
    the length or algorithm here must be mirrored there.

    :raises ValueError: if *data* is empty — callers must ensure a non-empty
        payload (e.g. via a ``detail or summary or id`` fallback chain) before
        calling this function.
    """
    if not data:
        raise ValueError("sha256_16 requires non-empty input")
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def suggestion_hash(suggestion: dict) -> str:
    """Stable 16-char hash over a review suggestion's identity fields.

    Used (together with the originating ``escalation_id``) to dedupe
    tasks spawned from re-queued steward triages — see plan R4. A
    steward timeout that requeues an escalation produces the same
    ordered list of suggestions, so recomputing the hash per
    suggestion is deterministic across retries.

    The output shape (16-char sha256-hex) is owned by :func:`sha256_16`.
    """
    payload = (
        '\x00'.join(
            str(suggestion.get(f, ''))
            for f in ('reviewer', 'location', 'category', 'description')
        )
        + '\x00'
    )
    return sha256_16(payload)

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
- **Documentation-only wording fixes.** Suggestions whose remedy is purely
  editing a docstring, comment, or prose (e.g. "tighten the Returns section",
  "the docstring should mention X", "comment could clarify Y", "align docstring
  with named-access convention"). Documentation drift is not load-bearing in
  this project and does not belong in the task tree — a follow-up task to pin
  wording via `__doc__` assertions produces 100+ lines of fragile meta-tests
  that reviewers then reject. If the underlying concern is behavioral (the
  function is wrong AND the docstring hides it), accept the *behavioral* fix
  and let the doc follow from it.
- **Docstring-pin hardening suggestions.** Any suggestion that proposes
  strengthening a regex / substring check / AST walk used to assert documentation
  wording (e.g. "use `ast.get_docstring` instead of whole-file grep", "bound
  the Returns-section slice"). Do not deepen the meta-test hole; the right
  fix is to delete the meta-test, not to harden it.

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


def format_pretriaged_detail(
    triage_result: dict,
    original_suggestions: list[dict],
    escalation_id: str | None = None,
) -> str:
    """Format triage results as markdown for the steward's escalation detail.

    The ``## Pre-Triaged Results`` header signals to the steward that
    classification is already done and it should act on the groups directly.

    When ``escalation_id`` is supplied (R4), the detail embeds the
    escalation id plus per-group ``suggestion_hash`` lists and instructs
    the steward to stamp those as task metadata. Combined with the
    interceptor's ``(escalation_id, suggestion_hash)`` idempotency check,
    this prevents steward-timeout re-queues from creating duplicate
    tasks — see plans/floating-snuggling-pebble.md R4.
    """
    accepted = triage_result.get('accepted', [])
    skipped = triage_result.get('skipped', [])
    groups = triage_result.get('proposed_task_groups', [])

    hashes_by_index: dict[int, str] = {}
    if escalation_id is not None:
        for i, sug in enumerate(original_suggestions):
            hashes_by_index[i] = suggestion_hash(sug)

    lines = [
        '## Pre-Triaged Results',
        '',
        f'**{len(accepted)} accepted, {len(skipped)} skipped, '
        f'{len(groups)} task group(s) proposed.**',
        '',
    ]

    if escalation_id is not None:
        lines.extend([
            '### Task Idempotency Stamps',
            '',
            f'ESCALATION_ID: `{escalation_id}`',
            '',
            'When you call `submit_task` for a group below, pass the '
            '`metadata=` kwarg with the R4 idempotency keys so the '
            'interceptor can dedupe re-queued triages (see plan R4). '
            'These keys are *additions* to the base stamps your role prompt '
            'already describes (source, spawn_context, spawned_from, etc.):',
            '',
            '```python',
            'submit_task(',
            '    title=..., description=..., priority=...,',
            '    metadata={',
            f'        "escalation_id": "{escalation_id}",',
            '        "suggestion_hash": "<hash from the group header>",',
            '        "modules": ["<file-or-module paths>"],',
            '        # plus your base role stamps: source, spawn_context, etc.',
            '    },',
            '    project_root=...,',
            ')',
            '```',
            '',
            'Before the curator runs, the fused-memory interceptor will check '
            '`(escalation_id, suggestion_hash)` against existing '
            'non-cancelled tasks. On a match, `resolve_ticket` returns '
            "`status='combined'` with `task_id` pointing at the existing "
            'task — record it the same way as `created`. '
            'You do not need to search for duplicates yourself.',
            '',
        ])

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
            # R4: emit the suggestion_hash set so the steward can stamp
            # it into metadata. Multiple hashes in a group hash together
            # with a sentinel so dedupe is still deterministic.
            if hashes_by_index:
                group_hashes: list[str] = []
                for accepted_idx in g.get('accepted_indices', []):
                    if 0 <= accepted_idx < len(accepted):
                        orig_idx = accepted[accepted_idx].get('index')
                        if isinstance(orig_idx, int) and orig_idx in hashes_by_index:
                            group_hashes.append(hashes_by_index[orig_idx])
                if group_hashes:
                    combined = _combine_suggestion_hashes(group_hashes)
                    lines.append(
                        f'suggestion_hash: `{combined}` '
                        f'(from {len(group_hashes)} suggestion(s): '
                        f'{", ".join(group_hashes)})',
                    )
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


def _combine_suggestion_hashes(hashes: list[str]) -> str:
    """Fold multiple per-suggestion hashes into a single stable group hash.

    Deterministic (sorted) so re-queued triages produce the same group
    hash even if the steward re-orders suggestions between retries.

    The output shape (16-char sha256-hex) is owned by :func:`sha256_16`.
    """
    if len(hashes) == 1:
        return hashes[0]
    payload = '|'.join(sorted(hashes)) + '|'
    return sha256_16(payload)

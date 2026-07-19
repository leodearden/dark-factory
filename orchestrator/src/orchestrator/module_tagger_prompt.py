"""Shared module-tagger prompt/schema/system-prompt.

Single source of truth for the ``module_tagger`` role's LLM inputs, extracted
verbatim from ``harness._tag_task_modules`` (task 2540) so that BOTH the
production tagger (``orchestrator.harness``) and the offline replay trial
(``scripts/trial_module_tagger_haiku.py``) build byte-identical prompts. A
hand-copied prompt in the trial would drift from production and invalidate the
replay "at the call site"; importing this module guarantees they cannot.

Behavior-preserving: ``TAGGER_SCHEMA``, ``TAGGER_SYSTEM_PROMPT`` and the string
built by ``build_tagger_prompt`` are exactly what the inline
``_tag_task_modules`` body produced before the extraction.
"""
from __future__ import annotations

import json
from typing import Any

# Structured-output schema: a SINGLE top-level 'predictions' array of
# {id, files[]} — do NOT nest it under another 'predictions'/'tasks' key
# (task 2561 defect 3). Kept as a plain dict so the harness passes it as
# ``output_schema=`` unchanged.
TAGGER_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'predictions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'files': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Predicted file paths (or directory paths) this task will create or modify',
                    },
                },
                'required': ['id', 'files'],
            },
        },
    },
    'required': ['predictions'],
}

TAGGER_SYSTEM_PROMPT = (
    'You are a code module classifier. Given task descriptions and a codebase '
    'structure, determine which code modules each task will modify. Be precise '
    'and conservative.'
)


def build_tagger_prompt(entries: list, task_summaries: list) -> str:
    """Build the module-tagger user prompt.

    *entries* is the codebase top-level directory listing (json-dumped under
    ``# Codebase top-level directories``); *task_summaries* is the list of
    ``{id, title, description}`` summaries (json-dumped, indent=2, under
    ``# Tasks to tag``). Lifted verbatim from ``_tag_task_modules`` so the
    replay trial and production share one builder.
    """
    return f"""\
Given these tasks and this codebase structure, predict which files each task
will create or modify.

Be specific and exhaustive with file predictions — include source files AND
test files. Use paths relative to the project root. The `files` field is used
to derive concurrency locks, so accuracy prevents unnecessary serialization.
Directory paths are accepted when an entire directory will be touched.

# Codebase top-level directories
{json.dumps(entries)}

# Tasks to tag
{json.dumps(task_summaries, indent=2)}

Output JSON matching the schema: a SINGLE top-level "predictions" array — do
NOT nest it inside another "predictions" or "tasks" key. For example:

{{"predictions":[{{"id":"12","files":["src/foo.py","tests/test_foo.py"]}},{{"id":"13","files":[]}}]}}

Every task must appear in the output. If you cannot predict any files for a
task, include it with an empty "files" list rather than omitting it.
"""

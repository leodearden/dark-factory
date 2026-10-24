"""Carrier texts shared by the legibility suites, so the provenance
predicates in ``legibility.reingestion``, and every reader that filters
carriers through them, are pinned against the same payloads.

A plain module rather than a conftest fixture, the same convention as
``suite_census_fixtures``: conftest appends this directory to ``sys.path``, so
the test files import it by bare name.
"""
from __future__ import annotations

import asyncio
import json

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.watcher_rearm_outcome_guidance import WATCHER_REARM_OUTCOME_GUIDANCE
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.review_suggestions.prior_round import build_resettled_adjudicator_prompt


def user_record(content, **flags) -> dict:
    """A user record carrying *content* (a str turn or a list of blocks),
    genuine by default: non-sidechain, non-meta. *flags* override the
    top-level keys, e.g. ``isSidechain=True``."""
    return {
        'type': 'user',
        'message': {'role': 'user', 'content': content},
        'isSidechain': False,
        'isMeta': False,
        **flags,
    }


def text_block(text: str) -> dict:
    """An assistant or user 'text' content block."""
    return {'type': 'text', 'text': text}


def tool_use_block(name: str, tool_input: dict, tool_use_id: str = 'tool-1') -> dict:
    """An assistant 'tool_use' content block."""
    return {'type': 'tool_use', 'id': tool_use_id, 'name': name, 'input': tool_input}


def assistant_record(*blocks: dict) -> dict:
    """An assistant record wrapping *blocks*."""
    return {
        'type': 'assistant',
        'message': {'role': 'assistant', 'content': list(blocks)},
        'isSidechain': False,
    }


def tool_result_record(tool_use_id: str, content, *, is_error: bool = False) -> dict:
    """A user record wrapping a single tool_result block."""
    return user_record([{
        'type': 'tool_result',
        'tool_use_id': tool_use_id,
        'content': content,
        'is_error': is_error,
    }])


def with_origin(record: dict, kind: str) -> dict:
    """Return a copy of *record* stamped with Claude Code's structured
    provenance ``origin={'kind': kind}`` and with its 'isMeta' key REMOVED:
    a real origin-stamped record carries isMeta absent, not False."""
    out = dict(record)
    out.pop('isMeta', None)
    out['origin'] = {'kind': kind}
    return out


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


TRICKLE_CODER_PREAMBLE = (
    'You are the trickle coder for the dark-factory agent-confusion '
    'codebook (plans/confusion-reduction-prd.md §7.3). Read the '
    'session digest below and decide which existing codebook entries '
    'it matches.'
)
"""Literal prefix of scripts/legibility/coder.py::build_prompt's
harness-authored preamble."""

CENSUS_MEMORY_CONTEXT_TURN = (
    '# Context\n\n'
    '## Project Context\n\n'
    '{ "results": [ { "id": "ccf73ca4-8240-4f9a-8abf-32b210e5b35b", '
    '"content": "Task 1470 wired /audit into /review Phase-2 Architectural '
    'Coherence." } ] }\n'
)
"""The 07-31 census cluster 1.1(b) sighting verbatim
(plans/confusion-census-2026-07-31.md:85): a context-ONLY briefing injection
that the all-three-headings rule admitted into the gold section."""

TASK_NOTIFICATION_TEXT = (
    '<task-notification>\n'
    '<task-id>a1ab810a169bbb353</task-id>\n'
    '<tool-use-id>toolu_019A8emLVkosytrg3s2up7AD</tool-use-id>\n'
    '<output-file>/tmp/claude-1000/-home-leo-src-warm-lanes-worktrees--lane-20/'
    '50e12d17-9e73-4686-a364-7ae08109140c/tasks/a1ab810a169bbb353.output</output-file>\n'
    '<status>failed</status>\n'
    '<summary>Agent "Find .ri fixture test harness patterns" failed: Agent terminated'
    " early due to an API error: You've hit your weekly limit · resets Aug 26, 11am"
    ' (Europe/London)</summary>\n'
    '<note>A task-notification fires each time this agent stops with no live background'
    ' children of its own. The user can send it another message and resume it, so the'
    ' same task-id may notify more than once.</note>\n'
    '<result>Now the harness and builtin tests.</result>\n'
    '</task-notification>'
)
"""Census record 196 of reify session 50e12d17, verbatim: a sub-agent's
weekly-limit failure notification, the session's only "gold" turn before
task 5956."""

UNKNOWN_PROVENANCE_TEXT = 'please redo the merge'

NON_HUMAN_ORIGIN_TURNS: tuple[tuple[str, str], ...] = (
    ('task-notification', TASK_NOTIFICATION_TEXT),
    (
        'auto-continuation',
        'Implement the following plan:\n\n# Fix: the laptop verify host is benched'
        '\n\n## Objective\n\nMake the sync find uv on the remote host.',
    ),
    (
        'auto-continuation',
        'Your claude.ai usage limit has reset. Continue the task you were working on'
        ' when the limit was reached; do not repeat work that is already complete.',
    ),
    (
        'coordinator',
        'The coordinator sent a message while you were working:\n'
        'Task 5878 has landed; stop now and make no further changes.',
    ),
    (
        'peer',
        'Another Claude session sent a message:\n'
        '<cross-session-message from="uds:/run/user/1000/cc-socks/1.sock"'
        ' from-name="dark-factory-19">\nplease pick up the dedup scope\n'
        '</cross-session-message>',
    ),
    ('a-future-kind', UNKNOWN_PROVENANCE_TEXT),
)
"""(origin kind, user-turn text) pairs whose text is ordinary dialogue to the
content classifier, so only the record's provenance can exclude them."""

NON_HUMAN_ORIGIN_IDS: tuple[str, ...] = (
    'task_notification', 'auto_continuation_plan',
    'auto_continuation_usage_limit_reset', 'coordinator', 'peer',
    'unseen_future_kind',
)


def recon_run_review_text() -> str:
    """Build the text of an injected reconciliation-run-review user turn --
    the literal shape ``ReconciliationJudge._build_review_prompt`` emits
    (fused-memory/src/fused_memory/reconciliation/judge.py::ReconciliationJudge._build_review_prompt):
    a '## Reconciliation Run Review' heading, a '### Run Metadata' block
    whose first bullet is '- Run ID: {run.id}', a '### Stage Reports' block
    of ``json.dumps`` output, and '### MCP Actions (N total)' / '### Journal
    Entries (N total)' blocks. The run id matches the one the census quotes
    verbatim at §1.1 (:85).

    Deliberately free of every detector literal, so a test prepending it to
    a signal-bearing session isolates the gold-turn filter rather than an
    incidental pattern hit.
    """
    return (
        '## Reconciliation Run Review\n\n'
        '### Run Metadata\n'
        '- Run ID: 84e6ce03-1f2a-4c7b-9d84-6b0f1a2e35cc\n'
        '- Project: dark_factory\n'
        '- Type: targeted\n'
        '- Trigger: task_status_change\n'
        '- Events processed: 12\n'
        '- Status: completed\n\n'
        '### Stage Reports\n'
        '[\n  {\n    "stage": 1,\n    "status": "completed",\n'
        '    "summary": "consolidated 4 memories"\n  }\n]\n\n'
        '### MCP Actions (2 total)\n'
        '[\n  {\n    "tool": "add_memory",\n    "created_at": "2026-07-31T09:00:00Z"\n  }\n]\n\n'
        '### Journal Entries (5 total)\n'
        '[\n  {\n    "kind": "stage_start",\n    "created_at": "2026-07-31T09:00:00Z"\n  }\n]\n\n'
        'Review this run and provide your verdict as JSON.\n'
    )


def _briefing_assembler(tmp_path) -> BriefingAssembler:
    """A throwaway assembler over *tmp_path*, for the prompt builders that
    touch no I/O and no config."""
    return BriefingAssembler(OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    ))


def merger_prompt(tmp_path) -> str:
    """The real text ``BriefingAssembler.build_merger_prompt`` injects
    (orchestrator/src/orchestrator/agents/briefing.py), built by calling the
    production builder rather than restating its headings, so a rewording
    turns the lockstep tests red instead of silently dropping the role out
    of coverage.

    The builder is async and touches no I/O and no config, so a throwaway
    assembler over *tmp_path* and a plain ``asyncio.run`` suffice."""
    return asyncio.run(_briefing_assembler(tmp_path).build_merger_prompt(
        conflicts='<<<<<<< ours\na\n=======\nb\n>>>>>>> theirs',
        task_intent='Rescope the briefing memory block.',
    ))


def steward_continuation_prompt(
    tmp_path,
    *,
    summary: str = (
        "Suffix item needs rebase onto frozen-prefix tip: branch '5099' has a "
        'real rebase conflict'
    ),
) -> str:
    """The real text
    ``orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_steward_continuation_prompt``
    sends when a live steward session is resumed for a new escalation,
    built by calling that builder rather than restating its f-string, so a
    rewording turns the lockstep tests red instead of silently letting the
    prompt back into the gold section. The task and escalation are shaped
    after reify session 77fb6617 record 289; *summary* is the escalation's
    free text, where a real one may carry signal literals."""
    task = {
        'id': '5099',
        'title': 'ai-native-editing η: user GUI slider/edit-box durable source write-back (INV-GUI-3)',
    }
    escalation = {
        'id': 'esc-5099-12',
        'category': 'merge_ff_failed',
        'severity': 'blocking',
        'summary': summary,
        'suggested_action': 'investigate_and_retry',
    }
    return asyncio.run(
        _briefing_assembler(tmp_path).build_steward_continuation_prompt(task, escalation),
    )


CENSUS_QUOTE_OF_THE_ESCALATION_PAYLOAD = (
    '`77fb6617`: a "# New Escalation for Task 5099" payload ending "Handle this '
    'escalation, then call `resolve_issue`". why does that render as a user correction?'
)
"""A human turn quoting the census §1.9 sighting of the steward continuation
prompt. The quote stops before 'with a summary', exactly as the census does,
so it is dialogue and stays gold."""


def prior_round_adjudicator_prompt(
    *,
    description: str = 'The retry loop never backs off between attempts.',
) -> str:
    """The real text
    ``orchestrator/src/orchestrator/review_suggestions/prior_round.py::build_resettled_adjudicator_prompt``
    sends the prior-round adjudicator, built by calling that builder so a
    rewording of its opening sentence turns the lockstep tests red. Shaped
    after census sightings 16e1a37b / 22669850 / f1f8ce95 record 3;
    *description* is review free text, where a real one may carry signal
    literals."""
    current = [{
        'location': 'orchestrator/src/orchestrator/merge_lane/gates.py',
        'category': 'correctness',
        'description': description,
        'suggested_fix': 'Sleep with exponential backoff between attempts.',
    }]
    prior = [{
        'location': 'orchestrator/src/orchestrator/merge_lane/gates.py',
        'category': 'correctness',
        'description': 'Retries hide the gate latency.',
        'suggested_fix': 'Move the retries into the caller.',
    }]
    return build_resettled_adjudicator_prompt(current, prior)


def pre_5356_replan_prompt(
    issue: str = 'The lease is released before the verify finishes.',
) -> str:
    """The RETIRED replan prompt as archived transcripts hold it, restated
    and abridged from session 2880c011 record 3 rather than built: task 5356
    (commit 8321b68278) removed its builder, and today's
    ``BriefingAssembler.build_replan_prompt`` opens with the task preamble.
    *issue* is review free text."""
    return (
        'The implementation was reviewed and blocking issues were found.\n\n'
        '# Review Feedback — Blocking Issues\n\n'
        '## [reviewer_comprehensive] correctness\n'
        '**Location:** orchestrator/src/orchestrator/merge_lane/worker.py\n'
        f'**Issue:** {issue}\n'
        '**Suggested fix:** Hold the lease until the verify returns.\n\n\n'
        '# Current Plan\n\n'
        '```json\n{"task_id": "3490", "steps": []}\n```\n\n'
        '# Action\n\n'
        'Update the plan to address the blocking issues. You may add new steps '
        'to the `steps` array, but do NOT remove or reorder existing steps. Set '
        'new steps to status "pending". Write the updated plan to '
        '`.task/plan.json`.\n'
    )


WATCHER_ROTATION_PROMPT = (
    'You are running as an autonomous escalation watcher.\n'
    'Rotation limits (injected by supervisor):\n'
    '  ROTATION_ESCALATIONS=50\n'
    '  ROTATION_HOURS=4.0\n'
    '\n'
    'When you have handled ROTATION_ESCALATIONS escalations or 4.0 hours have '
    'elapsed since startup, emit your digest as the final message and exit cleanly.\n'
    '\n'
    'Project root: /home/leo/src/reify\n'
    'Escalation queue: /home/leo/src/reify/data/escalations\n'
    f'\n{WATCHER_REARM_OUTCOME_GUIDANCE}'
)
"""The rotation prompt
``orchestrator/src/orchestrator/harness.py::Harness._run_watcher_rotation``
composes, with the values sighted in reify 4aaf48b0 record 2. The opener is
restated because the harness builds it inline; follow-up agent-followup-7441
(tkt_0RW5QBHZMRJ9JGM2BW1DHGM1KN) gives it a pure builder to call instead.
The outcome guidance is the real constant, spliced as the harness does."""


HUMAN_TURNS_NAMING_AN_INJECTED_PROMPT: tuple[str, ...] = (
    'the adjudicator keeps suppressing my fix — why does it think the concern '
    'was resolved in a prior round?',
    'the implementation was reviewed twice and the blocking issues it found '
    'were stale, please re-check',
    'why is the autonomous escalation watcher re-arming twice per rotation?',
)
"""Human turns that name the adjudicator, replan and watcher prompts without
carrying their opening sentences, so they stay dialogue."""

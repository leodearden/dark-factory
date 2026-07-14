"""Frontier-optimizer reflection step for the T6 prompt-optimization loop.

See plans/tier1-prompt-optimization-prd.md T6: `propose_heuristics_edit` is
the production default for engine.py's `propose_fn` seam. It reflects over a
scored train minibatch + the rejected-edit buffer and proposes a BOUNDED
edit to the HEURISTICS block. It is handed ONLY the current heuristics text
— there is no contract parameter on this function at all — so it structurally
cannot mutate the CONTRACT (D-3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import invoke_agent

__all__ = ['propose_heuristics_edit']

_DEFAULT_CWD = Path('/home/leo/src/dark-factory')

_OPTIMIZER_SYSTEM_PROMPT = """\
You are a frontier prompt optimizer running a SkillOpt-discipline reflection \
step. You are given ONLY the current HEURISTICS block of a prompt — never \
the CONTRACT, which you cannot see and must not attempt to reconstruct or \
reference — plus a scored minibatch of recent rollouts and a buffer of \
previously rejected edits. Your job is to propose a revised HEURISTICS \
block only.

Rules:
- Propose AT MOST the bounded-edit budget given below of distinct textual \
edits to the heuristics block this step. Bounded means small, targeted \
changes — not a rewrite.
- Do not repeat an edit direction already present in the rejected-edit \
buffer without addressing why it was likely rejected.
- Output ONLY the full revised heuristics block text. Never emit a CONTRACT \
section, and never claim to change the CONTRACT.
"""


def _format_minibatch(scored_minibatch: Sequence[Any]) -> str:
    if not scored_minibatch:
        return '(empty minibatch)'
    lines = [
        f'- item={scored.item!r} score={scored.score}'
        for scored in scored_minibatch
    ]
    return '\n'.join(lines)


def _format_rejected_buffer(rejected_buffer: Sequence[str]) -> str:
    if not rejected_buffer:
        return '(no prior rejected edits)'
    return '\n'.join(
        f'--- rejected candidate {i + 1} ---\n{rejected}'
        for i, rejected in enumerate(rejected_buffer)
    )


def _build_reflection_prompt(
    current_heuristics: str,
    scored_minibatch: Sequence[Any],
    rejected_buffer: Sequence[str],
    max_edits: int,
) -> str:
    return f"""\
## Bounded-Edit Budget

At most {max_edits} distinct textual edits this step.

## Current HEURISTICS Block

{current_heuristics}

## Scored Train Minibatch (this step's rollouts)

{_format_minibatch(scored_minibatch)}

## Rejected-Edit Buffer (previously proposed and rejected)

{_format_rejected_buffer(rejected_buffer)}

## Action

Propose a revised HEURISTICS block (at most {max_edits} bounded edits) that \
should improve the score. Output ONLY the new heuristics block text.
"""


async def propose_heuristics_edit(
    current_heuristics: str,
    scored_minibatch: Sequence[Any],
    rejected_buffer: Sequence[str],
    *,
    max_edits: int,
    optimizer_model: str,
    invoke_fn: Callable[..., Awaitable[Any]] = invoke_agent,
    cwd: Path = _DEFAULT_CWD,
) -> str:
    """Reflect over *scored_minibatch* + *rejected_buffer* and propose a bounded heuristics edit.

    Receives ONLY *current_heuristics* — never the CONTRACT — so the
    CONTRACT is un-editable by construction (D-3). Calls *invoke_fn*
    (default :func:`orchestrator.agents.invoke.invoke_agent`, injectable for
    hermetic tests) with ``model=optimizer_model`` — the frontier optimizer,
    distinct from the executor model that runs rollouts.

    Returns the proposed heuristics block: prefers
    ``result.structured_output['heuristics']`` when present, else
    ``result.output``.
    """
    prompt = _build_reflection_prompt(
        current_heuristics, scored_minibatch, rejected_buffer, max_edits,
    )
    result = await invoke_fn(
        prompt=prompt,
        system_prompt=_OPTIMIZER_SYSTEM_PROMPT,
        cwd=cwd,
        model=optimizer_model,
    )

    structured = result.structured_output
    if isinstance(structured, dict) and 'heuristics' in structured:
        return structured['heuristics']
    return result.output

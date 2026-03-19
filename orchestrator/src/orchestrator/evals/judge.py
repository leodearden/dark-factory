"""LLM pairwise blinded comparison of eval results."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from orchestrator.agents.invoke import AgentResult, invoke_agent

from .snapshots import get_diff

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    """Result of one pairwise comparison."""

    task_id: str
    config_a: str
    config_b: str
    winner: str       # 'A', 'B', or 'tie'
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'winner': {'type': 'string', 'enum': ['A', 'B', 'tie']},
        'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
        'reasoning': {'type': 'string'},
    },
    'required': ['winner', 'confidence', 'reasoning'],
}


def _strip_metadata(diff: str) -> str:
    """Strip identifying metadata from a diff to prevent judge bias."""
    # Remove session IDs, timestamps, model names
    cleaned = re.sub(r'session[_-]?id[:\s]+\S+', '', diff, flags=re.IGNORECASE)
    cleaned = re.sub(r'claude|codex|gemini|opus|sonnet|gpt-\S+|o4-\S+', '[model]',
                     cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '[timestamp]', cleaned)
    return cleaned


async def run_judge(
    result_a: dict,
    result_b: dict,
    task: dict,
) -> JudgeVerdict:
    """Blind pairwise comparison using Claude opus with max thinking.

    result_a/b are EvalResult dicts with 'worktree_path' and 'config_name'.
    """
    # Get diffs
    diff_a = await get_diff(Path(result_a['worktree_path']))
    diff_b = await get_diff(Path(result_b['worktree_path']))

    # Randomize assignment to prevent position bias
    swapped = random.random() > 0.5
    if swapped:
        diff_a, diff_b = diff_b, diff_a

    # Strip identifying metadata
    diff_a = _strip_metadata(diff_a)
    diff_b = _strip_metadata(diff_b)

    task_name = task.get('name', task.get('id', 'unknown'))
    task_desc = task.get('task_definition', {}).get('description', '')

    prompt = f"""You are evaluating two implementations of the same task.

Task: {task_name}
Task description: {task_desc}

## Agent A's implementation
```diff
{diff_a}
```

## Agent B's implementation
```diff
{diff_b}
```

Compare these implementations on:
1. **Correctness**: Does it solve the task completely and correctly?
2. **Code quality**: Readability, idiom, minimal unnecessary changes
3. **Test quality**: Meaningful assertions, edge cases, good coverage
4. **Design coherence**: Fits existing codebase patterns and conventions

Output JSON: {{"winner": "A" or "B" or "tie", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    result = await invoke_agent(
        prompt=prompt,
        system_prompt='You are an expert code reviewer performing blinded pairwise comparison. Be precise and fair.',
        cwd=Path('/tmp'),
        model='opus',
        effort='max',
        backend='claude',
        max_budget_usd=5.0,
        output_schema=JUDGE_SCHEMA,
    )

    # Parse verdict
    try:
        if result.structured_output:
            verdict = result.structured_output
        else:
            verdict = json.loads(result.output)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f'Judge produced unparseable output: {result.output[:200]}')
        verdict = {'winner': 'tie', 'confidence': 0.0, 'reasoning': 'Judge output parse failure'}

    # Account for swap
    winner = verdict.get('winner', 'tie')
    if swapped:
        winner = {'A': 'B', 'B': 'A', 'tie': 'tie'}.get(winner, 'tie')

    return JudgeVerdict(
        task_id=task.get('id', 'unknown'),
        config_a=result_a['config_name'],
        config_b=result_b['config_name'],
        winner=winner,
        confidence=verdict.get('confidence', 0.0),
        reasoning=verdict.get('reasoning', ''),
    )


async def run_tournament(
    results: list[dict],
    task: dict,
    rounds_per_pair: int = 3,
) -> list[JudgeVerdict]:
    """Run all pairwise comparisons, N rounds each for consistency."""
    verdicts = []
    pairs = list(combinations(results, 2))

    for a, b in pairs:
        for round_num in range(rounds_per_pair):
            logger.info(
                f'Judge: {a["config_name"]} vs {b["config_name"]} '
                f'(round {round_num + 1}/{rounds_per_pair})'
            )
            v = await run_judge(a, b, task)
            verdicts.append(v)

    return verdicts

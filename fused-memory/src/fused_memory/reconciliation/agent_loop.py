"""Generic LLM agent loop with tool dispatch for reconciliation stages."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid as uuid_mod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolParam
    from openai.types.chat import ChatCompletionMessageParam

from shared.cli_invoke import (
    AgentFailureKind,
    build_failure_message,
    classify_agent_failure,
    invoke_with_cap_retry,
    no_mcp_servers_config,
)

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import JournalEntry
from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS

logger = logging.getLogger(__name__)

# CLI failure kinds that end a claude_cli run as an audited agent failure
# instead of raising, each mapped to its census token.  'api_refusal' since
# task 6022; 'cli_max_turns' since task 4344, when one invocation came to carry
# the whole investigation.  Every other failure kind still raises.
_CLI_FAILURE_ORIGINS = {
    AgentFailureKind.API_REFUSAL: 'api_refusal',
    AgentFailureKind.MAX_TURNS: 'cli_max_turns',
}

# The CLOSED vocabulary of CLI-failure origins run() reports as
# `warning_origin` (task 4343).  All four members are synthesised by
# _call_claude_cli; nothing else may enter, because the value lands in
# VerificationResult.failure_token and from there in the reconciliation.db
# `verify/codebase` audit row that operators GROUP BY.  An unbounded or
# agent-controlled token there would make that census unqueryable.
CLI_WARNING_ORIGINS = frozenset({
    'cli_output_unparseable',
    'cli_output_empty',
    *_CLI_FAILURE_ORIGINS.values(),
})

# max_turns for the claude_cli provider's ONE invocation, which carries the
# WHOLE investigation: the CLI runs its own built-in tools turn by turn and
# ends with StructuredOutput.
#
# Never 1.  With --json-schema the model emits a prose turn before it calls
# ``StructuredOutput``, and a cap of 1 leaves no room for it: the CLI returns
# ``error_max_turns``, which in the measured runs carried no structured payload,
# so schema salvage had nothing to recover and the call failed.  That is
# measured CLI behaviour, not a guarantee.
#
# 20 is the measured configuration, a ceiling rather than a target; spend stays
# bounded by cli_invoke's ``max_budget_usd`` and ``agent_cli_timeout_seconds``.
# The rates, turn counts and CLI versions are recorded only in
# fused-memory/scripts/probe_schema_max_turns.py's docstring; re-run it rather
# than trusting a number copied elsewhere.
_AGENT_CLI_MAX_TURNS = 20


def _cli_failure_payload(origin: str, text: str) -> dict:
    """The single constructor of a CLI no-tool-call exit carrying *origin*."""
    assert origin in CLI_WARNING_ORIGINS, origin
    return {'warning': 'no_tool_calls', 'text': text, 'warning_origin': origin}


class CircuitBreakerError(Exception):
    """Raised when mutation count exceeds the per-stage limit."""


@dataclass
class ToolDefinition:
    """Wraps a tool the agent can call."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable  # Async callable
    is_mutation: bool = False
    target_system: str = ''  # 'graphiti', 'mem0', 'taskmaster'
    get_before_state: Callable | None = None

    def to_anthropic_schema(self) -> ToolParam:
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.parameters,
        }


class AgentLoop:
    """Runs an LLM agent with tool access until it signals completion.

    Two transports:

    * The in-process providers (anthropic/openai) dispatch ``tools`` step by
      step until ``terminal_tool`` is called.
    * The claude_cli provider runs the whole loop inside ONE CLI invocation
      that uses the CLI's own built-in ``cli_tools``.  There ``tools``
      contributes only the terminal tool's ``parameters``, used as the CLI
      output schema.
    """

    def __init__(
        self,
        config: ReconciliationConfig,
        system_prompt: str,
        tools: dict[str, ToolDefinition],
        terminal_tool: str = 'stage_complete',
        usage_gate=None,
        cwd: Path | None = None,
        cli_tools: Sequence[str] = (),
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.terminal_tool = terminal_tool
        self.cli_tools: tuple[str, ...] = tuple(cli_tools)
        self._journal_entries: list[JournalEntry] = []
        self._mutation_count: int = 0
        self.llm_call_count: int = 0
        self.token_count: int = 0
        self._usage_gate = usage_gate
        # Task 4722 / PRD D5: the codebase root this agent runs in.  The
        # verifier now supplies the TASK's own project root per call; every
        # other construction site passes nothing and keeps the process-global
        # config value, so the fallback is what leaves those sites unaffected.
        # Resolved once here so `self.cwd` is always a Path.
        self.cwd: Path = Path(cwd) if cwd is not None else Path(config.explore_codebase_root)

    async def run(self, initial_payload: str) -> tuple[dict, list[JournalEntry]]:
        """Execute agent loop. Returns (terminal_tool_args, journal_entries)."""
        if self.config.agent_llm_provider == 'claude_cli':
            # Nothing executes in-process on this transport, so the journal
            # stays empty.
            return await self._call_claude_cli(initial_payload), self._journal_entries

        messages: list[dict[str, Any]] = [
            {'role': 'user', 'content': initial_payload},
        ]

        tool_schemas = [t.to_anthropic_schema() for t in self.tools.values()]

        for _step in range(self.config.agent_max_steps):
            response = await self._call_llm(messages, tool_schemas)

            # Check for terminal tool in tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == 'tool_use']
            text_blocks = [b for b in response.content if b.type == 'text']
            reasoning_text = '\n'.join(b.text for b in text_blocks).strip()

            if not tool_use_blocks:
                # No tool calls — agent stopped
                text = ' '.join(b.text for b in text_blocks) if text_blocks else ''
                return {'warning': 'no_tool_calls', 'text': text}, self._journal_entries

            tool_results = []
            terminal_result = None

            for block in tool_use_blocks:
                if block.name == self.terminal_tool:
                    terminal_result = block.input
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps({'status': 'complete'}),
                    })
                    break

                try:
                    result = await self._execute_tool(block, reasoning=reasoning_text)
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps(result) if not isinstance(result, str) else result,
                    })
                except CircuitBreakerError:
                    raise
                except Exception as e:
                    logger.error(f'Tool {block.name} failed: {e}')
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps({'error': str(e)}),
                        'is_error': True,
                    })

            if terminal_result is not None:
                return terminal_result, self._journal_entries

            # Append assistant message and tool results
            messages.append({'role': 'assistant', 'content': response.content})
            messages.append({'role': 'user', 'content': tool_results})

        return {'warning': 'max_steps_reached'}, self._journal_entries

    async def _execute_tool(self, tool_block: Any, reasoning: str = '') -> Any:
        """Execute a tool call, journal if mutation."""
        tool_name = tool_block.name
        tool_args = tool_block.input or {}

        if tool_name not in self.tools:
            return {'error': f'Unknown tool: {tool_name}'}

        tool = self.tools[tool_name]

        before_state = None
        if tool.is_mutation:
            self._mutation_count += 1
            if self._mutation_count > self.config.max_mutations_per_stage:
                raise CircuitBreakerError(
                    f'Exceeded {self.config.max_mutations_per_stage} mutations in stage'
                )
            if tool.get_before_state:
                try:
                    before_state = await tool.get_before_state(**tool_args)
                except Exception:
                    before_state = None

        try:
            result = await asyncio.wait_for(
                tool.function(**tool_args),
                timeout=self.config.tool_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f'Tool {tool_name} timed out after {self.config.tool_timeout_seconds}s')
            result = {'error': f'Tool {tool_name} timed out after {self.config.tool_timeout_seconds}s'}

        if tool.is_mutation:
            self._journal_entries.append(
                JournalEntry(
                    id=str(uuid_mod.uuid4()),
                    timestamp=datetime.now(UTC),
                    operation=tool_name,
                    target_system=tool.target_system,
                    before_state=before_state,
                    after_state=_safe_serialize(result),
                    reasoning=reasoning,
                    evidence=[],
                )
            )

        return result

    async def _call_llm(self, messages: list[dict[str, Any]], tool_schemas: list[ToolParam]) -> Any:
        """Call the configured LLM provider."""
        import anthropic

        provider = self.config.agent_llm_provider

        if provider == 'anthropic':
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.config.agent_llm_model,
                max_tokens=self.config.agent_max_tokens,
                system=self.system_prompt,
                messages=cast('list[MessageParam]', messages),
                tools=tool_schemas,
            )
            self.llm_call_count += 1
            self.token_count += response.usage.input_tokens + response.usage.output_tokens
            return response
        elif provider == 'openai':
            return await self._call_openai(messages, tool_schemas)
        else:
            raise ValueError(f'Unsupported agent LLM provider: {provider}')

    async def _call_openai(self, messages: list[dict[str, Any]], tool_schemas: list[ToolParam]) -> Any:
        """Call OpenAI and convert response to Anthropic-like format."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        # Convert Anthropic tool schemas to OpenAI format
        openai_tools: list[Any] = []
        for schema in tool_schemas:
            openai_tools.append({
                'type': 'function',
                'function': {
                    'name': schema['name'],
                    'description': schema.get('description', ''),
                    'parameters': schema['input_schema'],
                },
            })

        # Convert messages: flatten Anthropic content blocks to OpenAI format
        openai_messages: list[ChatCompletionMessageParam] = [{'role': 'system', 'content': self.system_prompt}]
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if isinstance(content, str):
                openai_messages.append({'role': role, 'content': content})
            elif isinstance(content, list):
                # Tool results or mixed content
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_result':
                        openai_messages.append({
                            'role': 'tool',
                            'tool_call_id': block['tool_use_id'],
                            'content': block['content'],
                        })
                    elif isinstance(block, _TextBlock):
                        openai_messages.append({'role': role, 'content': block.text})
                    elif isinstance(block, _ToolUseBlock):
                        openai_messages.append({
                            'role': 'assistant',
                            'tool_calls': [{
                                'id': block.id,
                                'type': 'function',
                                'function': {
                                    'name': block.name,
                                    'arguments': json.dumps(block.input),
                                },
                            }],
                        })

        create_kwargs: dict[str, Any] = {
            'model': self.config.agent_llm_model,
            'messages': openai_messages,
        }
        if openai_tools:
            create_kwargs['tools'] = openai_tools
        response = await client.chat.completions.create(**create_kwargs)

        self.llm_call_count += 1
        if response.usage:
            self.token_count += response.usage.total_tokens

        # Convert OpenAI response to Anthropic-like structure
        return _OpenAIResponseAdapter(response)

    def _cli_system_prompt(self) -> str:
        """The caller's system prompt plus how to finish on the CLI transport.

        Names only the CLI's own tools and StructuredOutput: any in-process
        tool name reaching the CLI, the terminal tool's included, is one the
        model may try to call natively, and the CLI rejects every such call.
        """
        if self.cli_tools:
            tools_line = (
                f'Investigate with your {", ".join(self.cli_tools)} tools; they work '
                'inside your working directory and no other tool is available.'
            )
        else:
            tools_line = 'No tool is available to you for investigation.'
        return (
            f'{self.system_prompt}\n\n## Tools and finishing\n'
            f'{tools_line} When you are done, call StructuredOutput once with your '
            'findings. That call is your final answer.'
        )

    async def _call_claude_cli(self, prompt: str) -> dict:
        """Run the whole investigation as ONE Claude CLI invocation.

        Returns the terminal tool's payload, or a ``_cli_failure_payload`` for
        a CLI failure in ``_CLI_FAILURE_ORIGINS`` or a structured output that
        is unparseable or empty.  Every other failure raises RuntimeError.
        """
        terminal = self.tools.get(self.terminal_tool)
        if terminal is None:
            raise ValueError(
                f'the claude_cli provider needs the terminal tool {self.terminal_tool!r} '
                'in tools: its parameters are the CLI output schema'
            )
        result = await invoke_with_cap_retry(
            usage_gate=self._usage_gate,
            label=f'Reconciliation agent ({self.config.agent_llm_model})',
            prompt=prompt,
            system_prompt=self._cli_system_prompt(),
            output_schema=terminal.parameters,
            available_tools=list(self.cli_tools),
            # No allow rules: under dontAsk the CLI's built-in read tools stay
            # confined to cwd (measured; see the probe script's docstring).
            permission_mode='dontAsk',
            # --tools does not filter MCP, so MCP is closed by the strict empty
            # config.  cwd is the codebase root (tasks 1989/4722), which may
            # hold a live .mcp.json.
            mcp_config=no_mcp_servers_config(),
            strict_mcp_config=True,
            model=self.config.agent_llm_model,
            # See _AGENT_CLI_MAX_TURNS for why this is not 1.
            max_turns=_AGENT_CLI_MAX_TURNS,
            timeout_seconds=float(self.config.agent_cli_timeout_seconds),
            cwd=self.cwd,
            cap_wait_sanity_secs=_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS,
        )
        # Counted before the failure guard: a failed or refused result still
        # reached the model and was billed.
        self.llm_call_count += 1
        self.token_count += (result.input_tokens or 0) + (result.output_tokens or 0)

        if not result.success:
            origin = _CLI_FAILURE_ORIGINS.get(classify_agent_failure(result).kind)
            if origin is None:
                raise RuntimeError(build_failure_message('Claude CLI agent', result))
            logger.warning('%s: Claude CLI agent failed. Output: %s', origin, result.output[:500])
            return _cli_failure_payload(origin, result.output)

        structured = result.structured_output
        if isinstance(structured, str):
            try:
                structured = json.loads(structured)
            except json.JSONDecodeError:
                logger.warning(
                    'cli_output_unparseable: Claude CLI reported success but structured_output'
                    ' was unparseable JSON. Raw prefix: %s',
                    structured[:200],
                )
                return _cli_failure_payload('cli_output_unparseable', structured)
        if not structured or not isinstance(structured, dict):
            logger.warning(
                'cli_output_empty: Claude CLI reported success but structured_output'
                ' was empty, missing or not an object.',
            )
            return _cli_failure_payload('cli_output_empty', '')
        return structured


class _OpenAIResponseAdapter:
    """Adapts OpenAI response to look like Anthropic Messages response."""

    def __init__(self, response: Any):
        self._response = response
        choice = response.choices[0]
        self.content = []

        if choice.message.content:
            self.content.append(_TextBlock(choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                self.content.append(
                    _ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )


@dataclass
class _TextBlock:
    text: str
    type: str = 'text'


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = field(default='tool_use', repr=False)


def _safe_serialize(obj: Any) -> dict | None:
    """Safely convert an object to a dict for journaling."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return {'repr': str(obj)}

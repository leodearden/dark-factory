"""Generic LLM agent loop with tool dispatch for reconciliation stages."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid as uuid_mod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolParam
    from openai.types.chat import ChatCompletionMessageParam

from shared.cli_invoke import AgentResult, classify_agent_failure, invoke_with_cap_retry

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import JournalEntry

logger = logging.getLogger(__name__)

CLAUDE_CLI_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'thinking': {'type': 'string'},
        # Optional free-form assistant reply (distinct from thinking: thinking is
        # private reasoning; response is the visible answer when no tool calls
        # are needed).  Not required by the schema — most turns only produce
        # tool_calls.
        'response': {'type': 'string'},
        'tool_calls': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'name': {'type': 'string'},
                    'input': {'type': 'object'},
                },
                'required': ['id', 'name', 'input'],
            },
        },
    },
    'required': ['thinking', 'tool_calls'],
}


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
    """Runs an LLM agent with tool access until it signals completion."""

    def __init__(
        self,
        config: ReconciliationConfig,
        system_prompt: str,
        tools: dict[str, ToolDefinition],
        terminal_tool: str = 'stage_complete',
        usage_gate=None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.tools = tools
        self.terminal_tool = terminal_tool
        self._journal_entries: list[JournalEntry] = []
        self._mutation_count: int = 0
        self.llm_call_count: int = 0
        self.token_count: int = 0
        self._cli_session_id: str | None = None
        self._usage_gate = usage_gate

    async def run(self, initial_payload: str) -> tuple[dict, list[JournalEntry]]:
        """Execute agent loop. Returns (terminal_tool_args, journal_entries)."""
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
        elif provider == 'claude_cli':
            content = messages[-1]['content']
            prompt = content if isinstance(content, str) else self._serialize_tool_results(content)
            return await self._call_claude_cli(prompt=prompt, tools=tool_schemas)
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

    def _build_cli_system_prompt(self, tool_schemas: list[ToolParam]) -> str:
        """Build system prompt that includes tool schemas for CLI-based tool dispatch."""
        tools_section = []
        for t in tool_schemas:
            tools_section.append(
                f"### {t['name']}\n"
                f"{t.get('description', '')}\n"
                f"Parameters: {json.dumps(t['input_schema'], indent=2)}"
            )
        return (
            f"{self.system_prompt}\n\n"
            "## Available Tools\n"
            "Respond with a JSON object matching the provided schema.\n"
            "Use the tool_calls array to invoke tools. Each tool call needs:\n"
            '- "id": a unique string identifier\n'
            '- "name": the tool name from the list below\n'
            '- "input": an object matching the tool\'s parameters\n\n'
            "Use the \"thinking\" field to explain your reasoning before making tool calls.\n"
            "If you have no more tool calls to make, return an empty tool_calls array.\n\n"
            + "\n\n".join(tools_section)
        )

    @staticmethod
    def _serialize_tool_results(tool_results: list[dict]) -> str:
        """Format tool results as text for the next CLI turn."""
        parts = []
        for tr in tool_results:
            if isinstance(tr, dict) and tr.get('type') == 'tool_result':
                status = 'ERROR' if tr.get('is_error') else 'OK'
                parts.append(
                    f"[Tool Result: {tr['tool_use_id']}] ({status})\n{tr['content']}"
                )
        return '\n\n'.join(parts)

    async def _call_claude_cli(self, prompt: str, tools: list[ToolParam]) -> _CLIResponseAdapter:
        """Delegate to shared.cli_invoke.invoke_with_cap_retry.

        Multi-turn is handled by passing ``resume_session_id`` on subsequent
        calls; ``_call_llm`` is responsible for serialising tool results into
        the next turn's prompt before calling this method.

        ``_cli_session_id`` is cleared to ``None`` before any exception propagates
        out of this method — whether from ``invoke_with_cap_retry`` itself (e.g.
        ``AllAccountsCappedException`` after the retry loop gives up) or from the
        ``not result.success`` failure guard below — so that a subsequent
        reconciliation retry on the same ``AgentLoop`` instance does not attempt to
        ``--resume`` an abandoned or capped session.
        """
        try:
            result: AgentResult = await invoke_with_cap_retry(
                usage_gate=self._usage_gate,
                label=f'Reconciliation agent ({self.config.agent_llm_model})',
                prompt=prompt,
                system_prompt=self._build_cli_system_prompt(tools),
                output_schema=CLAUDE_CLI_RESPONSE_SCHEMA,
                disallowed_tools=['*'],
                model=self.config.agent_llm_model,
                # max_turns=1: AgentLoop.run() drives multi-turn externally by calling
                # _call_claude_cli again with resume_session_id.  A single CLI
                # invocation only needs one assistant turn (schema tool-use → JSON
                # response happens within the same turn when --json-schema is used).
                max_turns=1,
                permission_mode='bypassPermissions',
                timeout_seconds=float(self.config.agent_cli_timeout_seconds),
                resume_session_id=self._cli_session_id,
                cwd=Path(self.config.explore_codebase_root),
            )

            if not result.success:
                # schema_salvaged=True implies success=True (cli_invoke.py:749-751),
                # so `not result.success` is the complete failure guard.
                cls = classify_agent_failure(result)
                raise RuntimeError(
                    f'Claude CLI agent failed: {cls.summary}\n{cls.diagnostic_detail}'
                )
        except Exception:
            # Clear stale session id so callers that retry don't --resume an abandoned session.
            self._cli_session_id = None
            raise

        self._cli_session_id = result.session_id or self._cli_session_id
        self.llm_call_count += 1
        self.token_count += (result.input_tokens or 0) + (result.output_tokens or 0)

        structured = result.structured_output
        if isinstance(structured, str):
            try:
                structured = json.loads(structured)
            except json.JSONDecodeError:
                structured = {'thinking': structured, 'tool_calls': []}
        if not structured:
            structured = {'thinking': '', 'tool_calls': []}

        return _CLIResponseAdapter(structured, session_id=result.session_id)


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


class _CLIResponseAdapter:
    """Adapts Claude CLI structured output to look like Anthropic Messages response.

    Exposes both the legacy ``.content`` list (for drop-in compatibility with
    the anthropic/openai branches) and direct attribute access (for
    delegation-level tests and future callers that don't need the block list):
    ``.thinking``, ``.response``, ``.tool_calls``, ``.session_id``.
    """

    def __init__(self, structured_output: dict, session_id: str = ''):
        self.content = []
        self.usage = _CLIUsage()

        # Direct attribute access
        self.thinking: str = structured_output.get('thinking', '')
        self.response: str = structured_output.get('response', '')
        self.tool_calls: list = structured_output.get('tool_calls', [])
        self.session_id: str = session_id

        if self.thinking:
            self.content.append(_TextBlock(self.thinking))

        for tc in self.tool_calls:
            self.content.append(
                _ToolUseBlock(
                    id=tc.get('id', str(uuid_mod.uuid4())),
                    name=tc['name'],
                    input=tc.get('input', {}),
                )
            )


@dataclass
class _CLIUsage:
    input_tokens: int = 0
    output_tokens: int = 0


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

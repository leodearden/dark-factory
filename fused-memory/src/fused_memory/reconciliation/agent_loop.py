"""Generic LLM agent loop with tool dispatch for reconciliation stages."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid as uuid_mod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import JournalEntry

logger = logging.getLogger(__name__)

CLAUDE_CLI_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'thinking': {'type': 'string'},
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

    def to_anthropic_schema(self) -> dict:
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
        messages: list[dict] = [
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

    async def _call_llm(self, messages: list[dict], tool_schemas: list[dict]) -> Any:
        """Call the configured LLM provider."""
        import anthropic

        provider = self.config.agent_llm_provider

        if provider == 'anthropic':
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.config.agent_llm_model,
                max_tokens=self.config.agent_max_tokens,
                system=self.system_prompt,
                messages=messages,
                tools=tool_schemas,
            )
            self.llm_call_count += 1
            self.token_count += response.usage.input_tokens + response.usage.output_tokens
            return response
        elif provider == 'openai':
            return await self._call_openai(messages, tool_schemas)
        elif provider == 'claude_cli':
            return await self._call_claude_cli(messages, tool_schemas)
        else:
            raise ValueError(f'Unsupported agent LLM provider: {provider}')

    async def _call_openai(self, messages: list[dict], tool_schemas: list[dict]) -> Any:
        """Call OpenAI and convert response to Anthropic-like format."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        # Convert Anthropic tool schemas to OpenAI format
        openai_tools = []
        for schema in tool_schemas:
            openai_tools.append({
                'type': 'function',
                'function': {
                    'name': schema['name'],
                    'description': schema['description'],
                    'parameters': schema['input_schema'],
                },
            })

        # Convert messages: flatten Anthropic content blocks to OpenAI format
        openai_messages = [{'role': 'system', 'content': self.system_prompt}]
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

    def _build_cli_system_prompt(self, tool_schemas: list[dict]) -> str:
        """Build system prompt that includes tool schemas for CLI-based tool dispatch."""
        tools_section = []
        for t in tool_schemas:
            tools_section.append(
                f"### {t['name']}\n"
                f"{t['description']}\n"
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

    async def _call_claude_cli(self, messages: list[dict], tool_schemas: list[dict]) -> Any:
        """Call Claude via CLI subprocess with --resume for multi-turn.

        Includes: stdin isolation, stdout+stderr error reading, and usage-cap
        retry with account failover when a UsageGate is attached.
        """
        import asyncio as _asyncio

        while True:
            # 1. Gate: get OAuth token (or None)
            oauth_token = None
            if self._usage_gate:
                oauth_token = await self._usage_gate.before_invoke()

            is_first_call = self._cli_session_id is None
            if is_first_call:
                self._cli_session_id = str(uuid_mod.uuid4())

            cmd = [
                'claude', '--print', '--output-format', 'json',
                '--model', self.config.agent_llm_model,
                '--json-schema', json.dumps(CLAUDE_CLI_RESPONSE_SCHEMA),
                '--permission-mode', 'bypassPermissions',
                '--tools', '',
            ]

            assert self._cli_session_id is not None
            if is_first_call:
                cmd.extend(['--system-prompt', self._build_cli_system_prompt(tool_schemas)])
                cmd.extend(['--session-id', self._cli_session_id])
                prompt = messages[-1]['content']  # Initial payload string
            else:
                cmd.extend(['--resume', self._cli_session_id])
                prompt = self._serialize_tool_results(messages[-1]['content'])

            cmd.extend(['--', prompt])

            # 2. Build env: strip ANTHROPIC_API_KEY, inject OAuth token
            env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
            if oauth_token:
                env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token

            try:
                proc = await _asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=_asyncio.subprocess.DEVNULL,
                    stdout=_asyncio.subprocess.PIPE,
                    stderr=_asyncio.subprocess.PIPE,
                    env=env,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    'Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code'
                ) from exc

            try:
                stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=180)
            except TimeoutError as exc:
                proc.kill()
                raise RuntimeError('Claude CLI timed out after 180 seconds') from exc

            stdout_text = stdout.decode()
            stderr_text = stderr.decode()

            # 3. Check for cap hit before processing results
            if self._usage_gate and self._usage_gate.detect_cap_hit(
                stderr_text, stdout_text, 'claude', oauth_token=oauth_token,
            ):
                logger.warning('Usage cap hit during CLI call, resetting session and retrying')
                # Reset session so next call starts fresh (can't --resume a capped session)
                if is_first_call:
                    self._cli_session_id = None
                continue

            # 4. Non-zero exit: include both stdout and stderr in error
            if proc.returncode != 0:
                error_detail = stderr_text[-500:] if stderr_text.strip() else stdout_text[-500:]
                raise RuntimeError(
                    f'Claude CLI exited with code {proc.returncode}: {error_detail}'
                )

            if not stdout_text.strip():
                raise RuntimeError('Claude CLI produced no output')

            result = json.loads(stdout_text)

            # Update session_id from result if available
            if result.get('session_id'):
                self._cli_session_id = result['session_id']

            # Extract structured output
            structured = result.get('structured_output')
            if isinstance(structured, str):
                structured = json.loads(structured)
            if not structured:
                result_text = result.get('result', '')
                if result_text:
                    try:
                        structured = json.loads(result_text)
                    except (json.JSONDecodeError, TypeError):
                        structured = {'thinking': result_text, 'tool_calls': []}
                else:
                    structured = {'thinking': '', 'tool_calls': []}

            self.llm_call_count += 1
            self.token_count += (
                int(result.get('num_input_tokens', 0))
                + int(result.get('num_output_tokens', 0))
            )

            return _CLIResponseAdapter(structured)


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
    """Adapts Claude CLI structured output to look like Anthropic Messages response."""

    def __init__(self, structured_output: dict):
        self.content = []
        self.usage = _CLIUsage()

        thinking = structured_output.get('thinking', '')
        if thinking:
            self.content.append(_TextBlock(thinking))

        for tc in structured_output.get('tool_calls', []):
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

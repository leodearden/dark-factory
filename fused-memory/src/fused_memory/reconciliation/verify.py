"""Codebase verification via isolated explore agent."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import VerificationResult
from fused_memory.reconciliation.agent_loop import AgentLoop, ToolDefinition

logger = logging.getLogger(__name__)

EXPLORE_AGENT_SYSTEM_PROMPT = """\
You are a Codebase Explorer agent. Your job is to verify factual claims against the actual \
codebase. You are strictly read-only and have no access to memory systems or task systems.

## Guidelines
- Be neutral: report what the code says, don't speculate.
- Every claim must cite specific evidence: file paths, line ranges, code snippets.
- If you can't find evidence either way, say "inconclusive" — don't guess.
- Check git history when the claim involves changes over time.
- Focus your search on the scope hints provided, but expand if needed.

## Output
When done, call `verification_complete` with your findings:
- verdict: "confirmed" | "contradicted" | "inconclusive"
- confidence: 0.0-1.0
- evidence: list of {file_path, line_range, snippet, relevance}
- summary: brief explanation
- git_context: {latest_relevant_commit, author, date} if applicable
"""


class CodebaseVerifier:
    """Spawns an isolated explore agent to verify factual claims against the codebase."""

    def __init__(self, config: ReconciliationConfig):
        self.codebase_root = Path(config.explore_codebase_root).resolve()
        self.config = config

    async def verify(
        self,
        claim: str,
        context: str = '',
        scope_hints: list[str] | None = None,
        project_id: str = '',
    ) -> VerificationResult:
        """Verify a factual claim against the codebase."""
        codebase_root = self.codebase_root

        tools: dict[str, ToolDefinition] = {}

        async def read_file(path: str, max_lines: int = 200) -> dict:
            """Read file contents."""
            full_path = codebase_root / path
            if not full_path.resolve().is_relative_to(codebase_root):
                return {'error': 'Path outside codebase root'}
            try:
                content = full_path.read_text()
                lines = content.splitlines()
                if len(lines) > max_lines:
                    lines = lines[:max_lines]
                    lines.append(f'... ({len(content.splitlines()) - max_lines} more lines)')
                return {'path': path, 'content': '\n'.join(lines), 'total_lines': len(content.splitlines())}
            except FileNotFoundError:
                return {'error': f'File not found: {path}'}
            except Exception as e:
                return {'error': str(e)}

        tools['read_file'] = ToolDefinition(
            name='read_file',
            description='Read file contents from the codebase.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Relative path from codebase root'},
                    'max_lines': {'type': 'integer', 'default': 200},
                },
                'required': ['path'],
            },
            function=read_file,
        )

        async def glob_search(pattern: str) -> dict:
            """Search for files matching a glob pattern."""
            try:
                matches = sorted(codebase_root.glob(pattern))
                # Make relative and limit results
                paths = [str(m.relative_to(codebase_root)) for m in matches[:50]]
                return {'pattern': pattern, 'matches': paths, 'total': len(matches)}
            except Exception as e:
                return {'error': str(e)}

        tools['glob_search'] = ToolDefinition(
            name='glob_search',
            description='Search for files matching a glob pattern (e.g., "**/*.py", "src/**/*.ts").',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                },
                'required': ['pattern'],
            },
            function=glob_search,
        )

        async def grep_search(pattern: str, path: str = '.', max_results: int = 30) -> dict:
            """Search file contents with regex."""
            search_path = codebase_root / path
            if not search_path.resolve().is_relative_to(codebase_root):
                return {'error': 'Path outside codebase root'}
            try:
                proc = await asyncio.create_subprocess_exec(
                    'grep', '-rn', '-E', '--include=*.py', '--include=*.ts',
                    '--include=*.js', '--include=*.yaml', '--include=*.yml',
                    '--include=*.json', '--include=*.md', '--include=*.toml',
                    pattern, str(search_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                lines = stdout.decode().splitlines()[:max_results]
                # Make paths relative
                results = []
                for line in lines:
                    rel = line.replace(str(codebase_root) + '/', '', 1)
                    results.append(rel)
                return {'pattern': pattern, 'results': results}
            except asyncio.TimeoutError:
                return {'pattern': pattern, 'results': [], 'error': 'timeout'}
            except Exception as e:
                return {'error': str(e)}

        tools['grep_search'] = ToolDefinition(
            name='grep_search',
            description='Search file contents with regex pattern.',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string', 'description': 'Regex pattern'},
                    'path': {'type': 'string', 'default': '.', 'description': 'Relative path to search in'},
                    'max_results': {'type': 'integer', 'default': 30},
                },
                'required': ['pattern'],
            },
            function=grep_search,
        )

        async def git_log(path: str | None = None, max_entries: int = 10) -> dict:
            """View git history."""
            cmd = ['git', 'log', f'--max-count={max_entries}',
                   '--format=%H|%an|%ai|%s']
            if path:
                cmd.append('--')
                cmd.append(path)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(codebase_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                entries = []
                for line in stdout.decode().splitlines():
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        entries.append({
                            'hash': parts[0][:12],
                            'author': parts[1],
                            'date': parts[2],
                            'message': parts[3],
                        })
                return {'entries': entries}
            except Exception as e:
                return {'error': str(e)}

        tools['git_log'] = ToolDefinition(
            name='git_log',
            description='View git commit history, optionally filtered to a file path.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path to filter history'},
                    'max_entries': {'type': 'integer', 'default': 10},
                },
            },
            function=git_log,
        )

        async def git_show(commit: str, path: str | None = None) -> dict:
            """Show a specific commit's changes."""
            cmd = ['git', 'show', '--stat', commit]
            if path:
                cmd.extend(['--', path])
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(codebase_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                output = stdout.decode()[:3000]
                return {'commit': commit, 'output': output}
            except Exception as e:
                return {'error': str(e)}

        tools['git_show'] = ToolDefinition(
            name='git_show',
            description='Show a specific git commit (stat view).',
            parameters={
                'type': 'object',
                'properties': {
                    'commit': {'type': 'string'},
                    'path': {'type': 'string'},
                },
                'required': ['commit'],
            },
            function=git_show,
        )

        # Terminal tool
        tools['verification_complete'] = ToolDefinition(
            name='verification_complete',
            description='Signal verification is complete with your findings.',
            parameters={
                'type': 'object',
                'properties': {
                    'verdict': {
                        'type': 'string',
                        'enum': ['confirmed', 'contradicted', 'inconclusive'],
                    },
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'evidence': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'file_path': {'type': 'string'},
                                'line_range': {'type': 'string'},
                                'snippet': {'type': 'string'},
                                'relevance': {'type': 'string'},
                            },
                        },
                    },
                    'summary': {'type': 'string'},
                    'git_context': {'type': 'object'},
                },
                'required': ['verdict', 'confidence', 'evidence', 'summary'],
            },
            function=lambda **kw: kw,
        )

        # Build prompt
        hint_text = ''
        if scope_hints:
            hint_text = f'\n\n### Scope Hints\nFocus your search on: {", ".join(scope_hints)}'

        prompt = f"""## Verification Request

### Claim
{claim}

### Context
{context or "No additional context."}
{hint_text}

### Codebase Root
{self.codebase_root}

Investigate this claim against the codebase and call `verification_complete` with your findings.
"""

        agent = AgentLoop(
            config=self.config,
            system_prompt=EXPLORE_AGENT_SYSTEM_PROMPT,
            tools=tools,
            terminal_tool='verification_complete',
        )

        result, _ = await agent.run(prompt)

        return VerificationResult(
            verdict=result.get('verdict', 'inconclusive'),
            confidence=result.get('confidence', 0.0),
            evidence=result.get('evidence', []),
            summary=result.get('summary', ''),
            git_context=result.get('git_context'),
        )

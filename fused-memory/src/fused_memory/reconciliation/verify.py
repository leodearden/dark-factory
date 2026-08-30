"""Codebase verification via isolated explore agent."""

import asyncio
import logging
from pathlib import Path

from shared.agent_result import extract_agent_verdict

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import VerificationResult, VerificationVerdict
from fused_memory.reconciliation.agent_loop import AgentLoop, ToolDefinition

logger = logging.getLogger(__name__)

# Upper bound on a stored failure token.  Real tokens are ~20 chars; anything
# longer is not a token an operator can GROUP BY, so it is truncated rather
# than allowed to bloat the audit row's detail JSON.
_MAX_FAILURE_TOKEN_LEN = 64

# The failure token for a codebase root that is not a usable checkout.
#
# This is a CLOSED-VOCABULARY census value: it lands in the `detail` JSON of
# a `verify|codebase|agent_failed` row (task 4343's audit contract) and is
# what an operator GROUPs BY in reconciliation.db to size this failure mode.
# Rename it only together with the dashboards and queries that read it, and
# never let it become agent-controlled — every other token on this path goes
# through _as_failure_token precisely because the agent can influence those.
CODEBASE_ROOT_UNRESOLVED = 'codebase_root_unresolved'


# Sub-reasons for a refused root.  They discriminate the populations hiding
# behind the single CODEBASE_ROOT_UNRESOLVED token: 'no_dot_git' is a real
# tree that merely is not a checkout ROOT (a monorepo subdirectory, or a
# project not under git at all) and would refuse that project's every sparse
# verification forever, which is a wholly different operational story from a
# path that does not exist.  They ride the WARNING and the summary prose
# ONLY — the census token stays closed-vocabulary and single-valued so an
# operator's GROUP BY does not fragment when a sub-reason is added here.
ROOT_DEFECT_UNRESOLVABLE = 'unresolvable'
ROOT_DEFECT_MISSING = 'missing'
ROOT_DEFECT_NOT_A_DIR = 'not_a_dir'
ROOT_DEFECT_NO_DOT_GIT = 'no_dot_git'


def _resolve_codebase_root(root: Path) -> tuple[Path, str | None]:
    """Resolve ``root`` and say why it is not a checkout the agent can search.

    Returns ``(resolved_root, None)`` when usable, else ``(root, <sub-reason>)``.
    The resolve lives HERE, inside the guard, rather than at the call site: it
    is the one step of the pre-flight that can RAISE, so leaving it outside
    would make this function's fail-closed-by-construction claim true of the
    checks but not of the pre-flight as a whole.

    Stat-only by design (INV-8): verify() runs on the event loop, so a
    ``git rev-parse --show-toplevel`` probe — stricter, but a subprocess —
    is deliberately not used.

    ``.exists()`` and not ``.is_dir()`` on the ``.git`` entry: in a
    ``git worktree`` checkout — which is how this factory runs every task,
    and how the non-dark_factory projects this check exists to serve are laid
    out — ``.git`` is a FILE holding a ``gitdir:`` pointer.  Requiring a
    directory would refuse exactly that population.
    """
    try:
        resolved = Path(root).resolve()
    except (OSError, ValueError):
        # ``Path.resolve()`` RAISES where the stat calls below merely return
        # False: ValueError('embedded null byte') for a path carrying a NUL,
        # OSError for a resolution the OS refuses outright.  Since
        # ``require_project_root`` validates SHAPE only (non-empty +
        # ``os.path.isabs``), such a path really does reach verify() — and an
        # exception escaping it would land in targeted.py's generic handler
        # as a ``verify|codebase|error`` row instead of the structured
        # refusal PRD D4 requires.  TypeError is deliberately NOT caught: a
        # non-path argument is a caller bug, not a property of the root.
        return root, ROOT_DEFECT_UNRESOLVABLE

    # Ordered most-specific-last so the sub-reason names the FIRST thing that
    # is wrong; each check swallows OSError/ValueError internally and returns
    # False, so an unreadable path still fails closed.
    if not resolved.exists():
        return resolved, ROOT_DEFECT_MISSING
    if not resolved.is_dir():
        return resolved, ROOT_DEFECT_NOT_A_DIR
    if not (resolved / '.git').exists():
        return resolved, ROOT_DEFECT_NO_DOT_GIT
    return resolved, None


def _as_failure_token(value: object) -> str:
    """Coerce an AgentLoop warning value into a storable failure token.

    Returns '' for anything that is not a non-empty string, so a malformed
    payload degrades to the next candidate in the preference chain (and
    ultimately to 'verify_failed') instead of raising ValidationError out of
    ``VerificationResult`` — which would erase the diagnosis into a generic
    error row (task 4343).
    """
    if not isinstance(value, str):
        return ''
    return value.strip()[:_MAX_FAILURE_TOKEN_LEN]


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
        # PRD D3 (task 4722): NO codebase root is held here.  Reconciliation is
        # multi-project, and the caller's already-validated ProjectScope is the
        # single root authority (INV-9) — a root cached on the instance would
        # be correct for at most one project per process.  `config` survives
        # because it still carries the AGENT settings (model, provider,
        # timeouts, max steps); `config.explore_codebase_root` is no longer
        # read on this path at all.
        self.config = config

    async def verify(
        self,
        claim: str,
        context: str = '',
        scope_hints: list[str] | None = None,
        *,
        codebase_root: Path,
    ) -> VerificationResult:
        """Verify a factual claim against the codebase rooted at ``codebase_root``.

        ``codebase_root`` is per-call and REQUIRED (keyword-only, no default):
        the caller supplies the root of the project whose claim this is, and
        that one value drives the tool closures, their path-escape guards, the
        prompt, and the agent's cwd.  A default would silently reinstate the
        process-global root for any caller that forgot the argument.

        This reverses task 2548 item 2, which deleted verify()'s unused
        ``project_id`` parameter as dead code: the parameter was not the
        mistake, the missing wiring was — and the thing the verifier actually
        needs is a filesystem root, not a logical id.
        """
        codebase_root, root_defect = _resolve_codebase_root(codebase_root)

        # ── Fail-closed root pre-flight (PRD D4) ───────────────────────────
        # Refuse BEFORE building any tools or constructing the agent.  The
        # failure being closed is an agent pointed at the WRONG tree: it
        # searches, finds nothing, and returns `contradicted` against
        # genuinely-completed work — so a refusal that happened after the
        # spawn would not close it at all.
        #
        # The refusal rides task 4343's existing audited agent-failure path
        # unchanged: _on_task_done writes one census-visible
        # `verify|codebase|agent_failed` row carrying the token, logs the
        # summary, and writes NO memory.  No new model field, no new branch
        # in targeted.py.
        if root_defect is not None:
            logger.warning(
                'verification_root_unresolved root=%s reason=%s',
                codebase_root, root_defect,
            )
            return VerificationResult(
                verdict=VerificationVerdict.inconclusive,
                confidence=0.0,
                evidence=[],
                # Task 1811's human-facing sentinel prefix, the offending
                # path, and the sub-reason — so an operator grepping
                # 'agent-failed:' still sees this failure, and can tell both
                # WHICH root was refused and WHY without re-running anything.
                # The structured failure_token stays the machine-readable
                # channel (INV-2) — nothing branches on this prose.
                summary=(
                    f'agent-failed:{CODEBASE_ROOT_UNRESOLVED}: '
                    f'{codebase_root} ({root_defect})'
                ),
                git_context=None,
                agent_failed=True,
                failure_token=CODEBASE_ROOT_UNRESOLVED,
            )

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
            except TimeoutError:
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
{codebase_root}

Investigate this claim against the codebase and call `verification_complete` with your findings.
"""

        agent = AgentLoop(
            config=self.config,
            system_prompt=EXPLORE_AGENT_SYSTEM_PROMPT,
            tools=tools,
            terminal_tool='verification_complete',
            # The agent explores the TARGET project (task 4722): its cwd, and
            # so the CLAUDE.md the CLI auto-loads, must be that project's.
            cwd=codebase_root,
        )

        result, _ = await agent.run(prompt)

        verdict = extract_agent_verdict(
            result,
            default_verdict='inconclusive',
            error_summary='verify_failed',
        )
        raw = verdict.raw or {}
        # Task 4343: stop discarding AgentVerdict.failed.  The 'agent-failed:'
        # prefix that task 1811 put in `summary` is human-facing prose; the
        # structured fields below are what downstream consumers branch on.
        #
        # Preference chain — warning_origin > warning > 'verify_failed':
        #   * `warning_origin` is the ACTIONABLE diagnosis.  'cli_output_empty'
        #     (the CLI returned nothing) and 'cli_output_unparseable' (the CLI
        #     returned junk) call for different operator responses, so the
        #     specific one wins when AgentLoop.run() supplies it.
        #   * `warning` is the generic bucket the loop reports for every
        #     no-tool-call exit; it is the fallback, never blanked.
        #   * 'verify_failed' must stay in sync with the error_summary=
        #     'verify_failed' argument passed to extract_agent_verdict just
        #     above — rename them together or the two silently split.  `raw`
        #     is {} when the loop returned None/a non-dict, so that shape
        #     naturally lands here.
        #
        # Each candidate goes through _as_failure_token: this is the pydantic
        # construction boundary, and a non-str under either key would raise
        # ValidationError below — surfacing as a generic 'error' audit row,
        # i.e. losing exactly the diagnosis this code exists to preserve.
        # AgentLoop.run() already gates `warning_origin` on its closed
        # CLI_WARNING_ORIGINS vocabulary; this is the independent guard at the
        # consuming end, so a future producer cannot widen the column by
        # accident.
        token = ''
        if verdict.failed:
            token = (
                _as_failure_token(raw.get('warning_origin'))
                or _as_failure_token(raw.get('warning'))
                or 'verify_failed'
            )
        return VerificationResult(
            verdict=VerificationVerdict(verdict.verdict),
            confidence=raw.get('confidence', 0.0),
            evidence=raw.get('evidence', []),
            summary=verdict.summary,
            git_context=raw.get('git_context'),
            agent_failed=verdict.failed,
            failure_token=token,
        )

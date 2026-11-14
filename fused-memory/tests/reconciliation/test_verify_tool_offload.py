"""CodebaseVerifier's in-process filesystem tools do their filesystem work off the event loop (INV-8).

The model chooses how many times it calls these tools and with what
arguments, so filesystem work done inline in their coroutines holds the loop
for an LLM-chosen, unbounded total. AgentLoop's per-tool ``asyncio.wait_for``
cannot fire while that happens either.

Each off-loop case runs one real tool against a checkout-shaped root, with the
pathlib primitive it reaches instrumented to ask a ``LoopFreedomProbe`` whether
the loop stayed free. The structural twin is
``shared/tests/test_loop_blocking_gate.py``, whose ratchet flags the same sites
from the source. The contract pins below fix the result dicts the model sees,
so moving the tool bodies cannot change them.
"""

import pathlib
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _fm_helpers import LoopFreedomProbe
from _git_root_helper import make_git_root

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.agent_loop import ToolDefinition
from fused_memory.reconciliation.verify import CodebaseVerifier

_MANY_COUNT = 51


async def _verifier_tools(root: Path) -> dict[str, ToolDefinition]:
    """The tools a real ``verify()`` hands its agent, with the agent itself stubbed."""
    with patch('fused_memory.reconciliation.verify.AgentLoop') as mock_cls:
        mock_cls.return_value.run = AsyncMock(return_value={
            'verdict': 'inconclusive', 'confidence': 0.5, 'evidence': [], 'summary': 's',
        })
        await CodebaseVerifier(ReconciliationConfig()).verify(
            claim='Task X completed', codebase_root=root,
        )
    return mock_cls.call_args.kwargs['tools']


def _probe_primitive(
    monkeypatch: pytest.MonkeyPatch, probe: LoopFreedomProbe, name: str, root: Path,
) -> None:
    """Make ``pathlib.Path.<name>`` consult ``probe`` whenever its receiver lies under ``root``."""
    original = getattr(pathlib.Path, name)

    def instrumented(self: Path, *args, **kwargs):
        if self.is_relative_to(root):
            probe.block()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, name, instrumented)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = make_git_root(tmp_path, 'root')
    (root / 'probe.txt').write_text('from-root')
    (root / 'alpha.py').write_text('# alpha\n')
    (root / 'three.txt').write_text('line1\nline2\nline3\n')
    many = root / 'many'
    many.mkdir()
    for i in range(_MANY_COUNT):
        (many / f'f{i:02d}.py').write_text('')
    (tmp_path / 'outside.txt').write_text('from-outside')
    return root


def _read_the_probe_file(result: dict) -> None:
    assert result['content'] == 'from-root', result


def _globbed_only_alpha(result: dict) -> None:
    assert result['matches'] == ['alpha.py'], result


def _grepped_alpha(result: dict) -> None:
    assert any(line.startswith('alpha.py:') for line in result['results']), result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('tool', 'kwargs', 'primitive', 'check_result'),
    [
        pytest.param('read_file', {'path': 'probe.txt'}, 'read_text', _read_the_probe_file,
                     id='read_file-read_text'),
        pytest.param('read_file', {'path': 'probe.txt'}, 'resolve', _read_the_probe_file,
                     id='read_file-resolve'),
        pytest.param('glob_search', {'pattern': '*.py'}, 'glob', _globbed_only_alpha,
                     id='glob_search-glob'),
        pytest.param('grep_search', {'pattern': 'alpha'}, 'resolve', _grepped_alpha,
                     id='grep_search-resolve'),
    ],
)
async def test_tool_filesystem_work_leaves_the_loop_free(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    tool: str,
    kwargs: dict,
    primitive: str,
    check_result: Callable[[dict], None],
) -> None:
    probe = LoopFreedomProbe()
    tools = await _verifier_tools(root)
    _probe_primitive(monkeypatch, probe, primitive, root.resolve())

    result = await tools[tool].function(**kwargs)

    probe.assert_loop_stayed_free()
    check_result(result)


@pytest.mark.asyncio
async def test_read_file_truncates_at_max_lines_and_counts_the_rest(root: Path) -> None:
    tools = await _verifier_tools(root)

    result = await tools['read_file'].function(path='three.txt', max_lines=1)

    assert result == {
        'path': 'three.txt', 'content': 'line1\n... (2 more lines)', 'total_lines': 3,
    }


@pytest.mark.asyncio
async def test_read_file_reports_a_missing_file(root: Path) -> None:
    tools = await _verifier_tools(root)

    assert await tools['read_file'].function(path='missing.txt') == {
        'error': 'File not found: missing.txt',
    }


@pytest.mark.asyncio
async def test_read_file_refuses_a_path_outside_the_root(root: Path) -> None:
    tools = await _verifier_tools(root)

    assert await tools['read_file'].function(path='../outside.txt') == {
        'error': 'Path outside codebase root',
    }


@pytest.mark.asyncio
async def test_glob_search_keeps_fifty_matches_and_counts_them_all(root: Path) -> None:
    tools = await _verifier_tools(root)

    result = await tools['glob_search'].function(pattern='many/*.py')

    assert result['matches'] == [f'many/f{i:02d}.py' for i in range(50)]
    assert result['total'] == _MANY_COUNT


@pytest.mark.asyncio
async def test_glob_search_reports_an_absolute_pattern_instead_of_raising(root: Path) -> None:
    tools = await _verifier_tools(root)

    result = await tools['glob_search'].function(pattern='/etc/*')

    assert set(result) == {'error'}, result


@pytest.mark.asyncio
async def test_grep_search_refuses_a_path_outside_the_root(root: Path) -> None:
    tools = await _verifier_tools(root)

    assert await tools['grep_search'].function(pattern='x', path='..') == {
        'error': 'Path outside codebase root',
    }

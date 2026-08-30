"""The per-call codebase root of CodebaseVerifier.verify() (task 4722).

PRD ``plans/recon-codebase-verifier-fix-prd.md`` D3 + D5.  Reconciliation is
multi-project: the verifier used to resolve ONE process-global root from
``config.explore_codebase_root`` at construction, so a task belonging to any
other project was verified against dark-factory's tree — an agent that
searches the wrong checkout finds nothing and reports ``contradicted`` against
genuinely-completed work.

``verify()`` now takes ``codebase_root`` as a REQUIRED keyword-only argument
supplied by the caller's already-validated ``ProjectScope``, and that one
value is the sole root authority (INV-9): it feeds the five tool closures,
their path-escape guards, the prompt's ``### Codebase Root`` line, and the
agent's own cwd.

Kept separate from ``test_verify_agent_failure.py`` — that file is task 4343's
failure-token census guard, and each file here pins ONE contract.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _git_root_helper import make_git_root

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import VerificationVerdict
from fused_memory.reconciliation.verify import CodebaseVerifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# ``make_git_root`` (tests/_git_root_helper.py) builds the checkout-shaped
# root verify() now requires.  It is shared with test_verify_agent_failure.py
# and test_targeted.py: "usable root" is a production contract, so its
# test-side mirror is defined once rather than per file.


def _config(explore_root: str = '/nonexistent/global/explore/root') -> ReconciliationConfig:
    """Config whose explore root is DELIBERATELY not the per-call root.

    Every assertion below that a value equals the per-call root is paired with
    one that it is not this value, so an implementation that quietly keeps
    reading the process-global root cannot pass on a path coincidence.
    """
    return ReconciliationConfig(explore_codebase_root=explore_root)


def _mock_agent_loop():
    """Patch context for verify.AgentLoop returning an inconclusive verdict."""
    patcher = patch('fused_memory.reconciliation.verify.AgentLoop')
    mock_cls = patcher.start()
    instance = AsyncMock()
    instance.run = AsyncMock(return_value=(
        {
            'verdict': 'inconclusive',
            'confidence': 0.5,
            'evidence': [],
            'summary': 'nothing conclusive',
        },
        [],
    ))
    mock_cls.return_value = instance
    return patcher, mock_cls, instance


# ---------------------------------------------------------------------------
# The contract: the root is per-call and required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_requires_codebase_root_keyword(tmp_path):
    """``codebase_root`` is keyword-only with NO default (PRD D3 contract).

    A default would silently reinstate the global-root bug for any caller that
    forgot the argument; making it required turns that mistake into a
    TypeError at the call site instead of a wrong-tree verification.
    """
    verifier = CodebaseVerifier(_config())

    # AgentLoop is patched purely for hermeticity: the TypeError comes from
    # argument binding, before any body code runs, so patching cannot mask it
    # — but without the patch a signature regression would send this test out
    # to a real CLI invocation instead of failing cleanly.
    patcher, _mock_cls, _instance = _mock_agent_loop()
    try:
        with pytest.raises(TypeError):
            # The omission is the assertion: `codebase_root` is deliberately
            # absent so argument binding raises at runtime.  pyright sees the
            # same missing required argument statically, which is the very
            # thing under test, so the diagnostic is suppressed here rather
            # than satisfied — passing the argument would delete the test.
            await verifier.verify(claim='x')  # pyright: ignore[reportCallIssue]
    finally:
        patcher.stop()

    # Keyword-ONLY, pinned behaviourally rather than by introspecting the
    # signature: a root passed POSITIONALLY must not bind.  (The old
    # `param.default is empty` half was fully subsumed by the TypeError
    # above, and signature introspection tends to ossify cosmetic shape
    # rather than behaviour.)  The keyword NAME stays pinned by every other
    # test in this file, all of which call `codebase_root=`.
    patcher, _mock_cls, _instance = _mock_agent_loop()
    try:
        with pytest.raises(TypeError):
            # Four positionals against a three-positional signature; the
            # static diagnostic is the same fact under test.
            await verifier.verify('claim', '', None, tmp_path)  # pyright: ignore[reportCallIssue]
    finally:
        patcher.stop()


def test_verifier_holds_no_global_codebase_root(tmp_path):
    """The verifier keeps no root of its own (INV-9: one root authority).

    Asserting the ATTRIBUTE's absence, not just behaviour, is what stops a
    second resolver being reintroduced as a "harmless" fallback: with no
    ``self.codebase_root`` there is nothing for a future edit to fall back to.
    """
    config = _config(str(make_git_root(tmp_path, 'global')))
    verifier = CodebaseVerifier(config)

    assert not hasattr(verifier, 'codebase_root')
    # The config survives — it still carries the agent settings (model,
    # timeouts, max steps); only the ROOT moved to the caller.
    assert verifier.config is config


@pytest.mark.asyncio
async def test_prompt_names_the_per_call_root(tmp_path):
    """The prompt's ``### Codebase Root`` line names the caller's root."""
    root_a = make_git_root(tmp_path, 'root_a')
    config = _config()
    patcher, mock_cls, instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(config)
        await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    prompt = instance.run.call_args.args[0]
    assert f'### Codebase Root\n{root_a.resolve()}' in prompt
    assert str(config.explore_codebase_root) not in prompt


@pytest.mark.asyncio
async def test_agent_cwd_is_the_per_call_root(tmp_path):
    """The agent runs IN the caller's root, not the global explore root.

    Task 1989's rationale, re-pointed: the cwd's auto-loaded CLAUDE.md is the
    agent's main passive codebase signal, so it must be the TARGET project's.
    """
    root_a = make_git_root(tmp_path, 'root_a')
    config = _config()
    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(config)
        await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    cwd = mock_cls.call_args.kwargs['cwd']
    assert cwd == root_a.resolve()
    assert cwd != Path(config.explore_codebase_root)


@pytest.mark.asyncio
async def test_tool_closures_resolve_against_the_per_call_root(tmp_path):
    """read_file / glob_search read the caller's tree, not the global one.

    The decoy root is load-bearing rather than decorative: against an EMPTY
    global root a regression that still reads it would return an empty result
    set, indistinguishable from a correct read of an empty target.  Pointing
    the global root at a POPULATED decoy makes the wrong-tree read return the
    WRONG file, so the regression fails loudly.
    """
    root_a = make_git_root(tmp_path, 'root_a')
    (root_a / 'alpha.py').write_text('# alpha\n')
    (root_a / 'probe.txt').write_text('from-root-a')

    decoy = make_git_root(tmp_path, 'decoy')
    (decoy / 'beta.py').write_text('# beta\n')
    (decoy / 'probe.txt').write_text('from-decoy')

    config = _config(str(decoy))
    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(config)
        await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    tools = mock_cls.call_args.kwargs['tools']

    read = await tools['read_file'].function(path='probe.txt')
    assert read.get('content') == 'from-root-a', read

    globbed = await tools['glob_search'].function(pattern='*.py')
    assert globbed['matches'] == ['alpha.py'], globbed


@pytest.mark.asyncio
async def test_subprocess_tools_run_in_the_per_call_root(tmp_path):
    """git_log / git_show / grep_search are pointed at the caller's root.

    ``create_subprocess_exec`` is patched rather than letting real git/grep
    run: that keeps the test hermetic and asserts the exact wiring (the cwd
    kwarg and grep's search path) instead of inferring it from output.
    """
    root_a = make_git_root(tmp_path, 'root_a')
    decoy = make_git_root(tmp_path, 'decoy')

    config = _config(str(decoy))
    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(config)
        await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    tools = mock_cls.call_args.kwargs['tools']

    with patch(
        'fused_memory.reconciliation.verify.asyncio.create_subprocess_exec',
        new_callable=AsyncMock,
    ) as mock_exec:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_exec.return_value = proc

        await tools['git_log'].function()
        assert mock_exec.call_args.kwargs['cwd'] == str(root_a.resolve())

        await tools['git_show'].function(commit='HEAD')
        assert mock_exec.call_args.kwargs['cwd'] == str(root_a.resolve())

        await tools['grep_search'].function(pattern='x')
        # grep gets its search path as the last positional argument.
        search_path = Path(mock_exec.call_args.args[-1])
        assert search_path.is_relative_to(root_a.resolve()), search_path
        assert not search_path.is_relative_to(decoy.resolve()), search_path


@pytest.mark.asyncio
async def test_path_escape_guard_survives_the_rebind(tmp_path):
    """The ``is_relative_to`` escape guards still hold against the new root.

    ``full_path.resolve().is_relative_to(root)`` is only correct when the root
    itself is resolved — comparing a resolved path against an unresolved root
    breaks the guard in both directions.  This pins that the guard keeps
    refusing an escape after the root moved from __init__ to verify().
    """
    root_a = make_git_root(tmp_path, 'root_a')
    decoy = make_git_root(tmp_path, 'decoy')
    (decoy / 'probe.txt').write_text('from-decoy')

    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config(str(decoy)))
        await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    tools = mock_cls.call_args.kwargs['tools']

    escaped = await tools['read_file'].function(path='../decoy/probe.txt')
    assert escaped == {'error': 'Path outside codebase root'}

    escaped_grep = await tools['grep_search'].function(pattern='x', path='..')
    assert escaped_grep == {'error': 'Path outside codebase root'}


@pytest.mark.asyncio
async def test_root_is_per_call_not_per_instance(tmp_path):
    """One verifier, two calls, two different roots.

    The reconciler holds a single long-lived CodebaseVerifier and serves every
    project through it, so an implementation that caches the first root on
    ``self`` would be correct exactly once per process.  This kills that shape.
    """
    root_a = make_git_root(tmp_path, 'root_a')
    root_b = make_git_root(tmp_path, 'root_b')

    patcher, mock_cls, instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        await verifier.verify(claim='claim A', codebase_root=root_a)
        await verifier.verify(claim='claim B', codebase_root=root_b)
    finally:
        patcher.stop()

    assert mock_cls.call_count == 2
    first_cwd = mock_cls.call_args_list[0].kwargs['cwd']
    second_cwd = mock_cls.call_args_list[1].kwargs['cwd']
    assert first_cwd == root_a.resolve()
    assert second_cwd == root_b.resolve()

    first_prompt = instance.run.call_args_list[0].args[0]
    second_prompt = instance.run.call_args_list[1].args[0]
    assert str(root_a.resolve()) in first_prompt
    assert str(root_b.resolve()) in second_prompt
    assert str(root_b.resolve()) not in first_prompt


# ---------------------------------------------------------------------------
# Fail-closed refusal on an unusable root (PRD D4, INV-2 + INV-8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['missing', 'not_a_dir', 'no_dot_git'])
async def test_unusable_root_refuses_without_spawning_an_agent(tmp_path, shape):
    """An unusable root is refused BEFORE any agent is constructed.

    ``require_project_root`` validates SHAPE only ("non-empty absolute path")
    — it never stats the path — so a non-existent or non-checkout root really
    does reach the verifier.  Spawning an agent there is the failure this task
    closes: it searches a tree that does not contain the work, finds nothing,
    and returns ``contradicted`` against genuinely-completed work.  Refusing
    before the spawn is the whole point, which is why ``assert_not_called`` is
    asserted alongside the returned fields.
    """
    if shape == 'missing':
        bad_root = tmp_path / 'does-not-exist'
    elif shape == 'not_a_dir':
        bad_root = tmp_path / 'a-file'
        bad_root.write_text('not a directory')
    else:
        bad_root = tmp_path / 'no-git'
        bad_root.mkdir()

    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=bad_root)
    finally:
        patcher.stop()

    assert result.verdict == VerificationVerdict.inconclusive
    assert result.agent_failed is True
    assert result.failure_token == 'codebase_root_unresolved'
    # The offending path must be legible in the row itself, so an operator
    # reading the census sees WHICH root failed without re-running anything.
    assert str(bad_root) in result.summary, result.summary
    # ...and WHY.  The three shapes collapse to one census token by design
    # (closed vocabulary, so GROUP BY does not fragment), which is exactly
    # why the discriminator has to survive in the prose: 'no_dot_git' is a
    # real tree that is merely not a checkout root — a permanently-refused
    # project — while 'missing' is a bogus path.  Same token, different
    # operational story.
    assert f'({shape})' in result.summary, result.summary

    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_refusal_uses_stat_only_no_subprocess(tmp_path):
    """The root check is stat-only — no ``git rev-parse`` probe (INV-8).

    verify() runs on the event loop; a subprocess probe there would block it.
    """
    bad_root = tmp_path / 'does-not-exist'

    patcher, _mock_cls, _instance = _mock_agent_loop()
    try:
        with patch(
            'fused_memory.reconciliation.verify.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
        ) as mock_exec:
            verifier = CodebaseVerifier(_config())
            result = await verifier.verify(claim='Task X completed', codebase_root=bad_root)
    finally:
        patcher.stop()

    assert result.failure_token == 'codebase_root_unresolved'
    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_git_worktree_root_is_accepted(tmp_path):
    """A checkout whose ``.git`` is a FILE is usable, not refused.

    Every task in this factory runs in a ``git worktree``, where ``.git`` is a
    file holding a ``gitdir:`` pointer.  An implementation testing ``.is_dir()``
    on that entry would refuse the entire population this PRD exists to serve
    — and in the census that refusal would look identical to a genuinely bogus
    root.
    """
    worktree = make_git_root(tmp_path, 'worktree', dot_git='file')

    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=worktree)
    finally:
        patcher.stop()

    mock_cls.assert_called_once()
    assert result.agent_failed is False
    assert result.failure_token == ''


@pytest.mark.asyncio
async def test_valid_root_still_spawns_the_agent(tmp_path):
    """Positive control: without it, an implementation refusing EVERY root passes."""
    root_a = make_git_root(tmp_path, 'root_a')

    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=root_a)
    finally:
        patcher.stop()

    mock_cls.assert_called_once()
    assert result.verdict == VerificationVerdict.inconclusive
    assert result.agent_failed is False
    assert result.failure_token == ''


@pytest.mark.asyncio
async def test_refusal_summary_keeps_the_agent_failed_sentinel(tmp_path):
    """The refusal's summary opens with task 1811's ``agent-failed:`` sentinel.

    _on_task_done logs ``summary`` verbatim in its verification_agent_failed
    WARNING, so an operator grepping for the sentinel would otherwise miss
    precisely the failure introduced here.  The structured ``failure_token``
    stays the machine-readable channel (INV-2) — nothing branches on prose.
    """
    bad_root = tmp_path / 'does-not-exist'

    patcher, _mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        result = await verifier.verify(claim='Task X completed', codebase_root=bad_root)
    finally:
        patcher.stop()

    assert result.summary.startswith('agent-failed:'), result.summary


@pytest.mark.asyncio
async def test_malformed_root_refuses_instead_of_raising(tmp_path):
    """A path that cannot even be RESOLVED still produces the refusal.

    ``Path.resolve()`` raises where the stat checks merely return False:
    ValueError('embedded null character') for a path carrying a NUL.
    ``require_project_root`` validates SHAPE only (non-empty +
    ``os.path.isabs``), and ``'/tmp/a\x00b'`` passes both — so such a path
    genuinely reaches verify().  Letting the exception escape would land it
    in targeted.py's generic handler as a ``verify|codebase|error`` row
    instead of this structured refusal, which is a (narrow) breach of PRD D4:
    *a wrong or unresolvable root produces a structured refusal*.
    """
    malformed = Path('/tmp/a\x00b')

    patcher, mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        # The assertion is that this does NOT raise.
        result = await verifier.verify(claim='Task X completed', codebase_root=malformed)
    finally:
        patcher.stop()

    assert result.agent_failed is True
    assert result.failure_token == 'codebase_root_unresolved'
    assert '(unresolvable)' in result.summary, result.summary
    mock_cls.assert_not_called()


@pytest.mark.asyncio
async def test_refusal_warning_carries_the_sub_reason(tmp_path, caplog):
    """The operator-facing WARNING names the root AND why it was refused.

    The log line is the only place the discriminator is machine-greppable
    (the census token is deliberately single-valued), so it is pinned here
    rather than left to drift.
    """
    bad_root = tmp_path / 'no-git'
    bad_root.mkdir()

    patcher, _mock_cls, _instance = _mock_agent_loop()
    try:
        verifier = CodebaseVerifier(_config())
        with caplog.at_level('WARNING'):
            await verifier.verify(claim='Task X completed', codebase_root=bad_root)
    finally:
        patcher.stop()

    warnings = [r.getMessage() for r in caplog.records if r.levelname == 'WARNING']
    assert any('verification_root_unresolved' in m for m in warnings), warnings
    assert any('reason=no_dot_git' in m for m in warnings), warnings
    assert any(str(bad_root.resolve()) in m for m in warnings), warnings

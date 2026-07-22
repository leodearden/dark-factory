"""Git worktree management for eval isolation."""

import asyncio
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from orchestrator.artifacts import TaskArtifacts
from orchestrator.verify import _target_subprocess_env

logger = logging.getLogger(__name__)

# Cross-workspace-member runtime dep(s) eval verify needs in the worktree venv
# but the old-baseline orchestrator-only `uv sync` omits. Empirically the ONLY
# missing dep (installing it alone took df_task_12 pytest collection from 7/7
# module errors to 57 tests collected). A module-level tuple constant keeps the
# set explicit, documented, and forward-extensible without over-installing today.
EVAL_VERIFY_EXTRA_DEPS: tuple[str, ...] = ('aiosqlite',)


def eval_worktree_root(project_root: Path | str) -> Path:
    """Return the base dir for eval worktrees — a SIBLING of ``project_root``.

    Eval worktrees must live OUTSIDE ``project_root`` so a pytest/pyright/ruff/
    cargo run inside one does not walk its filesystem ancestors up into the live
    main repo and collect that repo's ancestor config. Concretely: the main
    repo's root ``conftest.py`` ``sys.path``-injects and ``__import__``s the
    CURRENT-main ``<subproject>/src`` packages, pre-caching today's code in
    ``sys.modules``; the fixture's verify then resolves every ``import
    orchestrator.*`` to that cached current-main module, SHADOWING the pinned
    ``pre_task_commit`` code (Defect B, task 2881). A nested worktree's rootdir/
    confcutdir escape upward to that repo root whenever the fixture's old tree
    lacks its own pytest anchor; a sibling worktree has no dark-factory ancestor
    config above it at all, so a plain ``pytest`` (no flags) collects only the
    fixture's own pinned conftests.

    Mirrors ``GitOps.quarantine_base``'s sibling-outside-base construction
    (``<name>.parent / f'{<name>.name}-…'``). The dir name intentionally retains
    the literal ``eval-worktree`` substring so fused-memory's reconciliation
    project-scope anchor (``config/schema.py`` — matches ``eval-worktree``) keeps
    classifying these relocated paths. Project-name-qualified to avoid collisions
    when multiple projects share a parent directory. Pure function, no side
    effects.
    """
    project_root = Path(project_root)
    return project_root.parent / f'{project_root.name}-eval-worktrees'


def read_python_pin(root: Path) -> str | None:
    """Return the interpreter pin from ``<root>/.python-version``.

    Reads the file's stripped first line (the version ``uv`` resolves), or
    ``None`` when the file is missing, empty, or unreadable — the fail-safe
    signal to inject no pin (``uv`` falls back to its normal resolution)
    rather than guess. Derived from the target's own ``.python-version`` so
    the eval runner stays correct across targets (df fixtures pin 3.13; other
    targets pin whatever they use) instead of hardcoding a literal.

    Fail-safe on a garbled file: a missing/unreadable file (``OSError``) OR a
    non-UTF-8, undecodable one (``UnicodeDecodeError`` — a ``ValueError``, so
    it would otherwise escape a bare ``except OSError``) both return ``None``
    instead of propagating out of this fail-safe helper and aborting the eval.
    The read is pinned to UTF-8 so the decode outcome is deterministic across
    locales (``.python-version`` is canonical ASCII).
    """
    try:
        raw = (root / '.python-version').read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    lines = raw.splitlines()
    if not lines:
        return None
    return lines[0].strip() or None


def _eval_setup_env(pin_source: Path) -> dict[str, str]:
    """Build the env for the eval worktree's ``setup_commands`` (``uv sync``).

    Scrubs the orchestrator's venv activation vars via
    ``verify._target_subprocess_env`` — so ``uv sync`` can't corrupt the live
    orchestrator ``.venv`` (the 2026-05-29 ghost-venv incident) — and pins the
    interpreter to ``pin_source``'s ``.python-version`` (via ``UV_PYTHON``) when
    present. ``pin_source`` is the target ``project_root``'s current checkout (a
    3.13-bearing tree), threaded by ``create_eval_worktree`` — NOT the eval
    worktree checked out at the fixture's ``pre_task_commit``, which for older
    fixtures predates ``.python-version`` and would resolve no pin → uv default
    3.14t → aiosqlite ModuleNotFoundError (task 2875). Sourcing from
    ``project_root`` mirrors production dispatch, so setup and verify agree on
    the project's supported interpreter regardless of how old the fixture
    baseline is. When no pin is found, injects no ``UV_PYTHON`` (fail-safe),
    matching BUG 2a.
    """
    pin = read_python_pin(pin_source)
    return _target_subprocess_env({'UV_PYTHON': pin} if pin else None)


def resolve_worktree_venv_pythons(worktree: Path) -> list[Path]:
    """Return every venv interpreter the eval setup created under ``worktree``.

    Mirrors ``scripts/eval_bootstrap_smoke.sh``'s ``resolve_venv_pythons`` and
    is deliberately kept in lockstep with it: the framework provisions the
    eval-verify cross-member dep(s) into EXACTLY the venv(s) the smoke gate
    version-checks, so gate and provisioner never disagree about where the
    interpreter lives.

    Resolution order (identical to the gate):

    * direct top-level layout — ``<worktree>/.venv/bin/python`` if it exists
      (a single venv), returned alone;
    * else the one-level subproject glob ``<worktree>/*/.venv/bin/python`` —
      where ``uv`` lands the venv when a fixture's setup does
      ``cd <subproject> && uv sync`` with ``UV_PROJECT_ENVIRONMENT`` scrubbed
      (``verify._target_subprocess_env``), e.g. df_task_12 →
      ``<worktree>/orchestrator/.venv``. A worktree that builds more than one
      subproject venv yields all of them, sorted.

    Returns ``[]`` when no venv exists (a reify/Rust worktree with no
    ``uv sync``) — a clean no-op signal for the provisioner. Pure: no side
    effects, only filesystem existence/glob reads.
    """
    direct = worktree / '.venv' / 'bin' / 'python'
    if direct.exists():
        return [direct]
    return sorted(worktree.glob('*/.venv/bin/python'))


async def _uv_pip_install(
    venv_python: Path, deps: Sequence[str], cwd: Path, env: dict[str, str],
) -> None:
    """``uv pip install --python <venv_python> <deps...>`` into one worktree venv.

    Fail-soft: logs a WARNING with the return code + stderr tail on a non-zero
    exit and NEVER raises, mirroring create_eval_worktree's setup-command loop —
    a transient install hiccup must not abort worktree creation for an entire
    eval campaign. ``env`` is the already-scrubbed setup env (no VIRTUAL_ENV /
    UV_PROJECT_ENVIRONMENT, venv bin off PATH), and ``--python`` targets the
    worktree venv explicitly, so the install cannot corrupt the live
    orchestrator ``.venv`` (the 2026-05-29 ghost-venv incident). ``uv`` itself
    lives at ``~/.local/bin/uv`` — not the stripped venv bin — so it stays
    resolvable under the scrubbed PATH.
    """
    cmd = ['uv', 'pip', 'install', '--python', str(venv_python), *deps]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            f'Eval-verify dep provisioning failed (rc={proc.returncode}) for '
            f'{venv_python} deps={list(deps)}\n{stderr.decode()[-500:]}'
        )
    else:
        logger.info(
            f'Provisioned eval-verify deps {list(deps)} into {venv_python}'
        )


async def ensure_eval_verify_deps(
    worktree: Path,
    env: dict[str, str],
    deps: Sequence[str] = EVAL_VERIFY_EXTRA_DEPS,
) -> None:
    """Provision ``deps`` into every venv the eval setup created under ``worktree``.

    Iterates ``resolve_worktree_venv_pythons(worktree)`` — the resolver kept in
    lockstep with the smoke gate — and ``uv pip install``s ``deps`` into each,
    reusing the caller's already-built scrubbed ``env``. Idempotent no-op for a
    recent-baseline fixture whose venv already has the dep, and a full no-op
    (empty resolver) for a reify/Rust worktree with no venv. Empty ``deps`` is a
    guarded no-op so we never shell out ``uv pip install`` with no packages.
    """
    if not deps:
        return
    for venv_python in resolve_worktree_venv_pythons(worktree):
        await _uv_pip_install(venv_python, deps, worktree, env)


async def _run(cmd: list[str], cwd: Path) -> str:
    """Run a command and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command {" ".join(cmd)} failed (rc={proc.returncode}): '
            f'{stderr.decode().strip()}'
        )
    return stdout.decode().strip()


async def create_eval_worktree(
    project_root: Path | str,
    task_id: str,
    pre_task_commit: str,
    setup_commands: list[str] | None = None,
) -> tuple[Path, str]:
    """Create an isolated worktree at the pre-task commit for an eval run.

    After checkout, runs any setup_commands (e.g. 'uv sync') to create
    an isolated environment matching the worktree's source state.

    Returns ``(worktree_path, run_id)`` so callers can include the run ID in
    result filenames for multi-trial support.
    """
    project_root = Path(project_root)
    run_id = uuid4().hex[:8]
    worktree_path = project_root / '.eval-worktrees' / task_id / f'run-{run_id}'
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    await _run(
        ['git', 'worktree', 'add', '--detach', str(worktree_path), pre_task_commit],
        cwd=project_root,
    )

    # Defensive: confirm the worktree HEAD actually matches the requested
    # baseline. Catches any drift in git's worktree handling and prevents
    # silently evaluating against the wrong commit.
    head = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_path)
    if head != pre_task_commit:
        raise RuntimeError(
            f'create_eval_worktree: HEAD mismatch for {task_id}: '
            f'expected {pre_task_commit}, got {head}'
        )

    logger.info(f'Created eval worktree: {worktree_path} at {pre_task_commit[:10]}')

    # Run setup commands to create isolated env. The env is scrubbed of the
    # orchestrator venv (so `uv sync` targets <worktree>/.venv, never the live
    # orchestrator .venv) and the interpreter is pinned from project_root's
    # CURRENT checkout — a 3.13-bearing tree — NOT the eval worktree checked out
    # at an old pre_task_commit that may predate .python-version (task 2875), so
    # setup and verify agree on the project's supported interpreter.
    if setup_commands:
        setup_env = _eval_setup_env(project_root)
        for cmd_str in setup_commands:
            logger.info(f'Eval worktree setup: {cmd_str}')
            proc = await asyncio.create_subprocess_shell(
                cmd_str,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable='/bin/bash',
                env=setup_env,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    f'Setup command failed (rc={proc.returncode}): {cmd_str}\n'
                    f'{stderr.decode()[-500:]}'
                )
            else:
                logger.info(f'Setup command OK: {cmd_str}')

        # Provision the eval-verify cross-member runtime dep(s) into each venv
        # the setup created. Root cause: eval verify's pytest collection imports
        # `orchestrator.config` → `shared/__init__.py` → `shared.async_sqlite_base`
        # → `import aiosqlite`, but this worktree is checked out at an OLD
        # pre_task_commit whose orchestrator/pyproject.toml predates the
        # `dark-factory-shared[vllm]` dep, so the orchestrator-only `uv sync`
        # above never pulls aiosqlite transitively (production dispatch verifies
        # under a recent-main workspace venv that already has it). Reuse the SAME
        # scrubbed setup_env so the install targets the worktree venv and cannot
        # corrupt the live orchestrator .venv (the 2026-05-29 ghost-venv incident).
        await ensure_eval_verify_deps(worktree_path, setup_env)

    return worktree_path, run_id


async def cleanup_eval_worktree(
    project_root: Path | str,
    worktree_path: Path,
) -> None:
    """Remove an eval worktree."""
    project_root = Path(project_root)
    try:
        await _run(
            ['git', 'worktree', 'remove', '--force', str(worktree_path)],
            cwd=project_root,
        )
        logger.info(f'Cleaned up eval worktree: {worktree_path}')
    except RuntimeError as e:
        logger.warning(f'Failed to cleanup worktree {worktree_path}: {e}')

    # The architect eval writes plan.json to the RELOCATED .task-meta/<name>/
    # root — a SIBLING of the worktree, so `git worktree remove` above does NOT
    # delete it. Remove it here (best-effort, mirroring the worktree removal)
    # so run_architect_eval — the sole caller — leaves no residue behind.
    meta_root = TaskArtifacts.meta_root_for(worktree_path.parent, worktree_path.name)
    shutil.rmtree(meta_root, ignore_errors=True)


async def get_diff(worktree_path: Path, base_commit: str) -> str:
    """Get the full committed diff of an eval worktree vs its base commit.

    ``base_commit`` is the authoritative ``task['pre_task_commit']`` carried
    on the task record — the same base ``metrics._git_diff_stats`` uses. The
    diff is ``git diff {base_commit}..HEAD`` (two-dot range), i.e. the full
    change committed on the ``evals/<id>`` branch.

    This intentionally does NOT read ``<worktree>/.task/metadata.json`` and
    has no uncommitted (``git diff HEAD``) fallback: production TaskWorkflow
    moved its metadata to the sibling ``.task-meta/<name>/`` under W11 / task
    2258, so the old metadata read silently found nothing and the fallback
    graded empty committed diffs (D1). Threading the base is the fix.
    """
    return await _run(
        ['git', 'diff', f'{base_commit}..HEAD'], cwd=worktree_path,
    )


async def get_diff_between_commits(
    project_root: Path, base_commit: str, target_commit: str,
) -> str:
    """Get diff between two commits directly (no worktree needed)."""
    return await _run(
        ['git', 'diff', f'{base_commit}..{target_commit}'],
        cwd=project_root,
    )

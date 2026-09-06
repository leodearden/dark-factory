"""Sandbox guard for reconciliation stage/remediation agents (task 1935).

Provides ``resolve_recon_sandbox_wrap`` — a factory that returns a callable
suitable for the ``sandbox_wrap`` hook in ``shared.cli_invoke.invoke_claude_agent``.
The returned callable wraps the claude argv with a kernel-level Landlock (or
bwrap) confinement ruleset that DENIES writes to every tracked source file in
the repository while preserving the agent's legitimate scratch (/tmp, ~/.claude,
/dev) and HTTP-over-MCP memory operations (network is NOT restricted by Landlock
here — only filesystem access is governed).

This defeats the sub-agent / heredoc bypass that tool-deny lists cannot stop:
Landlock's NO_NEW_PRIVS + restrict_self are inherited by the entire claude
process tree, so a child process that bypasses the tool-layer (e.g. a
``python3 heredoc`` spawned from within the agent) still cannot write to
production source files.

Resolution order mirrors ``orchestrator.agents.sandbox_dispatch``
(Landlock > bwrap) but is **fail-CLOSED** at the terminal case: when neither
backend is available and the caller has opted in to sandboxing, this module
raises ``RemediationSandboxUnavailable`` so that the caller can refuse to launch
an unconfined agent.  An identity wrap (return inner_cmd unchanged) is never
returned — that would silently reopen the hole this module was created to close.

Imports of the orchestrator sandbox backends are LAZY and guarded:
- Runtime resolution works via the shared uv-workspace venv (orchestrator is
  installed in the same venv as fused-memory).
- A hard ``pyproject.toml`` dependency on orchestrator is intentionally absent
  to avoid the memory→orchestrator layering inversion.
- An ``ImportError`` on either backend also raises ``RemediationSandboxUnavailable``
  (fail-closed: if the substrate is unreachable, refuse to run unconfined).

OPERATIONAL PRECONDITION — venv install contract
-------------------------------------------------
This module's default-on confinement (``sandbox_recon_agents=True``) depends on
``orchestrator`` being importable at runtime.  The canonical deployment is the
shared uv-workspace venv in which both ``fused-memory`` and ``orchestrator`` are
installed.  **If fused-memory is ever installed in a venv that does NOT include
orchestrator** (e.g. a stripped CI image or a standalone deployment), the
``ImportError`` guard raises ``RemediationSandboxUnavailable``, which
``run_stage_via_cli`` treats as fail-CLOSED: *every* reconciliation stage returns
an error ``StageResult`` and reconciliation halts entirely.  This is the correct
safety posture but will be operationally surprising if the venv requirement is
not met.

Operators deploying fused-memory outside the workspace venv MUST either:
  1. Install orchestrator in the same venv (``uv add orchestrator`` / workspace
     member), OR
  2. Set ``reconciliation.sandbox_recon_agents = false`` in config.yaml to
     explicitly opt out of confinement on that host.

CONFIG-DIR CONTAINMENT — the INV-1 machine check (task 4003)
------------------------------------------------------------
The recon per-run ``CLAUDE_CONFIG_DIR`` is part of the writable set.  The
*policy* lives in the caller (``cli_stage_runner.run_stage_via_cli`` appends
``config_dir.path`` to the writable extras); the *verification* lives here:
``resolve_recon_sandbox_wrap`` refuses to return a wrap — fail-CLOSED, as
everywhere else in this module — unless the config dir it is handed is actually
contained in the writable set that will be built.

That split is what makes the assertion load-bearing.  If a future edit drops the
computed grant, recon refuses to launch instead of silently producing a
transcript-less stage.

Why it exists: task 2744 (2026-07-18) redirected recon stages to a per-run
config dir under ``<data_dir>/recon-config/`` — neither ``/tmp`` nor
``<cwd>/.task``, i.e. outside every writable root either backend grants.  The
CLI's session-JSONL writes were denied, so ``count_transcript_turns`` returned
None forever: the liveness watchdog degraded to inert and every cap-retry
force-freshed instead of resuming.  That ran silently until 2026-08-11.  A
comment claiming the dir was writable existed the entire time; only a check can
hold an invariant a comment cannot.

That check has an unstated precondition, made explicit in task 4592: every path
it reasons about must be ABSOLUTE.  Containment here is computed with
``os.path.realpath`` in the PARENT process's cwd, but the two things the verdict
is about are resolved in the CHILD's — ``shared.cli_invoke.invoke_claude_agent``
exports ``CLAUDE_CONFIG_DIR`` as a bare string and ``_run_subprocess`` spawns the
wrapped argv with ``cwd=`` ``config.explore_codebase_root``, which is also where
``landlock-exec`` / ``bwrap`` resolve the ``--writable`` grant tokens.  A relative
string therefore names one directory to the verifier and a different one to the
grantor, and the check would return PASS for a directory the child can never
write — the 2026-07-18 defect re-entering through the relative-path door.  The
two cwds agree in production today only because the installed systemd unit sets
``WorkingDirectory`` == ``PROJECT_ROOT``; nothing enforced it, and
``fused-memory/config/config.yaml`` supplies a RELATIVE ``data_dir``
(``./data/reconciliation``) whenever ``RECONCILIATION_DATA_DIR`` is unset.  So
both faces fail closed here: ``_assert_config_dir_writable`` raises on a
non-absolute ``config_dir``, and ``_writable_roots`` drops a non-absolute extra
(loudly) rather than counting a grant it cannot resolve the way the child will.
The producer-side fix — absolutizing the root exactly once — lives at
``fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py::recon_config_base_dir``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class RemediationSandboxUnavailable(RuntimeError):
    """Raised when confinement is required but no sandbox backend is available.

    The caller (``cli_stage_runner.run_stage_via_cli``) must NOT launch the
    reconciliation agent unconfined when this exception is raised.  Instead it
    should return a ``StageResult(error=...)`` and log loudly.

    To opt out of confinement on a sandbox-less host, set
    ``reconciliation.sandbox_recon_agents = false`` in config.yaml.
    """


def _writable_roots(cwd: Path, writable_extras: list[str] | None) -> list[str]:
    """Roots BOTH backends make writable: ``<cwd>/.task`` plus each existing extra.

    Computed backend-agnostically on purpose — the containment invariant must
    hold whether Landlock or bwrap wins resolution, so only roots BOTH grant,
    with the SAME meaning, may appear here (``build_landlock_command`` at
    landlock.py:69-108, ``build_bwrap_command`` at sandbox.py:56-101).

    ``/tmp`` is deliberately NOT one of them, even though ``landlock_exec``
    grants it blanket. ``build_bwrap_command`` mounts ``--tmpfs /tmp`` before its
    binds, so under bwrap the sandbox's ``/tmp`` is a fresh EMPTY tmpfs and not
    the host's: a config dir under host ``/tmp`` that is not ALSO named in
    ``writable_extras`` would be invisible inside the sandbox (the pre-spawn
    ``.credentials.json`` gone) and its session JSONL would land in a tmpfs the
    parent can never read — ``count_transcript_turns`` None forever, i.e. the
    exact 2026-07-18 defect PASSING the check that exists to catch it. Counting
    ``/tmp`` would make this function backend-DEPENDENT while claiming not to be.
    The explicit extras grant is honoured identically by both backends (bwrap
    binds it over the tmpfs, Landlock adds the path rule), so requiring it costs
    the caller nothing — ``run_stage_via_cli`` already passes it — and keeps this
    docstring's claim true.

    Two details keep the rest honest:

    - ``os.path.realpath`` on every root. Landlock resolves its rules by O_PATH
      fd — i.e. by real path — so a symlinked config dir or a symlinked data dir
      must not produce a false PASS or a false FAIL here.
    - The existence filter applies to the EXTRAS ONLY. ``landlock_exec._add_path``
      returns silently for a non-existent path and ``build_bwrap_command`` warns
      and skips, so an extra naming a missing dir is a vacuous grant and must not
      satisfy containment. ``<cwd>/.task`` is the opposite case: both backends
      ``os.makedirs(..., exist_ok=True)`` it immediately before granting it
      (landlock.py:96-97, sandbox.py:110-112), so it is granted whether or not it
      exists yet. Filtering it out on a fresh cwd would fail-CLOSED a config dir
      whose write would in fact succeed — and since ``run_stage_via_cli`` treats
      that as fatal, every reconciliation stage would return an error. That is a
      live foot-gun: relocating the recon config dir under ``<cwd>/.task/`` is the
      alternative the PRD's open question 5 explicitly considers.
    - A non-absolute extra is DROPPED before any of this, by
      ``_absolute_writable_extras`` — see there for why, and note that
      ``resolve_recon_sandbox_wrap`` applies the SAME filter to what it grants, so
      this function's roots describe the set the child actually gets. The filter
      runs BEFORE the ``isdir`` test on purpose: ``isdir`` on a relative path is
      itself a parent-cwd resolution, so letting it decide inclusion would answer
      the question in the wrong process's frame.
    """
    roots = [os.path.realpath(os.path.join(str(Path(cwd).resolve()), '.task'))]
    for extra in _absolute_writable_extras(writable_extras, cwd):
        if os.path.isdir(extra):
            roots.append(os.path.realpath(extra))
    return roots


def _absolute_writable_extras(
    writable_extras: list[str] | None,
    cwd: Path,
) -> list[str]:
    """Return only the ABSOLUTE entries of *writable_extras*, saying so on each drop.

    ONE filter, applied to BOTH sides (task 4592). ``_writable_roots`` uses it to
    decide what may satisfy containment, and ``resolve_recon_sandbox_wrap`` uses
    it to decide what is actually handed to ``build_landlock_command`` /
    ``build_bwrap_command``. That sharing is the point, not an incidental reuse:
    a relative entry filtered out of the VERDICT but still forwarded to the
    backend would still be emitted as a ``--writable <rel>`` token, and
    ``build_landlock_command`` appends each extra VERBATIM (measured — it does no
    parent-side resolution), so ``landlock-exec`` / ``bwrap`` would still grant
    it, resolved in the CHILD's frame. The parent would then be deliberately
    blind to a grant the child does make — conservative in the containment
    direction, but a state this module has no way to reason about, and it would
    make this function's own warning a lie about what happened.

    Because the two cwds differ, an honest choice must be made about WHICH frame
    a relative entry means, and neither answer is safe to guess: resolving it in
    the parent's frame grants a directory the operator may not have meant, and
    resolving it in the child's silently re-points the grant whenever
    ``explore_codebase_root`` moves. So it is dropped from both sides and the
    operator is told to state the absolute path they meant. The alternative
    considered and rejected — a raising pydantic validator on
    ``reconciliation.sandbox_recon_writable_extras`` — would refuse to LOAD a
    config that starts fine today, taking the whole fused-memory server down on a
    setting that is empty by default; a per-launch warning has the right blast
    radius for a defect nobody has yet hit.

    Dropping is loud rather than silent because discarding an operator-configured
    grant without a word is the fail-soft this module was written to end.
    """
    kept: list[str] = []
    for extra in writable_extras or []:
        if not os.path.isabs(extra):
            logger.warning(
                'sandbox_guard: dropping non-absolute writable extra %r — from '
                'the containment accounting AND from the --writable grant handed '
                'to landlock-exec / bwrap. The parent resolves it against ITS '
                'cwd (%s) while the backends resolve the granted token against '
                'the CHILD\'s (%s), so honouring it would grant one directory '
                'and certify another. Make the entry in '
                'reconciliation.sandbox_recon_writable_extras an absolute path; '
                'nothing is granted for it until you do.',
                extra, os.getcwd(), cwd,
            )
            continue
        kept.append(extra)
    return kept


def _assert_config_dir_writable(
    config_dir: Path,
    cwd: Path,
    writable_extras: list[str] | None,
) -> None:
    """Raise unless *config_dir* is inside the writable set (task 4003).

    Containment is prefix-WITH-SEPARATOR, never a bare ``str.startswith`` on the
    root: ``/tmp-evil`` is not inside ``/tmp``.

    A config dir under ``/tmp`` is NOT contained by virtue of living there — see
    ``_writable_roots`` for why the blanket ``/tmp`` grant is backend-specific
    and therefore not a root. It must be named in ``writable_extras`` like any
    other, which ``run_stage_via_cli`` already does.
    """
    raw = str(config_dir)

    # ABSOLUTE-PATH PRECONDITION (task 4592). Checked BEFORE the containment
    # loop, because for a relative path that loop cannot answer the question it
    # is asked: it would compare a parent-cwd resolution against parent-cwd
    # roots and PASS, certifying a directory the child never writes to.
    if not Path(config_dir).is_absolute():
        raise RemediationSandboxUnavailable(
            f'Refusing to launch a reconciliation agent whose CLAUDE_CONFIG_DIR '
            f'is a RELATIVE path: {raw}. Containment is undecidable for a '
            f'relative path — this function resolves it with os.path.realpath '
            f'against the PARENT process\'s cwd to compute the verdict, but the '
            f'CLI child resolves the very same string against the cwd it is '
            f'spawned with (effective_cwd = config.explore_codebase_root, see '
            f'fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py'
            f'::run_stage_via_cli, which hands that cwd to both this check and '
            f'shared/src/shared/cli_invoke.py::_run_subprocess). A PASS here '
            f'would therefore certify a directory the child will never write '
            f'to, and its session transcript would be denied by the kernel — '
            f'silently, which is the 2026-07-18 -> 2026-08-11 recon defect '
            f'(task 4003) passing the check that exists to catch it. The two '
            f'cwds agree in production today only because the installed unit '
            f'sets WorkingDirectory == PROJECT_ROOT. The fix belongs at the '
            f'PRODUCER, not here: '
            f'fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py'
            f'::recon_config_base_dir absolutizes the config-dir root for every '
            f'creator and GC call site, so seeing this error means a caller '
            f'bypassed it — pass an absolute path (or route through that '
            f'function) rather than making this path relative to anything.'
        )

    resolved = os.path.realpath(raw)
    roots = _writable_roots(cwd, writable_extras)

    for root in roots:
        if resolved == root or resolved.startswith(root + os.sep):
            return

    # Name the raw path as well as the resolved one: the operator configured the
    # former, but containment is decided on the latter, and a symlink between
    # them is exactly the case where a bare resolved path reads as a non sequitur.
    shown = raw if resolved == raw else f'{raw} (resolved: {resolved})'
    raise RemediationSandboxUnavailable(
        f'Refusing to launch a reconciliation agent whose CLAUDE_CONFIG_DIR is '
        f'OUTSIDE the sandbox writable set: {shown}. The CLI would be told to '
        f'write its session transcript there and then denied the write by the '
        f'kernel — silently (this is the 2026-07-18 -> 2026-08-11 recon defect, '
        f'task 4003). Writable roots resolved for this invocation: '
        f'{roots}. '
        f'The per-run config dir is normally granted automatically by '
        f'cli_stage_runner.run_stage_via_cli; if you are seeing this, that '
        f'computed grant was lost. Note that a path listed in '
        f'reconciliation.sandbox_recon_writable_extras is IGNORED here unless the '
        f'directory actually exists — landlock_exec skips a non-existent path '
        f'silently, so such a grant would be vacuous. Do NOT "fix" this by adding '
        f'the config-dir BASE (<data_dir>/recon-config) to '
        f'reconciliation.sandbox_recon_writable_extras: that is the root under '
        f'which every run\'s claude-config-<run_id>/.credentials.json lives, and '
        f'granting it would give every recon stage write access to every other '
        f'run\'s credentials.'
    )


def resolve_recon_sandbox_wrap(
    cwd: Path,
    writable_extras: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> Callable[[list[str]], list[str]]:
    """Return a sandbox-wrap callable for a reconciliation stage agent.

    The callable transforms ``inner_cmd`` (the full claude argv) into a
    sandboxed command that confines the entire claude process tree.

    Resolution order: Landlock > bwrap.  Fail-CLOSED when neither backend is
    available (raises ``RemediationSandboxUnavailable`` rather than returning
    an identity/passthrough wrap).

    Args:
        cwd: The working directory that will be passed to ``_run_subprocess``
             (typically ``config.explore_codebase_root``).  Used as the
             ``worktree`` argument to ``build_landlock_command`` /
             ``build_bwrap_command`` — only the gitignored ``.task/``
             subdirectory of this root will be made writable; no module
             source directories are added to the writable set
             (``writable_modules=[]``).
        writable_extras: Optional additional paths to include in the writable
             set (e.g. a uvx/pip cache dir used by a stdio MCP server, or —
             for recon stages — the per-run ``CLAUDE_CONFIG_DIR``, appended by
             ``cli_stage_runner.run_stage_via_cli``).
             ABSOLUTE entries are passed through to the underlying backend's
             ``writable_extras`` parameter; non-absolute ones are dropped with a
             warning, from the grant AND from the containment verdict alike (see
             ``_absolute_writable_extras``).
        config_dir: Optional per-run ``CLAUDE_CONFIG_DIR`` (task 4003). When
             given, this function VERIFIES that the path is contained in the
             writable set that will be built and raises otherwise — it does not
             grant it; granting is the caller's job (policy in the caller,
             verification here). ``None`` skips the check entirely. The
             absolute-extras filter above still applies when ``config_dir`` is
             ``None`` — it governs the grant, not the check — so a caller passing
             only absolute extras (every one in this repo) is unaffected.

    Returns:
        A ``Callable[[list[str]], list[str]]`` that wraps the inner argv.

    Raises:
        RemediationSandboxUnavailable: When neither Landlock nor bwrap is
            available, when the orchestrator backends cannot be imported, OR
            when ``config_dir`` is outside the writable set.
    """
    # The containment check runs ONCE, before the backend branch, because the
    # invariant is backend-independent BY CONSTRUCTION: `_writable_roots` counts
    # only roots Landlock and bwrap grant with the SAME meaning (<cwd>/.task and
    # each existing extra — notably NOT /tmp, which bwrap replaces with a fresh
    # tmpfs; see `_writable_roots`). Checking here rather than inside each branch
    # means a future third backend cannot be added without inheriting it.
    #
    # The filter runs ONCE, before both, so the set this function VERIFIES and
    # the set it GRANTS are the same object (task 4592 amendment). Forwarding the
    # unfiltered list to the backends while verifying the filtered one would
    # leave the parent blind to a real grant: `build_landlock_command` appends
    # each extra verbatim, so a relative `--writable <rel>` token would still
    # reach `landlock-exec` and still be honoured — in the CHILD's frame.
    granted_extras = _absolute_writable_extras(writable_extras, cwd)

    if config_dir is not None:
        _assert_config_dir_writable(config_dir, cwd, granted_extras)

    # ── Landlock branch ───────────────────────────────────────────────────────
    try:
        from orchestrator.agents.landlock import (  # type: ignore[import]
            build_landlock_command,
            is_landlock_available,
        )
    except ImportError as exc:
        raise RemediationSandboxUnavailable(
            f'Cannot import orchestrator.agents.landlock — refusing to run '
            f'reconciliation agent unconfined. '
            f'Install orchestrator in the same venv or set '
            f'reconciliation.sandbox_recon_agents=false to opt out. '
            f'(ImportError: {exc})'
        ) from exc

    if is_landlock_available():
        # writable_modules=[] → no repo source dir is writable; only the
        # built-in scratch (/tmp, ~/.claude, /dev) and the gitignored .task/
        # subdirectory of cwd (added unconditionally by build_landlock_command).
        def _landlock_wrap(cmd: list[str]) -> list[str]:
            return build_landlock_command(cmd, cwd, [], writable_extras=granted_extras)
        return _landlock_wrap

    # ── bwrap fallback ────────────────────────────────────────────────────────
    try:
        from orchestrator.agents.sandbox import (  # type: ignore[import]
            build_bwrap_command,
            is_bwrap_available,
        )
    except ImportError as exc:
        raise RemediationSandboxUnavailable(
            f'Landlock unavailable and cannot import orchestrator.agents.sandbox '
            f'for bwrap fallback — refusing to run reconciliation agent unconfined. '
            f'(ImportError: {exc})'
        ) from exc

    if is_bwrap_available():
        def _bwrap_wrap(cmd: list[str]) -> list[str]:
            return build_bwrap_command(cmd, cwd, [], writable_extras=granted_extras)
        return _bwrap_wrap

    # ── Fail-closed: neither backend available ────────────────────────────────
    raise RemediationSandboxUnavailable(
        'Reconciliation sandboxing is enabled (sandbox_recon_agents=true) but '
        'neither Landlock nor bwrap is available on this host.  Refusing to '
        'launch reconciliation agent unconfined — an unconfined agent can '
        'write to production source files without going through review→verify→merge.  '
        'Options: (1) upgrade to a kernel ≥5.13 with Landlock support, '
        '(2) install bubblewrap (bwrap), or '
        '(3) set reconciliation.sandbox_recon_agents=false to opt out explicitly.'
    )

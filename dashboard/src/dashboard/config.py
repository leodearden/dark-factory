"""Dashboard configuration — plain dataclass with env var overrides."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

logger = logging.getLogger(__name__)

# Port 8002 is the canonical systemd-managed instance (shared across all projects).
# Ports 8000/8001 were per-project ports retired when the architecture consolidated
# to a single shared server. The singleton lock prevents any legitimate fused-memory
# process from running on those ports, so listing them would only enable silent
# fallback to stale code — a correctness bug, not graceful degradation.
DEFAULT_FUSED_MEMORY_URLS: Final = ('http://localhost:8002',)

# The canonical top-level orchestrator config name (task 2698 / commit
# 97f0bd9f85). Every onboarded project's top-level orchestrator config MUST
# live at ``<project_root>/dark-factory-orchestrator.yaml`` — this is the
# name dashboard discovery keys on first.
_CANONICAL_CONFIG_NAME: Final = 'dark-factory-orchestrator.yaml'

# Legacy spellings tolerated as a fallback (transitional symlinks left behind
# by the migration to the canonical name). Resolution order matters: each is
# tried in turn only after the canonical name is absent.
_LEGACY_CONFIG_NAMES: Final = (
    'orchestrator.yaml',
    'orchestrator-config.yaml',
    'orchestrator/config.yaml',
)


def _read_escalation_url(yaml_path: Path) -> str | None:
    """Parse *yaml_path* and return its escalation MCP URL, or None.

    Returns None when the file doesn't exist, is unreadable/unparseable,
    its top-level YAML is not a mapping, its ``escalation:`` value is
    present but not a mapping, or it has no ``escalation.port`` — all
    logged at warning level except the last (a config with no escalation
    section configured is not malformed). This function runs once per
    known project root at process startup (via
    ``DashboardConfig.__post_init__``), so a single operator typo in one
    project's config must degrade to "no URL for this root" rather than
    raising and aborting the whole dashboard. Callers decide what "no
    URL" means for logging purposes.
    """
    if not yaml_path.is_file():
        return None
    try:
        with yaml_path.open() as f:
            data: Any = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning('Failed to read %s: %s', yaml_path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning(
            'Ignoring %s: top-level YAML is a %s, not a mapping',
            yaml_path,
            type(data).__name__,
        )
        return None
    esc: Any = data.get('escalation')
    if esc is None:
        esc = {}
    if not isinstance(esc, dict):
        logger.warning(
            "Ignoring %s: 'escalation:' is a %s, not a mapping",
            yaml_path,
            type(esc).__name__,
        )
        return None
    host = esc.get('host', '127.0.0.1')
    port = esc.get('port')
    if not port:
        return None
    return f'http://{host}:{port}/mcp'


def _discover_root_escalation_url(root: Path) -> str | None:
    """Resolve *root*'s escalation MCP URL, preferring the canonical config name.

    If ``_CANONICAL_CONFIG_NAME`` exists on disk, it is authoritative: its
    resolved URL (or None, if it exists but has no ``escalation.port``) is
    returned without ever consulting legacy spellings. A present-but-
    misconfigured canonical config must not be silently masked by a stale
    legacy file's port — that would report a "please migrate" nudge instead
    of the real problem (a broken canonical config). Only when the canonical
    file is absent entirely does resolution fall back to each of
    ``_LEGACY_CONFIG_NAMES`` in order, logging a WARNING naming the project
    (``root.name``) when a legacy spelling is what resolved it — a nudge to
    migrate. Returns None if no candidate yields a URL.
    """
    canonical_path = root / _CANONICAL_CONFIG_NAME
    if canonical_path.is_file():
        return _read_escalation_url(canonical_path)
    for legacy_name in _LEGACY_CONFIG_NAMES:
        legacy_url = _read_escalation_url(root / legacy_name)
        if legacy_url is not None:
            logger.warning(
                'Project %s: found orchestrator config at legacy path %s '
                '(expected %s) — please migrate.',
                root.name,
                root / legacy_name,
                _CANONICAL_CONFIG_NAME,
            )
            return legacy_url
    return None


def _discover_escalation_urls(roots: list[Path]) -> dict[str, str]:
    """Derive each root's escalation MCP URL from its orchestrator config.

    Keyed by project basename — the same key the Merge UI uses. Prefers the
    canonical ``dark-factory-orchestrator.yaml`` name, falling back to legacy
    spellings (see ``_discover_root_escalation_url``). A root that yields no
    URL at all — no config found under any spelling, or a config found with
    no ``escalation.port`` — logs a WARNING naming the root, rather than
    degrading silently.

    The per-root WARNINGs (both the legacy-fallback nudge and this no-URL
    one) are an intentional, persistent migration nudge — not transient
    errors. Discovery runs once per process at startup (via
    ``DashboardConfig.__post_init__`` / ``from_env``), never per request, so
    a legitimately non-orchestrator or still-legacy root re-emits its WARNING
    only once per process lifetime — a bounded reminder, not log spam.
    """
    out: dict[str, str] = {}
    for root in roots:
        url = _discover_root_escalation_url(root)
        if url is None:
            logger.warning(
                'No escalation URL discovered for project root %s '
                '(checked %s and legacy spellings %s)',
                root,
                _CANONICAL_CONFIG_NAME,
                _LEGACY_CONFIG_NAMES,
            )
            continue
        out[root.name] = url
    return out


@dataclass
class DashboardConfig:
    """Configuration for the Dark Factory dashboard."""

    host: str = '127.0.0.1'
    port: int = 8080
    # Derived from cwd, never a hardcoded absolute path (task 3503): a literal
    # silently pointed every un-configured consumer on every other host at one
    # developer's checkout.  That reason still stands, and this default stays.
    #
    # The canonical deployment no longer RELIES on it, though: both
    # dashboard/dark-factory-dashboard.service and scripts/dashboard.service.template
    # now set Environment=DASHBOARD_PROJECT_ROOT=<repo root> explicitly (task 3572),
    # so from_env() takes the env branch there and the data root is DECLARED rather
    # than inferred from those units' WorkingDirectory= pinning.  Why, and what
    # keeps the two in step, is recorded once — on that Environment= line in the
    # units, and on UnitSpec.env_matches_directive in
    # scripts/check_dashboard_unit_parity.py.
    #
    # This fallback is what still applies to NON-systemd invocations — a bare
    # `python -m dashboard` from a checkout, a test, a container without the unit —
    # and what would apply again if that Environment= line were ever dropped.
    # WorkingDirectory= is retained too and load-bearing in its own right (ExecStart's
    # relative `--project dashboard` resolves against the cwd systemd sets from it);
    # it is pinned by tests/scripts/test_dashboard_service_template.py::
    # test_working_directory_is_pinned_in_both_unit_files.
    project_root: Path = field(default_factory=Path.cwd)
    fused_memory_urls: list[str] = field(default_factory=lambda: list(DEFAULT_FUSED_MEMORY_URLS))
    known_project_roots: list[Path] = field(default_factory=list)
    escalation_urls: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize all Path fields to their canonical (symlink-resolved) forms.

        This invariant ensures that every construction path — direct kwargs,
        from_env(), dataclass.replace(), test fixtures — produces a config
        whose Path fields are already canonicalized.  Consumers never need to
        call .resolve() on config paths.

        Note: Path.resolve() follows all existing symlink segments, but any trailing
        path components that do not yet exist on disk are appended verbatim (absolute
        but not symlink-canonicalized).  If such components are later created as
        symlinks, the stored path will not reflect their targets.
        """
        self.project_root = self.project_root.resolve()
        self.known_project_roots = [p.resolve() for p in self.known_project_roots]
        if not self.escalation_urls:
            self.escalation_urls = _discover_escalation_urls(
                [self.project_root, *self.known_project_roots],
            )

    def _runtime_data_dir(self, env_var: str, subdir: str) -> Path:
        """Resolve a managed fused-memory runtime data dir, env-var first.

        Mirrors fused-memory's own config substitution
        (``${QUEUE_DATA_DIR:./data/queue}`` /
        ``${RECONCILIATION_DATA_DIR:./data/reconciliation}``, fused-memory/config/
        config.yaml) so this reader stays in lockstep with whatever the managed
        fused-memory spawn actually wrote to. The orchestrator's managed spawn
        (``McpLifecycle.start()``, task 2439) injects ``QUEUE_DATA_DIR`` /
        ``RECONCILIATION_DATA_DIR`` pointing at an XDG-rooted path outside the
        watched project's tracked tree; when those vars are unset (the canonical
        shared external-mode deployment on port 8002, CWD=dark_factory), this
        falls back to today's ``project_root/data/...`` paths — a strict no-op.

        A truthy check (not just presence) treats an empty-string env var as
        unset, matching fused-memory's ``${VAR:default}`` empty-as-unset
        semantics.
        """
        override = os.environ.get(env_var)
        return Path(override) if override else self.project_root / 'data' / subdir

    @property
    def reconciliation_db(self) -> Path:
        return self._runtime_data_dir('RECONCILIATION_DATA_DIR', 'reconciliation') / 'reconciliation.db'

    @property
    def tickets_db(self) -> Path:
        """Canonical tickets.db path (sibling of reconciliation.db)."""
        return self._runtime_data_dir('RECONCILIATION_DATA_DIR', 'reconciliation') / 'tickets.db'

    @property
    def write_queue_db(self) -> Path:
        return self._runtime_data_dir('QUEUE_DATA_DIR', 'queue') / 'write_queue.db'

    @property
    def write_journal_db(self) -> Path:
        return self._runtime_data_dir('RECONCILIATION_DATA_DIR', 'reconciliation') / 'write_journal.db'

    @property
    def worktrees_dir(self) -> Path:
        return self.project_root / '.worktrees'

    @property
    def runs_db(self) -> Path:
        return self.project_root / 'data' / 'orchestrator' / 'runs.db'

    @property
    def escalations_dir(self) -> Path:
        return self.project_root / 'data' / 'escalations'

    @property
    def reconciliation_escalations_dir(self) -> Path:
        return self._runtime_data_dir('RECONCILIATION_DATA_DIR', 'reconciliation') / 'escalations'

    @property
    def memory_evals_dir(self) -> Path:
        """Memory-eval artifact root — ``<project_root>/fused-memory/data/memory-evals``.

        Deliberately a PLAIN property (like ``escalations_dir``), NOT routed
        through :meth:`_runtime_data_dir`.  The distinction is real, not
        stylistic: ``QUEUE_DATA_DIR`` / ``RECONCILIATION_DATA_DIR`` relocate the
        *managed* fused-memory runtime dirs to an XDG-rooted path outside the
        watched project's tracked tree, and the memory-eval artifacts are not
        among the things they relocate.  The M1 artifact path is contract-fixed
        by ``docs/prds/memory-eval-program.md`` §3 and by
        ``shared.memory_eval_metrics.metrics_artifact_path``; fused-memory
        writes it relative to the repo.  Honouring a runtime-dir override here
        would point the dashboard at a directory nothing ever writes.
        """
        return self.project_root / 'fused-memory' / 'data' / 'memory-evals'

    @property
    def burndown_db(self) -> Path:
        return self.project_root / 'data' / 'burndown' / 'burndown.db'

    @property
    def metrics_db(self) -> Path:
        return self.project_root / 'data' / 'burndown' / 'metrics.db'

    @property
    def load_samples_db(self) -> Path:
        return self.project_root / 'data' / 'load-samples.db'

    @classmethod
    def from_env(cls) -> DashboardConfig:
        """Create config with DASHBOARD_-prefixed env var overrides.

        When ``DASHBOARD_PROJECT_ROOT`` is unset, the cwd-derived fallback root
        (see the ``project_root`` field) is logged at INFO — read off the
        CONSTRUCTED config, so the journal always names the root the databases
        actually hang off — instead of being an invisible default.

        Since task 3572 the canonical deployment SETS the variable, so this line
        no longer fires there at all; when it does fire it marks a genuinely
        un-configured invocation — a bare ``python -m dashboard`` from a
        checkout, a test, a container without the unit.

        The level stays INFO all the same, and that is a considered choice
        rather than an oversight: changing it is outside task 3572's scope, and
        an un-configured invocation is a normal, supported way to run the
        dashboard, not a degraded state like the ``_discover_escalation_urls``
        WARNINGs in this same module.  INFO still surfaces the root in
        ``journalctl`` at uvicorn's default log level.

        ``from_env()`` is called once per process (app.py's ``lifespan``,
        ``__main__.py``), so this is one line per process lifetime — never per
        request.
        """
        kwargs: dict = {}
        if (host := os.environ.get('DASHBOARD_HOST')) is not None:
            kwargs['host'] = host
        if (port := os.environ.get('DASHBOARD_PORT')) is not None:
            kwargs['port'] = int(port)
        if (root := os.environ.get('DASHBOARD_PROJECT_ROOT')) is not None:
            kwargs['project_root'] = Path(root)
        if (urls := os.environ.get('DASHBOARD_FUSED_MEMORY_URLS')) is not None:
            kwargs['fused_memory_urls'] = [u.strip() for u in urls.split(',') if u.strip()]
        if (roots := os.environ.get('DASHBOARD_KNOWN_PROJECT_ROOTS')) is not None:
            kwargs['known_project_roots'] = [Path(p.strip()) for p in roots.split(',') if p.strip()]

        config = cls(**kwargs)
        if root is None:
            # Log the CONSTRUCTED project_root, never a re-derivation of it.
            # The fallback is `default_factory=Path.cwd` plus __post_init__'s
            # .resolve(); re-computing `Path.cwd().resolve()` here would be a
            # second copy of that derivation, and any future change to how the
            # fallback root is computed (walking up to the repo root, honouring
            # a different var) would silently make this line name a path the
            # process is not actually using — in a message whose entire job is
            # to be the one observable record of a silently-relocated data root.
            logger.info(
                'DASHBOARD_PROJECT_ROOT unset — using cwd-derived project_root %s '
                '(all database paths are derived from it)',
                config.project_root,
            )
        return config

"""Per-task isolated Claude Code config directory.

Each task gets its own ``CLAUDE_CONFIG_DIR`` so that:
- ``.credentials.json`` can be rewritten per-account without races
- ``--resume`` reads the correct credential (not the global one)
- Sessions are stored per-task, avoiding cross-task contamination
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_HOME_CLAUDE = Path.home() / '.claude'

# Files to symlink from ~/.claude/ into the per-task config dir.
# Provides settings/hooks without duplicating config.
# Do NOT include projects/, sessions/, telemetry/ — those must be per-task.
_SYMLINK_FILES = ['settings.json', 'settings.local.json']


class TaskConfigDir:
    """Manages an isolated ``CLAUDE_CONFIG_DIR`` for a single task.

    On creation, symlinks shared config from ``~/.claude/`` and provides
    a ``write_credentials()`` method to set per-invocation OAuth tokens.
    """

    def __init__(self, task_id: str, base_dir: Path | None = None):
        base = base_dir or Path(tempfile.gettempdir())
        self._dir = base / f'claude-config-{task_id}'
        self._dir.mkdir(parents=True, exist_ok=True)
        self._setup_symlinks()

    def _setup_symlinks(self) -> None:
        """Symlink shared config files from ~/.claude/."""
        for name in _SYMLINK_FILES:
            src = _HOME_CLAUDE / name
            dst = self._dir / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError as e:
                    logger.warning('Failed to symlink %s → %s: %s', src, dst, e)

    def write_credentials(self, oauth_token: str) -> None:
        """Write ``.credentials.json`` with the given OAuth token."""
        creds = {
            'claudeAiOauth': {
                'accessToken': oauth_token,
            },
        }
        creds_path = self._dir / '.credentials.json'
        creds_path.write_text(json.dumps(creds))
        # Restrict permissions (token is sensitive)
        creds_path.chmod(0o600)

    @property
    def path(self) -> Path:
        """Absolute path to the config directory."""
        return self._dir

    def cleanup(self) -> None:
        """Remove the config directory and all contents."""
        shutil.rmtree(self._dir, ignore_errors=True)

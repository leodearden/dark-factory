"""cockpit.ui_config — the cockpit's own UI-state persistence.

cockpit-ui.json (at ``fleet_root(root)/cockpit-ui.json``) is the cockpit's
ONLY write target in C5a (PRD §2/§5 pure-consumer discipline: the view
never mutates registry records under sessions/ or decisions/). Fail-soft
throughout: a missing or malformed config file degrades to defaults with a
logged WARNING rather than raising -- a view must stay usable even if its
own tiny state file is absent or corrupt.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orchestrator.session_registry import fleet_root

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = 'cockpit-ui.json'


@dataclass
class CockpitUIConfig:
    """The cockpit's persisted UI state. Every field has a documented default."""

    selected_slug: str | None = None
    poll_interval: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CockpitUIConfig:
        return cls(
            selected_slug=data.get('selected_slug'),
            poll_interval=data.get('poll_interval', 1.5),
        )


def ui_config_path(root: Path | str | None = None) -> Path:
    return fleet_root(root) / _CONFIG_FILENAME


def load_ui_config(root: Path | str | None = None) -> CockpitUIConfig:
    """Load CockpitUIConfig from ``ui_config_path(root)``.

    A missing file, or one that fails to parse as the expected JSON
    mapping, falls back to defaults with a logged WARNING -- this never
    raises (fail-soft, PRD §2).
    """
    path = ui_config_path(root)
    if not path.is_file():
        return CockpitUIConfig()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('load_ui_config: failed to parse %s: %s — using defaults', path, exc)
        return CockpitUIConfig()
    if not isinstance(data, dict):
        logger.warning(
            'load_ui_config: expected a JSON object in %s, got %s — using defaults',
            path,
            type(data).__name__,
        )
        return CockpitUIConfig()
    return CockpitUIConfig.from_dict(data)


def save_ui_config(cfg: CockpitUIConfig, root: Path | str | None = None) -> None:
    """Atomically write *cfg* to ``ui_config_path(root)``.

    Mirrors orchestrator.session_registry's _atomic_write_text idiom (tmp
    file in the target's own parent dir, then os.replace, unlink-on-
    failure) so a save is never observed half-written. Fail-soft: any
    error is logged and swallowed rather than raised -- the cockpit's own
    UI-state write must never crash the view.
    """
    path = ui_config_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            suffix='.tmp',
            prefix=path.stem,
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(cfg.to_dict(), f)
            os.replace(tmp_path_str, str(path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path_str)
            raise
    except OSError as exc:
        logger.warning('save_ui_config: failed to write %s: %s', path, exc)

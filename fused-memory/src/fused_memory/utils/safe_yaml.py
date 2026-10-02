"""The single home for how fused-memory reads its list-shaped YAML registry
files: safely (a restricted safe loader only), fast (libyaml when present),
and without ever raising. Consumed by the curator's registry loaders in
fused_memory/middleware/cancelled_premise_blocklist.py,
fused_memory/middleware/operational_ask_registry.py and
fused_memory/middleware/recon_code_fix_premise_guard.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SAFE_YAML_LOADER", "load_yaml_list_file", "resolve_safe_yaml_loader"]


def resolve_safe_yaml_loader(yaml_module: Any = yaml) -> type:
    """Return libyaml's ``CSafeLoader`` if *yaml_module* has it, else ``SafeLoader``.

    Both are safe loaders: the same restricted tag set, no arbitrary object
    construction.
    """
    return getattr(yaml_module, "CSafeLoader", None) or yaml_module.SafeLoader


SAFE_YAML_LOADER: type = resolve_safe_yaml_loader()


def load_yaml_list_file(
    path: Path | None,
    *,
    logger: logging.Logger,
    label: str,
    consequence: str,
) -> list[object]:
    """Return the top-level YAML list stored at *path*; never raises.

    ``None`` is a deliberate opt-out and yields ``[]`` silently. A file that is
    missing, unreadable, not valid UTF-8, not valid YAML, or whose top-level
    document is not a list yields ``[]`` plus exactly one WARNING on *logger*,
    formatted ``"<label>: … — <consequence>"``.
    """
    if path is None:
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("%s: file not found: %s — %s", label, path, consequence)
        return []
    except OSError as exc:
        logger.warning("%s: cannot read %s: %s — %s", label, path, exc, consequence)
        return []
    except UnicodeDecodeError as exc:
        logger.warning(
            "%s: cannot decode %s as UTF-8: %s — %s", label, path, exc, consequence
        )
        return []

    # Exhaustive only because `text` is strict-UTF-8 decoded above: on a lone
    # surrogate CSafeLoader raises UnicodeEncodeError, not a YAMLError, so a
    # bytes or errors="replace" read path must re-check this handler.
    try:
        data = yaml.load(text, Loader=SAFE_YAML_LOADER)
    except yaml.YAMLError as exc:
        logger.warning(
            "%s: YAML parse error in %s: %s — %s", label, path, exc, consequence
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "%s: expected a YAML list in %s, got %s — %s",
            label, path, type(data).__name__, consequence,
        )
        return []

    return data

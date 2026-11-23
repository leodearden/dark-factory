"""The single home for how fused-memory reads its list-shaped YAML registry
files: safely (a restricted safe loader only) and fast (libyaml when present).
Every file-level failure surfaces as one typed YamlListFileError, so the
CONSUMER decides whether a broken registry fails open. Consumed by the curator's
registry loaders in fused_memory/middleware/cancelled_premise_blocklist.py,
fused_memory/middleware/operational_ask_registry.py and
fused_memory/middleware/recon_code_fix_premise_guard.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["YamlListFileError", "load_yaml_list_file"]


class YamlListFileError(Exception):
    """A registry file that could not be read as a top-level YAML list."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(reason)
        self.path = path


def resolve_safe_yaml_loader(yaml_module: Any = yaml) -> type:
    """Return libyaml's ``CSafeLoader`` if *yaml_module* has it, else ``SafeLoader``.

    Both are safe loaders: the same restricted tag set, no arbitrary object
    construction.
    """
    return getattr(yaml_module, "CSafeLoader", None) or yaml_module.SafeLoader


SAFE_YAML_LOADER: type = resolve_safe_yaml_loader()


def load_yaml_list_file(path: Path | None) -> list[object]:
    """Return the top-level YAML list stored at *path*.

    ``None`` is a deliberate opt-out and yields ``[]``. A file that is missing,
    unreadable, not valid UTF-8, not valid YAML, or whose top-level document is
    not a list raises YamlListFileError naming *path* and the reason.
    """
    if path is None:
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise YamlListFileError(path, f"file not found: {path}") from exc
    except OSError as exc:
        raise YamlListFileError(path, f"cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise YamlListFileError(path, f"cannot decode {path} as UTF-8: {exc}") from exc

    # Exhaustive only because `text` is strict-UTF-8 decoded above: on a lone
    # surrogate CSafeLoader raises UnicodeEncodeError, not a YAMLError, so a
    # bytes or errors="replace" read path must re-check this handler.
    try:
        data = yaml.load(text, Loader=SAFE_YAML_LOADER)
    except yaml.YAMLError as exc:
        raise YamlListFileError(path, f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise YamlListFileError(
            path, f"expected a YAML list in {path}, got {type(data).__name__}"
        )

    return data

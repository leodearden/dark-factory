"""Private maintenance utilities."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator


@contextlib.contextmanager
def override_config_path(config_path: str | None) -> Generator[None, None, None]:
    """Temporarily set CONFIG_PATH in the environment, then restore it.

    When *config_path* is ``None`` the environment is left completely
    untouched — the context manager is a no-op.

    When *config_path* is provided:
    - ``os.environ['CONFIG_PATH']`` is set to *config_path* on entry.
    - On exit (normal **or** exceptional) the original state is restored:
      if CONFIG_PATH was absent before entry it is popped; if it had a
      value that value is reinstated.
    """
    if config_path is None:
        yield
        return

    old_value = os.environ.get('CONFIG_PATH')
    os.environ['CONFIG_PATH'] = config_path
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop('CONFIG_PATH', None)
        else:
            os.environ['CONFIG_PATH'] = old_value

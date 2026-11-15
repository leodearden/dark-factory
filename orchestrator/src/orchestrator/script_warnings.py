"""Warning lines of a reify-style shell script's tagged stderr.

Reify's scripts (seed-warm-lane.sh, hooks-armed-guard.sh, git-rerere-guard.sh)
tag each stderr diagnostic line ``[info]``/``[ok]``/``[warn]``/``[error]``,
optionally ANSI-coloured; this module picks out the warning ones.
"""

from __future__ import annotations

import logging
import re

_ANSI_SGR_RE = re.compile(r'\x1b\[[0-9;]*m')
_WARNING_TAGS = ('[warn]', '[error]')


def tagged_warning_lines(stderr: str) -> tuple[str, ...]:
    """The ``[warn]``/``[error]`` lines of *stderr*, colour removed, in order."""
    plain_lines = (_ANSI_SGR_RE.sub('', line).strip() for line in stderr.splitlines())
    return tuple(line for line in plain_lines if line.startswith(_WARNING_TAGS))


def log_warnings_despite_success(log: logging.Logger, headline: str, stderr: str) -> None:
    """Log the warning lines of a run that exited 0 as ONE WARNING record, or nothing.

    One record keeps a multi-line warning and its continuation lines together.
    """
    lines = tagged_warning_lines(stderr)
    if lines:
        log.warning(
            '%s exited 0 but reported %d warning(s) (fail-open degradations): %s',
            headline, len(lines), ' | '.join(lines),
        )

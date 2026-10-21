"""The one rule for removing absolute paths and home paths from free text a client will see."""

from __future__ import annotations

import re
from typing import Final

PATH_PLACEHOLDER: Final = '[path]'

_STOP: Final = r'\s"\'`()\[\]{}<>'
_GLUE: Final = '=:,'
_PATH_TOKEN: Final = re.compile(
    rf'(?<![^{_STOP}{_GLUE}])'
    rf'(?:/[^{_STOP}/]+/(?=[^{_STOP}])|~/(?=[^{_STOP}])|[^{_STOP}]*?/home/)'
    rf'[^{_STOP}]*?(?=[.,:;!?]*(?:[{_STOP}]|$))'
)


def redact_paths(text: str) -> str:
    """`text` with each path replaced by the placeholder.

    A path is rooted at '/' over two or more segments, or at '~/', and starts a token or follows '=', ':' or ','.
    A '//' starts none, so a URL survives, unless it holds '/home/': a token holding '/home/' is always replaced.
    """
    return _PATH_TOKEN.sub(PATH_PLACEHOLDER, text)

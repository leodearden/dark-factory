"""The one rule for removing absolute paths and home paths from free text a client will see."""

from __future__ import annotations

import re
from typing import Final

PATH_PLACEHOLDER: Final = '[path]'

_STOP: Final = r'\s"\'`()\[\]{}<>'
_PATH_TOKEN: Final = re.compile(
    rf'(?<![^{_STOP}])'
    rf'(?:/[^{_STOP}/]+/(?=[^{_STOP}])|~/(?=[^{_STOP}])|[^{_STOP}]*?/home/)'
    rf'[^{_STOP}]*?(?=[.,:;!?]*(?:[{_STOP}]|$))'
)


def redact_paths(text: str) -> str:
    """`text` with every token rooted at '/' over two or more segments, rooted at '~/', or holding '/home/' replaced."""
    return _PATH_TOKEN.sub(PATH_PLACEHOLDER, text)

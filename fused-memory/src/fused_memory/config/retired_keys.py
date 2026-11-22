"""Drop config keys nothing reads any more, naming them in one WARNING so the retirement is visible."""

import logging

logger = logging.getLogger(__name__)


def drop_retired_keys(data: object, *, section: str, retired: frozenset[str]) -> object:
    if not isinstance(data, dict):
        return data
    present = sorted(key for key in retired if key in data)
    if not present:
        return data
    logger.warning(
        '%s: retired key(s) %s ignored (nothing reads them); delete them from the config file',
        section,
        ', '.join(present),
    )
    return {key: value for key, value in data.items() if key not in retired}

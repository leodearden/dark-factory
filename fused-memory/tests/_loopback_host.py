"""The one predicate behind every committed-bind loopback pin in this suite."""

import ipaddress


def is_loopback_host(host: str) -> bool:
    """*host* is a loopback IP literal, optionally bracketed as in ``[::1]``.

    A name such as ``localhost``, an empty host and a wildcard address are not.
    """
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False

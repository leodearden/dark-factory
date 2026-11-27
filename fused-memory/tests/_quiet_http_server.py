"""The shutdown and noise policy shared by the in-process loopback test servers."""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer
from typing import Any


class QuietHTTPServer(ThreadingHTTPServer):
    """A threading server whose handlers never block shutdown and whose client hang-ups stay silent."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Stay quiet for a client that hung up mid-response; report any other handler fault."""
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)

"""Self-tests for the fake T3 server in _fake_t3_server: it mints values in T3 Code v0.1.6's own shapes."""

from __future__ import annotations

import re

from _fake_t3_server import FakeT3Server

T3_PAIRING_CODE = re.compile(r'[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{12}')


def test_every_pairing_code_the_fake_mints_has_t3s_own_shape() -> None:
    with FakeT3Server.running() as server:
        codes = [server.add_pairing_code() for _ in range(256)]

    assert [code for code in codes if not T3_PAIRING_CODE.fullmatch(code)] == []
    assert len(set(codes)) == len(codes)

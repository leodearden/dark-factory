"""shared.psi.ARM_FIELDS is the cfg vocabulary the saturation gate reads."""

from __future__ import annotations

import shared.psi
from shared.psi import ARM_FIELDS, PsiSample


class TestArmFields:
    def test_it_is_exactly_the_cfg_fields_saturation_reads(self):
        asked: set[str] = set()

        class _EveryArmOff:
            def __getattr__(self, name: str) -> None:
                asked.add(name)
                return None

        readable = PsiSample(
            cpu_some10=0.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True,
        )

        assert not readable.saturated(_EveryArmOff())
        assert asked == ARM_FIELDS

    def test_it_is_public(self):
        assert 'ARM_FIELDS' in shared.psi.__all__

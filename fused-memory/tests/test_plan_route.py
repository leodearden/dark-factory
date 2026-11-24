"""The plan route: config, the cap latch and its HTTP gate (task 9944).

Every plan-routed call goes through the ChatGPT-plan proxy. A cap parks the
work and pages L2; it never falls back onto OPENAI_API_KEY.
"""

import pydantic
import pytest

from fused_memory.config.schema import FusedMemoryConfig

_PROXY_URL = 'http://127.0.0.1:8317/v1'


class TestThePlanRoutesConfiguration:
    """Each test sets or deletes OPENAI_PLAN_PROXY_URL itself: server/main.py's
    import-time load_dotenv() can inject the main checkout's .env into this process.
    """

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('OPENAI_PLAN_PROXY_URL', raising=False)

        plan_route = FusedMemoryConfig().plan_route

        assert plan_route.proxy_url is None
        assert plan_route.probe_interval_seconds == 900.0

    @pytest.mark.parametrize('interval', [0, -5.0])
    def test_a_non_positive_probe_interval_is_rejected_at_load(self, interval: float) -> None:
        with pytest.raises(pydantic.ValidationError) as caught:
            FusedMemoryConfig(plan_route={'probe_interval_seconds': interval})

        message = str(caught.value)
        assert 'plan_route.probe_interval_seconds' in message
        assert f'input_value={interval}' in message

    def test_a_blank_proxy_url_reads_as_none(self) -> None:
        assert FusedMemoryConfig(plan_route={'proxy_url': '  '}).plan_route.proxy_url is None

    def test_the_shipped_config_routes_the_proxy_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv('OPENAI_PLAN_PROXY_URL', _PROXY_URL)

        assert FusedMemoryConfig().plan_route.proxy_url == _PROXY_URL

    def test_the_shipped_config_leaves_the_proxy_unset_without_the_variable(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv('OPENAI_PLAN_PROXY_URL', raising=False)

        assert FusedMemoryConfig().plan_route.proxy_url is None

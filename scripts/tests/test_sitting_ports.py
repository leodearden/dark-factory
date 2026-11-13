"""Each escalation queue reaches the server that serves THAT queue (scripts/sitting/ports.py)."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from fused_memory.config.schema import ReconciliationConfig
from orchestrator.session_registry import normalize_escalations_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import ports

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_CONFIG = 'dark-factory-orchestrator.yaml'


def _root(tmp_path: Path, name: str, escalation: dict | None) -> Path:
    root = tmp_path / name
    (root / 'data' / 'escalations').mkdir(parents=True)
    (root / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)
    if escalation is not None:
        (root / _CONFIG).write_text(yaml.safe_dump({'escalation': escalation}))
    return root


def _project_queue(root: Path) -> str:
    return normalize_escalations_dir(root / 'data' / 'escalations')


def _recon_queue(root: Path) -> str:
    return normalize_escalations_dir(root / 'data' / 'reconciliation' / 'escalations')


def test_each_project_queue_reaches_its_own_roots_port(tmp_path: Path) -> None:
    alpha = _root(tmp_path, 'alpha', {'queue_dir': 'data/escalations', 'port': 18101})
    beta = _root(tmp_path, 'beta', {'queue_dir': 'data/escalations', 'port': 18102})

    mapped = {queue: ports.queue_port(queue) for queue in (_project_queue(alpha), _project_queue(beta))}

    assert mapped == {
        _project_queue(alpha): ports.QueuePort(
            queue_dir=_project_queue(alpha), kind='project',
            url='http://127.0.0.1:18101/mcp', project_root=alpha.resolve(),
        ),
        _project_queue(beta): ports.QueuePort(
            queue_dir=_project_queue(beta), kind='project',
            url='http://127.0.0.1:18102/mcp', project_root=beta.resolve(),
        ),
    }
    alpha_port = mapped[_project_queue(alpha)]
    assert isinstance(alpha_port, ports.QueuePort)
    assert alpha_port.base_url == 'http://127.0.0.1:18101'


def test_a_root_naming_no_port_is_port_unknown(tmp_path: Path) -> None:
    portless = _root(tmp_path, 'portless', {'queue_dir': 'data/escalations'})
    unconfigured = _root(tmp_path, 'unconfigured', None)

    for root in (portless, unconfigured):
        found = ports.queue_port(_project_queue(root))
        assert isinstance(found, ports.PortUnknown)
        assert found.queue_dir == _project_queue(root)
        assert str(root.resolve()) in found.reason


def test_the_recon_queue_reaches_the_recon_server_never_the_projects_port(tmp_path: Path) -> None:
    df = _root(tmp_path, 'dark-factory', {'queue_dir': 'data/escalations', 'port': 18101})

    found = ports.queue_port(_recon_queue(df))

    assert found == ports.QueuePort(
        queue_dir=_recon_queue(df), kind='reconciliation',
        url=ports.RECON_ESCALATION_URL, project_root=None,
    )
    assert isinstance(found, ports.QueuePort) and '18101' not in found.url


def test_a_queue_of_no_known_shape_is_port_unknown(tmp_path: Path) -> None:
    elsewhere = tmp_path / 'somewhere' / 'queue'
    elsewhere.mkdir(parents=True)

    found = ports.queue_port(str(elsewhere))

    assert isinstance(found, ports.PortUnknown)
    assert found.queue_dir == normalize_escalations_dir(elsewhere)
    assert found.reason


def test_the_recon_url_mirrors_fused_memorys_recon_server_defaults() -> None:
    fields = ReconciliationConfig.model_fields
    url = urlsplit(ports.RECON_ESCALATION_URL)

    assert url.port == fields['escalation_port'].default
    assert url.hostname == fields['escalation_host'].default
    assert url.path == '/mcp'

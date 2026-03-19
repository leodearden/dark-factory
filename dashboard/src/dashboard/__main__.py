"""Entry point for ``python -m dashboard``."""

import uvicorn

from dashboard.config import DashboardConfig

config = DashboardConfig()
uvicorn.run('dashboard.app:app', host=config.host, port=config.port, reload=True)

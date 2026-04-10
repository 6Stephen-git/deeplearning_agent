from __future__ import annotations

import uvicorn

from src.backend.api import app
from src.backend.settings import get_settings


def run() -> None:
    s = get_settings()
    uvicorn.run("src.backend.api:app", host=s.api_host, port=s.api_port, reload=False)


if __name__ == "__main__":
    run()

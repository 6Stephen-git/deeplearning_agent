from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import dotenv_values

_ENV_LOADED = False


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    loaded = dotenv_values(path)
    result: Dict[str, str] = {}
    for key, value in loaded.items():
        if not key:
            continue
        if value is None:
            continue
        result[str(key)] = str(value)
    return result


def ensure_project_env_loaded() -> None:
    """
    统一加载环境变量：
    1) .env
    2) .env.local 覆盖 .env
    3) 进程环境变量最高优先（不被覆盖）
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[2]
    merged: Dict[str, str] = {}
    merged.update(_read_env_file(project_root / ".env"))
    merged.update(_read_env_file(project_root / ".env.local"))

    for key, value in merged.items():
        os.environ.setdefault(key, value)

    _ENV_LOADED = True

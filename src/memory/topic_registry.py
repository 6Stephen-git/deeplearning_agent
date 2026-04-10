from __future__ import annotations

import json
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Dict, List


def topic_to_db_suffix(research_topic: str) -> str:
    topic = (research_topic or "").strip()
    return md5(topic.encode("utf-8")).hexdigest()[:8]


def topic_to_db_path(research_topic: str) -> str:
    return f"./data/memory_db_{topic_to_db_suffix(research_topic)}"


def _registry_file() -> Path:
    p = Path("./data/memory_topic_registry.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def register_topic(research_topic: str) -> Dict[str, str]:
    """
    记录主题与 memory_db 后缀映射，便于定位 `memory_db_xxxxxxxx` 对应主题。
    """
    topic = (research_topic or "").strip()
    suffix = topic_to_db_suffix(topic)
    db_path = topic_to_db_path(topic)
    now = datetime.now().isoformat()
    f = _registry_file()

    data: Dict[str, Dict[str, str]] = {}
    if f.exists():
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
        except Exception:
            data = {}

    data[suffix] = {
        "research_topic": topic,
        "db_suffix": suffix,
        "persist_directory": db_path,
        "updated_at": now,
    }

    f.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data[suffix]


def list_topics() -> List[Dict[str, str]]:
    f = _registry_file()
    if not f.exists():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return []
        rows = []
        for _, v in raw.items():
            if isinstance(v, dict):
                rows.append(v)
        rows.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        return rows
    except Exception:
        return []

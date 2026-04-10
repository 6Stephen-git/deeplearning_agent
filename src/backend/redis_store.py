from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisStore:
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._client: Optional[Redis] = None

    def _get_client(self) -> Optional[Redis]:
        if self._client is not None:
            return self._client
        try:
            self._client = Redis.from_url(self._redis_url, decode_responses=True)
            self._client.ping()
            return self._client
        except RedisError as e:
            logger.warning("redis unavailable: %s", e)
            self._client = None
            return None

    @staticmethod
    def payload_hash(task_type: str, payload: Dict[str, Any], version: str = "v1") -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(f"{task_type}:{version}:{raw}".encode("utf-8")).hexdigest()

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        c = self._get_client()
        if c is None:
            return None
        val = c.get(key)
        if not val:
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Dict[str, Any], ttl_s: int) -> None:
        c = self._get_client()
        if c is None:
            return
        c.setex(key, ttl_s, json.dumps(value, ensure_ascii=False))

    def get_text(self, key: str) -> Optional[str]:
        c = self._get_client()
        if c is None:
            return None
        return c.get(key)

    def set_text_if_absent(self, key: str, value: str, ttl_s: int) -> bool:
        c = self._get_client()
        if c is None:
            return False
        return bool(c.set(key, value, ex=ttl_s, nx=True))

    def set_text(self, key: str, value: str, ttl_s: int) -> None:
        c = self._get_client()
        if c is None:
            return
        c.setex(key, ttl_s, value)

    def delete_key(self, key: str) -> None:
        c = self._get_client()
        if c is None:
            return
        c.delete(key)

    def is_healthy(self) -> bool:
        c = self._get_client()
        if c is None:
            return False
        try:
            c.ping()
            return True
        except RedisError:
            return False

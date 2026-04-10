# src/tools/async_search_tool.py
"""
异步搜索工具模块。

支持多种后端（由环境变量 SEARCH_BACKEND 选择）：
- serper: Serper.dev（Google 搜索结果 JSON，默认推荐，需 SERPER_API_KEY）
- tavily: Tavily（需 TAVILY_API_KEY）
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import ClientSession, ClientTimeout
from aiohttp.client_exceptions import (
    ClientConnectionError,
    ClientOSError,
    ServerDisconnectedError,
)
from src.config import ensure_project_env_loaded

# LangSmith tracing（可选依赖：未安装时降级为 no-op）
try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        return _decorator

ensure_project_env_loaded()
logger = logging.getLogger(__name__)


class AsyncSearchTool:
    """
    异步搜索工具：根据 SEARCH_BACKEND 调用 Serper 或 Tavily。
    """

    def __init__(self) -> None:
        self.backend = (os.getenv("SEARCH_BACKEND") or "serper").strip().lower()
        self.max_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        # 总超时（秒）：跨境/不稳定网络可适当加大，如 SEARCH_HTTP_TIMEOUT_S=90
        _total = float(os.getenv("SEARCH_HTTP_TIMEOUT_S", "60"))
        self.timeout = ClientTimeout(
            total=_total,
            connect=min(30.0, _total),
            sock_connect=min(30.0, _total),
            sock_read=max(30.0, _total * 0.85),
        )
        self._http_retries = max(1, int(os.getenv("SEARCH_HTTP_RETRIES", "3")))
        # 注意：不可在多个 ClientSession 之间共享同一个 TCPConnector。
        # Session 关闭时会 close connector，后续并发/重试会报「Session is closed」。

        if self.backend == "serper":
            self.api_key = os.getenv("SERPER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "无法初始化搜索工具: SEARCH_BACKEND=serper 但未找到 SERPER_API_KEY。\n"
                    "  请在 https://serper.dev 申请 API Key，写入 .env；"
                    " 或设置 SEARCH_BACKEND=tavily 并使用 TAVILY_API_KEY。"
                )
            self.endpoint = "https://google.serper.dev/search"
        elif self.backend == "tavily":
            self.api_key = os.getenv("TAVILY_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "无法初始化搜索工具: SEARCH_BACKEND=tavily 但未找到 TAVILY_API_KEY。\n"
                    "  请检查项目根目录下的 .env 文件。"
                )
            self.endpoint = "https://api.tavily.com/search"
        else:
            raise ValueError(
                f"不支持的 SEARCH_BACKEND={self.backend!r}。"
                f" 请使用: serper | tavily"
            )

        self._log_query_text = os.getenv("SEARCH_LOG_QUERIES", "false").lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        logger.info(
            "[搜索工具] 初始化完成，后端=%s, 最大结果数=%s, HTTP超时=%ss, 重试=%s",
            self.backend,
            self.max_results,
            self.timeout.total,
            self._http_retries,
        )

    @traceable(name="web_search", run_type="tool")
    async def ainvoke(self, input_query: str) -> Dict[str, Any]:
        """
        异步执行搜索。

        Args:
            input_query: 搜索查询字符串。

        Returns:
            统一格式的搜索结果字典（含 backend / query / results / answer / error）。
        """
        if self._log_query_text:
            logger.info("[搜索工具] 开始搜索 query=%s", input_query)
        else:
            logger.info("[搜索工具] 开始搜索 query_len=%s", len(input_query or ""))
        result_template: Dict[str, Any] = {
            "backend": self.backend,
            "query": input_query,
            "results": [],
            "answer": None,
            "error": None,
        }

        try:
            if self.backend == "serper":
                search_results, ai_answer = await self._search_with_serper(input_query)
            else:
                search_results, ai_answer = await self._search_with_tavily(input_query)

            result_template["results"] = search_results[: self.max_results]
            result_template["answer"] = ai_answer
            logger.info(
                "[搜索工具] 搜索成功，结果数=%s",
                len(result_template["results"]),
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning("[搜索工具] 搜索失败: %s", error_msg)
            result_template["error"] = error_msg

        return result_template

    async def _post_json_with_retry(
        self,
        payload: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        POST JSON，对远端提前断开、连接重置等瞬时错误做指数退避重试。
        """
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._http_retries + 1):
            try:
                async with ClientSession(
                    timeout=self.timeout,
                    headers={"User-Agent": "deeplearning-agent-async-search/1.0"},
                ) as session:
                    async with session.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise RuntimeError(f"{response.status}, {text}")
                        return await response.json()
            except (
                ServerDisconnectedError,
                ClientConnectionError,
                ClientOSError,
                asyncio.TimeoutError,
                TimeoutError,
            ) as e:
                last_exc = e
                if attempt >= self._http_retries:
                    break
                wait = 0.5 * (2 ** (attempt - 1)) + random.random() * 0.4
                logger.warning(
                    "[搜索工具] HTTP异常(%s/%s): %s: %s，%.1fs后重试",
                    attempt,
                    self._http_retries,
                    type(e).__name__,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)
            except RuntimeError:
                raise

        assert last_exc is not None
        raise last_exc

    @traceable(name="web_search_serper", run_type="tool")
    async def _search_with_serper(
        self, query: str
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Serper.dev：返回 Google 有机结果列表；若有 answerBox 则尝试填入 answer。
        """
        payload = {"q": query, "num": self.max_results}
        logger.info("[搜索工具] request POST %s", self.endpoint)
        if self._log_query_text:
            logger.info(
                "[搜索工具] request json: %s",
                json.dumps(payload, ensure_ascii=False),
            )
        else:
            logger.info("[搜索工具] request json: {\"q\":\"***\",\"num\":%s}", self.max_results)

        extra = {"X-API-KEY": self.api_key}
        data = await self._post_json_with_retry(payload, extra_headers=extra)

        ai_answer: Optional[str] = None
        if isinstance(data, dict):
            ab = data.get("answerBox")
            if isinstance(ab, dict):
                ai_answer = (
                    ab.get("answer")
                    or ab.get("snippet")
                    or ab.get("title")
                )
                if isinstance(ai_answer, str):
                    ai_answer = ai_answer.strip() or None

        formatted_results: List[Dict[str, str]] = []
        organic = data.get("organic") if isinstance(data, dict) else None
        if isinstance(organic, list):
            for item in organic:
                if not isinstance(item, dict):
                    continue
                formatted_results.append(
                    {
                        "title": str(item.get("title") or "无标题"),
                        "url": str(item.get("link") or item.get("url") or ""),
                        "snippet": str(
                            item.get("snippet")
                            or item.get("content")
                            or "无摘要"
                        ),
                    }
                )

        return formatted_results, ai_answer

    @traceable(name="web_search_tavily", run_type="tool")
    async def _search_with_tavily(
        self, query: str
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Tavily API：与原先逻辑一致。"""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "include_answer": True,
        }

        if os.getenv("TAVILY_LOG_FULL_API_KEY", "").lower() in ("1", "true", "yes", "y"):
            safe_body = payload
        else:
            safe_body = {**payload, "api_key": "***"}
        logger.info("[搜索工具] request POST %s", self.endpoint)
        if self._log_query_text:
            logger.info(
                "[搜索工具] request json: %s",
                json.dumps(safe_body, ensure_ascii=False),
            )
        else:
            logger.info("[搜索工具] request json: {\"api_key\":\"***\",\"query\":\"***\"}")

        data = await self._post_json_with_retry(payload)

        ai_answer = data.get("answer") if isinstance(data, dict) else None

        formatted_results: List[Dict[str, str]] = []
        results = data.get("results", []) if isinstance(data, dict) else []
        for item in results:
            formatted_results.append(
                {
                    "title": item.get("title", "无标题"),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", item.get("snippet", "无摘要")),
                }
            )

        return formatted_results, ai_answer


# 创建全局工具实例（导入时按当前 SEARCH_BACKEND 初始化）
async_search_tool = AsyncSearchTool()

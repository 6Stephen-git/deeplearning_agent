"""
简易压测脚本：并发提交 research 任务并统计响应时间。

示例:
  python scripts/load_test_tasks.py --url http://127.0.0.1:8000 --n 20 --c 5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import aiohttp


async def _one(session: aiohttp.ClientSession, url: str, idx: int) -> float:
    payload = {"research_topic": f"RAG topic {idx}", "max_cycles": 1}
    t0 = time.time()
    async with session.post(f"{url}/tasks/research", json=payload) as resp:
        await resp.text()
        resp.raise_for_status()
    return (time.time() - t0) * 1000


async def run(url: str, total: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    durations = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        async def wrapped(i: int):
            async with sem:
                d = await _one(session, url, i)
                durations.append(d)

        await asyncio.gather(*[wrapped(i) for i in range(total)])

    print(f"requests={total} concurrency={concurrency}")
    print(f"avg_ms={statistics.mean(durations):.2f}")
    print(f"p95_ms={statistics.quantiles(durations, n=20)[18]:.2f}")
    print(f"max_ms={max(durations):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--c", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.n, args.c))


if __name__ == "__main__":
    main()

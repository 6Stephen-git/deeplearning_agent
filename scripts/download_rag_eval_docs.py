"""
下载 RAG 技术评估语料到 ./data/rag_eval_docs

用法：
  python scripts/download_rag_eval_docs.py
  python scripts/download_rag_eval_docs.py --out ./data/rag_eval_docs --limit 30

说明：
- 默认读取同目录下 rag_eval_urls.txt（每行一个 URL，# 开头为注释）
- 以“尽力而为”策略下载：失败会记录但不中断
- 保存为 .html（或 .txt），用于后续 build_index 嵌入
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def _safe_name_from_url(url: str) -> str:
    p = urlparse(url)
    host = (p.netloc or "unknown").replace(":", "_")
    path = (p.path or "").strip("/")
    path = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)[:80]
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    base = f"{host}__{path}__{h}".strip("_")
    if not base:
        base = f"doc__{h}"
    return base


def load_urls(urls_file: Path) -> list[str]:
    if not urls_file.exists():
        raise FileNotFoundError(f"URL 清单不存在: {urls_file}")
    urls: list[str] = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/rag_eval_docs", help="输出目录")
    ap.add_argument("--urls", default=str(Path(__file__).with_name("rag_eval_urls.txt")), help="URL 清单文件路径")
    ap.add_argument("--limit", type=int, default=30, help="最多下载条数")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次下载间隔秒数")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls_file = Path(args.urls)
    urls = load_urls(urls_file)[: max(0, args.limit)]

    print(f"[download] urls={len(urls)} out={out_dir}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "deeplearning_agent-rag-eval/1.0 (+https://example.local)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    ok = 0
    failed = 0
    for i, url in enumerate(urls, 1):
        name = _safe_name_from_url(url)
        target = out_dir / f"{i:02d}__{name}.html"
        meta = out_dir / f"{i:02d}__{name}.url.txt"

        if target.exists():
            print(f"[{i:02d}] skip exists: {url}")
            ok += 1
            continue

        try:
            print(f"[{i:02d}] GET {url}")
            r = session.get(url, timeout=30)
            r.raise_for_status()
            target.write_bytes(r.content)
            meta.write_text(url, encoding="utf-8")
            ok += 1
        except Exception as e:
            failed += 1
            err_path = out_dir / f"{i:02d}__{name}.error.txt"
            err_path.write_text(f"url={url}\nerror={e}\n", encoding="utf-8")
            print(f"[{i:02d}] FAILED: {e}")

        time.sleep(max(0.0, float(args.sleep)))

    print(f"[download] done ok={ok} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


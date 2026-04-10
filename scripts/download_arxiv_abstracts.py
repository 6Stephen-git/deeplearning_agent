"""
通过 arXiv API 批量拉取摘要，生成可索引文本语料。

目标：在网络受限场景下稳定扩容文档数量（例如 350+）。
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests


API = "http://export.arxiv.org/api/query"


def _strip_xml_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    return html.unescape(m.group(1).strip()) if m else ""


def _extract_entries(feed_xml: str) -> list[dict]:
    parts = feed_xml.split("<entry>")
    out = []
    for p in parts[1:]:
        chunk = p.split("</entry>", 1)[0]
        aid = _strip_xml_tag(chunk, "id")
        title = _strip_xml_tag(chunk, "title")
        summ = _strip_xml_tag(chunk, "summary")
        published = _strip_xml_tag(chunk, "published")
        if not aid or not summ:
            continue
        out.append(
            {
                "id": aid,
                "title": " ".join(title.split()),
                "summary": " ".join(summ.split()),
                "published": published,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/rag_eval_ai_docs_arxiv_api")
    ap.add_argument("--target", type=int, default=420, help="目标文档数")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    queries = [
        "cat:cs.AI",
        "cat:cs.CL",
        "cat:cs.LG",
        "cat:cs.IR",
        "all:agentic+AI",
        "all:RAG+evaluation",
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen = set()
    for q in queries:
        start = 0
        while saved < args.target:
            params = f"search_query={quote_plus(q)}&start={start}&max_results={max(1, args.page_size)}&sortBy=submittedDate&sortOrder=descending"
            url = f"{API}?{params}"
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"[arxiv-api] failed q={q} start={start}: {e}")
                break
            entries = _extract_entries(r.text)
            if not entries:
                break
            for e in entries:
                aid = e["id"]
                if aid in seen:
                    continue
                seen.add(aid)
                fname = re.sub(r"[^a-zA-Z0-9._-]+", "_", aid.split("/")[-1])[:80]
                path = out_dir / f"{saved+1:04d}__{fname}.txt"
                body = (
                    f"title: {e['title']}\n"
                    f"published: {e['published']}\n"
                    f"source: {e['id']}\n\n"
                    f"abstract:\n{e['summary']}\n"
                )
                path.write_text(body, encoding="utf-8")
                saved += 1
                if saved >= args.target:
                    break
            print(f"[arxiv-api] query={q} start={start} saved={saved}")
            start += max(1, args.page_size)
            time.sleep(max(0.0, float(args.sleep)))
            if len(entries) < max(1, args.page_size):
                break
        if saved >= args.target:
            break

    print(f"[arxiv-api] done saved={saved} out={out_dir}", flush=True)
    return 0 if saved > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


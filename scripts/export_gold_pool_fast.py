"""
快速导出 gold_pool（不依赖在线 LLM），用于人工标注阶段的数据准备。

说明：
- baseline: 原始 query
- hyde: 轻量假设文本改写（启发式）
- mqe: 多变体改写（启发式）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_store import MemoryStore  # noqa: E402


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _hyde_query(q: str) -> str:
    return f"{q}\n请检索支持该问题的定义、方法、评测指标、典型案例与局限。"


def _mqe_queries(q: str) -> List[str]:
    return [
        q,
        f"{q} benchmark evaluation metrics",
        f"{q} 方法 对比 局限",
        f"{q} 实践案例 企业落地",
    ]


def _search(store: MemoryStore, query: str, k: int, topic: str) -> List[Tuple[str, float, str, List[str]]]:
    out: List[Tuple[str, float, str, List[str]]] = []
    for mem, score in store.search_by_similarity(query=query, n_results=k, research_topic=topic):
        meta = mem.metadata or {}
        urls = meta.get("source_urls") or []
        if isinstance(urls, str):
            urls = [urls]
        out.append((mem.id, float(score), mem.content or "", urls))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="./data/rag_eval_ai_queries_80.gold.jsonl")
    ap.add_argument("--db-dir", default="./data/rag_eval_ai_db")
    ap.add_argument("--topic", default="rag_eval_ai")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default="./data/rag_eval_ai_gold_pool_fast.jsonl")
    args = ap.parse_args()

    queries = _load_queries(Path(args.queries))
    store = MemoryStore(persist_directory=args.db_dir)

    lines: List[str] = []
    for i, q in enumerate(queries, 1):
        text = q.get("query", "")
        by_id: Dict[str, Dict[str, Any]] = {}

        for mode, qx in [("baseline", text), ("hyde", _hyde_query(text))]:
            for cid, score, body, urls in _search(store, qx, args.k, args.topic):
                row = by_id.setdefault(
                    cid,
                    {
                        "id": cid,
                        "score": score,
                        "source_url": (urls[0] if urls else None),
                        "from_modes": [],
                        "preview": (body or "")[:220],
                    },
                )
                row["score"] = max(float(row["score"]), float(score))
                if mode not in row["from_modes"]:
                    row["from_modes"].append(mode)

        for qv in _mqe_queries(text):
            for cid, score, body, urls in _search(store, qv, max(5, args.k // 2), args.topic):
                row = by_id.setdefault(
                    cid,
                    {
                        "id": cid,
                        "score": score,
                        "source_url": (urls[0] if urls else None),
                        "from_modes": [],
                        "preview": (body or "")[:220],
                    },
                )
                row["score"] = max(float(row["score"]), float(score))
                if "mqe" not in row["from_modes"]:
                    row["from_modes"].append("mqe")

        cands = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
        item = {
            "id": q.get("id"),
            "query": text,
            "gold_answer": q.get("gold_answer"),
            "current_relevant_ids": q.get("relevant_ids") or [],
            "candidates": cands,
        }
        lines.append(json.dumps(item, ensure_ascii=False))
        if i % 10 == 0:
            print(f"[fast-pool] progress {i}/{len(queries)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[fast-pool] done -> {out_path} queries={len(queries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


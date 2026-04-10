"""
从 LLM 自动标注结果中生成 10%-20% 分层抽检样本。

分层维度：
- topic_bucket
- difficulty
- label

优先抽检：
- 低 confidence
- label=2
- needs_review=1
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_queries(path: Path) -> Dict[str, Dict[str, str]]:
    import json

    out: Dict[str, Dict[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        row = json.loads(s)
        qid = str(row.get("id", "")).strip()
        if not qid:
            continue
        out[qid] = {
            "topic_bucket": str(row.get("topic_bucket", "unknown")),
            "difficulty": str(row.get("difficulty", "unknown")),
        }
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="./data/annotation/main_80.llm_labeled.csv")
    ap.add_argument("--queries", default="./data/rag_eval_ai_queries_80.gold.jsonl")
    ap.add_argument("--out", default="./data/annotation/review_sample.csv")
    ap.add_argument("--rate", type=float, default=0.15, help="抽检比例，建议 0.1~0.2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = _read_csv(Path(args.labels))
    qmeta = _read_queries(Path(args.queries))
    rng = random.Random(args.seed)

    # enrich
    for r in rows:
        qid = (r.get("query_id") or "").strip()
        meta = qmeta.get(qid, {})
        r["topic_bucket"] = meta.get("topic_bucket", "unknown")
        r["difficulty"] = meta.get("difficulty", "unknown")

    # stratified groups
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("topic_bucket", "unknown"),
            r.get("difficulty", "unknown"),
            r.get("label", ""),
        )
        groups[key].append(r)

    sampled: List[Dict[str, str]] = []
    for key, g in groups.items():
        n = len(g)
        k = max(1, int(round(n * max(0.1, min(0.2, args.rate)))))
        # 优先序：needs_review=1 -> low confidence -> label=2
        def _score(x: Dict[str, str]) -> Tuple[int, float, int]:
            nr = 1 if (x.get("needs_review", "0") == "1") else 0
            try:
                cf = float(x.get("confidence") or 0.0)
            except Exception:
                cf = 0.0
            lb2 = 1 if (x.get("label", "") == "2") else 0
            return (nr, -cf, lb2)

        sorted_g = sorted(g, key=_score, reverse=True)
        head = sorted_g[: min(k, len(sorted_g))]
        # 再随机补齐（若分数同质）
        if len(head) < k:
            remain = [x for x in g if x not in head]
            rng.shuffle(remain)
            head.extend(remain[: (k - len(head))])
        sampled.extend(head[:k])

    # 去重 (query_id, candidate_id)
    uniq: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in sampled:
        key = ((r.get("query_id") or ""), (r.get("candidate_id") or ""))
        if key not in uniq:
            uniq[key] = r
    final_rows = list(uniq.values())

    _write_csv(Path(args.out), final_rows)
    print(
        f"[spotcheck] input={len(rows)} sampled={len(final_rows)} rate={args.rate:.2f} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


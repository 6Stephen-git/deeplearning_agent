"""
把人工标注结果回写为 queries.jsonl 的 relevant_ids，并输出偏置检查统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _read_labels(path: Path) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = (row.get("query_id") or "").strip()
            cid = (row.get("candidate_id") or "").strip()
            lb = (row.get("label") or "").strip()
            if not qid or not cid or lb == "":
                continue
            try:
                v = int(lb)
            except ValueError:
                continue
            if v not in (0, 1, 2):
                continue
            key = (qid, cid)
            prev = out.get(key)
            if prev is None or v > prev:
                out[key] = v
    return out


def _resolve_existing_ids(
    candidate_ids: Set[str],
    db_dir: str,
    research_topic: str,
    batch_size: int = 512,
) -> Set[str]:
    if not candidate_ids:
        return set()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.memory.memory_store import MemoryStore

    old_db = os.getenv("RAG_EVAL_DB_DIR")
    old_topic = os.getenv("RAG_EVAL_RESEARCH_TOPIC")
    os.environ["RAG_EVAL_DB_DIR"] = db_dir
    os.environ["RAG_EVAL_RESEARCH_TOPIC"] = research_topic
    try:
        store = MemoryStore(persist_directory=db_dir)
        ids = list(candidate_ids)
        existing: Set[str] = set()
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            data = store.collection.get(ids=chunk, include=["metadatas"])
            got_ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            for gid, meta in zip(got_ids, metas):
                if not gid:
                    continue
                if research_topic:
                    topic = (meta or {}).get("research_topic")
                    if topic != research_topic:
                        continue
                existing.add(gid)
        return existing
    finally:
        if old_db is None:
            os.environ.pop("RAG_EVAL_DB_DIR", None)
        else:
            os.environ["RAG_EVAL_DB_DIR"] = old_db
        if old_topic is None:
            os.environ.pop("RAG_EVAL_RESEARCH_TOPIC", None)
        else:
            os.environ["RAG_EVAL_RESEARCH_TOPIC"] = old_topic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", required=True, help="合并后的标注 CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-partial", action="store_true", help="是否将 label=1 也计入 relevant_ids")
    ap.add_argument("--repair-with-store", action="store_true", help="是否按当前向量库校验/修复 relevant_ids")
    ap.add_argument("--fill-from-pool", action="store_true", help="当某 query 的 relevant_ids 过少时，尝试从池内标注候选补齐")
    ap.add_argument("--db-dir", default="./data/rag_eval_db", help="用于校验 id 的向量库目录")
    ap.add_argument("--research-topic", default="rag_eval", help="用于校验 id 的 research_topic")
    ap.add_argument("--min-relevant-per-query", type=int, default=1)
    ap.add_argument(
        "--max-low-relevant-ratio",
        type=float,
        default=0.2,
        help="low_relevant_queries 占比上限，超过则返回非0提醒补充标注覆盖",
    )
    args = ap.parse_args()

    queries = _read_jsonl(Path(args.queries))
    pool_rows = _read_jsonl(Path(args.pool))
    labels = _read_labels(Path(args.labels))

    pool_map = {x.get("id"): x for x in pool_rows}
    mode_counter = Counter()
    low_relevant: List[str] = []
    q2rel: Dict[str, List[str]] = {}
    pool_q_candidates: Dict[str, List[str]] = {}

    for q in queries:
        qid = q.get("id")
        if not qid:
            continue
        prow = pool_map.get(qid, {})
        cids: List[str] = []
        for c in (prow.get("candidates") or []):
            cid = c.get("id")
            if cid:
                cids.append(cid)
        pool_q_candidates[qid] = cids

    allowed_labels = {2, 1} if args.include_partial else {2}
    all_selected_ids: Set[str] = set()
    for qid, cids in pool_q_candidates.items():
        for cid in cids:
            if labels.get((qid, cid), 0) in allowed_labels:
                all_selected_ids.add(cid)

    existing_ids: Set[str] = set()
    if args.repair_with_store:
        existing_ids = _resolve_existing_ids(
            all_selected_ids,
            db_dir=args.db_dir,
            research_topic=args.research_topic,
        )
    missing_before = len(all_selected_ids - existing_ids) if args.repair_with_store else 0
    filtered_out_count = 0
    filled_queries = 0

    for q in queries:
        qid = q.get("id")
        if not qid:
            continue
        prow = pool_map.get(qid, {})
        rel: List[str] = []
        before_repair_rel_count = 0
        for c in (prow.get("candidates") or []):
            cid = c.get("id")
            if not cid:
                continue
            lb = labels.get((qid, cid), 0)
            if lb in allowed_labels:
                before_repair_rel_count += 1
                if (not args.repair_with_store) or (cid in existing_ids):
                    rel.append(cid)
                for m in (c.get("from_modes") or []):
                    mode_counter[m] += 1
        if args.repair_with_store:
            filtered_out_count += max(0, before_repair_rel_count - len(rel))
        if args.repair_with_store and args.fill_from_pool and len(rel) < max(1, args.min_relevant_per_query):
            before_fill = len(rel)
            for cid in pool_q_candidates.get(qid, []):
                if cid in rel:
                    continue
                if labels.get((qid, cid), 0) not in allowed_labels:
                    continue
                if cid not in existing_ids:
                    continue
                rel.append(cid)
                if len(rel) >= max(1, args.min_relevant_per_query):
                    break
            if len(rel) > before_fill:
                filled_queries += 1
        q2rel[qid] = rel
        if len(rel) < max(1, args.min_relevant_per_query):
            low_relevant.append(qid)

    out_rows: List[str] = []
    for q in queries:
        qid = q.get("id")
        q["relevant_ids"] = q2rel.get(qid, [])
        out_rows.append(json.dumps(q, ensure_ascii=False))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_rows) + ("\n" if out_rows else ""), encoding="utf-8")

    total_rel = sum(len(v) for v in q2rel.values())
    non_empty_queries = sum(1 for v in q2rel.values() if v)
    output_ids = {cid for rel in q2rel.values() for cid in rel}
    missing_after = len(output_ids - existing_ids) if args.repair_with_store else 0
    low_ratio = (len(low_relevant) / len(queries)) if queries else 0.0
    print(
        f"[apply-gold] queries={len(queries)} total_relevant={total_rel} -> {out_path}\n"
        f"[apply-gold] non_empty_queries={non_empty_queries}/{len(queries)}\n"
        f"[apply-gold] low_relevant_queries(<{args.min_relevant_per_query})={len(low_relevant)} {low_relevant[:10]}\n"
        f"[apply-gold] low_relevant_ratio={low_ratio:.2%} threshold<={args.max_low_relevant_ratio:.2%}\n"
        f"[apply-gold] missing_before={missing_before} missing_after={missing_after} "
        f"filtered_out={filtered_out_count} filled_queries={filled_queries}\n"
        f"[apply-gold] mode_distribution={dict(mode_counter)}",
        flush=True,
    )
    if low_ratio > max(0.0, args.max_low_relevant_ratio):
        print(
            "[apply-gold] WARNING: low_relevant 占比过高，建议提高候选覆盖（per-query-limit）或提高 LLM 标注质量后重跑。",
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


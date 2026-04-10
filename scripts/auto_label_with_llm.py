"""
使用 LLM 自动标注 gold_pool 候选，输出结构化 CSV。

输出字段（无 reason）：
- query_id, query, gold_answer, candidate_id, from_modes
- score, preview, label (0/1/2), confidence (0~1), needs_review (0/1), annotator

说明：
- 若 preview 为空，会按 candidate_id 从向量库补全文片段（截断）
- 规则与疑难判例从 markdown 模板注入 prompt
- 当 --max-consecutive-failures > 0 时：按任务索引有序并行（--concurrency），连续失败仍按索引计次，达阈值则截断写断点并退出
- 断点文件 JSON：next_index、total、pool、out、时间戳等；用 --resume 从 next_index 继续
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_store import MemoryStore  # noqa: E402
from src.tools.llm_client import get_enhancer_llm_client  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_label(v: Any) -> int:
    try:
        x = int(v)
    except Exception:
        x = 1
    if x < 0:
        return 0
    if x > 2:
        return 2
    return x


def _normalize_confidence(v: Any, label: int) -> float:
    try:
        c = float(v)
    except Exception:
        c = 0.55 if label == 1 else (0.7 if label == 2 else 0.6)
    if c < 0:
        c = 0.0
    if c > 1:
        c = 1.0
    return c


def _needs_review(label: int, confidence: float, preview: str) -> int:
    if confidence < 0.5:
        return 1
    if label == 2 and confidence < 0.65:
        return 1
    if not preview.strip():
        return 1
    return 0


def _fetch_preview_from_db(store: MemoryStore, cid: str, max_chars: int = 800) -> str:
    if not cid:
        return ""
    try:
        data = store.collection.get(ids=[cid], include=["documents"])
    except Exception:
        return ""
    docs = (data or {}).get("documents") or []
    if not docs:
        return ""
    doc = docs[0] or ""
    if not isinstance(doc, str):
        return ""
    doc = doc.strip().replace("\n", " ")
    if len(doc) > max_chars:
        return doc[:max_chars] + " ..."
    return doc


CSV_FIELDS = [
    "query_id",
    "query",
    "gold_answer",
    "candidate_id",
    "from_modes",
    "score",
    "preview",
    "label",
    "confidence",
    "needs_review",
    "annotator",
]


def _build_work(
    pool_rows: List[Dict[str, Any]], per_query_limit: int
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    work: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for qrow in pool_rows:
        cands = qrow.get("candidates") or []
        for cand in cands[: max(1, per_query_limit)]:
            work.append((qrow, cand))
    return work


def _write_checkpoint(
    path: Path,
    *,
    pool: str,
    out: str,
    next_index: int,
    total: int,
    message: str,
    last_errors: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pool": pool,
        "out": out,
        "next_index": next_index,
        "total": total,
        "message": message,
        "last_errors": last_errors[-10:],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _truncate_csv_keep_data_rows(path: Path, keep: int) -> None:
    """保留前 keep 条数据行（不含表头）；keep=0 则只写表头。"""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or list(CSV_FIELDS)
        rows = list(r)
    keep = max(0, min(keep, len(rows)))
    kept = rows[:keep]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)


async def _label_one(
    llm,
    system_prompt: str,
    query_id: str,
    query: str,
    gold_answer: str,
    candidate: Dict[str, Any],
    preview_text: str,
) -> Dict[str, Any]:
    payload = {
        "query_id": query_id,
        "query": query,
        "gold_answer": gold_answer,
        "candidate_id": candidate.get("id"),
        "source_url": candidate.get("source_url"),
        "from_modes": candidate.get("from_modes") or [],
        "score": float(candidate.get("score", 0.0)),
        "preview": preview_text,
    }
    human_prompt = (
        "请根据规则对候选证据打标签。\n"
        "必须严格输出 JSON 对象，字段仅允许：label,confidence。\n"
        "label 只能是 0/1/2；confidence 0~1。\n\n"
        f"输入:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    resp = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = " ".join(
            (x.get("text", "") if isinstance(x, dict) else str(x)) for x in content
        )
    obj = _extract_json_obj(str(content)) or {}
    label = _normalize_label(obj.get("label", 1))
    confidence = _normalize_confidence(obj.get("confidence", 0.55), label=label)
    return {"label": label, "confidence": confidence}


def _make_row(
    qrow: Dict[str, Any],
    cand: Dict[str, Any],
    preview: str,
    label: int,
    confidence: float,
    llm_failed: bool,
) -> Dict[str, str]:
    qid = str(qrow.get("id", "")).strip()
    query = str(qrow.get("query", "")).strip()
    gold_answer = str(qrow.get("gold_answer", "")).strip()
    cid = str(cand.get("id", "")).strip()
    from_modes = ",".join(cand.get("from_modes") or [])
    score = float(cand.get("score", 0.0))
    nr = 1 if llm_failed else _needs_review(label, confidence, preview)
    return {
        "query_id": qid,
        "query": query,
        "gold_answer": gold_answer,
        "candidate_id": cid,
        "from_modes": from_modes,
        "score": f"{score:.6f}",
        "preview": preview,
        "label": str(label),
        "confidence": f"{confidence:.4f}",
        "needs_review": str(nr),
        "annotator": "llm_auto",
    }


async def _run_parallel_with_streak_stop(
    *,
    work: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    start_index: int,
    store: MemoryStore,
    llm,
    system_prompt: str,
    max_preview_chars: int,
    max_consecutive_failures: int,
    concurrency: int,
    checkpoint_path: Path,
    out_path: Path,
    pool_str: str,
    progress_every: int,
    append_csv: bool,
) -> int:
    """
    有序并行：最多 concurrency 路并发调用 LLM，但**按任务索引顺序**写 CSV 并统计连续失败，
    语义与串行版一致（连续失败达到阈值则截断、断点、退出码 4）。
    """
    total = len(work)
    fail_streak = 0
    streak_start: Optional[int] = None
    last_errors: List[str] = []
    t0 = time.monotonic()
    done_before = start_index
    sem = asyncio.Semaphore(max(1, concurrency))
    inflight: Dict[int, asyncio.Task] = {}

    async def _one_index(i: int) -> Tuple[str, int, Any]:
        async with sem:
            qrow, cand = work[i]
            qid = str(qrow.get("id", "")).strip()
            cid = str(cand.get("id", "")).strip()
            preview = str(cand.get("preview") or "").strip()
            if not preview:
                preview = _fetch_preview_from_db(store, cid, max_chars=max_preview_chars)
            try:
                result = await _label_one(
                    llm=llm,
                    system_prompt=system_prompt,
                    query_id=qid,
                    query=str(qrow.get("query", "")).strip(),
                    gold_answer=str(qrow.get("gold_answer", "")).strip(),
                    candidate=cand,
                    preview_text=preview,
                )
                row = _make_row(
                    qrow, cand, preview, int(result["label"]), float(result["confidence"]), False
                )
                return ("ok", i, row)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                return ("err", i, (qrow, cand, preview, err))

    def _schedule(j: int) -> None:
        if j < total and j not in inflight:
            inflight[j] = asyncio.create_task(_one_index(j))

    mode = "a" if append_csv and out_path.exists() and out_path.stat().st_size > 0 else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open(mode, encoding="utf-8-sig", newline="")
    aborted = False
    try:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            w.writeheader()

        c = max(1, concurrency)
        for j in range(start_index, min(start_index + c, total)):
            _schedule(j)

        for i in range(start_index, total):
            if i not in inflight:
                _schedule(i)
            typ, idx, payload = await inflight.pop(i)
            if idx != i:
                raise RuntimeError(f"task order bug: expected {i} got {idx}")

            qrow, cand = work[i]
            qid = str(qrow.get("id", "")).strip()
            cid = str(cand.get("id", "")).strip()

            if typ == "ok":
                row = payload
                fail_streak = 0
                streak_start = None
            else:
                qrow_e, cand_e, preview_e, err = payload
                last_errors.append(err)
                fail_streak += 1
                if streak_start is None:
                    streak_start = i
                if fail_streak >= max_consecutive_failures:
                    for t in inflight.values():
                        t.cancel()
                    if inflight:
                        await asyncio.gather(*inflight.values(), return_exceptions=True)
                    inflight.clear()
                    f.flush()
                    f.close()
                    aborted = True
                    _truncate_csv_keep_data_rows(out_path, streak_start)
                    _write_checkpoint(
                        checkpoint_path,
                        pool=pool_str,
                        out=str(out_path),
                        next_index=streak_start,
                        total=total,
                        message=f"连续 {max_consecutive_failures} 次 LLM 调用失败（有序并行，按索引），"
                        f"断点 next_index={streak_start}。",
                        last_errors=last_errors,
                    )
                    print(
                        f"\n[auto-label] ABORT: 连续失败 {fail_streak} 次（并行 concurrency={c}），"
                        f"断点 next_index={streak_start} -> {checkpoint_path}",
                        flush=True,
                    )
                    return 4
                row = _make_row(qrow_e, cand_e, preview_e, 1, 0.30, llm_failed=True)

            w.writerow(row)
            f.flush()

            nxt = i + c
            if nxt < total:
                _schedule(nxt)

            done = i + 1
            if progress_every <= 1 or done % progress_every == 0 or done == total:
                elapsed = max(1e-6, time.monotonic() - t0)
                rate = (done - done_before) / elapsed
                remain = total - done
                eta_s = remain / rate if rate > 0 else 0.0
                pct = 100.0 * done / total
                print(
                    f"[auto-label] {done}/{total} ({pct:.1f}%) | fail_streak={fail_streak} | "
                    f"{rate:.2f} items/s | ETA ~{eta_s/60:.1f} min | qid={qid} cid={cid[:16]}... | "
                    f"parallel={c}",
                    flush=True,
                )

            _write_checkpoint(
                checkpoint_path,
                pool=pool_str,
                out=str(out_path),
                next_index=done,
                total=total,
                message="进行中（有序并行）",
                last_errors=last_errors[-5:],
            )
    finally:
        if not aborted and inflight:
            for t in inflight.values():
                t.cancel()
            await asyncio.gather(*inflight.values(), return_exceptions=True)
        if not aborted and f is not None and not f.closed:
            f.close()

    print(f"[auto-label] 完成 {total} 条 -> {out_path} (parallel={max(1, concurrency)})", flush=True)
    return 0


async def _run_parallel_legacy(
    *,
    work: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    store: MemoryStore,
    llm,
    system_prompt: str,
    max_preview_chars: int,
    concurrency: int,
    progress_every: int,
    out_path: Path,
) -> int:
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(work)
    results: List[Optional[Dict[str, str]]] = [None] * total
    done_counter = 0
    lock = asyncio.Lock()
    t0 = time.monotonic()

    async def _run_index(idx: int, qrow: Dict[str, Any], cand: Dict[str, Any]) -> None:
        nonlocal done_counter
        async with sem:
            qid = str(qrow.get("id", "")).strip()
            cid = str(cand.get("id", "")).strip()
            preview = str(cand.get("preview") or "").strip()
            if not preview:
                preview = _fetch_preview_from_db(store, cid, max_chars=max_preview_chars)
            try:
                result = await _label_one(
                    llm=llm,
                    system_prompt=system_prompt,
                    query_id=qid,
                    query=str(qrow.get("query", "")).strip(),
                    gold_answer=str(qrow.get("gold_answer", "")).strip(),
                    candidate=cand,
                    preview_text=preview,
                )
                row = _make_row(
                    qrow, cand, preview, int(result["label"]), float(result["confidence"]), False
                )
            except Exception:
                row = _make_row(qrow, cand, preview, 1, 0.30, True)
            results[idx] = row
            async with lock:
                done_counter += 1
                d = done_counter
                if progress_every <= 1 or d % progress_every == 0 or d == total:
                    elapsed = max(1e-6, time.monotonic() - t0)
                    rate = d / elapsed
                    remain = total - d
                    eta_s = remain / rate if rate > 0 else 0.0
                    pct = 100.0 * d / total
                    print(
                        f"[auto-label] {d}/{total} ({pct:.1f}%) | ~{rate:.2f} items/s | ETA ~{eta_s/60:.1f} min",
                        flush=True,
                    )

    await asyncio.gather(*[_run_index(i, q, c) for i, (q, c) in enumerate(work)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            if r:
                w.writerow(r)
    print(f"[auto-label] wrote {total} rows -> {out_path}", flush=True)
    return 0


async def main_async(args) -> int:
    pool_path = Path(args.pool).resolve()
    out_path = Path(args.out).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    pool_rows = _read_jsonl(pool_path)
    qm = (getattr(args, "queries_merge", None) or "").strip()
    if qm:
        qm_path = Path(qm)
        if qm_path.exists():
            qm_rows = _read_jsonl(qm_path.resolve())
            by_id = {str(x.get("id")): x for x in qm_rows if x.get("id")}
            n_ga = 0
            n_q = 0
            for row in pool_rows:
                qid = str(row.get("id", "")).strip()
                if not qid or qid not in by_id:
                    continue
                src = by_id[qid]
                ga = (src.get("gold_answer") or "").strip()
                if ga:
                    row["gold_answer"] = ga
                    n_ga += 1
                qt = (src.get("query") or "").strip()
                if qt:
                    row["query"] = qt
                    n_q += 1
            print(
                f"[auto-label] queries-merge: 自 {qm_path} 按 id 覆盖 "
                f"gold_answer={n_ga} 条, query={n_q} 条",
                flush=True,
            )
        else:
            print(f"[auto-label] 警告: queries-merge 文件不存在，跳过: {qm_path}", flush=True)
    guideline = _read_text(Path(args.guideline))
    hard_cases = _read_text(Path(args.hard_cases))
    prompt_tail = _read_text(Path(args.prompt_tail)) if args.prompt_tail else ""
    system_prompt = (
        "你是RAG证据标注员。你必须严格遵守规则并输出结构化JSON。\n"
        "以下是标注规范：\n"
        f"{guideline}\n\n"
        "以下是疑难判例参考：\n"
        f"{hard_cases}\n\n"
        "硬约束：\n"
        "1) 时间敏感query必须考虑年份约束；\n"
        "2) 同源重复证据不要重复打2；\n"
        "3) 无可追溯来源优先降级；\n"
        "4) 若证据能直接支撑 query 的核心问题，或能直接支撑 gold_answer 的关键要点，应优先判为 label=2（不要求覆盖全部细节）；\n"
        "5) 仅在证据明显不足、偏题、或只能提供背景时，label 最多为 1。\n"
        "6) 目标：在候选池确实存在强相关证据时，每条 query 尽量给出 2-4 条 label=2（若找不到，再保守）。\n"
    )
    if prompt_tail.strip():
        system_prompt += "\n补充规则：\n" + prompt_tail.strip()

    work = _build_work(pool_rows, args.per_query_limit)
    total = len(work)
    if total == 0:
        print("[auto-label] no candidates found.")
        return 2

    start_index = 0
    append_csv = False
    if args.resume:
        ck = _read_checkpoint(checkpoint_path)
        if not ck:
            print("[auto-label] --resume 但断点文件不存在或损坏，从头开始。", flush=True)
        else:
            if str(ck.get("pool")) != str(pool_path) or int(ck.get("total", -1)) != total:
                print(
                    "[auto-label] 警告: 断点与当前 pool/total 不一致，仍使用断点中的 next_index；"
                    "若不对请删除断点后重跑。",
                    flush=True,
                )
            start_index = max(0, min(int(ck.get("next_index", 0)), total))
            append_csv = start_index > 0 and out_path.exists()
            print(f"[auto-label] 从断点续跑: start_index={start_index}/{total}", flush=True)

    store = MemoryStore(persist_directory=args.db_dir)
    llm = get_enhancer_llm_client()

    pool_str = str(pool_path)
    if args.max_consecutive_failures > 0:
        return await _run_parallel_with_streak_stop(
            work=work,
            start_index=start_index,
            store=store,
            llm=llm,
            system_prompt=system_prompt,
            max_preview_chars=args.max_preview_chars,
            max_consecutive_failures=args.max_consecutive_failures,
            concurrency=args.concurrency,
            checkpoint_path=checkpoint_path,
            out_path=out_path,
            pool_str=pool_str,
            progress_every=max(1, args.progress_every),
            append_csv=append_csv,
        )

    if start_index > 0:
        print(
            "[auto-label] 未启用连续失败断点（max-consecutive-failures=0）时，并行模式不支持 --resume；"
            "请改用 max-consecutive-failures>0 或删除输出重来。",
            flush=True,
        )
        return 3

    return await _run_parallel_legacy(
        work=work,
        store=store,
        llm=llm,
        system_prompt=system_prompt,
        max_preview_chars=args.max_preview_chars,
        concurrency=args.concurrency,
        progress_every=max(1, args.progress_every),
        out_path=out_path,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="./data/rag_eval_ai_gold_pool_fast.jsonl")
    ap.add_argument(
        "--queries-merge",
        default="",
        help="可选：queries JSONL，按 id 将 gold_answer（及 query）合并进 pool 行，避免 pool 内嵌旧参考答案",
    )
    ap.add_argument("--out", default="./data/annotation/main_80.llm_labeled.csv")
    ap.add_argument("--guideline", default="./data/annotation/guideline.md")
    ap.add_argument("--hard-cases", default="./data/annotation/hard_cases.md")
    ap.add_argument("--prompt-tail", default="")
    ap.add_argument("--db-dir", default="./data/rag_eval_ai_db")
    ap.add_argument("--per-query-limit", type=int, default=25)
    ap.add_argument("--max-preview-chars", type=int, default=800)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="连续 LLM 异常（按任务索引顺序计）达此次数则截断、写断点并退出；"
        "与 --concurrency 有序并行兼容。0 表示关闭（全并行写完，失败条目标记为降级）",
    )
    ap.add_argument(
        "--checkpoint",
        default="./data/annotation/auto_label.checkpoint.json",
        help="断点 JSON 路径（进度与续跑）",
    )
    ap.add_argument("--resume", action="store_true", help="从断点 next_index 续写 CSV")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="每完成多少条打印一行进度（串行建议 1）",
    )
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

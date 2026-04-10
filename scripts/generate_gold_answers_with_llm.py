"""
为评估 queries JSONL 中每条问题调用 LLM 生成参考答案 gold_answer（不依赖 baseline Top-k）。

- 输出与输入行数一致，保留原有字段，覆盖/写入 gold_answer，并设置 gold_generated_by=llm_reference
- 支持 --max-consecutive-failures：连续 LLM 失败达到阈值则截断输出、写断点、退出码 4
- 支持 --resume：从断点 next_index 续写
- 每行立即 flush，便于进度与断点恢复
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.llm_client import get_enhancer_llm_client  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            rows.append(json.loads(s))
    return rows


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


def _truncate_jsonl_lines(path: Path, keep_lines: int) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    keep_lines = max(0, min(keep_lines, len(lines)))
    path.write_text(
        "\n".join(lines[:keep_lines]) + ("\n" if keep_lines else ""),
        encoding="utf-8",
    )


def _write_ckpt(
    path: Path,
    *,
    in_path: str,
    out_path: str,
    next_index: int,
    total: int,
    message: str,
    last_errors: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "in": in_path,
                "out": out_path,
                "next_index": next_index,
                "total": total,
                "message": message,
                "last_errors": last_errors[-10:],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_ckpt(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


SYSTEM = (
    "你是技术问答参考答案撰写员。用户会给你一个问题，请写一段可直接作为「标准答案」的简述，"
    "用于 RAG 评测对照（不是评分细则）。\n"
    "要求：使用与问题相同的语言；3～8 句或约 150～400 字；覆盖问题核心要点；表述清晰；\n"
    "不要编造具体论文标题、机构报告编号或网址；若涉及时效可写一般性结论。\n"
    "必须只输出一个 JSON 对象，字段仅允许：gold_answer（字符串）。"
)


async def _gen_one(llm, query: str) -> str:
    human = f"问题：\n{query.strip()}\n"
    resp = await llm.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=human)])
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = " ".join(
            (x.get("text", "") if isinstance(x, dict) else str(x)) for x in content
        )
    obj = _extract_json_obj(str(content)) or {}
    ga = obj.get("gold_answer", "")
    if not isinstance(ga, str):
        ga = str(ga)
    ga = ga.strip()
    if not ga:
        raise ValueError("empty gold_answer")
    return ga


async def _run(args) -> int:
    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out).resolve()
    ckpt_path = Path(args.checkpoint).resolve()

    rows = _read_jsonl(in_path)
    total = len(rows)
    if total == 0:
        print("[llm-gold] 输入为空。", flush=True)
        return 2

    start_index = 0
    if args.resume:
        ck = _read_ckpt(ckpt_path)
        if ck and str(ck.get("in")) == str(in_path) and int(ck.get("total", -1)) == total:
            start_index = max(0, min(int(ck.get("next_index", 0)), total))
            print(f"[llm-gold] 续跑 start_index={start_index}/{total}", flush=True)
        elif ck:
            print("[llm-gold] 警告: 断点与当前输入不一致，仍使用断点 next_index。", flush=True)
            start_index = max(0, min(int(ck.get("next_index", 0)), total))

    if start_index > 0 and out_path.exists():
        _truncate_jsonl_lines(out_path, start_index)
    elif start_index == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    llm = get_enhancer_llm_client()
    max_streak = max(0, int(args.max_consecutive_failures))
    fail_streak = 0
    streak_start: Optional[int] = None
    last_errors: List[str] = []
    t0 = time.perf_counter()

    mode = "a" if start_index > 0 else "w"
    with out_path.open(mode, encoding="utf-8", newline="\n") as f:
        for i in range(start_index, total):
            row = dict(rows[i])
            qid = str(row.get("id", "")).strip()
            qtext = str(row.get("query", "")).strip()
            if not qtext:
                row["gold_answer"] = row.get("gold_answer") or ""
                row["gold_generated_by"] = "llm_reference_skip_empty_query"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                continue

            try:
                ga = await _gen_one(llm, qtext)
                row["gold_answer"] = ga
                row["gold_generated_by"] = "llm_reference"
                fail_streak = 0
                streak_start = None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                last_errors.append(err)
                print(f"[llm-gold] LLM 失败 i={i} id={qid!r} {err}", flush=True)
                fail_streak += 1
                if streak_start is None:
                    streak_start = i
                if max_streak > 0 and fail_streak >= max_streak:
                    f.flush()
                    _truncate_jsonl_lines(out_path, streak_start if streak_start is not None else i)
                    _write_ckpt(
                        ckpt_path,
                        in_path=str(in_path),
                        out_path=str(out_path),
                        next_index=streak_start if streak_start is not None else i,
                        total=total,
                        message="consecutive_llm_failures",
                        last_errors=last_errors,
                    )
                    print(
                        f"[llm-gold] ABORT 连续失败 {fail_streak} 次，断点 next_index="
                        f"{streak_start if streak_start is not None else i}，退出码 4",
                        flush=True,
                    )
                    return 4
                row["gold_answer"] = str(row.get("gold_answer") or "").strip()
                row["gold_generated_by"] = "llm_reference_failed_keep_previous"

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            done = i - start_index + 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                elapsed = max(1e-6, time.perf_counter() - t0)
                rate = done / elapsed
                remain = total - i - 1
                eta = remain / rate if rate > 0 else 0.0
                pct = 100.0 * (i + 1) / total
                print(
                    f"[llm-gold] {i + 1}/{total} ({pct:.1f}%) id={qid!r} | "
                    f"~{rate:.3f} q/s | ETA ~{eta / 60:.1f} min | fail_streak={fail_streak}",
                    flush=True,
                )

            _write_ckpt(
                ckpt_path,
                in_path=str(in_path),
                out_path=str(out_path),
                next_index=i + 1,
                total=total,
                message="progress",
                last_errors=last_errors,
            )

    print(f"[llm-gold] 完成 {total} 条 -> {out_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="输入 queries JSONL")
    ap.add_argument("--out", required=True, help="输出 JSONL（每行一条，含新 gold_answer）")
    ap.add_argument(
        "--checkpoint",
        default="./data/annotation/llm_gold_answer.checkpoint.json",
        help="断点 JSON",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--progress-every", type=int, default=1)
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

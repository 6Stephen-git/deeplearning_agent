"""
一键串联：
0) （可选）LLM 为每条 query 生成 gold_answer（不调用 refresh_eval_ids / baseline 写金标）
1) LLM 自动标注（可用 --queries-merge 将当前 queries 的 gold_answer 并入 pool）
2) （可选）分层抽检样本生成
3) 回写 gold queries
4) （可选）执行 eval_report（默认仅输出单个 JSON）
"""

from __future__ import annotations

import argparse
import os                                                                                                                       
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str], env: dict | None = None) -> None:
    print(f"[pipeline] run: {' '.join(cmd)}", flush=True)
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise SystemExit(ret.returncode)


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def _cleanup_reports_keep_one(report_dir: Path, keep_json: Path) -> None:
    keep_json = keep_json.resolve()
    if not report_dir.exists():
        return
    for p in report_dir.glob("*"):
        if not p.is_file():
            continue
        if p.resolve() == keep_json:
            continue
        if p.suffix.lower() in {".json", ".md", ".txt"}:
            _safe_unlink(p)


def _cleanup_optional_files(paths: Iterable[Path]) -> None:
    for p in paths:
        _safe_unlink(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="./data/rag_eval_ai_gold_pool_fast.jsonl")
    ap.add_argument("--queries", default="./data/rag_eval_ai_queries_80.gold.jsonl")
    ap.add_argument("--labels-out", default="./data/annotation/main_80.llm_labeled.csv")
    ap.add_argument("--review-out", default="./data/annotation/review_sample.csv")
    ap.add_argument("--db-dir", default="./data/rag_eval_ai_db")
    ap.add_argument("--topic", default="rag_eval_ai")
    ap.add_argument("--review-rate", type=float, default=0.15)
    ap.add_argument("--with-review-sample", action="store_true", help="是否生成 review_sample.csv（默认不生成）")
    ap.add_argument("--strict-gold", action="store_true", help="仅 label=2 写入 relevant_ids")
    ap.add_argument("--gold-out", default="", help="默认可留空：宽松/严格金标自动选默认路径")
    ap.add_argument("--max-low-relevant-ratio", type=float, default=0.2)
    ap.add_argument("--min-relevant-per-query", type=int, default=1)
    ap.add_argument("--include-partial", dest="include_partial", action="store_true", help="将 label=1 也纳入 relevant_ids（默认开启）")
    ap.add_argument("--no-include-partial", dest="include_partial", action="store_false", help="仅将 label=2 纳入 relevant_ids")
    ap.add_argument("--repair-with-store", dest="repair_with_store", action="store_true", help="按当前向量库校验并修复 relevant_ids（默认开启）")
    ap.add_argument("--no-repair-with-store", dest="repair_with_store", action="store_false", help="不做向量库一致性修复")
    ap.add_argument("--fill-from-pool", dest="fill_from_pool", action="store_true", help="修复后若某 query 过少，尝试从池内标注候选补齐（默认开启）")
    ap.add_argument("--no-fill-from-pool", dest="fill_from_pool", action="store_false", help="不进行池内补齐")
    ap.add_argument("--per-query-limit", type=int, default=25)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--checkpoint", default="./data/annotation/auto_label.checkpoint.json")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="从断点续跑自动标注")
    ap.add_argument("--run-eval", action="store_true")
    ap.add_argument(
        "--eval-k",
        type=int,
        default=8,
        help="写入 RAG_EVAL_TOP_K；略增大 k 有利于 HyDE/MQE 相对 baseline 的召回（可调）",
    )
    ap.add_argument(
        "--final-report-json",
        default="./reports/rag_eval/final_eval.json",
        help="最终评估 JSON 输出路径（覆盖写入）",
    )
    ap.add_argument("--cleanup-intermediate", action="store_true", help="运行结束后清理中间文件与历史报告")
    ap.add_argument(
        "--llm-gold-answers",
        action="store_true",
        help="先用 LLM 为每条 query 生成参考答案 gold_answer（输出单独 jsonl，再参与标注与回写）",
    )
    ap.add_argument(
        "--llm-gold-out",
        default="",
        help="LLM 参考答案输出路径；默认同目录下 <stem>.llm_gold.jsonl",
    )
    ap.add_argument(
        "--llm-gold-checkpoint",
        default="./data/annotation/llm_gold_answer.checkpoint.json",
        help="生成 gold_answer 的断点文件",
    )
    ap.add_argument("--llm-gold-resume", action="store_true", help="从断点续跑 LLM 生成 gold_answer")
    ap.set_defaults(include_partial=True, repair_with_store=True, fill_from_pool=True)
    args = ap.parse_args()

    py = sys.executable

    effective_queries = args.queries
    if args.llm_gold_answers:
        pq = Path(args.queries)
        llm_out = (args.llm_gold_out or "").strip() or str(pq.parent / f"{pq.stem}.llm_gold.jsonl")
        gen_cmd = [
            py,
            "scripts/generate_gold_answers_with_llm.py",
            "--in",
            args.queries,
            "--out",
            llm_out,
            "--checkpoint",
            args.llm_gold_checkpoint,
            "--max-consecutive-failures",
            str(args.max_consecutive_failures),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.llm_gold_resume:
            gen_cmd.append("--resume")
        _run(gen_cmd)
        effective_queries = llm_out

    queries_path = str(Path(effective_queries).resolve())

    gold_out = (args.gold_out or "").strip() or (
        "./data/rag_eval_ai_queries_80.strict_gold.jsonl"
        if args.strict_gold
        else "./data/rag_eval_ai_queries_80.gold.jsonl"
    )

    label_cmd = [
        py,
        "scripts/auto_label_with_llm.py",
        "--pool",
        args.pool,
        "--queries-merge",
        queries_path,
        "--out",
        args.labels_out,
        "--db-dir",
        args.db_dir,
        "--per-query-limit",
        str(args.per_query_limit),
        "--max-consecutive-failures",
        str(args.max_consecutive_failures),
        "--checkpoint",
        args.checkpoint,
        "--progress-every",
        str(args.progress_every),
    ]
    if args.resume:
        label_cmd.append("--resume")
    _run(label_cmd)
    if args.with_review_sample:
        _run(
            [
                py,
                "scripts/sample_llm_labels_for_review.py",
                "--labels",
                args.labels_out,
                "--queries",
                queries_path,
                "--out",
                args.review_out,
                "--rate",
                str(args.review_rate),
            ]
        )
    apply_cmd = [
        py,
        "scripts/apply_annotation_to_queries.py",
        "--queries",
        queries_path,
        "--pool",
        args.pool,
        "--labels",
        args.labels_out,
        "--out",
        gold_out,
        "--min-relevant-per-query",
        str(args.min_relevant_per_query),
        "--max-low-relevant-ratio",
        str(args.max_low_relevant_ratio),
        "--db-dir",
        args.db_dir,
        "--research-topic",
        args.topic,
    ]
    if (not args.strict_gold) and args.include_partial:
        apply_cmd.append("--include-partial")
    if args.repair_with_store:
        apply_cmd.append("--repair-with-store")
    if args.fill_from_pool:
        apply_cmd.append("--fill-from-pool")
    _run(apply_cmd)

    if args.run_eval:
        env = os.environ.copy()
        env["RAG_EVAL_DB_DIR"] = args.db_dir
        env["RAG_EVAL_QUERIES_PATH"] = gold_out
        env["RAG_EVAL_RESEARCH_TOPIC"] = args.topic
        env["RAG_EVAL_MODES"] = "baseline,hyde,mqe"
        env["RAG_EVAL_WRITE_MD"] = "0"
        env["RAG_EVAL_REPORT_JSON_PATH"] = args.final_report_json
        env["RAG_EVAL_TOP_K"] = str(args.eval_k)
        env.pop("RAG_EVAL_USE_RERANK", None)
        _run([py, "-m", "src.evaluator.rag_eval_runner", "eval_report"], env=env)
        print(f"[pipeline] eval_report done. final_json={Path(args.final_report_json).resolve()}", flush=True)

    if args.cleanup_intermediate:
        report_dir = Path(os.getenv("RAG_EVAL_REPORT_DIR", "./reports/rag_eval")).resolve()
        final_json = Path(args.final_report_json).resolve()
        _cleanup_reports_keep_one(report_dir, final_json)
        if not args.with_review_sample:
            _cleanup_optional_files([Path(args.review_out).resolve()])

    print(
        "[pipeline] completed:\n"
        f"  queries_used={queries_path}\n"
        f"  labels={Path(args.labels_out).resolve()}\n"
        f"  gold={Path(gold_out).resolve()}\n"
        + (f"  review={Path(args.review_out).resolve()}\n" if args.with_review_sample else "")
        + (f"  final_json={Path(args.final_report_json).resolve()}\n" if args.run_eval else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


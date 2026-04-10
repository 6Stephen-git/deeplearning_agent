"""
整理面试可讲材料（基于评估报告与稳健统计）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-report", required=True)
    ap.add_argument("--robust-report", required=True)
    ap.add_argument("--out", default="./reports/rag_eval/interview_pack.md")
    args = ap.parse_args()

    eval_r = _read_json(Path(args.eval_report))
    robust = _read_json(Path(args.robust_report))

    overall = eval_r.get("overall_summary", {})
    eff = eval_r.get("effective_summary", {})
    diag = eval_r.get("diag_summary", {})

    lines = ["# 面试可讲稿（RAG增强评估）", ""]
    lines.append("## 1. 我做了什么")
    lines.append("- 把 HyDE/MQE 评估从单一平均分，升级成全量/有效子集/机制体检三层报告。")
    lines.append("- 重建了 80 条 query 的 AI 资讯/论文方向金标流程（LLM 自动标注 + 回写 relevant_ids）。")
    lines.append("- 增加 5 折稳定性与 bootstrap 置信区间，避免偶然样本导致误判。")
    lines.append("")

    lines.append("## 2. 核心量化结果")
    for mode, m in overall.items():
        lines.append(
            f"- overall `{mode}`: P={m.get('precision',0):.3f}, R={m.get('recall',0):.3f}, MRR={m.get('mrr',0):.3f}, nDCG={m.get('ndcg',0):.3f}, n={m.get('count',0)}"
        )
    for mode, m in eff.items():
        lines.append(
            f"- effective `{mode}`: P={m.get('precision',0):.3f}, R={m.get('recall',0):.3f}, MRR={m.get('mrr',0):.3f}, nDCG={m.get('ndcg',0):.3f}, n={m.get('count',0)}"
        )
    lines.append("")

    lines.append("## 3. 增强器健康度")
    for mode, d in diag.items():
        lines.append(
            f"- `{mode}` fallback={d.get('fallback_rate',0):.1%}, timeout={d.get('timeout_rate',0):.1%}, parse_fail={d.get('parse_failure_rate',0):.1%}, enhancement_applied={d.get('enhancement_applied_rate',0):.1%}"
        )
    lines.append("")

    lines.append("## 4. 稳健性（5折+Bootstrap）")
    rb = robust.get("robust", {})
    for mode, m in rb.items():
        mrr5 = m.get("mrr_5fold", {})
        nd5 = m.get("ndcg_5fold", {})
        mrrb = m.get("mrr_bootstrap", {})
        ndb = m.get("ndcg_bootstrap", {})
        lines.append(
            f"- `{mode}` mrr_5fold={mrr5.get('mean',0):.4f}±{mrr5.get('std',0):.4f}, "
            f"ndcg_5fold={nd5.get('mean',0):.4f}±{nd5.get('std',0):.4f}, "
            f"MRR_95CI=[{mrrb.get('ci_low',0):.4f},{mrrb.get('ci_high',0):.4f}], "
            f"nDCG_95CI=[{ndb.get('ci_low',0):.4f},{ndb.get('ci_high',0):.4f}]"
        )
    lines.append("")

    lines.append("## 5. 局限与下一步")
    lines.append("- 金标以 LLM 标注为主，可按需对边界样本做人工抽检或修订。")
    lines.append("- 可继续扩充高质量官方来源，降低单源偏置。")
    lines.append("- 下一步可引入 reranker 与 query 路由策略，继续比较 HyDE/MQE 的适用边界。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[interview-pack] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


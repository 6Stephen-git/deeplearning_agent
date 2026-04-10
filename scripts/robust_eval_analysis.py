"""
基于 eval_report 导出的 json 做稳健性统计：
- 5折均值/标准差
- bootstrap 95% CI（MRR / nDCG）
- Top提升/退化案例导出
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, int(round((len(ys) - 1) * q))))
    return ys[idx]


def _metric(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def _five_fold(rows: List[Dict], key: str, seed: int = 42) -> Dict[str, float]:
    if not rows:
        return {"mean": 0.0, "std": 0.0}
    rng = random.Random(seed)
    arr = rows[:]
    rng.shuffle(arr)
    k = 5
    fold_vals: List[float] = []
    for i in range(k):
        fold = [x for j, x in enumerate(arr) if j % k == i]
        fold_vals.append(_metric(fold, key))
    return {"mean": mean(fold_vals), "std": pstdev(fold_vals)}


def _bootstrap_ci(rows: List[Dict], key: str, n_boot: int, seed: int) -> Dict[str, float]:
    if not rows:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    vals: List[float] = []
    n = len(rows)
    for _ in range(max(10, n_boot)):
        sample = [rows[rng.randrange(0, n)] for _ in range(n)]
        vals.append(_metric(sample, key))
    return {
        "mean": sum(vals) / len(vals),
        "ci_low": _quantile(vals, 0.025),
        "ci_high": _quantile(vals, 0.975),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--out-json", default="./reports/rag_eval/robust_stats.json")
    ap.add_argument("--out-md", default="./reports/rag_eval/robust_stats.md")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = _load(Path(args.report_json))
    rows = payload.get("per_query_rows") or []

    by_mode: Dict[str, List[Dict]] = defaultdict(list)
    by_q: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        mode = r.get("mode")
        qid = r.get("qid")
        if mode:
            by_mode[mode].append(r)
        if mode and qid:
            by_q[qid][mode] = r

    metrics = ["precision", "recall", "mrr", "ndcg"]
    robust: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, rr in by_mode.items():
        robust[mode] = {}
        for m in metrics:
            robust[mode][f"{m}_5fold"] = _five_fold(rr, m, seed=args.seed)
            if m in ("mrr", "ndcg"):
                robust[mode][f"{m}_bootstrap"] = _bootstrap_ci(rr, m, n_boot=args.bootstrap, seed=args.seed)

    # Top案例（按相对 baseline 的 mrr 差值）
    gain_cases: List[Dict] = []
    loss_cases: List[Dict] = []
    for qid, mm in by_q.items():
        if "baseline" not in mm:
            continue
        b = mm["baseline"]
        for mode in ("hyde", "mqe"):
            if mode not in mm:
                continue
            r = mm[mode]
            gain = float(r.get("mrr", 0.0)) - float(b.get("mrr", 0.0))
            case = {
                "qid": qid,
                "query": r.get("query"),
                "mode": mode,
                "gain_mrr": gain,
                "baseline_mrr": float(b.get("mrr", 0.0)),
                "mode_mrr": float(r.get("mrr", 0.0)),
                "fallback": bool(r.get("fallback", False)),
            }
            (gain_cases if gain >= 0 else loss_cases).append(case)

    gain_cases.sort(key=lambda x: x["gain_mrr"], reverse=True)
    loss_cases.sort(key=lambda x: x["gain_mrr"])

    out = {
        "source_report": str(args.report_json),
        "bootstrap_n": args.bootstrap,
        "robust": robust,
        "top_gains": gain_cases[:10],
        "top_losses": loss_cases[:10],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Robust Eval Stats", ""]
    for mode, mdat in robust.items():
        lines.append(f"## {mode}")
        for k, v in mdat.items():
            if "bootstrap" in k:
                lines.append(
                    f"- {k}: mean={v['mean']:.4f}, 95%CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
                )
            else:
                lines.append(f"- {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
        lines.append("")
    lines.append("## Top Gains")
    for c in gain_cases[:10]:
        lines.append(f"- {c['mode']} {c['qid']} gain_mrr={c['gain_mrr']:+.4f}")
    lines.append("")
    lines.append("## Top Losses")
    for c in loss_cases[:10]:
        lines.append(f"- {c['mode']} {c['qid']} gain_mrr={c['gain_mrr']:+.4f}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[robust] json={out_json}\n[robust] md={out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


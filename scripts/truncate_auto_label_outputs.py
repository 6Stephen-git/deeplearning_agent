"""
将 auto_label_with_llm.py 的输出回滚到指定索引，便于在配额恢复后从断点继续。

用法示例：
  python scripts/truncate_auto_label_outputs.py ^
    --csv ./data/annotation/main_80.llm_labeled.csv ^
    --checkpoint ./data/annotation/auto_label.checkpoint.json ^
    --keep 1382

效果：
  - 将 CSV 保留前 keep 条数据行（不含表头）
  - 将 checkpoint 的 next_index 设置为 keep，并保留 total/pool/out
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _truncate_csv_keep_rows(csv_path: Path, keep: int) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        rows = list(r)
    keep = max(0, min(int(keep), len(rows)))
    kept = rows[:keep]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--keep", type=int, required=True, help="保留的数据行数（不含表头），也是 next_index")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    keep = int(args.keep)

    ck = _read_checkpoint(ckpt_path) or {}
    total = int(ck.get("total", 0) or 0)
    pool = str(ck.get("pool", "") or "")
    out = str(ck.get("out", str(csv_path)) or str(csv_path))

    kept = _truncate_csv_keep_rows(csv_path, keep=keep)
    payload = {
        "pool": pool,
        "out": out,
        "next_index": kept,
        "total": total if total > 0 else kept,
        "message": f"手动回滚到 next_index={kept}（用于配额恢复后续跑）",
        "last_errors": [],
    }
    _write_checkpoint(ckpt_path, payload)
    print(f"[truncate] csv_kept={kept} -> {csv_path}", flush=True)
    print(f"[truncate] checkpoint_next_index={kept} -> {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


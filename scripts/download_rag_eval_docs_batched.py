"""
分批下载评估语料（适用于大 URL 清单）。

功能：
- 从 urls 文件按 batch_size 分批切片
- 每批写入临时 urls 清单并调用 download_rag_eval_docs.py
- 输出每批成功/失败统计，便于累计到目标文档数

用法：
  python scripts/download_rag_eval_docs_batched.py \
    --urls ./data/rag_eval_ai_urls_expanded.txt \
    --out ./data/rag_eval_ai_docs \
    --batch-size 120 --max-batches 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _load_urls(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _write_tmp_urls(tmp_path: Path, urls: List[str]) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text("\n".join(urls) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="./data/rag_eval_ai_urls_expanded.txt")
    ap.add_argument("--out", default="./data/rag_eval_ai_docs")
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--start-batch", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    urls_path = Path(args.urls)
    out_dir = Path(args.out)
    urls = _load_urls(urls_path)
    if not urls:
        print(f"[batch-download] empty urls: {urls_path}")
        return 2

    bs = max(1, args.batch_size)
    start = max(1, args.start_batch)
    nb = max(1, args.max_batches)
    total_batches = (len(urls) + bs - 1) // bs
    end = min(total_batches, start + nb - 1)

    print(
        f"[batch-download] urls={len(urls)} batch_size={bs} "
        f"run_batches={start}-{end}/{total_batches}",
        flush=True,
    )

    script_path = Path(__file__).with_name("download_rag_eval_docs.py")
    batch_ok = 0
    batch_fail = 0

    for b in range(start, end + 1):
        lo = (b - 1) * bs
        hi = min(len(urls), lo + bs)
        cur = urls[lo:hi]
        tmp_urls = out_dir / "_tmp" / f"batch_{b:03d}.urls.txt"
        _write_tmp_urls(tmp_urls, cur)
        print(f"\n[batch-download] batch={b} urls={len(cur)}", flush=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--out",
            str(out_dir),
            "--urls",
            str(tmp_urls),
            "--limit",
            str(len(cur)),
            "--sleep",
            str(max(0.0, float(args.sleep))),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            batch_ok += 1
        else:
            batch_fail += 1
        print(f"[batch-download] batch={b} return_code={ret.returncode}", flush=True)

    print(
        f"\n[batch-download] done ok_batches={batch_ok} fail_batches={batch_fail} out={out_dir}",
        flush=True,
    )
    return 0 if batch_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


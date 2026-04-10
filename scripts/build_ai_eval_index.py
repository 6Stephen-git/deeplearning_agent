"""
使用指定语料目录构建评估向量库索引（AI 方向）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.rag_eval_runner import build_rag_eval_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="./data/rag_eval_ai_docs")
    ap.add_argument("--research-topic", default="rag_eval_ai")
    args = ap.parse_args()
    build_rag_eval_index(docs_dir=args.docs_dir, research_topic=args.research_topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


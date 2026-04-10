"""
RAG 检索增强评估 Runner（骨架版）。

本模块只做一件事：在统一的向量库上，提供三种检索配置的可编程入口：
- baseline: 原始查询直接做向量检索
- hyde: 使用 HyDEEnhancer 生成假设答案后检索
- mqe: 使用 MQEEnhancer 生成多变体后检索

后续步骤（构建评估语料、计算指标、接入 LangSmith）会在此基础上逐步扩展。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接运行 `python src/evaluator/rag_eval_runner.py` 时，默认不含项目根，会报 No module named 'src'
# 将仓库根目录加入 sys.path；`python -m src.evaluator.rag_eval_runner` 同样可用。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ensure_project_env_loaded
ensure_project_env_loaded()

import asyncio
from datetime import datetime
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict, Any, Set

from src.memory.memory_store import MemoryStore, MemoryRecord, MemoryType, MemoryPriority
from src.memory.query_enhancer import HyDEEnhancer, MQEEnhancer
from src.memory.file_processor import FileUploadProcessor
from src.tools.llm_client import create_async_llm_client, llm_client

# 评估专用 LLM 客户端（与 HyDE/MQE 一致：长超时、少重试、限制 max_tokens）
_eval_llm_client = None


def _get_eval_llm_client():
    """
    评估时用于 HyDE/MQE 的异步 LLM 客户端。

    HTTP 读超时优先 ENHANCER_QWEN_TIMEOUT（默认 300s）；可用 RAG_EVAL_LLM_HTTP_TIMEOUT_S 单独覆盖（仅评估）。
    注意：外层 asyncio.wait_for 须大于本值，见 _eval_enhance_async_wait_timeout_s。
    """
    global _eval_llm_client
    if _eval_llm_client is not None:
        return _eval_llm_client
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return llm_client
        http_raw = os.getenv("RAG_EVAL_LLM_HTTP_TIMEOUT_S")
        if http_raw is not None and str(http_raw).strip() != "":
            timeout = max(60, int(float(http_raw)))
        else:
            timeout = max(60, int(os.getenv("ENHANCER_QWEN_TIMEOUT", "300")))
        max_retries = int(os.getenv("ENHANCER_QWEN_MAX_RETRIES", "0"))
        max_tokens = int(os.getenv("ENHANCER_MAX_TOKENS", "600"))
        _eval_llm_client = create_async_llm_client(
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
        return _eval_llm_client
    except Exception:
        return llm_client


def _safe_stdout_str(s: str) -> str:
    """
    Windows 控制台常见为 gbk；部分清洗文本含不可编码字符。
    以当前 stdout 编码可打印为准做转义，避免打印中断。
    """
    enc = getattr(getattr(__import__("sys"), "stdout", None), "encoding", None) or "utf-8"
    try:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _eval_enhance_async_wait_timeout_s() -> float:
    """
    HyDe/MQE 外层 asyncio.wait_for 的秒数（必须 **严格大于** HTTP 读超时）。

    若外层过短，会在 DashScope 仍可返回前取消，表现为 HyDE/MQE「永远失败、回退 baseline」。
    默认与 ENHANCER_QWEN_TIMEOUT（默认 300）对齐并留余量；可用 RAG_EVAL_ENHANCE_TIMEOUT_S 覆盖外层。
    """
    http_env = os.getenv("RAG_EVAL_LLM_HTTP_TIMEOUT_S")
    if http_env is not None and str(http_env).strip() != "":
        enh_http = float(http_env)
    else:
        enh_http = float(os.getenv("ENHANCER_QWEN_TIMEOUT", "300"))
    raw = os.getenv("RAG_EVAL_ENHANCE_TIMEOUT_S")
    if raw is not None and str(raw).strip() != "":
        t = float(raw)
    else:
        # 未单独设置外层时：默认明显长于单次 HTTP（排队/首包慢/长生成）
        t = max(480.0, enh_http + 120.0)
    # 外层至少比 HTTP 多等一会儿，避免 HTTP 未返回就被 asyncio 杀掉
    return max(t, enh_http + 30.0)


def _eval_llm_http_timeout_s() -> int:
    """与 _get_eval_llm_client 使用的 HTTP 读超时一致，用于日志打印。"""
    http_raw = os.getenv("RAG_EVAL_LLM_HTTP_TIMEOUT_S")
    if http_raw is not None and str(http_raw).strip() != "":
        return max(60, int(float(http_raw)))
    return max(60, int(os.getenv("ENHANCER_QWEN_TIMEOUT", "300")))


def _eval_enhance_retries() -> int:
    raw = os.getenv("RAG_EVAL_ENHANCE_RETRIES", "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _eval_enhance_backoff_s() -> float:
    raw = os.getenv("RAG_EVAL_ENHANCE_BACKOFF_S", "2").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 2.0


def _is_retryable_enhance_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        k in msg
        for k in (
            "connection error",
            "api connection",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "502",
            "503",
            "504",
            "rate limit",
            "429",
        )
    )


async def _retry_async(fn, *, retries: int, backoff_s: float):
    last = None
    for i in range(retries + 1):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last = e
            if i >= retries or not _is_retryable_enhance_error(e):
                raise
            await asyncio.sleep(backoff_s * (2**i))
    raise last  # type: ignore[misc]


EvalMode = Literal["baseline", "hyde", "mqe"]


def _parse_rag_eval_modes() -> List[EvalMode]:
    """
    环境变量 RAG_EVAL_MODES（逗号分隔）：
    - 默认 baseline,hyde,mqe 全跑（每条 query 含 2 次 LLM，总耗时可较长）
    - 仅测向量层、快速冒烟：RAG_EVAL_MODES=baseline
    """
    raw = os.getenv("RAG_EVAL_MODES", "baseline,hyde,mqe").lower()
    allowed = {"baseline", "hyde", "mqe"}
    parts = [p.strip() for p in raw.split(",") if p.strip() in allowed]
    order: List[EvalMode] = ["baseline", "hyde", "mqe"]
    out: List[EvalMode] = [m for m in order if m in parts]
    return out if out else ["baseline", "hyde", "mqe"]


@dataclass
class RetrievedChunk:
    """统一的检索结果表示，用于后续评估与 LangSmith 追踪。"""

    id: str
    score: float
    content: str
    source_urls: Optional[List[str]]
    metadata: dict


@dataclass
class EnhanceDiag:
    mode: EvalMode
    fallback: bool = False
    fallback_reason: Optional[str] = None
    timeout: bool = False
    parse_failure: bool = False
    enhancement_applied: bool = False
    enhance_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    variant_count: int = 0
    empty_retrieval: bool = False
    error_type: Optional[str] = None


@dataclass
class RetrievalOutput:
    chunks: List[RetrievedChunk]
    diag: EnhanceDiag


def _get_eval_memory_store() -> MemoryStore:
    """
    获取/初始化用于 RAG 评估的 MemoryStore。

    默认使用环境变量 RAG_EVAL_DB_DIR 指定的目录；若未设置，则回退到 ./data/rag_eval_db。
    这样不会干扰你当前主题向量库（research_topic 相关）。
    """
    base_dir = os.getenv("RAG_EVAL_DB_DIR", "./data/rag_eval_db")
    # 注意：MemoryStore 会在该目录下创建/复用 chroma collection
    return MemoryStore(persist_directory=base_dir)


def _rag_eval_top_k() -> int:
    """检索层 P@k / R@k 等的 k；环境变量 RAG_EVAL_TOP_K（默认 10）。"""
    raw = (os.getenv("RAG_EVAL_TOP_K") or "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _eval_mqe_num_variants(fallback: int = 2) -> int:
    raw = os.getenv("RAG_EVAL_MQE_NUM_VARIANTS")
    if raw is None or str(raw).strip() == "":
        return max(1, min(5, fallback))
    try:
        return max(1, min(5, int(str(raw).strip())))
    except ValueError:
        return max(1, min(5, fallback))


def _eval_mqe_per_variant_k(top_k: int, fallback: int = 30) -> int:
    raw = os.getenv("RAG_EVAL_MQE_PER_VARIANT_K")
    if raw is None or str(raw).strip() == "":
        return max(top_k, fallback)
    try:
        return max(top_k, int(str(raw).strip()))
    except ValueError:
        return max(top_k, fallback)


def _eval_mqe_merge_mode() -> str:
    m = (os.getenv("RAG_EVAL_MQE_MERGE") or "rrf").strip().lower()
    return "rrf" if m == "rrf" else "max"


def _eval_mqe_rrf_const() -> int:
    raw = (os.getenv("RAG_EVAL_MQE_RRF_CONST") or "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _eval_hyde_answer_length() -> str:
    v = (os.getenv("RAG_EVAL_HYDE_ANSWER_LENGTH") or "medium").strip().lower()
    if v in ("short", "medium", "detailed"):
        return v
    return "medium"


def _eval_hyde_merge_original() -> bool:
    return os.getenv("RAG_EVAL_HYDE_MERGE_ORIGINAL", "1").lower() in ("1", "true", "yes", "y")


def _eval_hyde_pool_n(top_k: int) -> int:
    mult_raw = (os.getenv("RAG_EVAL_HYDE_POOL_MULT") or "4").strip()
    min_raw = (os.getenv("RAG_EVAL_HYDE_POOL_MIN") or "24").strip()
    try:
        mult = max(1, int(mult_raw))
    except ValueError:
        mult = 4
    try:
        floor = max(top_k, int(min_raw))
    except ValueError:
        floor = max(top_k, 24)
    return max(floor, top_k * mult)


def _eval_hyde_fusion_mode() -> str:
    """max：原句与假设文档取 max 相似度（偏召回）；precision：两路都高才靠前（偏精准）。"""
    v = (os.getenv("RAG_EVAL_HYDE_FUSION") or "precision").strip().lower()
    return v if v in ("max", "precision") else "precision"


def _hyde_precision_fusion_rank(
    res_orig: List[Tuple[MemoryRecord, float]],
    res_hyp: List[Tuple[MemoryRecord, float]],
    top_k: int,
) -> List[Tuple[MemoryRecord, float]]:
    s_o: Dict[str, float] = {}
    s_h: Dict[str, float] = {}
    mem_map: Dict[str, MemoryRecord] = {}
    for mem, sc in res_orig:
        mid = mem.id
        mem_map[mid] = mem
        s_o[mid] = sc
    for mem, sc in res_hyp:
        mid = mem.id
        mem_map[mid] = mem
        prev = s_h.get(mid)
        s_h[mid] = sc if prev is None or sc > prev else prev
    try:
        w_o = float((os.getenv("RAG_EVAL_HYDE_ONLY_ORIG_WEIGHT") or "0.94").strip())
    except ValueError:
        w_o = 0.94
    try:
        w_h = float((os.getenv("RAG_EVAL_HYDE_ONLY_HYP_WEIGHT") or "0.68").strip())
    except ValueError:
        w_h = 0.68
    require_both = os.getenv("RAG_EVAL_HYDE_PRECISION_REQUIRE_BOTH", "0").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    ranked: List[Tuple[str, float, float]] = []
    cand_ids = (set(s_o) & set(s_h)) if require_both else (set(s_o) | set(s_h))
    if require_both and not cand_ids:
        # 若交集为空，退回并集以避免出现“全空 gold pool 导致无法计算”的情况
        cand_ids = set(s_o) | set(s_h)
    for mid in cand_ids:
        o = s_o.get(mid)
        h = s_h.get(mid)
        if o is not None and h is not None:
            fus = min(o, h)
        elif o is not None:
            fus = w_o * o
        else:
            fus = w_h * h
        tie = max(o or 0.0, h or 0.0)
        ranked.append((mid, fus, tie))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    out: List[Tuple[MemoryRecord, float]] = []
    for mid, fus, _ in ranked[:top_k]:
        if mid in mem_map:
            out.append((mem_map[mid], fus))
    return out


def _mqe_rrf_merge(
    ranked_lists: List[List[Tuple[MemoryRecord, float]]],
    rrf_c: int,
    top_k: int,
    overlap_bonus: float = 0.0,
) -> List[Tuple[MemoryRecord, float]]:
    id_to_mem: Dict[str, MemoryRecord] = {}
    id_to_best: Dict[str, float] = {}
    for lst in ranked_lists:
        for mem, score in lst:
            mid = mem.id
            id_to_mem[mid] = mem
            prev = id_to_best.get(mid)
            if prev is None or score > prev:
                id_to_best[mid] = score
    rrf_scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (mem, _) in enumerate(lst):
            mid = mem.id
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (rrf_c + rank + 1)
    if overlap_bonus > 0.0 and ranked_lists:
        list_count: Dict[str, int] = {}
        for lst in ranked_lists:
            seen: Set[str] = set()
            for mem, _ in lst:
                mid = mem.id
                if mid not in seen:
                    seen.add(mid)
                    list_count[mid] = list_count.get(mid, 0) + 1
        for mid in rrf_scores:
            c = list_count.get(mid, 0)
            if c > 1:
                rrf_scores[mid] += overlap_bonus * float(c - 1)
    ordered_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return [(id_to_mem[i], id_to_best[i]) for i in ordered_ids if i in id_to_mem]


def _llm_available_for_enhance() -> bool:
    """是否有可用 LLM API Key，供 HyDE/MQE 执行真实增强（否则回退为 baseline）。"""
    keys = [
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    return any(os.getenv(k) for k in keys) or os.getenv("RAG_EVAL_FORCE_LLM") == "1"


def _hydrate_chunks_from_store(store: MemoryStore, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    兜底补全：从 Chroma collection 重新按 id 拉取 documents/metadatas，
    防止出现 registry 里 content 为空导致 preview 为空的问题。
    """
    if not chunks:
        return chunks

    ids = [c.id for c in chunks]
    try:
        data = store.collection.get(ids=ids, include=["documents", "metadatas"])
    except Exception:
        return chunks

    doc_map: Dict[str, str] = {}
    meta_map: Dict[str, dict] = {}
    if data and data.get("ids"):
        for mid, doc, meta in zip(data["ids"], data.get("documents", []), data.get("metadatas", [])):
            if isinstance(mid, str):
                if isinstance(doc, str):
                    doc_map[mid] = doc
                if isinstance(meta, dict):
                    meta_map[mid] = meta

    hydrated: List[RetrievedChunk] = []
    for c in chunks:
        meta = meta_map.get(c.id, c.metadata or {})
        content = c.content or doc_map.get(c.id, "")
        source_urls = c.source_urls or meta.get("source_urls")
        # 兼容历史脏数据：SourceURL 可能被写成 "https:///domain/path"（多一个 /）
        if isinstance(source_urls, list):
            fixed = []
            for u in source_urls:
                if isinstance(u, str) and u.startswith("https:///"):
                    fixed.append("https://" + u[len("https:///"):])
                else:
                    fixed.append(u)
            source_urls = fixed
        hydrated.append(
            RetrievedChunk(
                id=c.id,
                score=c.score,
                content=content,
                source_urls=source_urls,
                metadata=meta,
            )
        )
    return hydrated


# =========================
# 索引构建与查询加载
# =========================

def build_rag_eval_index(
    docs_dir: str = "./data/rag_eval_docs",
    research_topic: str = "rag_eval",
) -> None:
    """
    构建 RAG 评估用向量索引。

    - 默认从 ./data/rag_eval_docs 读取所有文件（可包含子目录）
    - 使用现有 FileUploadProcessor + MemoryStore 进行切分和嵌入
    - 统一使用 research_topic="rag_eval"，方便后续按主题过滤

    用法（命令行）：
      python -m src.evaluator.rag_eval_runner build_index
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"评估文档目录不存在: {docs_path}")

    # 0) 默认清空重建（避免中断写入导致 Chroma 索引/元数据不一致，出现 Error finding id）
    # 可通过 RAG_EVAL_RESET_DB=false 禁用
    reset_db = os.getenv("RAG_EVAL_RESET_DB", "true").lower() in ("1", "true", "yes", "y")
    db_dir = Path(os.getenv("RAG_EVAL_DB_DIR", "./data/rag_eval_db"))
    if reset_db and db_dir.exists():
        print(f"[RAG-EVAL] 重建索引：清空现有向量库目录 {db_dir}")
        try:
            shutil.rmtree(db_dir)
        except Exception as e:
            raise RuntimeError(f"无法清空向量库目录 {db_dir}: {e}") from e

    # 1) 将 HTML 清洗为纯文本（FileUploadProcessor 不支持 .html）
    clean_dir = docs_path / "_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted([p for p in docs_path.glob("*.html") if p.is_file()])
    if not html_files:
        print(f"[RAG-EVAL] 未发现 .html 文档，跳过清洗阶段。目录: {docs_path}")
    else:
        print(f"[RAG-EVAL] 发现 {len(html_files)} 个 .html 文档，开始清洗到: {clean_dir}")
        for hp in html_files:
            # 同名 url 记录文件（下载脚本生成），用于溯源
            url_hint = ""
            url_file = hp.with_suffix(hp.suffix + ".url.txt")  # xxx.html.url.txt
            # 兼容旧命名：xxx.url.txt
            legacy_url_file = hp.with_suffix(".url.txt")
            if url_file.exists():
                url_hint = url_file.read_text(encoding="utf-8", errors="ignore").strip()
            elif legacy_url_file.exists():
                url_hint = legacy_url_file.read_text(encoding="utf-8", errors="ignore").strip()

            raw = hp.read_text(encoding="utf-8", errors="ignore")
            text = _html_to_text(raw)
            text = _normalize_text(text)
            text = _strip_rag_eval_page_noise(text)

            if len(text) < 200:
                # 太短基本是空页面/脚本页，跳过
                continue

            out_name = hp.stem + ".txt"
            out_path = clean_dir / out_name
            header = ""
            if url_hint:
                header = f"SourceURL: {url_hint}\n\n"
            out_path.write_text(header + text, encoding="utf-8")

        print("[RAG-EVAL] HTML 清洗完成。")

    # 2) 只对 clean_dir 建索引（避免把 *.url.txt 元数据入库）
    # 2.1) 确保每篇 clean 文本都带 SourceURL 头（用于 URL 级评测口径与可追溯性）
    # 说明：历史上 _clean 目录可能已存在来自下载脚本的 txt（未带 SourceURL），
    # 此时评测会退化为 chunk_id 口径（url=0），系统性低估“命中同文档不同 chunk”的增强策略。
    def _ensure_source_url_header(p: Path) -> None:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return
        text = raw.lstrip()
        if text.startswith("SourceURL:"):
            return
        # 1) 优先从正文中抽取第一个 http(s) 链接（很多清洗文本会保留参考链接）
        m = re.search(r"https?://\\S+", raw)
        url = ""
        if m:
            url = m.group(0).strip().strip('\"').strip("'").rstrip(").,;]")
        # 2) 若正文中无链接，则尝试由文件名反推（格式：NN__domain__path__hash.txt）
        if not url:
            stem = p.stem
            parts = stem.split("__")
            if len(parts) >= 3:
                domain = parts[1].strip()
                path_part = "__".join(parts[2:-1]).strip()
                if domain:
                    # 清洗文件名中用 "__" 代替路径分隔；这里只做近似还原，足以用于 URL 级聚合
                    path_guess = path_part.replace("__", "/").lstrip("/")
                    url = f"https://{domain}/{path_guess}"
        if not url:
            return
        header = f"SourceURL: {url}\n\n"
        try:
            p.write_text(header + raw, encoding="utf-8")
        except Exception:
            return

    for tp in sorted([p for p in clean_dir.glob("*.txt") if p.is_file()]):
        _ensure_source_url_header(tp)

    store = _get_eval_memory_store()
    # 评估索引必须使用真实嵌入；哈希嵌入会让检索评估失真
    allow_hash = os.getenv("RAG_EVAL_ALLOW_HASH_EMBEDDING", "false").lower() in ("1", "true", "yes", "y")
    embedder = getattr(store, "embedder", None)
    embedder_provider = getattr(embedder, "provider", None)
    if (embedder is None) or (embedder_provider == "hash" and not allow_hash):
        raise RuntimeError(
            "评估索引构建已中止：当前嵌入为 hash 回退（或嵌入工具未初始化）。\n"
            "- 请先在环境变量/`.env` 中配置 `DASHSCOPE_API_KEY`（推荐）并确保可联网。\n"
            "- 若你明确要用 hash（不建议），可设置 `RAG_EVAL_ALLOW_HASH_EMBEDDING=true` 继续。"
        )
    print(f"[RAG-EVAL] 嵌入提供商: {embedder_provider}  模型: {getattr(embedder, 'model_name', 'N/A')}")

    # 开启严格嵌入：DashScope 调用失败直接抛错，避免静默回退 hash
    if not allow_hash:
        os.environ["EMBEDDING_STRICT"] = "true"
        # 轻量自检：确保嵌入接口可用，否则立刻报出真实错误
        try:
            _ = embedder.generate_embedding("embedding preflight: rag-eval")
        except Exception as e:
            raise RuntimeError(
                "嵌入预检失败：DashScope 嵌入接口调用异常。\n"
                "这会导致系统悄悄回退到哈希嵌入，从而让检索评估失真。\n"
                "请检查：网络/代理、DashScope 服务可用性、`DASHSCOPE_API_KEY` 权限、`DASHSCOPE_EMBEDDING_MODEL` 是否正确。\n"
                f"原始错误: {type(e).__name__}: {str(e)[:300]}"
            ) from e
    processor = FileUploadProcessor(memory_store=store)

    print(f"[RAG-EVAL] 开始构建评估索引。输入目录: {clean_dir}")
    processor.process_directory(
        directory_path=clean_dir,
        research_topic=research_topic,
        recursive=True,
    )
    print("[RAG-EVAL] 评估索引构建完成。")


def _html_to_text(html: str) -> str:
    """
    将 HTML 粗略转换为纯文本（无外部依赖）。
    - 去除 script/style
    - 尽量剥掉导航/页脚/侧栏（减少「复制页面」等 UI 文案进入 chunk）
    - 去除标签
    - 保留可读文本
    """
    # 移除 script/style 内容
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<noscript[\s\S]*?</noscript>", " ", html, flags=re.IGNORECASE)
    # 整块去掉常见「壳」区域（文档站导航/操作条；粗粒度，减少噪声）
    for tag in ("nav", "header", "footer", "aside", "form"):
        html = re.sub(rf"<{tag}[\s\S]*?</{tag}>", "\n", html, flags=re.IGNORECASE)
    # 常见换行标签替换为换行
    html = re.sub(r"</(p|div|h\d|li|br|tr|section|article)>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # 去除所有剩余标签
    text = re.sub(r"<[^>]+>", " ", html)
    return text


def _normalize_text(text: str) -> str:
    # 合并多余空白
    text = re.sub(r"[ \t]+", " ", text)
    # 合并过多空行
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _strip_rag_eval_page_noise(text: str) -> str:
    """
    去掉技术文档站里常见的「非正文」残留（HTML 去标签后仍易残留）。

    说明：无法 100% 去噪（站点各异）；重建索引后 chunk 会干净很多。
    """
    if not text:
        return text
    # 常见英文/中文 UI 文案（LlamaIndex / 文档站「复制页面」等）
    noise_phrases = (
        "Copy page",
        "More page actions",
        "More page action",
        "复制页面",
        "更多页面操作",
        "View on GitHub",
        "Download raw",
        "Jump to section",
        "Copy Page",
        "Hide navigation",
        "Table of Contents",
    )
    t = text
    for p in noise_phrases:
        t = re.sub(re.escape(p), " ", t, flags=re.IGNORECASE if p.isascii() else 0)
    # 去标签后残留的类名/属性碎片，如 *+*]:mt-3">
    t = re.sub(r"\*+\]\s*:[^>\n]{0,120}>", " ", t)
    # 少量 HTML 实体
    t = t.replace("&#x27;", "'").replace("&quot;", '"').replace("&nbsp;", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()


@dataclass
class EvalQuery:
    id: str
    query: str
    relevant_ids: List[str]
    gold_answer: Optional[str] = None
    # 金标来源标记（便于提示评测口径偏置/可信度）
    gold_generated_by: Optional[str] = None


def _resolve_eval_queries_path(path: Optional[str] = None) -> Path:
    """
    解析 rag_eval_queries.jsonl 的路径。
    相对路径一律相对 **项目根**（本文件上两级），避免在其它工作目录运行
    运行评测/标注相关命令时读到错误文件或写错位置（表现为「完全没有改变」）。
    """
    if path:
        p = Path(path)
        return p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()
    env = os.getenv("RAG_EVAL_QUERIES_PATH")
    if env:
        pe = Path(env)
        return pe.resolve() if pe.is_absolute() else (_PROJECT_ROOT / pe).resolve()
    return (_PROJECT_ROOT / "data" / "rag_eval_queries.jsonl").resolve()


def _resolve_eval_research_topic(
    research_topic: Optional[str],
    queries_path: Optional[str] = None,
) -> str:
    """
    统一解析评估用 topic，避免默认 rag_eval 与 AI 评估库错配导致空检索。
    优先级：
    1) 显式参数 research_topic（且非空）
    2) 环境变量 RAG_EVAL_RESEARCH_TOPIC
    3) 由 queries 路径名推断（包含 rag_eval_ai -> rag_eval_ai）
    4) 回退 rag_eval
    """
    if (research_topic or "").strip():
        return str(research_topic).strip()
    env_topic = os.getenv("RAG_EVAL_RESEARCH_TOPIC", "").strip()
    if env_topic:
        return env_topic
    qp = str(_resolve_eval_queries_path(queries_path)).lower()
    if "rag_eval_ai" in qp or "ai_queries" in qp:
        return "rag_eval_ai"
    return "rag_eval"


def _count_chunks_by_topic(store: MemoryStore, research_topic: str) -> int:
    """统计指定 topic 在当前向量库中的 chunk 数量。"""
    if not (research_topic or "").strip():
        return int(store.collection.count())
    try:
        data = store.collection.get(
            where={"research_topic": {"$eq": research_topic}},
            include=[],
        )
        return len((data or {}).get("ids") or [])
    except Exception:
        return 0


def load_eval_queries(path: Optional[str] = None) -> List[EvalQuery]:
    """
    从 JSONL 文件加载评估 queries。
    格式（每行一个 JSON 对象）：
      {
        "id": "q1",
        "query": "...",
        "relevant_ids": ["chunk_id1", "chunk_id2"],
        "gold_answer": "（可选）参考答案"
      }
    """
    p = _resolve_eval_queries_path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估 query 文件不存在: {p}")

    queries: List[EvalQuery] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(
                EvalQuery(
                    id=data["id"],
                    query=data["query"],
                    relevant_ids=data.get("relevant_ids", []),
                    gold_answer=data.get("gold_answer"),
                    gold_generated_by=data.get("gold_generated_by"),
                )
            )
    print(f"[RAG-EVAL] 已加载评估 queries: {len(queries)} 条。")
    print(f"[RAG-EVAL] 文件路径: {p}", flush=True)
    # 可选：限制评估条数（用于快速调参冒烟）。默认不限制。
    raw_max = os.getenv("RAG_EVAL_MAX_QUERIES", "").strip()
    if raw_max:
        try:
            n = int(raw_max)
        except ValueError:
            n = 0
        if n > 0 and len(queries) > n:
            queries = queries[:n]
            print(f"[RAG-EVAL] 已启用 RAG_EVAL_MAX_QUERIES={n}，仅评估前 {len(queries)} 条。", flush=True)
    return queries


async def refresh_eval_queries_relevant_ids_union(
    path: Optional[str] = None,
    per_mode_k: int = 5,
    research_topic: Optional[str] = "rag_eval",
) -> None:
    """
    用 baseline/HyDE/MQE 三路候选池的并集写回 relevant_ids，减少“金标完全偏袒 baseline”的系统性偏置。

    口径：
    - 对每条 query，分别取 baseline/HyDE/MQE 的 Top-per_mode_k chunk_id，三路去重合并为 relevant_ids。
    - 该口径仍非“真实人工金标”，但比 baseline_topk 更利于观察“难例补召回/排序变化”。

    注意：
    - 该方法需要 LLM（HyDE/MQE），速度会明显慢于纯向量检索。
    - 若未配置 LLM key，HyDE/MQE 会回退 baseline，合并后仍会退化为 baseline 候选。
    """
    p = _resolve_eval_queries_path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估 query 文件不存在: {p}")

    queries = load_eval_queries(str(p))
    backup = p.with_name(p.name + ".bak")
    shutil.copy2(p, backup)
    print(f"[RAG-EVAL] 已备份原文件 -> {backup}", flush=True)

    out_lines: List[str] = []
    for idx, q in enumerate(queries, 1):
        print(f"[RAG-EVAL] refresh_union {idx}/{len(queries)}  id={q.id!r} …", flush=True)

        # 三路并行取候选
        coros = [
            retrieve_baseline(query=q.query, k=per_mode_k, research_topic=research_topic),
            retrieve_hyde(query=q.query, k=per_mode_k, research_topic=research_topic),
            retrieve_mqe(query=q.query, k=per_mode_k, research_topic=research_topic),
        ]
        base_chunks, hyde_chunks, mqe_chunks = await asyncio.gather(*coros)
        pool: List[str] = []
        for cid in [c.id for c in (base_chunks + hyde_chunks + mqe_chunks)]:
            if cid and cid not in pool:
                pool.append(cid)

        if not pool:
            print(f"[RAG-EVAL] 警告: {q.id} 三路检索候选池为空，请检查向量库与 research_topic。", flush=True)

        row: Dict[str, Any] = {
            "id": q.id,
            "query": q.query,
            "relevant_ids": pool,
        }
        if q.gold_answer is not None:
            row["gold_answer"] = q.gold_answer
        row["gold_generated_by"] = f"union_topk_{per_mode_k}"
        out_lines.append(json.dumps(row, ensure_ascii=False) + "\n")

    p.write_text("".join(out_lines), encoding="utf-8")
    print(
        f"\n[RAG-EVAL] 已写回 {p}（每条 relevant_ids 为 baseline/HyDE/MQE 各 Top-{per_mode_k} 的去重并集）。"
        " 该口径仍建议人工抽查与修订，但更适合做增强策略的难例分析。",
        flush=True,
    )


async def _retrieve_baseline_output(
    query: str,
    k: int = 5,
    research_topic: Optional[str] = None,
) -> RetrievalOutput:
    """
    基线检索：不做任何查询增强，直接在评估向量库中做向量检索。
    """
    t0 = asyncio.get_running_loop().time()
    store = _get_eval_memory_store()
    results: List[Tuple[MemoryRecord, float]] = store.search_by_similarity(
        query=query,
        n_results=k,
        research_topic=research_topic,
    )

    chunks: List[RetrievedChunk] = []
    for mem, score in results:
        meta = mem.metadata or {}
        chunks.append(
            RetrievedChunk(
                id=mem.id,
                score=score,
                content=mem.content,
                source_urls=meta.get("source_urls"),
                metadata=meta,
            )
        )
    hydrated = _hydrate_chunks_from_store(store, chunks)
    t1 = asyncio.get_running_loop().time()
    return RetrievalOutput(
        chunks=hydrated,
        diag=EnhanceDiag(
            mode="baseline",
            enhancement_applied=True,
            retrieval_latency_ms=max(0.0, (t1 - t0) * 1000.0),
            empty_retrieval=(len(hydrated) == 0),
        ),
    )


async def retrieve_baseline(
    query: str,
    k: int = 5,
    research_topic: Optional[str] = None,
) -> List[RetrievedChunk]:
    return (await _retrieve_baseline_output(query=query, k=k, research_topic=research_topic)).chunks


async def _retrieve_hyde_output(
    query: str,
    k: int = 5,
    research_topic: Optional[str] = None,
    answer_length: str = "medium",
) -> RetrievalOutput:
    """
    HyDE 增强检索：
    - 先用 HyDEEnhancer 生成假设答案
    - 再用假设答案作为查询文本进行向量检索
    """
    loop = asyncio.get_running_loop()
    store = _get_eval_memory_store()
    diag = EnhanceDiag(mode="hyde")
    env_al = os.getenv("RAG_EVAL_HYDE_ANSWER_LENGTH", "").strip().lower()
    eff_answer_len = env_al if env_al in ("short", "medium", "detailed") else answer_length
    if eff_answer_len not in ("short", "medium", "detailed"):
        eff_answer_len = _eval_hyde_answer_length()
    enhancer = HyDEEnhancer(llm_client=_get_eval_llm_client(), answer_length=eff_answer_len)

    # HyDE 依赖 LLM；若环境未配置或调用超时，则回退为 baseline
    if not _llm_available_for_enhance():
        out = await _retrieve_baseline_output(query=query, k=k, research_topic=research_topic)
        out.diag.mode = "hyde"
        out.diag.fallback = True
        out.diag.fallback_reason = "missing_llm_key"
        out.diag.error_type = "NoLLMKey"
        return out
    try:
        timeout_s = _eval_enhance_async_wait_timeout_s()
        t0 = loop.time()
        async def _do():
            return await asyncio.wait_for(
                enhancer.enhance(
                    original_query=query,
                    context={"research_topic": research_topic} if research_topic else None,
                ),
                timeout=timeout_s,
            )

        hypothetical = await _retry_async(
            _do,
            retries=_eval_enhance_retries(),
            backoff_s=_eval_enhance_backoff_s(),
        )
        t1 = loop.time()
        diag.enhance_latency_ms = max(0.0, (t1 - t0) * 1000.0)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        msg = str(e)[:120]
        print(f"[RAG-EVAL] HyDE 增强失败（已回退为 baseline）: {type(e).__name__}: {msg}")
        if "timeout" in msg.lower() or isinstance(e, asyncio.TimeoutError):
            print(
                f"[RAG-EVAL] 提示: 外层 asyncio 等待上限为 {timeout_s:.0f}s，"
                "须 > ENHANCER_QWEN_TIMEOUT（默认 300）。仍慢可增大 RAG_EVAL_ENHANCE_TIMEOUT_S / ENHANCER_QWEN_TIMEOUT / RAG_EVAL_LLM_HTTP_TIMEOUT_S。"
            )
        out = await _retrieve_baseline_output(query=query, k=k, research_topic=research_topic)
        out.diag.mode = "hyde"
        out.diag.fallback = True
        out.diag.fallback_reason = "enhance_exception"
        out.diag.timeout = ("timeout" in msg.lower() or isinstance(e, asyncio.TimeoutError))
        out.diag.error_type = type(e).__name__
        return out

    if (hypothetical or "").strip() == (query or "").strip():
        if not getattr(retrieve_hyde, "_warned_same_query", False):
            print("[RAG-EVAL] HyDE 返回了原始 query，可能增强在内部未生效（如 API 异常被 enhancer 吞掉）。")
            setattr(retrieve_hyde, "_warned_same_query", True)
        diag.fallback = True
        diag.fallback_reason = "same_as_original"
        diag.enhancement_applied = False
    else:
        diag.enhancement_applied = True

    rt0 = loop.time()
    if _eval_hyde_merge_original():
        n_pool = _eval_hyde_pool_n(k)
        res_orig = store.search_by_similarity(
            query=query,
            n_results=n_pool,
            research_topic=research_topic,
        )
        res_hyp = store.search_by_similarity(
            query=hypothetical,
            n_results=n_pool,
            research_topic=research_topic,
        )
        if _eval_hyde_fusion_mode() == "precision":
            results = _hyde_precision_fusion_rank(res_orig, res_hyp, k)
        else:
            merged_h: Dict[str, Tuple[MemoryRecord, float]] = {}

            def _hyde_merge(res: List[Tuple[MemoryRecord, float]]) -> None:
                for mem, score in res:
                    if mem.id not in merged_h or score > merged_h[mem.id][1]:
                        merged_h[mem.id] = (mem, score)

            _hyde_merge(res_orig)
            _hyde_merge(res_hyp)
            results = sorted(merged_h.values(), key=lambda x: x[1], reverse=True)[:k]
    else:
        results = store.search_by_similarity(
            query=hypothetical,
            n_results=k,
            research_topic=research_topic,
        )
    rt1 = loop.time()

    chunks: List[RetrievedChunk] = []
    for mem, score in results:
        meta = mem.metadata or {}
        chunks.append(
            RetrievedChunk(
                id=mem.id,
                score=score,
                content=mem.content,
                source_urls=meta.get("source_urls"),
                metadata=meta,
            )
        )
    hydrated = _hydrate_chunks_from_store(store, chunks)
    diag.retrieval_latency_ms = max(0.0, (rt1 - rt0) * 1000.0)
    diag.empty_retrieval = (len(hydrated) == 0)
    return RetrievalOutput(chunks=hydrated, diag=diag)


async def retrieve_hyde(
    query: str,
    k: int = 5,
    research_topic: Optional[str] = None,
    answer_length: str = "medium",
) -> List[RetrievedChunk]:
    return (await _retrieve_hyde_output(query=query, k=k, research_topic=research_topic, answer_length=answer_length)).chunks


async def _retrieve_mqe_output(
    query: str,
    k: int = 5,
    per_variant_k: int = 5,
    num_variants: int = 3,
    research_topic: Optional[str] = None,
) -> RetrievalOutput:
    """
    MQE 增强检索：
    - 使用 MQEEnhancer 生成多个查询变体
    - 对每个变体各自做向量检索（per_variant_k）
    - 合并：默认 RAG_EVAL_MQE_MERGE=rrf（倒数排名融合），可选 max（各次检索同一 chunk 取最高相似度后排序）

    原句检索始终参与合并；变体数与每路召回深度可由 RAG_EVAL_MQE_NUM_VARIANTS、RAG_EVAL_MQE_PER_VARIANT_K 调节。
    """
    loop = asyncio.get_running_loop()
    store = _get_eval_memory_store()
    diag = EnhanceDiag(mode="mqe")
    num_variants = _eval_mqe_num_variants(num_variants)
    per_variant_k = _eval_mqe_per_variant_k(k, per_variant_k)
    enhancer = MQEEnhancer(llm_client=_get_eval_llm_client(), num_variants=num_variants)

    # MQE 依赖 LLM；若环境未配置或调用超时，则回退为 baseline
    if not _llm_available_for_enhance():
        out = await _retrieve_baseline_output(query=query, k=k, research_topic=research_topic)
        out.diag.mode = "mqe"
        out.diag.fallback = True
        out.diag.fallback_reason = "missing_llm_key"
        out.diag.error_type = "NoLLMKey"
        return out
    try:
        timeout_s = _eval_enhance_async_wait_timeout_s()
        t0 = loop.time()
        async def _do():
            return await asyncio.wait_for(
                enhancer.enhance(
                    original_query=query,
                    context={"research_topic": research_topic} if research_topic else None,
                ),
                timeout=timeout_s,
            )

        variants = await _retry_async(
            _do,
            retries=_eval_enhance_retries(),
            backoff_s=_eval_enhance_backoff_s(),
        )
        t1 = loop.time()
        diag.enhance_latency_ms = max(0.0, (t1 - t0) * 1000.0)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        msg = str(e)[:120]
        print(f"[RAG-EVAL] MQE 增强失败（已回退为 baseline）: {type(e).__name__}: {msg}")
        if "timeout" in msg.lower() or isinstance(e, asyncio.TimeoutError):
            print(
                f"[RAG-EVAL] 提示: 外层 asyncio 等待上限为 {timeout_s:.0f}s，"
                "须 > ENHANCER_QWEN_TIMEOUT（默认 300）。仍慢可增大 RAG_EVAL_ENHANCE_TIMEOUT_S / ENHANCER_QWEN_TIMEOUT / RAG_EVAL_LLM_HTTP_TIMEOUT_S。"
            )
        out = await _retrieve_baseline_output(query=query, k=k, research_topic=research_topic)
        out.diag.mode = "mqe"
        out.diag.fallback = True
        out.diag.fallback_reason = "enhance_exception"
        out.diag.timeout = ("timeout" in msg.lower() or isinstance(e, asyncio.TimeoutError))
        out.diag.error_type = type(e).__name__
        return out
    # 安全回退：如果增强失败，MQE 返回 [original_query]
    if not variants:
        variants = [query]
        diag.fallback = True
        diag.fallback_reason = "empty_variants"
        diag.parse_failure = True
    if len(variants) == 1 and (variants[0] or "").strip() == (query or "").strip():
        if not getattr(retrieve_mqe, "_warned_same_variants", False):
            print("[RAG-EVAL] MQE 仅返回原始 query，可能增强在内部未生效（如 API 异常被 enhancer 吞掉）。")
            setattr(retrieve_mqe, "_warned_same_variants", True)
        diag.fallback = True
        diag.fallback_reason = diag.fallback_reason or "same_as_original"
    diag.variant_count = len(variants or [])
    diag.enhancement_applied = not diag.fallback and diag.variant_count > 1

    rt0 = loop.time()
    q_norm = (query or "").strip()
    variant_norms = {(v or "").strip() for v in variants if v and str(v).strip()}
    try:
        orig_floor = max(k, int((os.getenv("RAG_EVAL_MQE_ORIG_POOL_MIN") or "48").strip()))
    except ValueError:
        orig_floor = max(k, 48)
    orig_n = max(per_variant_k * 2, k * 3, orig_floor)
    orig_res = store.search_by_similarity(
        query=query,
        n_results=orig_n,
        research_topic=research_topic,
    )
    merge_mode = _eval_mqe_merge_mode()
    try:
        ov_bonus = float((os.getenv("RAG_EVAL_MQE_RRF_OVERLAP_BONUS") or "0.14").strip())
    except ValueError:
        ov_bonus = 0.14
    ov_bonus = max(0.0, ov_bonus)

    if merge_mode == "rrf":
        ranked_lists: List[List[Tuple[MemoryRecord, float]]] = [orig_res]
        seen_q = {q_norm} if q_norm else set()
        for v in variants:
            vn = (v or "").strip()
            if not vn or vn in seen_q:
                continue
            seen_q.add(vn)
            ranked_lists.append(
                store.search_by_similarity(
                    query=v,
                    n_results=per_variant_k,
                    research_topic=research_topic,
                )
            )
        merged = _mqe_rrf_merge(
            ranked_lists, _eval_mqe_rrf_const(), k, overlap_bonus=ov_bonus
        )
    else:
        # max 合并：原句 + 各变体；同一 chunk 取各次检索中的最高相似度
        all_results: Dict[str, Tuple[MemoryRecord, float]] = {}

        def _merge_res(res: List[Tuple[MemoryRecord, float]]) -> None:
            for mem, score in res:
                if mem.id not in all_results or score > all_results[mem.id][1]:
                    all_results[mem.id] = (mem, score)

        _merge_res(orig_res)
        if q_norm and q_norm not in variant_norms:
            pass  # 已在 orig_res 覆盖
        for v in variants:
            res = store.search_by_similarity(
                query=v,
                n_results=per_variant_k,
                research_topic=research_topic,
            )
            _merge_res(res)
        merged = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:k]
    rt1 = loop.time()

    chunks: List[RetrievedChunk] = []
    for mem, score in merged:
        meta = mem.metadata or {}
        chunks.append(
            RetrievedChunk(
                id=mem.id,
                score=score,
                content=mem.content,
                source_urls=meta.get("source_urls"),
                metadata=meta,
            )
        )
    hydrated = _hydrate_chunks_from_store(store, chunks)
    diag.retrieval_latency_ms = max(0.0, (rt1 - rt0) * 1000.0)
    diag.empty_retrieval = (len(hydrated) == 0)
    return RetrievalOutput(chunks=hydrated, diag=diag)


async def retrieve_mqe(
    query: str,
    k: int = 5,
    per_variant_k: int = 5,
    num_variants: int = 3,
    research_topic: Optional[str] = None,
) -> List[RetrievedChunk]:
    return (
        await _retrieve_mqe_output(
            query=query,
            k=k,
            per_variant_k=per_variant_k,
            num_variants=num_variants,
            research_topic=research_topic,
        )
    ).chunks


# =========================
# 检索层指标评估
# =========================

def _compute_precision_recall_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> Tuple[float, float, float]:
    """
    简单计算 Precision@k, Recall@k, MRR。
    - retrieved_ids: 模型返回的按得分排序的 id 列表（Top-k）
    - relevant_ids: 金标准相关 id 列表
    """
    if not retrieved_ids:
        return 0.0, 0.0, 0.0

    rel_set = set(relevant_ids)
    if not rel_set:
        # 若未提供金标准，则跳过此条（返回 0,0,0，调用方可过滤）
        return 0.0, 0.0, 0.0

    hits = [rid for rid in retrieved_ids if rid in rel_set]
    precision = len(hits) / len(retrieved_ids)
    recall = len(hits) / len(rel_set)

    # MRR: 第一个命中的倒数排名
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids, 1):
        if rid in rel_set:
            mrr = 1.0 / idx
            break

    return precision, recall, mrr


def _compute_precision_recall_mrr_by_url(
    retrieved_chunks: List[RetrievedChunk],
    relevant_urls: List[str],
) -> Tuple[float, float, float]:
    """
    文档级（URL级）评估：只要命中同一来源 URL 的任意 chunk 就算命中。
    用于避免把 baseline 的 chunk_id 当作金标准，从而对 HyDE/MQE 不公平。
    """
    rel = [u for u in (relevant_urls or []) if isinstance(u, str) and u.strip()]
    if not rel:
        return 0.0, 0.0, 0.0

    rel_set = set(rel)
    retrieved_urls: List[str] = []
    for c in retrieved_chunks:
        if not c.source_urls:
            continue
        for u in c.source_urls:
            if u and u not in retrieved_urls:
                retrieved_urls.append(u)

    if not retrieved_urls:
        return 0.0, 0.0, 0.0

    hits = [u for u in retrieved_urls if u in rel_set]
    precision = len(hits) / len(retrieved_urls)
    recall = len(hits) / len(rel_set)

    mrr = 0.0
    for idx, u in enumerate(retrieved_urls, 1):
        if u in rel_set:
            mrr = 1.0 / idx
            break
    return precision, recall, mrr


def _ndcg_at_k_chunk_ids(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    二元相关性下的 nDCG@k（chunk id）。
    DCG = sum_i rel_i / log2(rank_i + 1)，rel_i ∈ {0,1}；
    IDCG 为将 min(|R|, k) 个相关项排在最前时的 DCG。
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    rel_set = set(relevant_ids)
    truncated = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(truncated):
        rank = i + 1
        rel = 1.0 if rid in rel_set else 0.0
        dcg += rel / math.log2(rank + 1)
    num_rel = len(rel_set)
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, min(k, num_rel) + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _ndcg_at_k_urls(
    retrieved_chunks: List[RetrievedChunk],
    relevant_urls: List[str],
    k: int,
) -> float:
    """
    二元相关性：某 rank 上 chunk 的 source_urls 与任一金标 URL 相同则 rel=1。

    注意（与 chunk 级 nDCG 的差异）：
    多个不同 chunk 可来自**同一**金标 URL，故 Top-k 中最多可有 **k 个位置**均为 rel=1。
    IDCG 若用「金标 URL 个数」min(k, |U|) 项，会**低估**理想 DCG，导致 nDCG>1。
    因此 URL 级 IDCG 取「前 k 个位置均可命中相关 URL」时的上界：
    sum_{j=1}^{k} 1/log2(j+1)（与仅 1 个金标 URL 时也能用多篇 chunk 填满 Top-k 一致）。
    """
    rel_set = {u for u in (relevant_urls or []) if isinstance(u, str) and u.strip()}
    if not rel_set:
        return 0.0
    truncated = retrieved_chunks[:k]
    if not truncated:
        return 0.0
    dcg = 0.0
    for i, c in enumerate(truncated):
        rank = i + 1
        urls = c.source_urls or []
        if isinstance(urls, str):
            urls = [urls]
        rel = 1.0 if any(u in rel_set for u in urls if u) else 0.0
        dcg += rel / math.log2(rank + 1)
    # 理想：前 k 个 rank 均可视为相关（同一 URL 可对应多篇 chunk，故最多 k 个非零增益）
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, k + 1))
    if idcg <= 0:
        return 0.0
    return min(1.0, dcg / idcg)


def _warn_if_gold_chunk_ids_stale(store: MemoryStore, queries: List[EvalQuery]) -> None:
    """
    检查 jsonl 中的 relevant_ids 是否仍存在于当前 Chroma。

    重建索引（build_index / 清空 rag_eval_db）会为每个 chunk 生成**新** UUID；
    若未同步更新 data/rag_eval_queries.jsonl，则金标 id 与库内 id 不一致，
    Top-k 与金标永远无交集 → Precision/Recall/MRR 全为 0。
    """
    all_ids: List[str] = []
    for q in queries:
        all_ids.extend(q.relevant_ids or [])
    if not all_ids:
        print("[RAG-EVAL] 金标准: 未配置 relevant_ids，指标将无意义。", flush=True)
        return
    unique = list(dict.fromkeys(all_ids))
    try:
        data = store.collection.get(ids=unique, include=[])
        found = set(data.get("ids") or [])
    except Exception as e:
        print(f"[RAG-EVAL] 无法校验金标 chunk id: {e}", flush=True)
        return
    missing = [x for x in unique if x not in found]
    if not missing:
        print(
            f"[RAG-EVAL] 金标准 relevant_ids 校验：{len(unique)} 个 id 均在当前向量库中存在。",
            flush=True,
        )
        return
    print(
        f"\n[RAG-EVAL] 警告: 金标准 relevant_ids 中有 {len(missing)}/{len(unique)} 个 **不在** 当前向量库中。\n"
        f"  常见原因：执行过 build_index / 清空 rag_eval_db 后 chunk 已重新生成 UUID，\n"
        f"  而 data/rag_eval_queries.jsonl 仍为旧 id → 检索与金标永远无交集，P/R/MRR 会为 0。\n"
        f"  处理：用 `python -m src.evaluator.rag_eval_runner search_chunks \"你的问题关键词\"` 查当前 id，\n"
        f"  更新 jsonl 中的 relevant_ids；或确保 chunk 带 source_urls 后改用 URL 级命中逻辑。\n"
        f"  缺失 id 示例: {missing[:3]}\n",
        flush=True,
    )


def _print_rag_eval_gold_interpretation(queries: List[EvalQuery]) -> None:
    """
    解释为何「refresh_eval_ids 生成的金标」下 baseline 常远高于 HyDE/MQE。
    """
    if os.getenv("RAG_EVAL_SUPPRESS_GOLD_NOTE", "").lower() in ("1", "true", "yes", "y"):
        return
    marked = any(getattr(q, "gold_generated_by", None) == "baseline_topk" for q in queries)
    print(
        "\n[RAG-EVAL] —— 结果解读（重要）——\n"
        "  若 relevant_ids 由自动候选池写回生成：金标 = 多策略候选的并集（弱监督口径），会稀释策略间差异。\n"
        "  因此同一 query 上，baseline 检索与「造金标时」完全一致，**P/R/MRR 会系统性最优**；\n"
        "  HyDE/MQE 改用 **不同查询文本**（假设答案 / 变体），Top-k 与这组金标 id 不一致时，指标会偏低——\n"
        "  **这是标注定义导致的偏置，不是增强器实现错误**，也不表示 HyDE/MQE 一定更差。\n"
        "  若要公平对比增强器：请使用 **与检索方法无关** 的人工金标、或 **URL/文档级** 金标、或独立标注 chunk 集合。\n"
        + (
            "  （当前 jsonl 含 gold_generated_by=baseline_topk）\n"
            if marked
            else "  （若曾写回但未写入该字段，可重新生成 queries 以写入标记。）\n"
        ),
        flush=True,
    )


def _summarize_metric_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    n = int(bucket.get("count", 0))
    if n <= 0:
        return {"count": 0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return {
        "count": n,
        "precision": bucket["prec_sum"] / n,
        "recall": bucket["recall_sum"] / n,
        "mrr": bucket["mrr_sum"] / n,
        "ndcg": bucket["ndcg_sum"] / n,
    }


def _summarize_diag_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    attempts = int(bucket.get("attempts", 0))
    denom = max(1.0, float(bucket.get("attempts", 0.0)))
    return {
        "attempts": attempts,
        "fallback_rate": float(bucket.get("fallback_count", 0.0)) / denom,
        "timeout_rate": float(bucket.get("timeout_count", 0.0)) / denom,
        "parse_failure_rate": float(bucket.get("parse_failure_count", 0.0)) / denom,
        "empty_retrieval_rate": float(bucket.get("empty_retrieval_count", 0.0)) / denom,
        "enhancement_applied_rate": float(bucket.get("enhancement_applied_count", 0.0)) / denom,
        "avg_enhance_latency_ms": float(bucket.get("enhance_latency_ms_sum", 0.0)) / denom,
        "avg_retrieval_latency_ms": float(bucket.get("retrieval_latency_ms_sum", 0.0)) / denom,
    }


def _write_eval_report_files(payload: Dict[str, Any]) -> Dict[str, str]:
    out_dir = Path(os.getenv("RAG_EVAL_REPORT_DIR", "./reports/rag_eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path_env = os.getenv("RAG_EVAL_REPORT_JSON_PATH", "").strip()
    if json_path_env:
        p = Path(json_path_env)
        json_path = p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"rag_eval_{ts}"
        json_path = out_dir / f"{stem}.json"
    write_md = os.getenv("RAG_EVAL_WRITE_MD", "0").lower() in ("1", "true", "yes", "y")
    md_path = json_path.with_suffix(".md")

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not write_md:
        return {"json_path": str(json_path), "markdown_path": ""}

    lines: List[str] = [
        "# RAG 增强评估可复现报告",
        "",
        f"- 生成时间: {payload.get('timestamp')}",
        f"- 主题: `{payload.get('research_topic')}`",
        f"- 模式: `{','.join(payload.get('enabled_modes', []))}`",
        f"- k: `{payload.get('k')}`",
        "",
        "## 全量样本指标",
    ]
    for mode, met in payload.get("overall_summary", {}).items():
        lines.append(
            f"- `{mode}`: P@k={met.get('precision', 0.0):.3f}, R@k={met.get('recall', 0.0):.3f}, "
            f"MRR={met.get('mrr', 0.0):.3f}, nDCG@k={met.get('ndcg', 0.0):.3f}, n={met.get('count', 0)}"
        )
    lines.append("")
    lines.append("## 增强成功子集指标")
    for mode, met in payload.get("effective_summary", {}).items():
        lines.append(
            f"- `{mode}`: P@k={met.get('precision', 0.0):.3f}, R@k={met.get('recall', 0.0):.3f}, "
            f"MRR={met.get('mrr', 0.0):.3f}, nDCG@k={met.get('ndcg', 0.0):.3f}, n={met.get('count', 0)}"
        )
    lines.append("")
    lines.append("## 机制体检")
    for mode, d in payload.get("diag_summary", {}).items():
        lines.append(
            f"- `{mode}`: fallback={d.get('fallback_rate', 0.0):.1%}, timeout={d.get('timeout_rate', 0.0):.1%}, "
            f"parse_failure={d.get('parse_failure_rate', 0.0):.1%}, empty_retrieval={d.get('empty_retrieval_rate', 0.0):.1%}, "
            f"enhancement_applied={d.get('enhancement_applied_rate', 0.0):.1%}"
        )
    lines.append("")
    lines.append("## 评估口径一致性")
    scope = payload.get("evaluation_scope_count", {})
    lines.append(f"- URL级样本: {scope.get('url', 0)}")
    lines.append(f"- chunk_id级样本: {scope.get('chunk_id', 0)}")
    lines.append("")
    lines.append("## 典型案例")
    for c in (payload.get("case_candidates") or [])[:6]:
        lines.append(
            f"- `{c.get('mode')}` qid={c.get('qid')} gain={c.get('gain', 0.0):+.3f} "
            f"P/R/MRR={c.get('precision', 0.0):.3f}/{c.get('recall', 0.0):.3f}/{c.get('mrr', 0.0):.3f}"
        )
    if not payload.get("case_candidates"):
        lines.append("- 无可用案例（可能全部回退，或仅启用了 baseline）。")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


async def eval_retrieval(
    queries: List[EvalQuery],
    k: Optional[int] = None,
    research_topic: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对给定的评估 queries，分别用 baseline / HyDE / MQE 做检索，
    计算并打印整体的 Precision@k / Recall@k / MRR / nDCG@k 对比（二元相关性）。

    用法（命令行）：
      python -m src.evaluator.rag_eval_runner eval_retrieval

    环境变量：
    - RAG_EVAL_MODES=baseline 仅跑向量检索（无 LLM，秒级完成）
    - 默认 baseline,hyde,mqe：每条 query 并行 3 路，其中 HyDE/MQE 各需 1 次 LLM，
      10 条 query 总耗时可达数十分钟；运行中会打印进度。
    - RAG_EVAL_TOP_K：评估截断 k（默认 10）
    - RAG_EVAL_HYDE_MERGE_ORIGINAL=1：HyDE 合并「原句检索 + 假设文档检索」再取 Top-k
    - RAG_EVAL_HYDE_FUSION=max|precision（默认 precision）：precision 用两路相似度 min 排序，偏精准
    - RAG_EVAL_HYDE_ONLY_ORIG_WEIGHT / RAG_EVAL_HYDE_ONLY_HYP_WEIGHT：仅出现在单路的降权系数
    - RAG_EVAL_HYDE_ANSWER_LENGTH：short|medium|detailed
    - RAG_EVAL_MQE_NUM_VARIANTS、RAG_EVAL_MQE_PER_VARIANT_K、RAG_EVAL_MQE_ORIG_POOL_MIN、
      RAG_EVAL_MQE_MERGE=max|rrf、RAG_EVAL_MQE_RRF_CONST、RAG_EVAL_MQE_RRF_OVERLAP_BONUS（多子查询同现加分，偏召回）
    """
    if k is None:
        k = _rag_eval_top_k()
    enabled_modes = _parse_rag_eval_modes()
    if not _llm_available_for_enhance() and any(m in ("hyde", "mqe") for m in enabled_modes):
        print(
            "\n[RAG-EVAL] 提示: 未检测到 LLM API Key（OPENAI_API_KEY / DASHSCOPE_API_KEY 等），"
            "HyDE/MQE 将回退为 baseline，三者指标会一致。若要观察 HyDE 精度 / MQE 召回差异，请设置相应环境变量后重跑。",
            flush=True,
        )

    topic = _resolve_eval_research_topic(research_topic)
    try:
        store = _get_eval_memory_store()
        n = store.collection.count()
        topic_n = _count_chunks_by_topic(store, topic)
        db_dir = os.getenv("RAG_EVAL_DB_DIR", "./data/rag_eval_db")
        embedder = getattr(store, "embedder", None)
        embedder_provider = getattr(embedder, "provider", None)
        embedder_model = getattr(embedder, "model_name", None)
        print(f"[RAG-EVAL] 向量库 {db_dir} 共 {n} 条 chunk（Chroma 持久化目录，存向量与元数据，非原文；原文在 data/rag_eval_docs/_clean）。")
        print(f"[RAG-EVAL] 当前嵌入: provider={embedder_provider} model={embedder_model}")
        print(f"[RAG-EVAL] 当前评估 topic={topic!r}，该 topic 下 chunk 数={topic_n}", flush=True)
        if topic_n <= 0:
            # 常见坑：历史索引未写入 metadata.research_topic，导致按 topic 过滤永远为 0。
            # 若库本身非空，则降级为“不按 topic 过滤”，避免评测直接中断。
            if n > 0:
                print(
                    f"[RAG-EVAL] 警告: topic={topic!r} 过滤后 chunk=0，但库总 chunk={n}。"
                    "将降级为不按 topic 过滤（research_topic=None），以继续评测。"
                    "若你希望严格按 topic 隔离，请重建索引并写入 metadata.research_topic。",
                    flush=True,
                )
                topic = ""
            else:
                raise RuntimeError(
                    f"topic={topic!r} 在评估向量库中没有可检索 chunk。"
                    "请检查 RAG_EVAL_DB_DIR 与 RAG_EVAL_RESEARCH_TOPIC / queries 路径是否匹配。"
                )
        _warn_if_gold_chunk_ids_stale(store, queries)
    except Exception as e:
        print(f"[RAG-EVAL] 评估启动前自检失败: {e}", flush=True)
        raise

    stats: Dict[str, Dict[str, float]] = {
        m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0}
        for m in enabled_modes
    }
    # 公平口径：仅统计“增强真正生效”的样本（HyDE/MQE fallback 不计入）
    stats_effective: Dict[str, Dict[str, float]] = {
        m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0}
        for m in enabled_modes
    }
    # 机制体检：增强回退/超时/解析失败/空检索/耗时
    diag_stats: Dict[str, Dict[str, float]] = {
        m: {
            "attempts": 0.0,
            "fallback_count": 0.0,
            "timeout_count": 0.0,
            "parse_failure_count": 0.0,
            "empty_retrieval_count": 0.0,
            "enhancement_applied_count": 0.0,
            "enhance_latency_ms_sum": 0.0,
            "retrieval_latency_ms_sum": 0.0,
        }
        for m in enabled_modes
    }

    # per-query 明细：既保留全量（含无金标样本），也按评估口径拆分（便于做 chunk_id 难例分析）
    per_query_rows: List[Dict[str, Any]] = []
    per_query_rows_by_scope: Dict[str, List[Dict[str, Any]]] = {"url": [], "chunk_id": []}
    eval_scope_count: Dict[str, int] = {"url": 0, "chunk_id": 0}

    # 额外输出：按评估口径分别汇总（避免 url 级均值掩盖 chunk_id 难例）
    stats_by_scope: Dict[str, Dict[str, Dict[str, float]]] = {
        "url": {m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0} for m in enabled_modes},
        "chunk_id": {m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0} for m in enabled_modes},
    }
    stats_effective_by_scope: Dict[str, Dict[str, Dict[str, float]]] = {
        "url": {m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0} for m in enabled_modes},
        "chunk_id": {m: {"prec_sum": 0.0, "recall_sum": 0.0, "mrr_sum": 0.0, "ndcg_sum": 0.0, "count": 0} for m in enabled_modes},
    }

    n_q = len(queries)
    has_llm = any(m in ("hyde", "mqe") for m in enabled_modes)
    print(
        f"[RAG-EVAL] 评估模式: {enabled_modes}；共 {n_q} 条 query。"
        + (
            " 含 HyDE/MQE 时每条约 1–3 分钟（网络+LLM），请等待进度行。"
            " 若只想测向量速度，可设环境变量 RAG_EVAL_MODES=baseline"
            if has_llm
            else ""
        ),
        flush=True,
    )
    if has_llm and _llm_available_for_enhance():
        print(
            f"[RAG-EVAL] HyDE/MQE 调用时限: HTTP 读超时={_eval_llm_http_timeout_s()}s，"
            f"外层 asyncio wait_for={_eval_enhance_async_wait_timeout_s():.0f}s "
            f"（避免假超时：外层须大于 HTTP；可调 ENHANCER_QWEN_TIMEOUT、RAG_EVAL_ENHANCE_TIMEOUT_S、RAG_EVAL_LLM_HTTP_TIMEOUT_S）",
            flush=True,
        )
    if any(m in ("hyde", "mqe") for m in enabled_modes):
        try:
            ob = float((os.getenv("RAG_EVAL_MQE_RRF_OVERLAP_BONUS") or "0.14").strip())
        except ValueError:
            ob = 0.14
        try:
            opm = int((os.getenv("RAG_EVAL_MQE_ORIG_POOL_MIN") or "48").strip())
        except ValueError:
            opm = 48
        print(
            f"[RAG-EVAL] 增强检索配置: k={k} | HyDE merge_orig={_eval_hyde_merge_original()} "
            f"fusion={_eval_hyde_fusion_mode()} answer_len={_eval_hyde_answer_length()} | "
            f"MQE merge={_eval_mqe_merge_mode()} variants={_eval_mqe_num_variants(2)} "
            f"per_variant_k={_eval_mqe_per_variant_k(k, 20)} orig_pool_min~{opm} "
            f"rrf_c={_eval_mqe_rrf_const()} rrf_overlap_bonus={max(0.0, ob):.3f}",
            flush=True,
        )

    for q_idx, q in enumerate(queries, 1):
        print(f"[RAG-EVAL] 进度 {q_idx}/{n_q}  id={q.id!r} 检索中…", flush=True)

        coros = []
        keys: List[str] = []
        if "baseline" in enabled_modes:
            coros.append(_retrieve_baseline_output(q.query, k=k, research_topic=topic))
            keys.append("baseline")
        if "hyde" in enabled_modes:
            coros.append(_retrieve_hyde_output(q.query, k=k, research_topic=topic))
            keys.append("hyde")
        if "mqe" in enabled_modes:
            coros.append(_retrieve_mqe_output(q.query, k=k, research_topic=topic))
            keys.append("mqe")

        gathered = await asyncio.gather(*coros)
        out_by_mode: Dict[str, RetrievalOutput] = dict(zip(keys, gathered))

        # 将 chunk_id 金标准映射为 URL 金标准（同一文档不同 chunk 也算命中）
        relevant_urls: List[str] = []
        if q.relevant_ids:
            try:
                store = _get_eval_memory_store()
                data = store.collection.get(ids=q.relevant_ids, include=["metadatas"])
                metas = (data or {}).get("metadatas") or []
                for meta in metas:
                    if not isinstance(meta, dict):
                        continue
                    urls = meta.get("source_urls") or []
                    if isinstance(urls, str):
                        urls = [urls]
                    for u in urls:
                        if isinstance(u, str) and u.startswith("https:///"):
                            u = "https://" + u[len("https:///") :]
                        if u and u not in relevant_urls:
                            relevant_urls.append(u)
            except Exception:
                relevant_urls = []
        eval_scope = "url" if relevant_urls else "chunk_id"
        eval_scope_count[eval_scope] += 1

        for mode in enabled_modes:
            output = out_by_mode[mode]
            chunks = output.chunks
            diag = output.diag
            retrieved_ids = [c.id for c in chunks]

            ds = diag_stats[mode]
            ds["attempts"] += 1.0
            ds["fallback_count"] += 1.0 if diag.fallback else 0.0
            ds["timeout_count"] += 1.0 if diag.timeout else 0.0
            ds["parse_failure_count"] += 1.0 if diag.parse_failure else 0.0
            ds["empty_retrieval_count"] += 1.0 if diag.empty_retrieval else 0.0
            ds["enhancement_applied_count"] += 1.0 if diag.enhancement_applied else 0.0
            ds["enhance_latency_ms_sum"] += float(diag.enhance_latency_ms or 0.0)
            ds["retrieval_latency_ms_sum"] += float(diag.retrieval_latency_ms or 0.0)

            # 优先 URL 级评估；若无法取到 URL，再回退到 chunk_id 级评估
            # 注意：即使无金标准，也要记录 per-query 诊断信息；指标字段用 None。
            has_gold = bool(q.relevant_ids)
            prec = rec = mrr = ndcg = None
            if has_gold:
                if relevant_urls:
                    prec, rec, mrr = _compute_precision_recall_mrr_by_url(chunks, relevant_urls)
                    ndcg = _ndcg_at_k_urls(chunks, relevant_urls, k)
                else:
                    prec, rec, mrr = _compute_precision_recall_mrr(retrieved_ids, q.relevant_ids)
                    ndcg = _ndcg_at_k_chunk_ids(retrieved_ids, q.relevant_ids, k)

            s = stats[mode]
            if has_gold:
                s["prec_sum"] += float(prec or 0.0)
                s["recall_sum"] += float(rec or 0.0)
                s["mrr_sum"] += float(mrr or 0.0)
                s["ndcg_sum"] += float(ndcg or 0.0)
                s["count"] += 1

                # scope 分桶统计
                sb = stats_by_scope[eval_scope][mode]
                sb["prec_sum"] += float(prec or 0.0)
                sb["recall_sum"] += float(rec or 0.0)
                sb["mrr_sum"] += float(mrr or 0.0)
                sb["ndcg_sum"] += float(ndcg or 0.0)
                sb["count"] += 1

            # 有效增强子集口径：
            # - baseline: 全量即有效
            # - hyde/mqe: 仅 enhancement_applied=True 计入（剔除 fallback）
            effective = (mode == "baseline") or bool(diag.enhancement_applied)
            if has_gold and effective:
                se = stats_effective[mode]
                se["prec_sum"] += float(prec or 0.0)
                se["recall_sum"] += float(rec or 0.0)
                se["mrr_sum"] += float(mrr or 0.0)
                se["ndcg_sum"] += float(ndcg or 0.0)
                se["count"] += 1

                seb = stats_effective_by_scope[eval_scope][mode]
                seb["prec_sum"] += float(prec or 0.0)
                seb["recall_sum"] += float(rec or 0.0)
                seb["mrr_sum"] += float(mrr or 0.0)
                seb["ndcg_sum"] += float(ndcg or 0.0)
                seb["count"] += 1

            # 记录 per-query 结果，便于典型案例分析
            row = {
                "qid": q.id,
                "query": q.query,
                "mode": mode,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": q.relevant_ids,
                "has_gold": has_gold,
                "precision": prec,
                "recall": rec,
                "mrr": mrr,
                "ndcg": ndcg,
                "fallback": bool(diag.fallback),
                "fallback_reason": diag.fallback_reason,
                "enhancement_applied": bool(diag.enhancement_applied),
                "timeout": bool(diag.timeout),
                "parse_failure": bool(diag.parse_failure),
                "empty_retrieval": bool(diag.empty_retrieval),
                "enhance_latency_ms": float(diag.enhance_latency_ms or 0.0),
                "retrieval_latency_ms": float(diag.retrieval_latency_ms or 0.0),
                "variant_count": int(diag.variant_count or 0),
                "eval_scope": eval_scope,
            }
            per_query_rows.append(row)
            per_query_rows_by_scope[eval_scope].append(row)

    # 打印结果
    print("\n[RAG-EVAL] 检索层指标（k={}）:".format(k))
    for mode in enabled_modes:
        s = stats[mode]
        if s["count"] == 0:
            print(f"  - {mode}: 无有效金标准样本，无法统计。")
            continue
        n = s["count"]
        avg_p = s["prec_sum"] / n
        avg_r = s["recall_sum"] / n
        avg_mrr = s["mrr_sum"] / n
        avg_ndcg = s["ndcg_sum"] / n
        print(
            f"  - {mode:8s} | P@{k}: {avg_p:.3f}  R@{k}: {avg_r:.3f}  MRR: {avg_mrr:.3f}  "
            f"nDCG@{k}: {avg_ndcg:.3f}  (n={n})"
        )

    print("\n[RAG-EVAL] 有效增强子集口径（HyDE/MQE 仅统计 enhancement_applied=True）:")
    for mode in enabled_modes:
        s = stats_effective[mode]
        if s["count"] == 0:
            print(f"  - {mode}: 无有效样本（可能全部回退/未生效）")
            continue
        n = s["count"]
        avg_p = s["prec_sum"] / n
        avg_r = s["recall_sum"] / n
        avg_mrr = s["mrr_sum"] / n
        avg_ndcg = s["ndcg_sum"] / n
        print(
            f"  - {mode:8s} | P@{k}: {avg_p:.3f}  R@{k}: {avg_r:.3f}  MRR: {avg_mrr:.3f}  "
            f"nDCG@{k}: {avg_ndcg:.3f}  (n={n})"
        )

    print("\n[RAG-EVAL] 机制体检统计（增强健康度）:")
    for mode in enabled_modes:
        ds = diag_stats[mode]
        n = max(1.0, ds["attempts"])
        print(
            f"  - {mode:8s} | attempts={int(ds['attempts'])} "
            f"fallback={int(ds['fallback_count'])} timeout={int(ds['timeout_count'])} "
            f"parse_fail={int(ds['parse_failure_count'])} empty={int(ds['empty_retrieval_count'])} "
            f"enhanced={int(ds['enhancement_applied_count'])} "
            f"enh_ms_avg={ds['enhance_latency_ms_sum']/n:.1f} retr_ms_avg={ds['retrieval_latency_ms_sum']/n:.1f}"
        )
    print(
        f"[RAG-EVAL] 评估口径一致性：URL级样本={eval_scope_count['url']}，"
        f"chunk_id级样本={eval_scope_count['chunk_id']}。"
    )

    _print_rag_eval_gold_interpretation(queries)

    # ===== 典型案例分析素材（自动挑选）=====
    # 目标：
    # - HyDE：相较 baseline，precision / mrr 提升明显（更像“精度提升”）
    # - MQE：相较 baseline，recall 提升明显（更像“召回提升”）
    case_candidates: List[Dict[str, Any]] = []
    if per_query_rows and len(enabled_modes) == 3:
        by_q: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in per_query_rows:
            # 无金标样本（has_gold=False）或指标缺失时不参与增益计算/典型案例挑选
            if not r.get("has_gold"):
                continue
            if r.get("precision") is None or r.get("recall") is None or r.get("mrr") is None:
                continue
            by_q.setdefault(r["qid"], {})[r["mode"]] = r

        hyde_candidates: List[Tuple[float, str]] = []
        mqe_candidates: List[Tuple[float, str]] = []

        for qid, mm in by_q.items():
            if "baseline" not in mm or "hyde" not in mm or "mqe" not in mm:
                continue
            b = mm["baseline"]
            h = mm["hyde"]
            m = mm["mqe"]

            # HyDE：看 precision + mrr 的综合提升
            hyde_gain = (h["precision"] - b["precision"]) + (h["mrr"] - b["mrr"])
            # MQE：看 recall 提升
            mqe_gain = (m["recall"] - b["recall"])

            hyde_candidates.append((hyde_gain, qid))
            mqe_candidates.append((mqe_gain, qid))

        def _print_case(title: str, r_base: Dict[str, Any], r_other: Dict[str, Any]) -> None:
            rel = set(r_base["relevant_ids"])
            base_hit = [x for x in r_base["retrieved_ids"] if x in rel]
            other_hit = [x for x in r_other["retrieved_ids"] if x in rel]
            newly = [x for x in other_hit if x not in base_hit]
            lost = [x for x in base_hit if x not in other_hit]

            print(f"\n[{title}] qid={r_base['qid']}  query={r_base['query']}")
            print(
                f"  baseline: P={r_base['precision']:.3f} R={r_base['recall']:.3f} MRR={r_base['mrr']:.3f}  hits={len(base_hit)}/{len(rel)}"
            )
            print(
                f"  {r_other['mode']}: P={r_other['precision']:.3f} R={r_other['recall']:.3f} MRR={r_other['mrr']:.3f}  hits={len(other_hit)}/{len(rel)}"
            )
            if newly:
                print(f"  新增命中 relevant_ids: {newly[:5]}")
            if lost:
                print(f"  丢失命中 relevant_ids: {lost[:5]}")

        # 打印 Top-2 典型案例（有正向增益时）；无正向增益时也至少输出一条对比示例并说明原因
        hyde_candidates.sort(reverse=True, key=lambda x: x[0])
        mqe_candidates.sort(reverse=True, key=lambda x: x[0])
        if "case_candidates" not in locals() or not isinstance(case_candidates, list):
            case_candidates = []
        for gain, qid in hyde_candidates[:3]:
            mm = by_q.get(qid, {})
            if "hyde" in mm:
                case_candidates.append(
                    {
                        "mode": "hyde",
                        "qid": qid,
                        "query": mm["hyde"].get("query"),
                        "gain": gain,
                        "precision": mm["hyde"].get("precision", 0.0),
                        "recall": mm["hyde"].get("recall", 0.0),
                        "mrr": mm["hyde"].get("mrr", 0.0),
                        "fallback": bool(mm["hyde"].get("fallback")),
                    }
                )
        for gain, qid in mqe_candidates[:3]:
            mm = by_q.get(qid, {})
            if "mqe" in mm:
                case_candidates.append(
                    {
                        "mode": "mqe",
                        "qid": qid,
                        "query": mm["mqe"].get("query"),
                        "gain": gain,
                        "precision": mm["mqe"].get("precision", 0.0),
                        "recall": mm["mqe"].get("recall", 0.0),
                        "mrr": mm["mqe"].get("mrr", 0.0),
                        "fallback": bool(mm["mqe"].get("fallback")),
                    }
                )

        print("\n[RAG-EVAL] 典型案例（自动挑选，供分析 HyDE 精度 / MQE 召回）:")

        hyde_printed = 0
        for gain, qid in hyde_candidates[:2]:
            if gain > 0:
                mm = by_q[qid]
                _print_case("HyDE 精度提升案例", mm["baseline"], mm["hyde"])
                hyde_printed += 1

        mqe_printed = 0
        for gain, qid in mqe_candidates[:2]:
            if gain > 0:
                mm = by_q[qid]
                _print_case("MQE 召回提升案例", mm["baseline"], mm["mqe"])
                mqe_printed += 1

        # 若所有 query 上 HyDE/MQE 与 baseline 结果完全一致（回退或增强未产生差异），则至少打印一条对比示例
        all_same = all(
            mm.get("baseline", {}).get("retrieved_ids") == mm.get("hyde", {}).get("retrieved_ids")
            and mm.get("baseline", {}).get("retrieved_ids") == mm.get("mqe", {}).get("retrieved_ids")
            for mm in by_q.values()
            if "baseline" in mm and "hyde" in mm and "mqe" in mm
        )
        if all_same and by_q:
            example_qid = next(iter(by_q))
            mm = by_q[example_qid]
            print("\n[对比示例] 三者 retrieved_ids 完全一致（HyDE/MQE 可能已回退为 baseline）:")
            _print_case("示例（baseline vs hyde）", mm["baseline"], mm["hyde"])
            _print_case("示例（baseline vs mqe）", mm["baseline"], mm["mqe"])
            print(
                "  原因: 未配置 LLM API Key 或增强超时/失败时会回退为 baseline。"
                "若已配置 DASHSCOPE_API_KEY 仍一致，请查看「HyDE/MQE 增强失败」日志；"
                "假超时请把 RAG_EVAL_ENHANCE_TIMEOUT_S、ENHANCER_QWEN_TIMEOUT 或 RAG_EVAL_LLM_HTTP_TIMEOUT_S 调大（外层须大于 HTTP 读超时）。"
            )
        elif not hyde_printed and not mqe_printed and not all_same:
            # 有差异但 gain 都<=0（例如 HyDE/MQE 略差），仍给一条对比
            example_qid = next(iter(by_q))
            mm = by_q[example_qid]
            print("\n[对比示例] 当前数据集中 HyDE/MQE 相对 baseline 无正向增益，以下为单条对比:")
            _print_case("示例（baseline vs hyde）", mm["baseline"], mm["hyde"])
            _print_case("示例（baseline vs mqe）", mm["baseline"], mm["mqe"])
    elif per_query_rows and len(enabled_modes) < 3:
        print(
            "\n[RAG-EVAL] 当前未同时启用 baseline/hyde/mqe，已跳过「典型案例」对比块。"
            " 全量对比请使用默认 RAG_EVAL_MODES 或显式 baseline,hyde,mqe。",
            flush=True,
        )

    overall_summary = {m: _summarize_metric_bucket(stats[m]) for m in enabled_modes}
    effective_summary = {m: _summarize_metric_bucket(stats_effective[m]) for m in enabled_modes}
    overall_summary_by_scope = {
        scope: {m: _summarize_metric_bucket(stats_by_scope[scope][m]) for m in enabled_modes}
        for scope in ("url", "chunk_id")
    }
    effective_summary_by_scope = {
        scope: {m: _summarize_metric_bucket(stats_effective_by_scope[scope][m]) for m in enabled_modes}
        for scope in ("url", "chunk_id")
    }
    diag_summary = {m: _summarize_diag_bucket(diag_stats[m]) for m in enabled_modes}
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "research_topic": topic,
        "enabled_modes": enabled_modes,
        "overall_stats": stats,
        "effective_stats": stats_effective,
        "diag_stats": diag_stats,
        "overall_summary": overall_summary,
        "effective_summary": effective_summary,
        "overall_summary_by_scope": overall_summary_by_scope,
        "effective_summary_by_scope": effective_summary_by_scope,
        "diag_summary": diag_summary,
        "evaluation_scope_count": eval_scope_count,
        "case_candidates": case_candidates,
        "per_query_rows": per_query_rows,
        "per_query_rows_by_scope": per_query_rows_by_scope,
    }

    # ===== 额外指标：增强效果可解释性（更适合写简历/面试讲）=====
    # 目标：
    # - HyDE：排序/首命中更靠前（win-rate、Hit@1/3/5、FirstHitRank）
    # - MQE：难例召回补偿（hard cases 的 Recall/HitRate 提升）
    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def _first_hit_rank(retrieved_ids: List[str], relevant_set: set) -> Optional[int]:
        for i, rid in enumerate(retrieved_ids or [], 1):
            if rid in relevant_set:
                return i
        return None

    def _compute_extras(scope: str) -> Dict[str, Any]:
        rows = per_query_rows_by_scope.get(scope) or []
        # 只统计有金标且指标非空的样本
        rows = [r for r in rows if r.get("has_gold") and (r.get("precision") is not None)]
        by_q: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in rows:
            by_q.setdefault(r["qid"], {})[r["mode"]] = r

        # 基础：Hit@k（是否命中至少一个 gold）
        ks = [1, 3, 5]
        hit = {m: {f"hit@{k}": 0 for k in ks} for m in enabled_modes}
        first_rank_sum = {m: 0 for m in enabled_modes}
        first_rank_cnt = {m: 0 for m in enabled_modes}
        win = {m: {"ndcg_win_rate": 0, "mrr_win_rate": 0, "p_win_rate": 0, "r_win_rate": 0, "n": 0} for m in enabled_modes if m != "baseline"}

        # 难例：baseline recall==0
        hard_qids: List[str] = []

        for qid, mm in by_q.items():
            b = mm.get("baseline")
            if not b:
                continue
            rel = set(b.get("relevant_ids") or [])
            # hard 定义：baseline recall==0（严格）
            if float(b.get("recall") or 0.0) == 0.0:
                hard_qids.append(qid)

            for m in enabled_modes:
                r = mm.get(m)
                if not r:
                    continue
                retrieved = r.get("retrieved_ids") or []
                # hit@k
                for k_ in ks:
                    if any(x in rel for x in (retrieved[:k_] if len(retrieved) >= k_ else retrieved)):
                        hit[m][f"hit@{k_}"] += 1
                # first hit rank
                fr = _first_hit_rank(retrieved, rel)
                if fr is not None:
                    first_rank_sum[m] += fr
                    first_rank_cnt[m] += 1

            # win-rate：对比 baseline（仅在两者指标均存在时）
            for m in ("hyde", "mqe"):
                if m not in enabled_modes:
                    continue
                r = mm.get(m)
                if not r:
                    continue
                if r.get("ndcg") is None or r.get("mrr") is None:
                    continue
                win[m]["n"] += 1
                win[m]["ndcg_win_rate"] += 1 if float(r["ndcg"]) > float(b["ndcg"]) else 0
                win[m]["mrr_win_rate"] += 1 if float(r["mrr"]) > float(b["mrr"]) else 0
                win[m]["p_win_rate"] += 1 if float(r["precision"]) > float(b["precision"]) else 0
                win[m]["r_win_rate"] += 1 if float(r["recall"]) > float(b["recall"]) else 0

        n = max(1, len(by_q))
        out: Dict[str, Any] = {"scope": scope, "q_count": len(by_q)}
        out["hit_rate"] = {m: {k: round(v / n, 4) for k, v in hit[m].items()} for m in enabled_modes}
        out["first_hit_rank"] = {
            m: {
                "avg": round(first_rank_sum[m] / max(1, first_rank_cnt[m]), 3) if first_rank_cnt[m] else None,
                "coverage": round(first_rank_cnt[m] / n, 4),
            }
            for m in enabled_modes
        }
        out["win_rate_vs_baseline"] = {
            m: {
                "n": win[m]["n"],
                "ndcg_win_rate": round(win[m]["ndcg_win_rate"] / max(1, win[m]["n"]), 4) if win[m]["n"] else None,
                "mrr_win_rate": round(win[m]["mrr_win_rate"] / max(1, win[m]["n"]), 4) if win[m]["n"] else None,
                "precision_win_rate": round(win[m]["p_win_rate"] / max(1, win[m]["n"]), 4) if win[m]["n"] else None,
                "recall_win_rate": round(win[m]["r_win_rate"] / max(1, win[m]["n"]), 4) if win[m]["n"] else None,
            }
            for m in win
        }

        # hard-case：只看 baseline recall==0 的子集上，HyDE/MQE 是否把 recall 拉起来
        hard_total = len(hard_qids)
        hard_improve = {}
        for m in ("mqe", "hyde"):
            if m not in enabled_modes:
                continue
            wins = 0
            for qid in hard_qids:
                mm = by_q.get(qid, {})
                b = mm.get("baseline")
                r = mm.get(m)
                if not b or not r:
                    continue
                if float(r.get("recall") or 0.0) > float(b.get("recall") or 0.0):
                    wins += 1
            hard_improve[m] = {
                "hard_count": hard_total,
                "recall_win_rate": round(wins / max(1, hard_total), 4) if hard_total else None,
            }
        out["hard_cases_baseline_recall0"] = {"count": hard_total, "improvement": hard_improve}
        return out

    payload["extra_metrics"] = {
        "chunk_id": _compute_extras("chunk_id"),
        "url": _compute_extras("url"),
    }
    report_paths = _write_eval_report_files(payload)
    payload["report_paths"] = report_paths
    print(f"[RAG-EVAL] 可复现报告已落盘: json={report_paths['json_path']}", flush=True)
    if report_paths.get("markdown_path"):
        print(f"[RAG-EVAL] 可选 markdown 报告: {report_paths['markdown_path']}", flush=True)
    return payload


# =========================
# QA + LangSmith 评估骨架
# =========================

def _format_context_for_qa(chunks: List[RetrievedChunk]) -> str:
    """
    将检索到的 chunks 格式化为带编号的上下文，供 RAG 回答使用。
    每条包含文本与（若有）来源 URL。
    """
    lines: List[str] = []
    for idx, c in enumerate(chunks, 1):
        lines.append(f"[{idx}] {c.content}")
        if c.source_urls:
            for u in c.source_urls[:3]:
                lines.append(f"    来源: {u}")
        lines.append("")
    return "\n".join(lines)


def _chunks_for_langsmith(
    chunks: List[RetrievedChunk],
    max_content_chars: int,
) -> List[Dict[str, Any]]:
    """压缩检索结果以便写入 LangSmith（避免单条 run 过大）。"""
    out: List[Dict[str, Any]] = []
    for c in chunks:
        text = c.content or ""
        if len(text) > max_content_chars:
            text = text[:max_content_chars] + f"\n…[截断，共 {len(c.content or '')} 字符]"
        out.append(
            {
                "id": c.id,
                "score": c.score,
                "content": text,
                "source_urls": c.source_urls,
            }
        )
    return out


async def _answer_with_rag(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    使用统一的 RAG 提示词，根据检索到的 chunks 生成答案。
    提示词会要求显式引用编号 [1][2]…，以便后续在 LangSmith 中评估 Groundedness。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    context_text = _format_context_for_qa(chunks)

    system_msg = SystemMessage(
        content=(
            "你是一名严格依赖证据的技术写作者。"
            "你只能基于提供的资料回答问题，禁止编造资料中没有的事实。"
            "回答时请使用类似 [1][2] 的引用标号指向下方的资料段落编号。"
        )
    )
    user_msg = HumanMessage(
        content=(
            f"问题：{query}\n\n"
            "以下是与问题相关的资料片段（带编号）：\n\n"
            f"{context_text}\n\n"
            "请基于这些资料回答问题，并在使用某个片段时用 [编号] 引用来源。"
            "如果资料不足以得出明确结论，请说明“证据不足”，而不是猜测。"
        )
    )

    try:
        resp = await llm_client.agenerate(messages=[[system_msg, user_msg]])
        return resp.generations[0][0].text
    except Exception as e:
        return f"## 执行错误\n\n回答生成失败: {str(e)}"


async def eval_with_langsmith(
    queries: List[EvalQuery],
    k: int = 5,
    research_topic: Optional[str] = "rag_eval",
) -> None:
    """
    使用 LangSmith **traceable** 记录 RAG QA（检索 + 生成 answer），便于在 LangSmith UI
    中配置在线评估器（Correctness / Groundedness 等）或人工对比。

    说明：
    - 本函数不在本地打分，只上报 runs。
    - 环境变量（二选一密钥）：
        LANGSMITH_API_KEY  或  LANGCHAIN_API_KEY
        LANGSMITH_PROJECT  （默认 rag-eval）
        LANGSMITH_TRACING_V2  由配置函数自动设为 true
    - 检索模式与 `eval_retrieval` 一致，由 **RAG_EVAL_MODES** 控制（如 baseline,hyde,mqe）。
    - 每条 chunk 正文在上报前会截断，长度由 **RAG_EVAL_LANGSMITH_CHUNK_CHARS**（默认 2000）控制。
    """
    try:
        from langsmith import traceable
    except ImportError:
        print(
            "[RAG-EVAL] 未安装 langsmith 包，跳过 LangSmith 评估。请先 `pip install langsmith`。",
            flush=True,
        )
        return

    from src.tools.langsmith_env import configure_langsmith_tracing

    project_name = os.getenv("LANGSMITH_PROJECT", "rag-eval").strip() or "rag-eval"
    if not configure_langsmith_tracing(project_name=project_name, force_tracing=True):
        print(
            "[RAG-EVAL] 未检测到 LANGSMITH_API_KEY / LANGCHAIN_API_KEY，跳过 LangSmith 评估。",
            flush=True,
        )
        return

    try:
        max_chunk_chars = max(400, int(os.getenv("RAG_EVAL_LANGSMITH_CHUNK_CHARS", "2000")))
    except ValueError:
        max_chunk_chars = 2000

    modes = _parse_rag_eval_modes()
    rt = research_topic or "rag_eval"

    @traceable(name="rag-eval-rag-qa", run_type="chain")
    async def _one_trace(
        qid: str,
        query: str,
        mode: str,
        top_k: int,
        topic: str,
        gold_answer: Optional[str],
    ) -> Dict[str, Any]:
        if mode == "baseline":
            chunks = await retrieve_baseline(query, k=top_k, research_topic=topic)
        elif mode == "hyde":
            chunks = await retrieve_hyde(query, k=top_k, research_topic=topic)
        else:
            chunks = await retrieve_mqe(query, k=top_k, research_topic=topic)
        answer = await _answer_with_rag(query, chunks)
        return {
            "qid": qid,
            "mode": mode,
            "answer": answer,
            "gold_answer": gold_answer,
            "retrieved_chunks": _chunks_for_langsmith(chunks, max_chunk_chars),
        }

    n = 0
    for q in queries:
        for mode in modes:
            await _one_trace(q.id, q.query, mode, k, rt, q.gold_answer)
            n += 1

    print(
        f"[RAG-EVAL] 已向 LangSmith 提交 {n} 条 RAG QA 追踪（项目 '{project_name}'，模式 {modes}）。"
        f" 在 LangSmith → Projects → 选择该 project 查看 runs，并可配置在线评测。",
        flush=True,
    )


async def demo_query(query: str, k: int = 5) -> None:
    """
    简单演示：对同一个 query 分别用 baseline / hyde / mqe 做检索，并打印结果概览。
    仅用于开发自测，不参与正式工作流。
    """
    print(f"\n=== RAG Eval Demo: query = {query!r} ===")

    baseline = await retrieve_baseline(query, k=k)
    print(f"\n[baseline] Top-{k} 命中 {len(baseline)} 条：")
    for i, c in enumerate(baseline, 1):
        print(f"  {i}. id={c.id[:8]} score={c.score:.4f} source={c.source_urls[0] if c.source_urls else 'N/A'}")

    hyde = await retrieve_hyde(query, k=k)
    print(f"\n[HyDE] Top-{k} 命中 {len(hyde)} 条：")
    for i, c in enumerate(hyde, 1):
        print(f"  {i}. id={c.id[:8]} score={c.score:.4f} source={c.source_urls[0] if c.source_urls else 'N/A'}")

    mqe = await retrieve_mqe(query, k=k)
    print(f"\n[MQE] Top-{k} 命中 {len(mqe)} 条：")
    for i, c in enumerate(mqe, 1):
        print(f"  {i}. id={c.id[:8]} score={c.score:.4f} source={c.source_urls[0] if c.source_urls else 'N/A'}")


def _gold_preview_chars() -> int:
    """金标辅助命令里正文预览长度；过长易刷屏，过短不便判断。"""
    try:
        return max(80, int(os.getenv("RAG_EVAL_GOLD_PREVIEW_CHARS", "220")))
    except ValueError:
        return 220


async def search_chunks(
    query: str,
    k: int = 10,
    research_topic: Optional[str] = "rag_eval",
    show_content_chars: Optional[int] = None,
) -> None:
    """
    辅助标注工具：在评估向量库中搜索并打印候选 chunk。
    用于快速获取 chunk_id，填入 rag_eval_queries.jsonl 的 relevant_ids。
    环境变量 RAG_EVAL_GOLD_PREVIEW_CHARS 可调整预览字符数（默认 220）；完整正文在向量库中。
    """
    if show_content_chars is None:
        show_content_chars = _gold_preview_chars()
    chunks = await retrieve_baseline(query=query, k=k, research_topic=research_topic)
    print(f"\n=== RAG Eval Search: query = {_safe_stdout_str(query)!r} | top={k} | topic={research_topic} ===")
    for i, c in enumerate(chunks, 1):
        preview = (c.content or "").strip().replace("\u200b", "").replace("\r", " ").replace("\n", " ")
        if len(preview) > show_content_chars:
            preview = preview[:show_content_chars] + "..."
        first_url = c.source_urls[0] if c.source_urls else "N/A"
        print(f"\n[{i}] id={c.id}  score={c.score:.4f}")
        print(f"    url={first_url}")
        print(f"    preview={_safe_stdout_str(preview)}")


async def gold_label_candidates(
    queries: Optional[List[EvalQuery]] = None,
    path: Optional[str] = None,
    k: Optional[int] = None,
    research_topic: Optional[str] = "rag_eval",
    show_content_chars: Optional[int] = None,
) -> None:
    """
    人工金标「第一步」：对 jsonl 中每条 query 用 baseline 拉 Top-k 候选，打印 id/url/预览。

    用法：
      python -m src.evaluator.rag_eval_runner gold_candidates

    环境变量：
      RAG_EVAL_QUERIES_PATH — 评估 jsonl 路径（默认 data/rag_eval_queries.jsonl）
      RAG_EVAL_GOLD_CANDIDATES_K — 每条 query 打印条数（默认 10）

    将输出与当前 jsonl 中的 relevant_ids 对照，人工增删 UUID 后保存，再跑 eval_retrieval。

    若 jsonl 里写了 **gold_answer**，会一并打印，作为「应对题意概括」的**对照 rubric**（不是向量库检索出来的）。
    金标含义：选**能支撑该题答案的段落 chunk**，不是选「和 gold_answer 字面一致」的句子。

    预览仅为截断显示；完整 chunk 正文存于向量库。预览长度：环境变量 RAG_EVAL_GOLD_PREVIEW_CHARS（默认 220）。
    查看某 id 全文：`python -m src.evaluator.rag_eval_runner dump_chunk <uuid>`
    """
    if queries is None:
        queries = load_eval_queries(path)
    if k is None:
        k = int(os.getenv("RAG_EVAL_GOLD_CANDIDATES_K", "10"))
    if show_content_chars is None:
        show_content_chars = _gold_preview_chars()

    print(
        f"\n[RAG-EVAL] gold_candidates: {len(queries)} 条 query | top-{k} | topic={research_topic!r}\n"
        "对照下方候选，在 jsonl 中编辑 relevant_ids；需要换关键词时再用 search_chunks。\n"
        f"（preview 截断为前 {show_content_chars} 字符；看全文: dump_chunk <uuid>；可调 RAG_EVAL_GOLD_PREVIEW_CHARS）\n",
        flush=True,
    )
    print(
        "[RAG-EVAL] 重要（方法学）: 下列列表仅为 **baseline 向量检索** 的 Top-k，"
        "不是金标全集。若你只从这里勾选 relevant_ids，等于默认「所有正确答案都在 baseline 结果里」，"
        "会系统性低估 HyDE/MQE 等 **提高召回** 的方法（它们召回的新文档若不在你的金标集合里，Recall 不会涨）。\n"
        "公平做法：用 gold_answer + 知识判断，配合 search_chunks / dump_chunk 在全库中找应标为相关的 id；"
        "或运行 **gold_pool**（合并 baseline+HyDE+MQE 候选池，仍非全库）扩大候选面。\n",
        flush=True,
    )

    for qi, q in enumerate(queries, 1):
        print("\n" + "=" * 72)
        print(f"[{qi}/{len(queries)}] id={q.id!r}")
        print(f"query: {_safe_stdout_str(q.query)}")
        if (q.gold_answer or "").strip():
            print(f"参考答案 gold_answer（jsonl 对照用，非检索生成）: {_safe_stdout_str(q.gold_answer.strip())}")
        cur = q.relevant_ids or []
        print(f"当前 jsonl relevant_ids ({len(cur)} 个): {cur}")
        chunks = await retrieve_baseline(query=q.query, k=k, research_topic=research_topic)
        if not chunks:
            print("  (baseline 返回 0 条，请检查向量库与 research_topic)", flush=True)
            continue
        for i, c in enumerate(chunks, 1):
            preview = (c.content or "").strip().replace("\u200b", "").replace("\r", " ").replace("\n", " ")
            if len(preview) > show_content_chars:
                preview = preview[:show_content_chars] + "..."
            first_url = c.source_urls[0] if c.source_urls else "N/A"
            print(f"\n  [{i}] id={c.id}  score={c.score:.4f}")
            print(f"      url={first_url}")
            print(f"      preview={_safe_stdout_str(preview)}")
    print("\n" + "=" * 72 + "\n[RAG-EVAL] gold_candidates 结束。下一步：编辑 jsonl → baseline eval_retrieval。\n", flush=True)


async def gold_annotation_pool(
    queries: Optional[List[EvalQuery]] = None,
    path: Optional[str] = None,
    k: Optional[int] = None,
    research_topic: Optional[str] = "rag_eval",
    show_content_chars: Optional[int] = None,
) -> None:
    """
    人工金标「宽候选池」：对每条 query 并行跑 baseline / HyDE / MQE 各 Top-k，按 chunk id 去重合并后打印。

    比「仅 gold_candidates」更公平：HyDE/MQE 单独召回的 chunk 会出现在池里，便于写入 relevant_ids。
    仍非全库：若某文档三种方法都未进 Top-k，需用 search_chunks 等自行发现。

    用法：
      python -m src.evaluator.rag_eval_runner gold_pool

    环境变量：RAG_EVAL_GOLD_CANDIDATES_K（每路各取几条，默认 10）；需 LLM，比 gold_candidates 慢。
    """
    if queries is None:
        queries = load_eval_queries(path)
    if k is None:
        k = int(os.getenv("RAG_EVAL_GOLD_CANDIDATES_K", "10"))
    if show_content_chars is None:
        show_content_chars = _gold_preview_chars()

    print(
        f"\n[RAG-EVAL] gold_pool: {len(queries)} 条 query | 每路 top-{k} | baseline+HyDE+MQE 去重合并 | topic={research_topic!r}\n"
        "用于扩大金标候选面；仍应结合 gold_answer / 关键词检索补全 relevant_ids。\n",
        flush=True,
    )

    for qi, q in enumerate(queries, 1):
        print("\n" + "=" * 72)
        print(f"[{qi}/{len(queries)}] id={q.id!r}")
        print(f"query: {_safe_stdout_str(q.query)}")
        if (q.gold_answer or "").strip():
            print(f"参考答案 gold_answer: {_safe_stdout_str(q.gold_answer.strip())}")
        print(f"当前 jsonl relevant_ids: {q.relevant_ids or []}")

        bl, hy, mq = await asyncio.gather(
            retrieve_baseline(q.query, k=k, research_topic=research_topic),
            retrieve_hyde(q.query, k=k, research_topic=research_topic),
            retrieve_mqe(q.query, k=k, research_topic=research_topic),
        )
        by_id: Dict[str, Tuple[RetrievedChunk, Set[str]]] = {}
        for mode, lst in (("baseline", bl), ("hyde", hy), ("mqe", mq)):
            for c in lst:
                if c.id not in by_id:
                    by_id[c.id] = (c, {mode})
                else:
                    best, modes = by_id[c.id]
                    modes.add(mode)
                    if c.score > best.score:
                        by_id[c.id] = (c, modes)
                    else:
                        by_id[c.id] = (best, modes)

        merged = sorted(by_id.values(), key=lambda x: x[0].score, reverse=True)
        print(f"\n  合并后共 {len(merged)} 个不同 chunk id（三路去重）\n", flush=True)
        for j, (c, modes) in enumerate(merged, 1):
            preview = (c.content or "").strip().replace("\u200b", "").replace("\r", " ").replace("\n", " ")
            if len(preview) > show_content_chars:
                preview = preview[:show_content_chars] + "..."
            first_url = c.source_urls[0] if c.source_urls else "N/A"
            tag = ",".join(sorted(modes))
            print(f"\n  [{j}] id={c.id}  score={c.score:.4f}  from=[{tag}]")
            print(f"      url={first_url}")
            print(f"      preview={_safe_stdout_str(preview)}")
    print("\n" + "=" * 72 + "\n[RAG-EVAL] gold_pool 结束。\n", flush=True)


def _default_ai_domain_queries() -> List[Dict[str, Any]]:
    return [
        {
            "id": "aiq1",
            "query": "OpenAI 最近发布的模型中，哪一些更适合代码生成，为什么？",
            "relevant_ids": [],
            "gold_answer": "应比较近期模型在代码基准、工具调用、长上下文和成本上的差异，并给出适用场景。",
            "domain": "ai_news_paper",
        },
        {
            "id": "aiq2",
            "query": "DeepSeek 最新论文里关于推理效率优化的核心方法是什么？",
            "relevant_ids": [],
            "gold_answer": "应提炼论文中的关键优化策略、实验设置与局限。",
            "domain": "ai_news_paper",
        },
        {
            "id": "aiq3",
            "query": "多模态大模型近一年在视频理解方向有哪些代表性进展？",
            "relevant_ids": [],
            "gold_answer": "应覆盖代表模型、任务提升点与可复现证据来源。",
            "domain": "ai_news_paper",
        },
        {
            "id": "aiq4",
            "query": "2025 年以来 AI Agent 框架在企业落地上的主要瓶颈是什么？",
            "relevant_ids": [],
            "gold_answer": "应从稳定性、成本、评估与安全治理四个维度给出证据。",
            "domain": "ai_news_paper",
        },
        {
            "id": "aiq5",
            "query": "RAG 结合长上下文模型后，检索阶段是否仍然必要？",
            "relevant_ids": [],
            "gold_answer": "应比较无检索方案与检索增强方案在可追溯性、成本和准确率上的差异。",
            "domain": "ai_news_paper",
        },
        {
            "id": "aiq6",
            "query": "近期 AI 安全对齐论文对“幻觉”治理提出了哪些可操作方法？",
            "relevant_ids": [],
            "gold_answer": "应给出方法类别、实验结果和在真实应用中的约束条件。",
            "domain": "ai_news_paper",
        },
    ]


def _write_queries_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def export_gold_pool_jsonl(
    queries: Optional[List[EvalQuery]] = None,
    in_path: Optional[str] = None,
    out_path: Optional[str] = None,
    k: Optional[int] = None,
    research_topic: Optional[str] = "rag_eval",
) -> Dict[str, Any]:
    """
    将 baseline+HyDE+MQE 的候选池导出为 jsonl，供人工标注 relevant_ids 使用。
    """
    if queries is None:
        queries = load_eval_queries(in_path)
    if k is None:
        k = int(os.getenv("RAG_EVAL_GOLD_CANDIDATES_K", "10"))
    if out_path:
        out = Path(out_path)
        if not out.is_absolute():
            out = (_PROJECT_ROOT / out).resolve()
    else:
        out = (_PROJECT_ROOT / "data" / "rag_eval_gold_pool.jsonl").resolve()

    lines: List[str] = []
    for q in queries:
        bl, hy, mq = await asyncio.gather(
            _retrieve_baseline_output(q.query, k=k, research_topic=research_topic),
            _retrieve_hyde_output(q.query, k=k, research_topic=research_topic),
            _retrieve_mqe_output(q.query, k=k, research_topic=research_topic),
        )
        by_id: Dict[str, Dict[str, Any]] = {}
        for mode, outm in (("baseline", bl), ("hyde", hy), ("mqe", mq)):
            for c in outm.chunks:
                row = by_id.setdefault(
                    c.id,
                    {
                        "id": c.id,
                        "score": c.score,
                        "source_url": (c.source_urls[0] if c.source_urls else None),
                        "from_modes": [],
                        "preview": (c.content or "")[:220],
                    },
                )
                row["score"] = max(float(row["score"]), float(c.score))
                if mode not in row["from_modes"]:
                    row["from_modes"].append(mode)

        candidates = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
        item = {
            "id": q.id,
            "query": q.query,
            "gold_answer": q.gold_answer,
            "current_relevant_ids": q.relevant_ids,
            "candidates": candidates,
        }
        lines.append(json.dumps(item, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[RAG-EVAL] 已导出 gold_pool 候选 -> {out}", flush=True)
    return {"out_path": str(out), "query_count": len(queries), "k": k}


async def rebuild_ai_domain_gold(
    queries_out: Optional[str] = None,
    pool_out: Optional[str] = None,
    research_topic: Optional[str] = "rag_eval",
) -> Dict[str, Any]:
    """
    AI 资讯/论文解读方向：重建 query+gold 工作流。
    1) 生成 AI 领域 query 集（空 relevant_ids，待人工标注）
    2) 导出 gold_pool 候选（baseline+HyDE+MQE 合并）
    """
    if queries_out:
        qpath = Path(queries_out)
        if not qpath.is_absolute():
            qpath = (_PROJECT_ROOT / qpath).resolve()
    else:
        qpath = (_PROJECT_ROOT / "data" / "rag_eval_queries_ai_domain.jsonl").resolve()
    seed_rows = _default_ai_domain_queries()
    _write_queries_jsonl(qpath, seed_rows)
    print(f"[RAG-EVAL] AI 领域 query 集已生成 -> {qpath}", flush=True)
    qs = load_eval_queries(str(qpath))
    pool_result = await export_gold_pool_jsonl(
        queries=qs,
        in_path=str(qpath),
        out_path=pool_out or str((_PROJECT_ROOT / "data" / "rag_eval_ai_domain_gold_pool.jsonl").resolve()),
        research_topic=research_topic,
    )
    return {
        "queries_path": str(qpath),
        "pool_path": pool_result["out_path"],
        "query_count": len(qs),
    }


async def dump_chunk(
    chunk_id: str,
    max_chars: Optional[int] = None,
) -> None:
    """
    按 chunk id 从评估向量库打印**完整**正文（及 metadata），用于金标时读全段再决定。

    用法：
      python -m src.evaluator.rag_eval_runner dump_chunk <uuid>
    环境变量 RAG_EVAL_DUMP_CHUNK_MAX_CHARS 限制最大输出字符（默认 50000，防刷屏）。
    """
    if not (chunk_id or "").strip():
        print("[RAG-EVAL] dump_chunk: 请传入 chunk uuid")
        return
    if max_chars is None:
        try:
            max_chars = int(os.getenv("RAG_EVAL_DUMP_CHUNK_MAX_CHARS", "50000"))
        except ValueError:
            max_chars = 50000
    max_chars = max(1000, max_chars)

    store = _get_eval_memory_store()
    try:
        data = store.collection.get(ids=[chunk_id.strip()], include=["documents", "metadatas"])
    except Exception as e:
        print(f"[RAG-EVAL] dump_chunk 查询失败: {e}")
        return
    ids = (data or {}).get("ids") or []
    if not ids:
        print(f"[RAG-EVAL] dump_chunk: id 不在向量库中: {chunk_id!r}")
        return
    docs = (data or {}).get("documents") or []
    metas = (data or {}).get("metadatas") or []
    body = docs[0] if docs else ""
    meta = metas[0] if metas and isinstance(metas[0], dict) else {}
    if isinstance(body, str) and len(body) > max_chars:
        total = len(body)
        body = body[:max_chars] + f"\n\n... [截断，共 {total} 字符；增大 RAG_EVAL_DUMP_CHUNK_MAX_CHARS]"
    print(f"\n=== dump_chunk id={chunk_id} ===")
    print(f"metadata: {meta}")
    print("--- content ---")
    print(_safe_stdout_str(body) if body else "(空)")
    print("--- end ---\n", flush=True)


if __name__ == "__main__":
    """
    简单命令行入口：
      1) 构建索引:
         python -m src.evaluator.rag_eval_runner build_index
      2) 检索 demo:
         python -m src.evaluator.rag_eval_runner demo "你的问题"
      3) 评估检索指标（全量较慢；仅向量可设 set RAG_EVAL_MODES=baseline 后执行）:
         python -m src.evaluator.rag_eval_runner eval_retrieval
      4) 将 RAG（检索+回答）追踪记录到 LangSmith（需 LANGSMITH_API_KEY；模式同 RAG_EVAL_MODES）:
         python -m src.evaluator.rag_eval_runner eval_langsmith
      5) 搜索并打印候选 chunk（辅助标注 relevant_ids）:
         python -m src.evaluator.rag_eval_runner search_chunks "你的关键词"
      6) 重建索引后，用当前库的 baseline Top-k 刷新 jsonl 中的 relevant_ids（先备份 .bak）:
         python -m src.evaluator.rag_eval_runner refresh_eval_ids_union
         可选环境变量: RAG_EVAL_QUERIES_PATH, RAG_EVAL_REFRESH_K（默认 3）
      7) 人工金标：批量打印每条评估 query 的 baseline Top-k 候选（便于对照 jsonl 勾选/修改 relevant_ids）:
         python -m src.evaluator.rag_eval_runner gold_candidates
         可选: RAG_EVAL_GOLD_CANDIDATES_K（默认 10）；有 gold_answer 时会打印对照
      7b) 金标「宽候选池」：baseline+HyDE+MQE 三路各 Top-k 去重合并（更公平评估召回增强；需 LLM，较慢）:
         python -m src.evaluator.rag_eval_runner gold_pool
      8) 按 chunk id 打印向量库中全文（金标审阅用）:
         python -m src.evaluator.rag_eval_runner dump_chunk <uuid>
      9) 评估并自动导出可复现报告（json+md）:
         python -m src.evaluator.rag_eval_runner eval_report
      10) 导出可标注 gold_pool 候选（jsonl）:
         python -m src.evaluator.rag_eval_runner export_gold_pool
      11) 重建 AI资讯/论文解读方向 query+gold_pool:
         python -m src.evaluator.rag_eval_runner rebuild_ai_gold
    """
    import sys

    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
        if cmd == "build_index":
            build_rag_eval_index()
        elif cmd == "demo":
            q = "检索增强生成 RAG 的评估方法有哪些？"
            if len(sys.argv) > 2:
                q = " ".join(sys.argv[2:])
            asyncio.run(demo_query(q))
        elif cmd == "eval_retrieval":
            qs = load_eval_queries()
            asyncio.run(eval_retrieval(qs))
        elif cmd == "eval_report":
            qs = load_eval_queries()
            result = asyncio.run(eval_retrieval(qs))
            rp = result.get("report_paths", {})
            print(f"[RAG-EVAL] 报告已写出: json={rp.get('json_path')}", flush=True)
        elif cmd == "eval_langsmith":
            qs = load_eval_queries()
            asyncio.run(eval_with_langsmith(qs))
        elif cmd == "search_chunks":
            q = "RAG evaluation metrics"
            if len(sys.argv) > 2:
                q = " ".join(sys.argv[2:])
            asyncio.run(search_chunks(q))
        elif cmd == "refresh_eval_ids":
            print(
                "命令 refresh_eval_ids 已被移除（过去容易与 baseline_topk 回写混淆）。\n"
                "请改用：python -m src.evaluator.rag_eval_runner refresh_eval_ids_union\n",
                flush=True,
            )
        elif cmd == "refresh_eval_ids_union":
            k = int(os.getenv("RAG_EVAL_GOLD_CANDIDATES_K", "5"))
            asyncio.run(refresh_eval_queries_relevant_ids_union(per_mode_k=k))
        elif cmd == "gold_candidates":
            asyncio.run(gold_label_candidates())
        elif cmd == "gold_pool":
            asyncio.run(gold_annotation_pool())
        elif cmd == "export_gold_pool":
            asyncio.run(export_gold_pool_jsonl())
        elif cmd == "dump_chunk":
            cid = ""
            if len(sys.argv) >= 3:
                cid = sys.argv[2].strip()
            asyncio.run(dump_chunk(cid))
        elif cmd == "rebuild_ai_gold":
            out = asyncio.run(rebuild_ai_domain_gold())
            print(
                "[RAG-EVAL] AI 方向 gold 重建已完成:\n"
                f"  - queries: {out['queries_path']}\n"
                f"  - gold_pool: {out['pool_path']}\n"
                "下一步建议:\n"
                "  1) 设置 RAG_EVAL_QUERIES_PATH 指向该 queries 文件\n"
                "  2) 在 gold_pool 文件中人工标注 relevant_ids\n"
                "  3) 使用该 queries 文件运行 eval_report",
                flush=True,
            )
        else:
            print(f"未知命令: {cmd}")
    else:
        # 默认行为：跑一个 demo
        q = "检索增强生成 RAG 的评估方法有哪些？"
        asyncio.run(demo_query(q))


#!/usr/bin/env python3
"""
深度研究助手测试脚本。
用于验证整个异步研究助手工作流的完整功能。
"""
import asyncio
import logging
import sys

from src.state import GraphState
import argparse
from typing import List


def _setup_research_console_logging(level: int = logging.INFO) -> None:
    """
    将根日志输出到 stdout，便于在终端看到规划/执行/聚合等各节点 logger.info。
    第三方库（Chroma/HTTP）默认提到 WARNING，避免刷屏。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in (
        "chromadb",
        "chromadb.telemetry",
        "httpcore",
        "httpx",
        "openai",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="深度研究助手")
    parser.add_argument("research_topic", nargs="?", default=None, help="研究主题（可选；不填会提示输入）")
    parser.add_argument(
        "--upload",
        action="append",
        default=[],
        help="上传文档路径（文件或目录）。可重复传入多次，例如：--upload a.pdf --upload docs/",
    )
    parser.add_argument("--list-uploads", action="store_true", help="列出该研究主题下已上传的文档")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="输出 DEBUG 级日志（非常冗长，仅排障用）",
    )
    return parser


def _resolve_research_topic(raw_topic: str | None) -> str:
    if raw_topic and raw_topic.strip():
        return raw_topic.strip()

    default_topic = "agent应用的伦理问题"
    print(f"[提示] 未提供 research_topic，自动使用默认主题: {default_topic}")
    return default_topic


def _get_persist_directory_for_topic(research_topic: str) -> str:
    from hashlib import md5

    db_suffix = md5(research_topic.encode()).hexdigest()[:8]
    return f"./data/memory_db_{db_suffix}"


def _prepare_topic_memory_store(research_topic: str):
    from src.memory.memory_store import MemoryStore

    return MemoryStore(persist_directory=_get_persist_directory_for_topic(research_topic))


def _handle_uploads_and_listing(research_topic: str, upload_paths: List[str], list_uploads: bool) -> None:
    # 在研究开始前处理外部文件 → 嵌入 → 写入该主题的向量库
    memory_store = _prepare_topic_memory_store(research_topic)

    if list_uploads:
        from src.memory.file_processor import UploadedDocumentManager

        mgr = UploadedDocumentManager(memory_store=memory_store)
        docs = mgr.get_uploaded_documents(research_topic=research_topic)
        if not docs:
            print(f"[上传文档] 主题“{research_topic}”下暂无已上传文档。")
        else:
            print(f"[上传文档] 主题“{research_topic}”下已上传文档:")
            for d in docs:
                print(f"- {d.get('file_name')}  chunks={d.get('total_chunks')}  upload_time={d.get('upload_time')}")

    if upload_paths:
        from pathlib import Path
        from src.memory.file_processor import FileUploadProcessor

        processor = FileUploadProcessor(memory_store=memory_store)
        for raw_path in upload_paths:
            p = Path(raw_path)
            if p.is_dir():
                processor.process_directory(p, research_topic=research_topic, recursive=True)
            else:
                processor.process_uploaded_file(p, research_topic=research_topic)


async def main(research_topic: str, *, log_level: int = logging.INFO):
    """
    主测试函数：初始化状态，运行研究助手图，并打印结果。

    Args:
        research_topic: 要研究的话题。
        log_level: 根日志级别，默认 INFO；`--debug` 时为 DEBUG。
    """
    from src.tools.langsmith_env import try_enable_langsmith_for_research

    _setup_research_console_logging(level=log_level)
    logging.getLogger(__name__).info("控制台日志已启用（各节点将输出 INFO 级流程日志）")

    try_enable_langsmith_for_research()

    print("=" * 60)
    print("深度研究助手 - 工作流测试")
    print("=" * 60)
    print(f"研究主题: {research_topic}")
    print()

    # 1. 导入图应用（在函数内导入，避免循环依赖）
    from src.graph import app

    # 2. 初始化状态（根据GraphState定义必需字段）
    # 2. 初始化状态（根据 GraphState 定义必需字段）
    initial_state: GraphState = {
        "research_topic": research_topic,
        "sub_tasks": [],  # 将由规划节点填充
        "active_tasks": [],  # 将由执行节点填充
        "task_results": [],  # 将由聚合节点填充
        "need_deeper_research": False,
        "current_cycle": 1,  # 将从1开始计数
        "max_cycles": 3,  # 最大研究轮次（默认缩短；最大轮次仍有报告门控兜底）
        "final_report": None,  # 将由报告节点填充
        "messages": [],  # LangGraph消息历史
        "working_memory": None,  # 工作记忆实例，将由记忆初始化节点创建
        # 以下是新增的智能评估相关字段
        "task_quality_profiles": [],  # 任务质量画像列表，由聚合节点填充
        "deficiency_report": None,  # 轮次缺陷报告，由聚合节点填充
        "targeted_instructions": [],  # 定向研究指令，由聚合节点生成
        "last_cycle_score": None,  # 上一轮研究评分，由聚合节点填充
    }

    # 3. 运行图（异步调用）
    print("启动研究流程...")
    try:
        # 注意：app 是 LangGraph 编译后的可调用对象，支持异步 invoke
        # app.ainvoke()是异步调用方法，用于执行图工作流。功能：启动langgraph的工作流，异步运行所有节点，返回最终状态。
        final_state = await app.ainvoke(initial_state)
    except Exception as e:
        print(f"工作流执行失败: {e}")
        # 退出程序
        sys.exit(1)

    # 4. 输出关键结果
    print()
    print("=" * 60)
    print("执行摘要")
    print("=" * 60)
    print(f"研究主题: {final_state['research_topic']}")
    print(f"总研究轮次: {final_state['current_cycle']}")
    print(f"生成子任务数: {len(final_state['sub_tasks'])}")
    print(f"有效结果数: {len([r for r in final_state['task_results'] if r is not None])}")
    print()

    # 5. 打印最终报告
    if final_state["final_report"]:
        print("=" * 60)
        print("最终研究报告")
        print("=" * 60)
        try:
            # 在 GBK 控制台下安全输出，自动忽略无法编码的字符（如 emoji）
            safe_report = final_state["final_report"].encode("gbk", errors="ignore").decode("gbk", errors="ignore")
            print(safe_report)
        except Exception:
            # 回退：直接打印，必要时让异常抛出
            print(final_state["final_report"])
    else:
        print("未生成最终报告。")

    # 6. 可选：保存报告到文件
    try:
        with open(f"report_{research_topic[:20]}.md", "w", encoding="utf-8") as f:
            f.write(final_state["final_report"] or "# 报告生成失败")
        print(f"\n报告已保存至: report_{research_topic[:20]}.md")
    except:
        pass


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    topic = _resolve_research_topic(args.research_topic)
    log_level = logging.DEBUG if args.debug else logging.INFO

    # 在开始研究相关主题之前，先判断是否传入文件/是否需要列出上传文档
    _handle_uploads_and_listing(
        research_topic=topic,
        upload_paths=args.upload,
        list_uploads=args.list_uploads,
    )

    # 如果只是列出上传文档（未传入任何新文件），则不启动研究流程
    if args.list_uploads and not args.upload:
        sys.exit(0)

    # 运行异步主函数
    asyncio.run(main(topic, log_level=log_level))
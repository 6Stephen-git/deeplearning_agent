#!/usr/bin/env python3
"""
MemoryStore 功能验证脚本。
用于测试集成ChromaDB的记忆存储与向量检索功能是否正常工作。
"""
import asyncio
import sys
import os
from datetime import datetime

# 将项目根目录添加到Python路径，以便导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory.memory_store import MemoryStore, MemoryType, MemoryPriority

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
async def main():
    """主测试函数"""
    print("=" * 60)
    print("🧠 MemoryStore 功能验证测试")
    print("=" * 60)

    # 1. 初始化 MemoryStore
    # 使用一个独立的测试数据库路径，避免污染正式数据
    test_db_path = "./data/test_memory_db"
    print(f"[1/5] 初始化 MemoryStore，测试数据库位于: {test_db_path}")

    try:
        memory_store = MemoryStore(persist_directory=test_db_path)
        print("   ✅ MemoryStore 初始化成功")
    except Exception as e:
        print(f"   ❌ MemoryStore 初始化失败: {e}")
        return

    # 2. 添加测试记忆
    print("\n[2/5] 添加测试记忆...")

    # 定义一些测试记忆，涵盖不同类型和主题
    test_memories = [
        {
            "content": "深度学习模型，特别是Transformer架构，在自然语言处理任务上取得了突破性进展。",
            "type": MemoryType.FACT,
            "topic": "人工智能",
            "priority": MemoryPriority.HIGH,
            "source_urls": ["https://example.com/ai-progress"]
        },
        {
            "content": "使用梯度下降优化神经网络时，学习率的选择对训练稳定性和收敛速度至关重要。",
            "type": MemoryType.METHOD,
            "topic": "机器学习",
            "priority": MemoryPriority.CRITICAL,
            "source_urls": ["https://example.com/ai-progress"]
        },
        {
            "content": "LangGraph是一个基于状态图的框架，用于构建复杂、可靠的多智能体应用程序。",
            "type": MemoryType.FACT,
            "topic": "智能体开发",
            "priority": MemoryPriority.HIGH,
            "source_urls": ["https://example.com/ai-progress"]
        },
        {
            "content": "在构建检索增强生成系统时，将用户查询与向量数据库中的文档进行语义匹配是关键步骤。",
            "type": MemoryType.INSIGHT,
            "topic": "RAG系统",
            "priority": MemoryPriority.MEDIUM,
            "source_urls": ["https://example.com/ai-progress"]
        },
        {
            "content": "气候变化导致全球平均气温上升，极端天气事件频率增加。",
            "type": MemoryType.FACT,
            "topic": "气候变化",
            "priority": MemoryPriority.HIGH,
            "source_urls": ["https://example.com/ai-progress"]
        },
    ]

    added_ids = []
    for i, mem in enumerate(test_memories, 1):
        try:
            mem_id = memory_store.add_memory(
                content=mem["content"],
                memory_type=mem["type"],
                priority=mem["priority"],
                research_topic=mem["topic"],
                confidence=0.9,
                metadata={"test_batch": "validation_1"}
            )
            added_ids.append(mem_id)
            print(f"   添加记忆 {i}: {mem['content'][:50]}...")
        except Exception as e:
            print(f"   ❌ 添加记忆失败: {e}")

    print(f"   ✅ 成功添加 {len(added_ids)} 条测试记忆")

    # 3. 测试向量相似性检索
    print("\n[3/5] 测试向量相似性检索...")

    test_queries = [
        ("神经网络训练方法", "机器学习", None),  # 查询，主题过滤，类型过滤
        ("什么是LangGraph", "智能体开发", MemoryType.FACT),
        ("全球变暖的影响", "气候变化", None),
    ]

    for query, topic, mem_type in test_queries:
        print(f"\n   查询: '{query}' (主题: {topic})")
        results = memory_store.search_by_similarity(
            query=query,
            n_results=3,
            research_topic=topic,
            memory_type=mem_type
        )

        if not results:
            print("     ⚠️  未检索到相关记忆")
            continue

        for mem, score in results:
            print(f"    相关度 {score:.3f}: {mem.content[:60]}...")

    # 4. 测试混合检索
    print("\n[4/5] 测试混合检索 (语义+过滤)...")

    hybrid_results = memory_store.search_memories(
        query="人工智能模型",
        research_topic="人工智能",
        memory_type=MemoryType.FACT,
        limit=5,
        use_hybrid=True  # 启用混合模式
    )

    print(f"   混合检索到 {len(hybrid_results)} 条结果:")
    for i, mem in enumerate(hybrid_results, 1):
        print(f"   {i}. {mem.content[:70]}...")
        print(f"      类型: {mem.memory_type.value}, 主题: {mem.metadata.get('research_topic')}")

    # 5. 检查统计信息
    print("\n[5/5] 检查记忆库统计...")
    stats = memory_store.get_stats(detail=True)

    print(f"   记忆总数: {stats['total_memories']}")
    print(f"   按类型分布:")
    for mem_type, count in stats['by_type'].items():
        if count > 0:
            print(f"     - {mem_type}: {count}")

    print(f"   按主题分布:")
    for topic, count in stats.get('by_topic', {}).items():
        if count > 0:
            print(f"     - {topic}: {count}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

    # 提示清理（可选）
    print("\n提示: 测试数据库已保存至:", test_db_path)
    print("如需清理，可手动删除该目录。")


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())
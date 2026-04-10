#!/usr/bin/env python3
"""
简化版 MemoryStore 功能测试（Windows兼容版）。
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory.memory_store import MemoryStore, MemoryType, MemoryPriority


def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试 MemoryStore 基本功能")

    # 创建一个临时目录（Windows兼容）
    temp_dir = tempfile.mkdtemp(prefix="memory_test_")
    print(f"   使用临时目录: {temp_dir}")

    try:
        # 使用临时目录，而不是:memory:
        memory_store = MemoryStore(persist_directory=temp_dir)

        # 测试1: 添加记忆
        print("\n1. 测试添加记忆...")
        mem_id = memory_store.add_memory(
            content="这是一个测试记忆",
            memory_type=MemoryType.FACT,
            priority=MemoryPriority.MEDIUM,
            research_topic="测试",
            source_urls=["https://test.com"]  # 非空列表
        )
        print(f"   添加成功，记忆ID: {mem_id[:8]}")

        # 测试2: 添加不带source_urls的记忆（验证修复）
        print("\n2. 测试添加不带source_urls的记忆...")
        mem_id2 = memory_store.add_memory(
            content="这是另一个测试记忆，没有source_urls",
            memory_type=MemoryType.INSIGHT,
            priority=MemoryPriority.HIGH,
            research_topic="测试"
            # 注意：不传递source_urls参数
        )
        print(f"   添加成功，记忆ID: {mem_id2[:8]}")

        # 测试3: 检索记忆
        print("\n3. 测试检索记忆...")
        memories = memory_store.search_memories(
            query="测试记忆",
            limit=5
        )
        print(f"   检索到 {len(memories)} 条记忆")
        for i, mem in enumerate(memories, 1):
            print(f"   {i}. {mem.content[:50]}...")
            print(f"      类型: {mem.memory_type.value}, 优先级: {mem.priority.value}")

        # 测试4: 统计信息
        print("\n4. 测试统计信息...")
        stats = memory_store.get_stats()
        print(f"   记忆总数: {stats['total_memories']}")

        return len(memories) > 0

    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"\n   已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"\n   警告: 清理临时目录失败: {e}")


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if success:
            print("\n✅ 测试通过！")
            sys.exit(0)
        else:
            print("\n❌ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
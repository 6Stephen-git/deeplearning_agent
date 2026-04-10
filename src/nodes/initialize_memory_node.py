# src/nodes/initialize_memory_node.py
"""
记忆初始化节点。
职责：在研究开始时，创建并初始化工作记忆系统。
这是工作流的新入口点，确保所有后续节点都能访问工作记忆。
"""
import logging
from src.state import GraphState
from src.memory.memory_store import MemoryStore
from src.memory.working_memory import WorkingMemory
from src.memory.topic_registry import register_topic, topic_to_db_path

logger = logging.getLogger(__name__)


async def initialize_memory_node(state: GraphState) -> GraphState:
    """
    记忆初始化节点主函数。

    1. 创建MemoryStore实例（连接到持久化向量数据库）
    2. 创建WorkingMemory实例（绑定到当前研究会话）
    3. 将工作记忆实例存入图状态，供后续节点使用

    Args:
        state: 图状态，必须包含research_topic。

    Returns:
        更新后的状态，新增working_memory字段。
    """
    research_topic = state["research_topic"]
    logger.info(f"[记忆初始化] 开始初始化工作记忆系统，研究主题: {research_topic}")

    try:
        # 1. 创建记忆存储（长期记忆）
        # 使用研究主题的哈希作为数据库路径的一部分，实现主题隔离
        # 计算研究主题的MD5哈希值，并取前8个字符，得到一个简短、唯一且确定的字符串（db_suffix）
        # 实现主题级的数据隔离。不同研究主题的记忆会存储在不同路径的独立数据库中。
        reg = register_topic(research_topic)
        db_suffix = reg["db_suffix"]
        memory_store = MemoryStore(
            persist_directory=topic_to_db_path(research_topic)
        )

        # 2. 创建工作记忆（短期记忆）
        # 使用时间戳+主题哈希作为会话ID，确保唯一性
        from datetime import datetime
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{db_suffix}"

        working_memory = WorkingMemory(
            session_id=session_id,
            # 将上面创建的memory_store实例作为参数传入WorkingMemory。
            # 这使得工作记忆对象获得了访问和操作对应主题的长期记忆库的能力，打通了短期上下文与长期知识之间的通道。
            memory_store=memory_store,
            # 设置工作记忆的上下文缓冲区最大容量为15项。这是一个经验值，旨在平衡信息丰富性与上下文窗口限制。
            # 当缓冲区满时，最旧的上下文项会被移除（FIFO），模拟人类工作记忆的有限性，并防止输入给LLM的提示词无限膨胀。
            max_context_items=15
        )

        # 3. 将用户查询作为初始上下文
        working_memory.add_context(
            item_type="user_query",
            content=f"研究主题: {research_topic}",
            metadata={"research_topic": research_topic}
        )

        # 4. 检索相关历史记忆
        logger.info(f"[记忆初始化] 检索与主题相关的历史记忆...")
        related_memories = await working_memory.retrieve_relevant_memories(
            current_topic=research_topic,
            strategy="hybrid",
            limit=3
        )

        # 5. 更新图状态
        state["working_memory"] = working_memory

        # 记录初始化统计
        memory_stats = memory_store.get_stats()
        logger.info(
            f"[记忆初始化] 完成。会话ID: {session_id}, "
            f"长期记忆库大小: {memory_stats['total_memories']}, "
            f"检索到相关记忆: {len(related_memories)}条"
        )

        return state

    except Exception as e:
        logger.error(f"[记忆初始化] 失败: {e}")
        # 优雅降级：创建最小化的工作记忆
        state["working_memory"] = None
        return state

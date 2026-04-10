#!/usr/bin/env python3
"""
嵌入工具功能验证测试。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory.embedding_tool import EmbeddingTool


def main():
    """主测试函数"""
    print("🔧 测试嵌入工具功能")
    print("-" * 40)

    # 1. 初始化工具
    print("1. 初始化嵌入工具...")
    tool = EmbeddingTool()
    print(f"   提供商: {tool.provider}")
    print(f"   模型: {tool.model_name}")
    print(f"   维度: {tool.embedding_dim}")

    # 2. 测试单条文本
    print("\n2. 测试单条文本嵌入...")
    test_text = "深度学习是人工智能的一个重要分支"
    embedding = tool.generate_embedding(test_text)
    print(f"   文本: '{test_text}'")
    print(f"   向量长度: {len(embedding)}")
    print(f"   前5维: {embedding[:5]}")

    # 3. 测试批量文本
    print("\n3. 测试批量文本嵌入...")
    test_texts = [
        "自然语言处理研究计算机和人类语言之间的交互",
        "机器学习使计算机能够从数据中学习而无需明确编程",
        "计算机视觉使机器能够从图像和视频中获取信息"
    ]
    embeddings = tool.generate_embeddings_batch(test_texts)
    print(f"   批量数量: {len(embeddings)}")
    for i, (text, emb) in enumerate(zip(test_texts, embeddings), 1):
        print(f"   {i}. '{text[:20]}...' -> 向量长度: {len(emb)}")

    # 4. 验证向量质量
    print("\n4. 验证向量质量...")
    if tool.provider == "dashscope":
        # 计算相似度（余弦相似度）
        import numpy as np
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"   相似度(0-1,2): {similarity:.3f}")

        if similarity > 0.1:  # DashScope嵌入应该有合理相似度
            print("   ✅ 嵌入质量正常")
        else:
            print("   ⚠️  嵌入相似度过低，可能有问题")
    else:
        print("   ℹ️  使用哈希嵌入，无语义相似度")

    print("\n" + "=" * 40)
    print("✅ 嵌入工具测试完成")


if __name__ == "__main__":
    main()
# src/evaluator/schemas.py
"""
评估器模块的数据结构定义。
包含任务质量画像、指标分数等核心数据模型。
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MetricScore(BaseModel):
    """单一质量指标的分数与解释"""
    name: str = Field(..., description="指标名称，如'来源多样性'")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="0-100的评分")
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="该指标在综合评分中的权重")
    evidence: Optional[str] = Field(default=None, description="评分依据或证据摘要")
    # 示例: name="来源多样性", score=40.0, weight=0.2, evidence="仅引用自'arxiv.org'单一域名"

    def __str__(self) -> str:
        return f"{self.name}: {self.score:.1f} (权重:{self.weight:.2f})"


class TaskQualityProfile(BaseModel):
    """
    单个研究子任务的质量画像。
    与 `task_results` 列表中的元素一一对应。
    """
    # 基础标识
    task_id: int = Field(..., description="对应的任务ID")
    research_cycle: int = Field(..., description="所属的研究轮次")

    # 核心质量指标 (可配置、可扩展)
    metrics: Dict[str, MetricScore] = Field(
        default_factory=dict,
        description="质量指标字典，key为指标名，value为MetricScore对象"
    )

    # 智能分析标签
    tags: List[str] = Field(
        default_factory=list,
        description="从内容中分析出的标签，如'存在观点冲突'、'来源权威性高'、'信息新颖'"
    )

    # 关键发现摘要 (用于生成缺陷报告)
    key_findings: List[str] = Field(
        default_factory=list,
        description="从评估中提取的关键发现，如'观点冲突焦点：A方法与B方法的效率对比'"
    )

    # 综合评分
    composite_score: float = Field(default=0.0, ge=0.0, le=100.0)

    # 决策建议
    storage_suggestion: str = Field(
        default="PENDING",
        description="存储建议: IMMEDIATE(立即存储)/VERIFY(需核实)/REJECT(丢弃)"
    )
    next_research_suggestion: Optional[str] = Field(
        default=None,
        description="对此任务的下一步研究建议，如'探索更多学术机构(.edu)来源'"
    )

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    diagnostic_version: str = Field(default="1.0.0")

    def calculate_composite(self) -> float:
        """计算加权综合评分"""
        if not self.metrics:
            self.composite_score = 0.0
            return 0.0

        total_weight = sum(metric.weight for metric in self.metrics.values())
        if total_weight <= 0:
            self.composite_score = 0.0
            return 0.0

        weighted_sum = sum(metric.score * metric.weight for metric in self.metrics.values())
        self.composite_score = weighted_sum / total_weight
        return self.composite_score

    def get_metric_score(self, metric_name: str) -> float:
        """安全获取指定指标的分数，不存在则返回0"""
        return self.metrics.get(metric_name, MetricScore(name=metric_name, score=0.0, weight=0.0)).score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于存储和传输"""
        return {
            "task_id": self.task_id,
            "research_cycle": self.research_cycle,
            "composite_score": self.composite_score,
            "metrics": {name: {"score": m.score, "weight": m.weight, "evidence": m.evidence}
                       for name, m in self.metrics.items()},
            "tags": self.tags,
            "key_findings": self.key_findings,
            "storage_suggestion": self.storage_suggestion,
            "next_research_suggestion": self.next_research_suggestion
        }


class CycleDeficiencyReport(BaseModel):
    """
    研究轮次缺陷报告。
    由 ResearchCycleDiagnoser 生成，汇总本轮所有任务的问题。
    """
    cycle_number: int = Field(..., description="报告对应的研究轮次")
    generated_at: datetime = Field(default_factory=datetime.now)

    # 任务分类
    task_categories: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="按问题分类的任务ID字典，键为类别名，值为任务ID列表"
    )

    # 报告文本
    report_text: str = Field(default="", description="人类可读的缺陷报告")

    # 轮次统计
    average_composite_score: float = Field(default=0.0)
    total_tasks: int = Field(default=0)
    valid_tasks: int = Field(default=0)

    # 全局决策建议
    global_continue_suggestion: bool = Field(default=False, description="是否建议继续下一轮研究")
    primary_deficiency: str = Field(default="", description="本轮最主要缺陷类别")

    def get_tasks_by_category(self, category: str) -> List[int]:
        """获取指定类别的任务ID列表"""
        return self.task_categories.get(category, [])

    def has_deficiency(self) -> bool:
        """判断本轮是否存在需要改进的缺陷（非high_quality类别有任务）"""
        deficient_categories = set(self.task_categories.keys()) - {"high_quality", "invalid"}
        for category in deficient_categories:
            if self.task_categories.get(category):
                return True
        return False
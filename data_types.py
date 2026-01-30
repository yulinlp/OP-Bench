"""
OverPersonalized数据类型定义

定义了评测框架中使用的所有核心数据类型。

作者: ylhu
版本: 2.0.0
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TestCase:
    """单个测试用例
    
    表示一个完整的测试用例，包含问题、任务类型等信息。
    
    字段:
        test_id: 测试用例唯一标识
        task_type: 任务类型（irrelevance_easy, irrelevance_hard, sycophancy, diversity）
        person_name: 测试对象（用户）名称
        questions: 问题列表
        topic: 主题（可选）
        context: 对话上下文（可选）
        metadata: 元数据（可选）
    """
    test_id: str
    task_type: str  # irrelevance_easy, irrelevance_hard, sycophancy, diversity
    person_name: str
    questions: List[str]
    topic: Optional[str] = None
    context: Optional[List[Dict[str, str]]] = None  # 多轮对话上下文
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    test_case: TestCase
    responses: List[str]  # agent对每个问题的回答
    timestamps: List[str]  # 每次调用的时间戳
    latencies: List[float]  # 每次调用的延迟（秒）
    error_messages: List[Optional[str]]  # 错误信息（如果有）
    raw_responses: List[Optional[Dict[str, Any]]]  # 原始API响应
    scores: Optional[Dict[str, float]] = None  # 评分结果


@dataclass
class EvaluationConfig:
    """评测配置"""
    agent_api_base: str = "https://api.openai.com/v1"
    agent_api_key: str = ""
    agent_model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    max_workers: int = 5  # 生成阶段的并发数
    scoring_workers: Optional[int] = None  # 评分阶段的并发数，None表示使用max_workers
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    use_both_personas: bool = False  # 是否使用每组中的两个角色（默认仅第一个）
    agent_port_mapping: Optional[Dict[str, str]] = None  # 角色到端口的映射

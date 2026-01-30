"""
OverPersonalized评测框架

该模块提供了用于评测大语言模型过度个性化行为的统一框架，支持：
- 生成模型回复 (generate)
- 评分模型回复 (score)

评测任务类型：
- irrelevance_easy: 简单无关性测试
- irrelevance_hard: 复杂无关性测试
- sycophancy: 谄媚性测试（支持fact、value、memory三种子类型）
- diversity: 多样性测试

进度显示：
- 按任务类型分别显示进度条
- 支持串行和并行执行模式
- 显示详细的统计信息

"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime
import yaml
import openai
import requests
from tqdm import tqdm

# 导入数据类型
from data_types import TestCase, TestResult, EvaluationConfig


class Evaluator:
    """过度个性化评测器
    
    该类提供了完整的评测流程：
    1. 从benchmark数据加载测试用例
    2. 使用OpenAI格式API调用被测agent
    3. 存储测试结果和响应数据
    4. 预留评分系统接口
    
    Attributes:
        config: 评测配置
        benchmark_path: benchmark数据文件路径
        results_dir: 结果存储目录
        scorer: 评分器（预留接口）
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        benchmark_path: str = "./data/locomo10_overpersonalized.json",
        results_dir: str = "./results",
        scorer: Optional[Callable] = None
    ):
        """初始化评测器
        
        Args:
            config: 评测配置
            benchmark_path: benchmark数据文件路径
            results_dir: 结果存储目录
            scorer: 评分器函数（预留）
        """
        self.config = config
        self.benchmark_path = benchmark_path
        self.results_dir = results_dir
        self.scorer = scorer
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=config.agent_api_key,
            base_url=config.agent_api_base
        )
        
        # 加载benchmark数据
        self.benchmark_data = self._load_benchmark_data()
        
    def _load_benchmark_data(self) -> List[Dict[str, Any]]:
        """加载benchmark数据"""
        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载benchmark数据失败: {e}")
            return []
    
    def extract_test_cases(self, task_types: Optional[List[str]] = None) -> List[TestCase]:
        """从benchmark数据中提取测试用例
        
        Args:
            task_types: 要提取的任务类型列表，None表示提取所有类型
            
        Returns:
            List[TestCase]: 测试用例列表
        """
        if task_types is None:
            task_types = ["irrelevance_easy", "irrelevance_hard", "sycophancy", "diversity"]
            
        test_cases = []
        
        for person_data in self.benchmark_data:
            # person_data 形如 {"A": {...}, "B": {...}}
            # 根据配置选择人物：默认仅使用第一个；若 use_both_personas=True 则全用
            items = list(person_data.items())
            if not getattr(self.config, "use_both_personas", False) and items:
                items = [items[0]]
            for person_name, person_info in items:
                tasks = person_info.get("tasks", {})
                
                for task_type in task_types:
                    if task_type not in tasks:
                        continue
                        
                    task_data = tasks[task_type]
                    
                    if task_type == "irrelevance_easy":
                        # 新格式：直接是问题列表，不再按topic分组
                        questions = []
                        for item in task_data:
                            if isinstance(item, dict) and "question" in item:
                                questions.append(item["question"])
                            elif isinstance(item, str):
                                questions.append(item)
                        
                        if questions:
                            test_id = f"{person_name}_{task_type}"
                            test_case = TestCase(
                                test_id=test_id,
                                task_type=task_type,
                                person_name=person_name,
                                questions=questions,
                                metadata={
                                    "profile": person_info.get("profile", ""),
                                    "topics": person_info.get("topics", [])
                                }
                            )
                            test_cases.append(test_case)
                    elif task_type == "irrelevance_hard":
                        questions = [item["question"] for item in task_data]
                        # 单轮QA任务
                        test_id = f"{person_name}_{task_type}"
                        test_case = TestCase(
                            test_id=test_id,
                            task_type=task_type,
                            person_name=person_name,
                            questions=questions,
                            metadata={
                                "profile": person_info.get("profile", ""),
                                "topics": person_info.get("topics", [])
                            }
                        )
                        test_cases.append(test_case)
                    elif task_type == "sycophancy":
                        # 谄媚任务：问题列表，每个问题有type和explanation
                        questions = [item["question"] for item in task_data if "question" in item]
                        if questions:
                            test_id = f"{person_name}_{task_type}"
                            test_case = TestCase(
                                test_id=test_id,
                                task_type=task_type,
                                person_name=person_name,
                                questions=questions,
                                metadata={
                                    "profile": person_info.get("profile", ""),
                                    "topics": person_info.get("topics", []),
                                    "observations": person_info.get("observation", [])
                                }
                            )
                            test_cases.append(test_case)
                    elif task_type == "diversity":
                        # 多样性任务
                        # for idx, topic_item in enumerate(task_data):
                        #     topic = topic_item.get("topic", "")
                        questions_raw = task_data[0].get("questions", [])
                        
                        # 处理questions字段，可能是字符串数组或对象数组
                        questions = []
                        for q in questions_raw:
                            if isinstance(q, str):
                                questions.append(q)
                            elif isinstance(q, dict) and "question" in q:
                                questions.append(q["question"])
                            else:
                                # 如果无法解析，跳过这个问题
                                continue
                        
                        test_id = f"{person_name}_{task_type}"
                        test_case = TestCase(
                            test_id=test_id,
                            task_type=task_type,
                            person_name=person_name,
                            topic="",
                            questions=questions,
                            metadata={
                                "profile": person_info.get("profile", ""),
                                "topics": person_info.get("topics", [])
                            }
                        )
                        test_cases.append(test_case)
        return test_cases
    
    def _call_agent_api(self, messages: List[Dict[str, str]], person_name: str = None) -> Tuple[str, float, Optional[str], Optional[Dict[str, Any]]]:
        """调用agent API
        
        Args:
            messages: OpenAI格式的消息列表
            person_name: 角色名字，用于LDAgent模型端口切换
            
        Returns:
            Tuple[str, float, Optional[str], Optional[Dict]]: (响应内容, 延迟, 错误信息, 原始响应)
        """
        start_time = time.time()
        
        # 根据角色名字动态切换端口（仅对LDAgent开头的配置档案）
        client = self.client
        if (hasattr(self.config, 'agent_port_mapping') and 
            self.config.agent_port_mapping and 
            person_name):
            
            # 从配置中获取角色对应的端口
            port = self.config.agent_port_mapping.get(person_name)
            if port:
                # 构建新的API base URL
                base_url = self.config.agent_api_base
                # 使用正则表达式或简单的字符串处理来替换端口号
                import re
                # 匹配 http://host:port 或 https://host:port 的模式
                pattern = r'^(https?://[^:]+):\d+(.*)$'
                match = re.match(pattern, base_url)
                if match:
                    # 替换端口号，保留路径部分
                    base_url = match.group(1) + ":" + port + match.group(2)
                else:
                    # 如果没有匹配到端口号模式，直接添加端口
                    base_url = base_url.rstrip("/") + ":" + port
                
                # 创建新的客户端实例
                client = openai.OpenAI(
                    api_key=self.config.agent_api_key,
                    base_url=base_url
                )
                print(f"🔄 为角色 {person_name} 切换到端口 {port} (API Base: {base_url})")
        
        for attempt in range(self.config.retry_attempts):
            try:
                # 显示API调用信息
                if attempt == 0:  # 只在第一次尝试时显示
                    print(f"📡 调用API: 模型={self.config.agent_model}, 消息数={len(messages)}")
                    if person_name:
                        print(f"👤 角色: {person_name}")
                
                response = client.chat.completions.create(
                    model=self.config.agent_model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                    extra_body={"enable_thinking": False} 
                )
                
                latency = time.time() - start_time
                content = response.choices[0].message.content
                
                # 显示成功信息
                if attempt > 0:  # 只在重试成功时显示
                    print(f"✅ 重试成功! 延迟: {latency:.2f}秒")
                elif attempt == 0:  # 只在第一次成功时显示
                    print(f"✅ API调用成功! 延迟: {latency:.2f}秒")
                
                # 转换为可序列化的dict
                raw_response = {
                    "id": response.id,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ]
                }
                
                return content, latency, None, raw_response
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = f"API调用失败 (尝试 {attempt + 1}/{self.config.retry_attempts}): {error_type}: {str(e)}"
                print(f"❌ {error_msg}")
                
                # 显示更详细的错误信息
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json() if hasattr(e.response, 'json') else str(e.response.text)
                        print(f"🔍 错误详情: {error_detail}")
                    except:
                        print(f"🔍 响应状态码: {e.response.status_code if hasattr(e.response, 'status_code') else '未知'}")
                
                if attempt == self.config.retry_attempts - 1:
                    # 最后一次尝试失败
                    latency = time.time() - start_time
                    print(f"💥 所有重试失败，放弃调用")
                    return "", latency, error_msg, None
                else:
                    # 重试前等待
                    print(f"⏳ 等待 {self.config.retry_delay * (attempt + 1):.1f} 秒后重试...")
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # 理论上不会到达这里
        latency = time.time() - start_time
        return "", latency, "未知错误", None
    
    def _execute_single_test(self, test_case: TestCase) -> TestResult:
        """执行单个测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            TestResult: 测试结果
        """
        responses = []
        timestamps = []
        latencies = []
        error_messages = []
        raw_responses = []
        
        # 准备基础上下文
        base_messages = []
        if test_case.context:
            base_messages = test_case.context.copy()
        
        # 对每个问题进行测试
        for question in test_case.questions:
            timestamp = datetime.now().isoformat()
            timestamps.append(timestamp)
            
            # 构建消息
            messages = base_messages.copy()
            messages.append({"role": "user", "content": question})
            
            # 调用API
            response, latency, error, raw_response = self._call_agent_api(messages, test_case.person_name)
            
            responses.append(response)
            latencies.append(latency)
            error_messages.append(error)
            raw_responses.append(raw_response)
            
            # 如果是多轮对话，将assistant的回答添加到上下文中
            if test_case.context and response:
                base_messages.append({"role": "user", "content": question})
                base_messages.append({"role": "assistant", "content": response})
        
        return TestResult(
            test_id=test_case.test_id,
            test_case=test_case,
            responses=responses,
            timestamps=timestamps,
            latencies=latencies,
            error_messages=error_messages,
            raw_responses=raw_responses
        )
    
    def generate(
        self,
        task_id: str,
        task_types: Optional[List[str]] = None,
        max_test_cases: Optional[int] = None,
        save_intermediate: bool = False
    ) -> str:
        """生成模型回复并保存完整内容
        
        Args:
            task_id: 任务ID，用于文件命名
            task_types: 要评测的任务类型列表
            max_test_cases: 最大测试用例数量（用于调试）
            save_intermediate: 是否保存中间结果
            
        Returns:
            str: 保存的结果文件路径
        """
        print(f"=== 开始生成模型回复 (任务ID: {task_id}) ===")
        print("正在提取测试用例...")
        test_cases = self.extract_test_cases(task_types)
        
        if max_test_cases:
            test_cases = test_cases[:max_test_cases]
            
        print(f"共提取到 {len(test_cases)} 个测试用例")
        
        # 执行生成
        results = self._execute_tests(test_cases, save_intermediate, apply_scoring=False)
        
        # 保存完整的生成结果
        filepath = self._save_generation_results(results, task_id)
        
        print(f"✅ 生成完成！结果已保存到: {filepath}")
        return filepath
    
    def score(
        self,
        task_id: str,
        generation_file: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        max_test_cases: Optional[int] = None
    ) -> str:
        """对生成的结果进行评分
        
        Args:
            task_id: 任务ID，用于文件命名
            generation_file: 生成结果文件路径，如果None则自动找最新的
            task_types: 任务类型筛选
            max_test_cases: 最大测试用例数
            
        Returns:
            str: 保存的评分文件路径
        """
        print(f"=== 开始评分 (任务ID: {task_id}) ===")
        
        if not self.scorer:
            raise ValueError("错误: 未设置评分器，无法进行评分")
        
        # 自动找到生成文件
        if not generation_file:
            import glob
            pattern = os.path.join(self.results_dir, "generation_*.json")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError("错误: 未找到生成结果文件，请先调用generate()方法")
            generation_file = max(files, key=os.path.getmtime)
            print(f"自动选择最新生成文件: {os.path.basename(generation_file)}")
        
        # 加载生成结果
        results = self.load_results(generation_file)
        print(f"成功加载 {len(results)} 个测试结果")
        
        # 应用筛选
        if task_types:
            results = [r for r in results if r.test_case.task_type in task_types]
            print(f"筛选后剩余 {len(results)} 个测试结果")
        
        if max_test_cases:
            results = results[:max_test_cases]
            print(f"限制数量后处理 {len(results)} 个测试结果")
        
        # 应用评分
        self.apply_scoring_to_results(results)
        
        # 保存评分结果
        filepath = self._save_score_results(results, task_id)
        
        print(f"✅ 评分完成！结果已保存到: {filepath}")
        return filepath
    
        
    def _execute_tests(
        self,
        test_cases: List[TestCase],
        save_intermediate: bool = True,
        apply_scoring: bool = True
    ) -> List[TestResult]:
        """执行测试用例"""
        # 生成时间戳标识
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 按任务类型分组测试用例
        task_groups = {}
        for test_case in test_cases:
            task_type = test_case.task_type
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(test_case)
        
        # 显示任务分组信息
        print(f"📋 任务分组统计:")
        for task_type, cases in task_groups.items():
            print(f"  {task_type}: {len(cases)} 个测试用例")
        print()
        
        # 批量执行测试
        all_results = []
        
        if self.config.max_workers == 1:
            # 串行执行 - 按任务类型分别显示进度
            for task_type, task_cases in task_groups.items():
                print(f"🔄 执行 {task_type} 任务...")
                for test_case in tqdm(task_cases, desc=f"  {task_type}"):
                    result = self._execute_single_test(test_case)
                    all_results.append(result)
                    
                    # 中间保存
                    if save_intermediate and len(all_results) % 10 == 0:
                        self._save_intermediate_results(all_results, timestamp)
                print(f"✅ {task_type} 任务完成 ({len(task_cases)} 个用例)")
                print()
        else:
            # 并行执行 - 按任务类型分别显示进度
            print(f"使用 {self.config.max_workers} 个线程并行执行...")
            
            for task_type, task_cases in task_groups.items():
                print(f"🔄 执行 {task_type} 任务...")
                
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    # 提交当前任务类型的所有任务
                    future_to_test = {
                        executor.submit(self._execute_single_test, test_case): test_case 
                        for test_case in task_cases
                    }
                    
                    # 收集当前任务类型的结果
                    task_results = []
                    with tqdm(total=len(task_cases), desc=f"  {task_type}") as pbar:
                        for future in as_completed(future_to_test):
                            result = future.result()
                            task_results.append(result)
                            all_results.append(result)
                            pbar.update(1)
                            
                            # 中间保存
                            if save_intermediate and len(all_results) % 20 == 0:
                                self._save_intermediate_results(all_results, timestamp)
                
                print(f"✅ {task_type} 任务完成 ({len(task_cases)} 个用例)")
                print()
        
        # 评分（如果需要并且提供了评分器）
        if apply_scoring and self.scorer:
            print("🎯 正在进行评分...")
            self._apply_scoring_with_progress(all_results, task_groups)
        
        # 注意：保存结果现在由调用方负责，这里不再保存
        
        return all_results
    
    def _save_intermediate_results(self, results: List[TestResult], timestamp: str) -> None:
        """保存中间结果"""
        try:
            filename = f"intermediate_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            self._save_results_to_file(results, filepath)
            print(f"💾 中间结果已保存: {len(results)} 个测试用例")
        except Exception as e:
            print(f"⚠️  保存中间结果失败: {e}")
    
    def _save_results_to_file(self, results: List[TestResult], filepath: str) -> None:
        """将结果保存到文件"""
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            serializable_result = {
                "test_id": result.test_id,
                "test_case": {
                    "test_id": result.test_case.test_id,
                    "task_type": result.test_case.task_type,
                    "person_name": result.test_case.person_name,
                    "topic": result.test_case.topic,
                    "questions": result.test_case.questions,
                    "context": result.test_case.context,
                    "metadata": result.test_case.metadata
                },
                "responses": result.responses,
                "timestamps": result.timestamps,
                "latencies": result.latencies,
                "error_messages": result.error_messages,
                "raw_responses": result.raw_responses,
                "scores": result.scores
            }
            serializable_results.append(serializable_result)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def apply_scoring_to_results(self, results: List[TestResult]) -> None:
        """对已有结果应用评分器"""
        if not self.scorer:
            print("警告: 未设置评分器")
            return
            
        print(f"正在对 {len(results)} 个测试结果进行评分...")
        
        # 确定评分并发数
        scoring_workers = self.config.scoring_workers if self.config.scoring_workers is not None else self.config.max_workers
        
        if scoring_workers == 1:
            # 串行评分
            print("使用串行评分模式")
            self._apply_scoring_serial(results)
        else:
            # 并行评分
            print(f"使用并行评分模式，并发数: {scoring_workers}")
            self._apply_scoring_parallel(results, scoring_workers)
                
        print("评分完成！")
    
    def _apply_scoring_serial(self, results: List[TestResult]) -> None:
        """串行评分"""
        for result in results:
            try:
                # 检查评分器是否需要evaluator实例
                import inspect
                scorer_signature = inspect.signature(self.scorer)
                if "evaluator_instance" in scorer_signature.parameters:
                    scores = self.scorer(result, evaluator_instance=self)
                else:
                    scores = self.scorer(result)
                result.scores = scores
            except Exception as e:
                print(f"评分失败 {result.test_id}: {e}")
                result.scores = {"error": str(e)}
    
    def _apply_scoring_parallel(self, results: List[TestResult], max_workers: int) -> None:
        """并行评分"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def score_single_result(result: TestResult) -> TestResult:
            """对单个结果进行评分"""
            try:
                # 检查评分器是否需要evaluator实例
                import inspect
                scorer_signature = inspect.signature(self.scorer)
                if "evaluator_instance" in scorer_signature.parameters:
                    scores = self.scorer(result, evaluator_instance=self)
                else:
                    scores = self.scorer(result)
                result.scores = scores
                return result
            except Exception as e:
                print(f"评分失败 {result.test_id}: {e}")
                result.scores = {"error": str(e)}
                return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有评分任务
            future_to_result = {
                executor.submit(score_single_result, result): result 
                for result in results
            }
            
            # 收集结果
            with tqdm(total=len(results), desc="评分进度") as pbar:
                for future in as_completed(future_to_result):
                    result = future.result()
                    pbar.update(1)
    
    def _apply_scoring(self, results: List[TestResult]) -> None:
        """内部应用评分器方法"""
        self.apply_scoring_to_results(results)
    
    def _apply_scoring_with_progress(self, results: List[TestResult], task_groups: dict) -> None:
        """带进度显示的评分方法"""
        # 按任务类型分组结果
        result_groups = {}
        for result in results:
            task_type = result.test_case.task_type
            if task_type not in result_groups:
                result_groups[task_type] = []
            result_groups[task_type].append(result)
        
        # 显示评分分组信息
        print(f"📊 评分任务统计:")
        for task_type, task_results in result_groups.items():
            print(f"  {task_type}: {len(task_results)} 个结果")
        print()
        
        # 确定评分并发数
        scoring_workers = self.config.scoring_workers if self.config.scoring_workers is not None else self.config.max_workers
        
        if scoring_workers == 1:
            # 串行评分 - 按任务类型分别显示进度
            print("使用串行评分模式")
            for task_type, task_results in result_groups.items():
                print(f"🎯 评分 {task_type} 任务...")
                for result in tqdm(task_results, desc=f"  {task_type}"):
                    self._score_single_result(result)
                print(f"✅ {task_type} 评分完成 ({len(task_results)} 个结果)")
                print()
        else:
            # 并行评分 - 按任务类型分别显示进度
            print(f"使用并行评分模式，并发数: {scoring_workers}")
            
            for task_type, task_results in result_groups.items():
                print(f"🎯 评分 {task_type} 任务...")
                
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=scoring_workers) as executor:
                    # 提交当前任务类型的所有评分任务
                    future_to_result = {
                        executor.submit(self._score_single_result, result): result 
                        for result in task_results
                    }
                    
                    # 收集当前任务类型的结果
                    with tqdm(total=len(task_results), desc=f"  {task_type}") as pbar:
                        for future in as_completed(future_to_result):
                            result = future.result()
                            pbar.update(1)
                
                print(f"✅ {task_type} 评分完成 ({len(task_results)} 个结果)")
                print()
        
        print("🎉 所有评分完成！")
    
    def _score_single_result(self, result: TestResult) -> TestResult:
        """对单个结果进行评分"""
        try:
            # 检查评分器是否需要evaluator实例
            import inspect
            scorer_signature = inspect.signature(self.scorer)
            if "evaluator_instance" in scorer_signature.parameters:
                scores = self.scorer(result, evaluator_instance=self)
            else:
                scores = self.scorer(result)
            result.scores = scores
            return result
        except Exception as e:
            print(f"评分失败 {result.test_id}: {e}")
            result.scores = {"error": str(e)}
            return result
    
    def load_results(self, filepath: str) -> List[TestResult]:
        """从文件加载测试结果
        
        Args:
            filepath: 结果文件路径
            
        Returns:
            List[TestResult]: 测试结果列表
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = []
        for item in data:
            # 检查是否是简洁格式或完整格式
            if "test_case" in item:
                # 完整格式
                test_case = TestCase(**item["test_case"])
                result = TestResult(
                    test_id=item["test_id"],
                    test_case=test_case,
                    responses=item["responses"],
                    timestamps=item["timestamps"],
                    latencies=item["latencies"],
                    error_messages=item["error_messages"],
                    raw_responses=item["raw_responses"],
                    scores=item.get("scores")
                )
            else:
                # 简洁格式，需要重构数据
                test_case = TestCase(
                    test_id=item["test_id"],
                    task_type=item["task_type"],
                    person_name=item["person_name"],
                    topic=item["topic"],
                    questions=item["questions"],
                    context=item.get("context"),
                    metadata={"profile_summary": item.get("profile_summary", "")}
                )
                
                # 从success列表重构error_messages
                success_list = item.get("success", [])
                error_messages = [None if success else "API调用失败" for success in success_list]
                
                # 构造缺失的字段
                timestamps = [f"2024-01-01T00:00:0{i}" for i in range(len(item["responses"]))]
                raw_responses = [None] * len(item["responses"])
                
                result = TestResult(
                    test_id=item["test_id"],
                    test_case=test_case,
                    responses=item["responses"],
                    timestamps=timestamps,
                    latencies=item.get("latencies", [0.0] * len(item["responses"])),
                    error_messages=error_messages,
                    raw_responses=raw_responses,
                    scores=item.get("scores")
                )
            
            results.append(result)
        
        return results
    
    def _save_generation_results(self, results: List[TestResult], task_id: str) -> str:
        """保存生成结果"""
        filename = f"generation_{task_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        self._save_results_to_file(results, filepath)
        return filepath
    
    def _get_question_type(self, result: TestResult, question_index: int) -> Optional[str]:
        """获取问题的type信息
        
        Args:
            result: 测试结果
            question_index: 问题索引
            
        Returns:
            Optional[str]: 问题的type，如果无法获取则返回None
        """
        try:
            # 从benchmark数据中查找对应的问题type
            person_name = result.test_case.person_name
            task_type = result.test_case.task_type
            question = result.test_case.questions[question_index]
            
            # 在benchmark数据中查找匹配的问题
            for person_data in self.benchmark_data:
                if person_name in person_data:
                    tasks = person_data[person_name].get("tasks", {})
                    if task_type in tasks:
                        task_data = tasks[task_type]
                        # 查找匹配的问题
                        for item in task_data:
                            if isinstance(item, dict) and item.get("question") == question:
                                return item.get("type", None)
            return None
        except Exception:
            return None
    
    def _save_score_results(self, results: List[TestResult], task_id: str) -> str:
        """保存评分结果JSON文件"""
        # 构建评分数据
        score_results = []
        task_stats = {}
        user_stats = {}
        type_topic_stats = {}
        all_scores = []
        
        for result in results:
            # 构建测试用例评分数据
            test_data = {
                "test_id": result.test_id,
                "task_type": result.test_case.task_type,
                "person_name": result.test_case.person_name,
                "topic": result.test_case.topic,
                "questions_and_answers": []
            }
            
            # 添加问题、回答和评分
            for i, (question, response) in enumerate(zip(result.test_case.questions, result.responses)):
                qa_data = {
                    "question": question,
                    "answer": response,
                    "success": result.error_messages[i] is None,
                    "latency": result.latencies[i] if i < len(result.latencies) else 0.0
                }
                
                # 添加评分
                if result.scores:
                    for score_key, score_value in result.scores.items():
                        if f"response_{i}_" in score_key and score_key.endswith("_score"):
                            qa_data["score"] = score_value
                            
                            # 统计评分 - 按任务类型
                            task_type = result.test_case.task_type
                            if task_type not in task_stats:
                                task_stats[task_type] = []
                            task_stats[task_type].append(score_value)
                            all_scores.append(score_value)
                            
                            # 统计评分 - 按用户
                            person_name = result.test_case.person_name
                            if person_name not in user_stats:
                                user_stats[person_name] = []
                            user_stats[person_name].append(score_value)
                            
                            # 统计评分 - 按type/topic（多样性任务除外）
                            if task_type != "diversity":
                                # 对于sycophancy和irrelevance_hard，按type分组
                                if task_type in ["sycophancy", "irrelevance_hard"]:
                                    # 从原始数据中获取question的type信息
                                    question_type = self._get_question_type(result, i)
                                    if question_type:
                                        type_key = f"{task_type}_{question_type}"
                                        if type_key not in type_topic_stats:
                                            type_topic_stats[type_key] = []
                                        type_topic_stats[type_key].append(score_value)
                                # 对于irrelevance_easy，由于新格式不再按topic分组，统一归类
                                elif task_type == "irrelevance_easy":
                                    type_key = f"{task_type}_all"
                                    if type_key not in type_topic_stats:
                                        type_topic_stats[type_key] = []
                                    type_topic_stats[type_key].append(score_value)
                            break
                
                test_data["questions_and_answers"].append(qa_data)
            
            # 添加整体评分
            if result.scores and "primary_score" in result.scores:
                test_data["overall_score"] = result.scores["primary_score"]
            
            score_results.append(test_data)
        
        # 计算统计信息
        statistics = {
            "by_task_type": {},
            "by_user": {},
            "by_type_topic": {},
            "overall": {}
        }
        
        # 按任务类型统计
        for task_type, scores in task_stats.items():
            if scores:
                statistics["by_task_type"][task_type] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # 按用户统计
        for user_name, scores in user_stats.items():
            if scores:
                statistics["by_user"][user_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # 按type/topic统计
        for type_topic_key, scores in type_topic_stats.items():
            if scores:
                statistics["by_type_topic"][type_topic_key] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # 总体统计
        if all_scores:
            statistics["overall"] = {
                "average": sum(all_scores) / len(all_scores),
                "min": min(all_scores),
                "max": max(all_scores),
                "count": len(all_scores)
            }
        
        # 保存数据
        final_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "statistics": statistics,
            "results": score_results
        }
        
        filename = f"scores_{task_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    


# ============================================================================
# 配置管理工具
# ============================================================================

def load_config_from_yaml(config_path: str = "./config.yaml", profile: str = "default") -> EvaluationConfig:
    """从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        profile: 配置档案名称
        
    Returns:
        EvaluationConfig: 配置对象
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            configs = yaml.safe_load(f)
        
        if profile not in configs:
            print(f"警告: 配置档案 '{profile}' 不存在，使用默认配置")
            profile = "default"
        
        config_data = configs.get(profile, configs.get("default", {}))
        
        # 从环境变量覆盖API密钥（如果设置了）
        api_key = os.getenv("AGENT_API_KEY", config_data.get("agent_api_key", ""))
        if api_key:
            config_data["agent_api_key"] = api_key
        
        # 添加角色到端口的映射（仅对LDAgent开头的配置档案）
        if profile.startswith("BASE"):
            config_data["agent_port_mapping"] = None
        else:
            config_data["agent_port_mapping"] = configs.get("AGENTS", {})
        
        return EvaluationConfig(**config_data)
        
    except FileNotFoundError:
        print(f"配置文件不存在: {config_path}，使用默认配置")
        return EvaluationConfig()
    except Exception as e:
        print(f"加载配置失败: {e}，使用默认配置")
        return EvaluationConfig()

# ============================================================================
# 主程序入口和示例用法
# ============================================================================

if __name__ == "__main__":
    """简单演示用法，推荐使用main.py作为主程序"""
    
    print("提示: 推荐使用 main.py 作为命令行工具")
    print("示例: python main.py full --task-id demo --task-types diversity --max-cases 2")
    print()
    
    # 导入评分器
    from scorers import combined_scorer
    
    # 简单演示
    config = load_config_from_yaml()
    evaluator = Evaluator(
        config=config,
        benchmark_path="./data/locomo10_overpersonalized.json",
        results_dir="./results",
        scorer=combined_scorer
    )
    
    print("快速演示:")
    task_id = "quick_demo"
    generation_file = evaluator.generate(task_id, ["diversity"], max_test_cases=1)
    score_file = evaluator.score(task_id, generation_file)
    
    print(f"\n✅ 演示完成！")
    print(f"📁 生成: {generation_file}")
    print(f"📊 评分: {score_file}")
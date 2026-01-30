"""
LoCoMo过度个性化Benchmark构建器

该模块提供了用于构建LoCoMo过度个性化基准测试的统一接口，支持多种任务类型的生成：
- Irrelevance (无关性): Easy/Hard 模式
- Diversity (多样性)
- Sycophancy (谄媚性)

功能整合：
- 初始化：基于原始LoCoMo数据抽取 persona/profile 与直接话题
- 任务构建：无关性测试、多样性测试、谄媚性测试等多种测试场景

"""

import json
import os
import random
from typing import Any, Dict, List, Tuple

import openai
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

LOCOMO_DATA_DIR = ""

class BenchmarkBuilder:
    """LoCoMo过度个性化Benchmark构建器
    
    该类提供了构建过度个性化基准测试的完整功能，包括：
    1. 从原始LoCoMo数据中抽取用户persona和话题
    2. 构建多种类型的测试任务（无关性、冗余性等）
    3. 生成符合用户表达习惯的测试问题
    4. 合成多轮对话上下文
    
    Attributes:
        PREDEFINED_TOPICS: 预定义话题列表，用于计算话题相关性
        data_path: 原始数据文件路径
        output_path: 输出文件路径
        prompt_dir: prompt模板文件目录
        openai_model: 使用的OpenAI模型名称
    """

    # 预定义话题列表，用于计算与用户话题的相关性
    PREDEFINED_TOPICS: List[str] = [
        'Ordinary Life', 'School Life', 'Culture & Education', 'Attitude & Emotion',
        'Relationship', 'Tourism', 'Health', 'Work', 'Politics', 'Finance'
    ]

    def __init__(
        self,
        data_path: str = "./data/locomo10.json",
        output_path: str = "./data/locomo10_overpersonalized.json",
        prompt_dir: str = "./prompts",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        """初始化BenchmarkBuilder实例
        
        Args:
            data_path: 原始LoCoMo数据文件路径
            output_path: 输出增强数据文件路径
            prompt_dir: prompt模板文件目录路径
            openai_model: 使用的OpenAI模型名称
        """
        # 基础配置
        self.data_path = data_path
        self.output_path = output_path
        self.prompt_dir = prompt_dir
        self.openai_model = openai_model

        # 初始化OpenAI客户端
        self._client = openai.OpenAI()
        self.deepseek_client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_API_BASE"))
        self.deepseek_model = "deepseek-reasoner"

        # 加载原始数据到内存，避免重复读取
        self._original_data = self._read_json(self.data_path)

        # 初始化状态：优先加载现有文件，否则从原始数据抽取
        self._state: List[Dict[str, Any]] = []
        if os.path.exists(self.output_path):
            print(f"加载现有文件: {self.output_path}")
            loaded = self._safe_read(self.output_path) or []
            self._state = self._to_enhanced_structure(loaded)
        else:
            print(f"从原始数据抽取persona与话题: {self.output_path}")
            self._state = self._extract_persona_topics()
            self.save()

    # ============================================================================
    # 初始化阶段：抽取 persona & topics
    # ============================================================================
    
    def _extract_persona_topics(self) -> List[Dict[str, Any]]:
        """从原始LoCoMo数据中抽取用户persona和话题
        
        该方法基于原始对话数据，使用LLM抽取每个用户的profile和直接话题，
        同时整理每个角色的observation信息，为后续任务构建提供基础数据。
        支持缓存检查：如果已有用户信息，只补充observation。
        
        Returns:
            List[Dict[str, Any]]: 增强结构的数据列表，每个元素包含用户的
                topics、profile、observation和空的tasks字段
        """
        data = self._read_json(self.data_path)
        
        # 检查是否已有缓存数据
        existing_data = None
        if os.path.exists(self.output_path):
            existing_data = self._safe_read(self.output_path)
            if existing_data and len(existing_data) == len(data):
                print("检测到已有用户数据，只补充observation信息...")
                return self._supplement_observations_only(existing_data, data)
        
        # 如果没有缓存，执行完整的抽取流程
        print("执行完整的persona和话题抽取...")
        prompt_template = self._read_text(os.path.join(self.prompt_dir, "extract_persona_topic.txt"))

        persona_topics_raw: List[Dict[str, Any]] = []
        for content in tqdm(data, desc="抽取persona与话题"):
            # 清理 speaker 名称
            speaker_a = content["conversation"]["speaker_a"].strip()
            speaker_b = content["conversation"]["speaker_b"].strip()
            session_summary = content["session_summary"]

            messages = [
                {
                    "role": "user",
                    "content": prompt_template.format(
                        name_A=speaker_a,
                        name_B=speaker_b,
                        session_summary=session_summary,
                    ),
                }
            ]
            response = self._client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            persona_topics_raw.append(parsed)
        
        # 返回增强结构（为每人补一个空的 tasks 和整理 observation），不落盘
        enhanced: List[Dict[str, Any]] = []
        for i, item in enumerate(persona_topics_raw):
            content = data[i]  # 对应的原始数据
            out_item: Dict[str, Any] = {}
            
            for person_name, src in item.items():
                if not isinstance(src, dict):
                    continue
                
                # 整理该角色的所有observation
                observations = self._extract_observations_for_person(content, person_name)
                
                out_item[person_name] = {
                    "topics": src.get("topics", []),
                    "profile": src.get("profile", ""),
                    "observation": observations,  # 整理后的observation列表
                    "tasks": {},
                }
            enhanced.append(out_item)
        return enhanced
    
    def _supplement_observations_only(self, existing_data: List[Dict[str, Any]], original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """仅补充observation信息到已有数据中
        
        Args:
            existing_data: 已有的用户数据
            original_data: 原始LoCoMo数据
            
        Returns:
            List[Dict[str, Any]]: 补充了observation的数据
        """
        enhanced: List[Dict[str, Any]] = []
        
        for i, existing_item in enumerate(tqdm(existing_data, desc="补充observation信息")):
            content = original_data[i]  # 对应的原始数据
            out_item: Dict[str, Any] = {}
            
            for person_name, person_data in existing_item.items():
                if not isinstance(person_data, dict):
                    continue
                
                # 保留已有的数据，只补充observation
                observations = self._extract_observations_for_person(content, person_name)
                
                out_item[person_name] = {
                    "topics": person_data.get("topics", []),
                    "profile": person_data.get("profile", ""),
                    "observation": observations,  # 新补充的observation列表
                    "tasks": person_data.get("tasks", {}),  # 保留已有的tasks
                }
            enhanced.append(out_item)
        return enhanced
    
    def _extract_observations_for_person(self, content: Dict[str, Any], person_name: str) -> List[str]:
        """从原始数据中提取指定角色的所有observation
        
        Args:
            content: 原始数据项
            person_name: 角色名称
            
        Returns:
            List[str]: 该角色的所有observation文本列表
        """
        observations = []
        if "observation" in content:
            for session_key, session_obs in content["observation"].items():
                if person_name in session_obs:
                    for obs_item in session_obs[person_name]:
                        if isinstance(obs_item, list) and len(obs_item) >= 1:
                            observations.append(obs_item[0])  # 取observation文本，忽略evidence ID
        return observations

    # ============================================================================
    # 任务构建：Irrelevance - Easy（单轮QA）
    # ============================================================================
    
    def build_irrelevance_easy(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        """构建Irrelevance-Easy任务
        
        该任务类型用于测试模型在无关话题上的过度个性化行为：
        1. 计算用户话题与预定义话题的相关性，筛选出无关话题
        2. 基于用户profile和历史对话，生成与无关话题相关的问题
        3. 这些问题应该符合用户表达习惯，但不应该触发个性化响应
        
        Args:
            overwrite: 是否覆盖已存在的任务数据
            
        Returns:
            List[Dict[str, Any]]: 更新后的状态数据
        """
        data = self._state
        lock = Lock()
        total = sum(len(item) for item in data)

        def _job(work_item):
            i, enhanced_item = work_item
            for person_name, person_data in enhanced_item.items():
                if not overwrite and self._task_exists(enhanced_item, person_name, "irrelevance_easy"):
                    continue
                    
                user_topics: List[str] = person_data.get("topics", [])
                unrelated_topics, _ = self._calculate_topic_similarity(user_topics, self.PREDEFINED_TOPICS)

                topic_to_questions = []
                for topic in unrelated_topics:
                    topic_to_questions.extend(self._generate_questions(
                        topic=topic,
                        num_questions=3,
                        mode="irrelevance"
                    ))

                with lock:
                    tasks = enhanced_item[person_name].setdefault("tasks", {})
                    tasks["irrelevance_easy"] = topic_to_questions
                    enhanced_item[person_name]["tasks"] = tasks

        work = list(enumerate(self._state))
        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(_job, w) for w in work]
            for _ in tqdm(as_completed(futures), total=total, desc="构建Irrelevance-Easy任务"):
                futures.pop().result()

        self.save()
        return self._state

    # ============================================================================
    # 任务构建：Irrelevance - Hard（单轮QA）
    # ============================================================================
    
    def build_irrelevance_hard(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        """构建Irrelevance-Hard任务
        
        该任务类型用于测试模型在部分相关话题上的过度个性化行为：
        1. 基于用户直接话题生成衍生话题（如运动健身、体能提升等）
        2. 这些衍生话题与用户话题部分相关，但不需要个性化响应
        3. 生成符合用户表达习惯的钓鱼问题，可能诱导模型过度个性化
        
        Args:
            overwrite: 是否覆盖已存在的任务数据
            
        Returns:
            List[Dict[str, Any]]: 更新后的状态数据
        """
        lock = Lock()
        total = len(self._state)

        def _build_single(index_item: tuple[int, dict]) -> None:
            idx, enhanced_item = index_item
            for person_name, person_data in enhanced_item.items():
                if not overwrite and self._task_exists(enhanced_item, person_name, "irrelevance_hard"):
                    continue
                user_profile: str = person_data.get("profile", "")
                baiting_questions = self._generate_questions(
                    num_questions=5,
                    user_profile=user_profile,
                    mode="irrelevance_v2",
                )
                # 线程安全地写回
                with lock:
                    tasks_out = enhanced_item[person_name].setdefault("tasks", {})
                    tasks_out["irrelevance_hard"] = baiting_questions
                    enhanced_item[person_name]["tasks"] = tasks_out

        # 为了 tqdm 进度条，把 enumerate 转成 list
        work = list(enumerate(self._state))
        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(_build_single, w) for w in work]
            for _ in tqdm(as_completed(futures), total=total, desc="构建Irrelevance-Hard"):
                # 只要抛异常就让它直接炸出来
                futures.pop().result()

        # 全部线程结束后一次性落盘
        self.save()
        return self._state

    def build_diversity(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        """Diversity：对同一兴趣 facet 提多个字面不同、答案应不同的问题，
        若 agent 回答相似度高于阈值，则判为多样性不足。"""
        lock = Lock()
        total = len(self._state)
        def _job(args: Tuple[int, Dict[str, Any]]):
            idx, item = args
            for name, data in item.items():
                if not overwrite and self._task_exists(item, name, "diversity"):
                    continue
                user_topics = data.get("topics", [])
                if not user_topics:
                    continue
                # 为每个用户话题生成多样性任务
                diversity_tasks = []
                questions_all = []
                for topic in user_topics:
                    questions = []
                    # 1. 让 LLM 把话题拆成 3 个 derived topics
                    derived_topics = self._generate_derived_topics(topic, num=3)
                    # 2. 每个 derived topic 生成 3 个不同问法
                    for derived_topic in derived_topics:
                        derived_topic_questions = self._generate_questions(
                            topic=derived_topic, num_questions=3, mode="diversity"
                        )
                        questions.extend(derived_topic_questions)
                    random.shuffle(questions)
                    questions_all.extend(questions)
                diversity_tasks.append({"questions": questions_all})
                # 3. 线程安全写回
                with lock:
                    tasks = data.setdefault("tasks", {})
                    tasks["diversity"] = diversity_tasks
                    data["tasks"] = tasks

        work = list(enumerate(self._state))
        with ThreadPoolExecutor() as exe:
            futures = [exe.submit(_job, w) for w in work]
            for _ in tqdm(as_completed(futures), total=total, desc="Diversity"):
                futures.pop().result()
        self.save()
        return self._state

    def build_sycophancy(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        """构建sycophancy（谄媚个性化）任务集

        包含三类子任务：
        - fact：答案谄媚，以牺牲事实为代价迎合用户观点
        - memory：幻觉谄媚，虚构不存在的记忆
        - value：价值谄媚，在冲突情境中自动站队

        构建流程：
        - fact / value：输入用户profile，由LLM直接生成钓鱼问题
        - memory：输入真实记忆，生成若干虚假记忆+钓鱼问题

        Args:
            overwrite: 是否覆盖已有Sycophancy任务
            
        Returns:
            List[Dict[str, Any]]: 更新后的增强数据结构
        """
        lock = Lock()
        total = len(self._state)

        def _job(args: Tuple[int, Dict[str, Any]]):
            idx, item = args
            for name, data in item.items():
                if not overwrite and self._task_exists(item, name, "sycophancy"):
                    continue

                user_profile = data.get("profile", "")
                user_observations = data.get("observation", [])

                # ---------------------------
                # 答案谄媚 (Fact-level) & 价值谄媚 (Value-level)
                # ---------------------------
                baiting_questions = self._generate_questions(
                    num_questions=10,
                    user_profile=user_profile,
                    mode="sycophancy"
                )

                # ---------------------------
                # 2️⃣ 幻觉谄媚 (Memory-level)
                # ---------------------------
                # 基于用户observation直接生成钓鱼问题
                baiting_questions_v2 = self._generate_questions(
                    user_observations=user_observations,
                    num_questions=10,
                    mode="sycophancy_v2"
                )

                sycophancy_tasks = baiting_questions + baiting_questions_v2
                random.shuffle(sycophancy_tasks)

                with lock:
                    tasks = data.setdefault("tasks", {})
                    tasks["sycophancy"] = sycophancy_tasks
                    data["tasks"] = tasks

        work = list(enumerate(self._state))
        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(_job, w) for w in work]
            for _ in tqdm(as_completed(futures), total=total, desc="构建Sycophancy任务"):
                futures.pop().result()

        self.save()
        return self._state

    # ============================================================================
    # 私有帮助方法：文件操作
    # ============================================================================
    
    def _read_json(self, path: str) -> Any:
        """读取JSON文件
        
        Args:
            path: 文件路径
            
        Returns:
            Any: 解析后的JSON数据
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: str, data: Any) -> None:
        """写入JSON文件
        
        Args:
            path: 文件路径
            data: 要写入的数据
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _read_text(self, path: str) -> str:
        """读取文本文件
        
        Args:
            path: 文件路径
            
        Returns:
            str: 文件内容
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ============================================================================
    # 私有帮助方法：LLM交互
    # ============================================================================
    
    def _calculate_topic_similarity(
        self, user_topics: List[str], predefined_topics: List[str]
    ) -> Tuple[List[str], List[float]]:
        """计算用户话题与预定义话题的相关性
        
        使用LLM判断预定义话题与用户直接话题的相关性，返回无关话题列表和相似度分数。
        
        Args:
            user_topics: 用户话题列表
            predefined_topics: 预定义话题列表
            
        Returns:
            Tuple[List[str], List[float]]: (无关话题列表, 相似度分数列表)
        """
        user_topics_text = ", ".join(user_topics)
        prompt_template = self._read_text(os.path.join(self.prompt_dir, "calculate_topic_similarity.txt"))
        prompt = prompt_template.format(
            user_topics_text=user_topics_text,
            predefined_topics=predefined_topics
        )

        try:
            response = self._client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            unrelated_topics: List[str] = []
            similarities: List[float] = []
            for topic in predefined_topics:
                is_related = bool(result.get(topic, False))
                similarities.append(1.0 if is_related else 0.0)
                if not is_related:
                    unrelated_topics.append(topic)
            return unrelated_topics, similarities
        except Exception:
            return predefined_topics, [0.0] * len(predefined_topics)

    def _generate_questions(
        self,
        topic: str=None,
        num_questions: int = 3,
        user_profile: str = "",
        fake_memories: Dict[str, List[str]] = {},
        user_observations: List[str] = [],
        mode: str = "normal",   
    ) -> List[Dict[str, Any]]:
        """为指定话题生成符合用户表达习惯的问题
        
        根据用户的历史对话和profile信息，生成与指定话题相关的问题。
        支持普通模式和钓鱼模式，钓鱼模式会生成可能诱导过度个性化的问题。
        
        Args:
            topic: 目标话题
            num_questions: 生成问题数量
            user_profile: 用户profile信息
            fake_memories: 虚假记忆字典（用于memory模式）
            mode: 生成模式
            
        Returns:
            List[Dict[str, Any]]: 生成的问题列表，每个问题包含question、type、explanation等字段
        """
        
        if mode == "irrelevance_v2":
            prompt_template = self._read_text(os.path.join(self.prompt_dir, "generate_questions_irrelevance_v2.txt"))
            prompt = prompt_template.format(
                num_questions=num_questions,
                user_profile=user_profile
            )
        elif mode == "irrelevance":
            prompt_template = self._read_text(os.path.join(self.prompt_dir, "generate_questions_irrelevance.txt"))
            prompt = prompt_template.format(
                num_questions=num_questions,
                topic=topic
            )
        elif mode == "diversity":
            prompt_template = self._read_text(os.path.join(self.prompt_dir, "generate_questions_diversity.txt"))
            prompt = prompt_template.format(
                num_questions=num_questions,
                topic=topic
            )
        elif mode == "sycophancy_v2":
            prompt_template = self._read_text(os.path.join(self.prompt_dir, "generate_questions_sycophancy_v2.txt"))
            observations_text = "\n".join([f"- {obs}" for obs in user_observations])
            prompt = prompt_template.format(
                user_observations=observations_text,
                num_questions=num_questions
            )
        elif mode == "sycophancy":
            prompt_template = self._read_text(os.path.join(self.prompt_dir, "generate_questions_sycophancy.txt"))
            prompt = prompt_template.format(
                num_questions=num_questions,
                user_profile=user_profile,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # 根据模式选择模型
        client, model = self._get_client_and_model_for_mode(mode)
        print(f"Using model: {model}")
        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                print(result)
                questions = result.get("questions", [])
                return questions if isinstance(questions, list) else []
            except Exception as e:
                print(e)
                continue
    
    def _get_client_and_model_for_mode(self, mode: str) -> Tuple[openai.OpenAI, str]:
        """根据模式选择合适的模型
        
        Args:
            mode: 生成模式
            
        Returns:
            str: 模型名称
        """
        # 特定模式使用deepseek-v3
        deepseek_modes = {"irrelevance_v2", "sycophancy_v2", "sycophancy"}
        
        if mode in deepseek_modes:
            return self.deepseek_client, self.deepseek_model
        else:
            return self._client, self.openai_model
    
    def _generate_derived_topics(self, topic: str, num: int = 6) -> List[str]:
        """让 LLM 把 topic 拆成 num 个子 facet，返回 list[str]"""
        prompt = self._read_text(os.path.join(self.prompt_dir, "generate_derived_topics.txt"))
        prompt = prompt.format(seed=topic, num=num)
        while True:
            try:
                r = self._client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return json.loads(r.choices[0].message.content)["derived_topics"]
            except Exception as e:
                print(e)
                continue
    

    # ============================================================================
    # 私有帮助方法：数据处理与验证
    # ============================================================================
    
    def _safe_read(self, path: str) -> Any:
        """安全读取JSON文件
        
        Args:
            path: 文件路径
            
        Returns:
            Any: 解析后的JSON数据，失败时返回None
        """
        try:
            return self._read_json(path)
        except Exception:
            return None

    def _to_enhanced_structure(self, current_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换为增强数据结构
        
        轻量归一化处理：
        - 若已是增强结构（含 tasks），确保 tasks 为 dict，其余原样保留
        - 若是 persona 列表（无 tasks），为每人补一个空 tasks
        
        Args:
            current_data: 当前数据列表
            
        Returns:
            List[Dict[str, Any]]: 增强结构的数据列表
        """
        if not current_data:
            return []

        def item_has_tasks(item: Dict[str, Any]) -> bool:
            for _, v in item.items():
                if isinstance(v, dict) and "tasks" in v:
                    return True
            return False

        if any(item_has_tasks(x) for x in current_data):
            fixed: List[Dict[str, Any]] = []
            for item in current_data:
                out: Dict[str, Any] = {}
                for name, pdata in item.items():
                    if not isinstance(pdata, dict):
                        continue
                    out[name] = {
                        "topics": pdata.get("topics", []),
                        "profile": pdata.get("profile", ""),
                        "observation": pdata.get("observation", []),
                        "tasks": pdata.get("tasks", {}) if isinstance(pdata.get("tasks", {}), dict) else {},
                    }
                fixed.append(out)
            return fixed

        enhanced: List[Dict[str, Any]] = []
        for item in current_data:
            out_item: Dict[str, Any] = {}
            for person_name, src in item.items():
                if not isinstance(src, dict):
                    continue
                out_item[person_name] = {
                    "topics": src.get("topics", []),
                    "profile": src.get("profile", ""),
                    "observation": src.get("observation", []),
                    "tasks": {},
                }
            enhanced.append(out_item)
        return enhanced

    # ============================================================================
    # 持久化与状态管理
    # ============================================================================
    
    def save(self) -> None:
        """将内存中的最新结构持久化到输出文件"""
        self._write_json(self.output_path, self._state)

    def _task_exists(self, enhanced_item: Dict[str, Any], person_name: str, task_key: str) -> bool:
        """检查指定用户是否已存在非空任务结果
        
        Args:
            enhanced_item: 增强数据项
            person_name: 用户名称
            task_key: 任务键名
            
        Returns:
            bool: 是否存在非空任务结果
        """
        try:
            tasks = enhanced_item[person_name].get("tasks", {})
            val = tasks.get(task_key)
            if isinstance(val, list):
                return len(val) > 0
            return bool(val)
        except Exception:
            return False

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    """主程序入口
    
    执行完整的Benchmark构建流程：
    1. 初始化：抽取persona和topics
    2. 构建各类任务：Irrelevance Easy/Hard、Redundancy Easy/Hard
    """
    # 初始化构建器
    builder = BenchmarkBuilder(
        data_path="./data/locomo10.json",
        output_path="./data/locomo10_overpersonalized.json",
        prompt_dir="./prompts",
        openai_model="gpt-4o-mini",
    )

    # 分步执行任务构建（可按需注释某一步）
    print("开始构建Irrelevance-Easy任务...")
    builder.build_irrelevance_easy()
    
    print("开始构建Irrelevance-Hard任务...")
    builder.build_irrelevance_hard()
    
    print("开始构建Diversity任务...")
    builder.build_diversity()
    
    print("开始构建Sycophancy任务...")
    builder.build_sycophancy(True)
    
    print("所有任务构建完成！")


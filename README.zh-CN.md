# OP-Bench：面向记忆增强个性化对话代理的过度个性化基准

<p align="center">
  <a href="https://arxiv.org/abs/2601.13722">
    <img src="https://img.shields.io/badge/arXiv-2601.13722-b31b1b.svg?style=flat-square" alt="arXiv：2601.13722">
  </a>
  <a href="https://2026.emnlp.org/">
    <img src="https://img.shields.io/badge/EMNLP-2026-6f42c1.svg?style=flat-square" alt="EMNLP 2026">
  </a>
  <a href="#website-coming-soon">
    <img src="https://img.shields.io/badge/%E7%BD%91%E7%AB%99-%E5%BB%BA%E8%AE%BE%E4%B8%AD-2563eb.svg?style=flat-square" alt="网站建设中">
  </a>
</p>

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/English-README-1f6feb.svg?style=flat-square" alt="英文 README">
  </a>
  <a href="README.zh-CN.md">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README.zh--CN-16a34a.svg?style=flat-square" alt="中文 README">
  </a>
</p>

<p align="center">
  <strong>一个用于衡量长期记忆何时导致对话代理过度、过频或不恰当地进行个性化的诊断性基准。</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.13722">论文</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="data/README.md">数据</a>
  ·
  <a href="#引用">引用</a>
</p>

## 概览

记忆增强对话代理旨在利用长期用户信息提供有帮助的个性化服务。
OP-Bench 关注这种能力的另一面：**过度个性化**，即代理在没有必要时
引入个人信息、为了迎合用户而牺牲准确性或中立性，或在不同问题之间
反复生成相似的个性化内容。

OP-Bench 包含从长时程对话历史中构建并经过人工核验的 1,700 个实例。
它使用模型评估指标和基于嵌入的指标，衡量三类失败模式：
无关性、谄媚性和重复性。

| 基本信息 | |
| --- | --- |
| 基准规模 | 1,700 个经过核验的实例 |
| 用户数量 | 20 |
| 失败模式 | 3 个主要类别、6 个子类别 |
| 评估方向 | 分数越高表示过度个性化越少 |
| 论文实验 | 6 个语言模型、5 种记忆设置、36 个配置 |
| 开源内容 | 基准数据、提示词、评估器、选定的代理实现、MemOS 适配器和交互式演示 |

### 失败模式

| 类别 | 衡量内容 | 子类别 |
| --- | --- | --- |
| **无关性（Irrelevance）** | 用户问题不需要个性化时，回答是否仍然注入个人信息 | 完全无关；诱导型（表面相关但实际具有误导性） |
| **谄媚性（Sycophancy）** | 个性化是否导致代理过度附和，而不是保持事实准确或价值中立 | 事实层面；价值层面；记忆层面 |
| **重复性（Repetition）** | 对语义不同的问题，代理是否生成几乎相同的个性化回答 | 单一重复性设置 |

<p align="center">
  <img src="docs/figures/figure-1-taxonomy.png" alt="OP-Bench 三类失败模式：无关性、谄媚性和重复性" width="720">
</p>
<p align="center"><em>三类过度个性化行为示意图。</em></p>

## 数据构建

基准数据通过以下三个阶段构建：

1. **初始化：** 从长时程 LoCoMo 对话中提取结构化用户画像和用户主题。
2. **任务构建：** 针对无关性、谄媚性和重复性生成可控测试问题，
   并覆盖相应子类别。
3. **人工审核：** 对候选实例进行独立审核，对存在分歧的样本进行
   裁决，只保留经过核验的实例。

<p align="center">
  <img src="docs/figures/figure-3-pipeline.png" alt="OP-Bench 从初始化到任务构建再到人工审核的三阶段流程" width="100%">
</p>
<p align="center"><em>OP-Bench 数据构建流程（论文图 3）。</em></p>

最终数据集的分布如下：

| 类别 | 子类别 | 数量 | 占比 |
| --- | --- | ---: | ---: |
| 无关性 | 完全无关 | 318 | 18.7% |
| 无关性 | 诱导型 | 100 | 5.9% |
| 谄媚性 | 事实层面 | 100 | 5.9% |
| 谄媚性 | 价值层面 | 100 | 5.9% |
| 谄媚性 | 记忆层面 | 200 | 11.8% |
| 重复性 | — | 882 | 51.9% |
| **总计** | — | **1,700** | **100%** |

## 实验发现

OP-Bench 的分数定义为**越高越好**：分数越高，说明回答越能避免
相应的过度个性化行为。

### 主要结果

下表展示论文主要结果表中 GPT-4o-mini 对应的结果。括号中的百分比
表示相对于无记忆 <code>BASE</code> 设置的相对下降幅度。

| 记忆设置 | OP-Bench 平均分 |
| --- | ---: |
| <code>BASE</code> | **83.10** |
| <code>RAG</code> | 55.96（↓32.7%） |
| <code>Mem0</code> | 46.32（↓44.3%） |
| <code>MemU</code> | 40.46（↓51.3%） |
| <code>MEMOS</code> | 41.86（↓49.6%） |

在不同模型和记忆系统上，论文报告的相对下降幅度为
**26.2%–61.1%**，参照设置为 <code>BASE</code>。

### 不同模型的对比

下面四个面板复现论文中不同模型系列的雷达图，上方为统一图例。

<p align="center">
  <img src="docs/figures/figure-2-legend.png" alt="BASE、RAG、Mem0、MEMOS 和 MemU 的图例" width="620">
</p>
<p align="center">
  <img src="docs/figures/figure-2-gpt4o-mini.png" alt="GPT-4o-mini 雷达图" width="47%">
  <img src="docs/figures/figure-2-gemini-2-5-flash.png" alt="Gemini-2.5-flash 雷达图" width="47%">
</p>
<p align="center">
  <img src="docs/figures/figure-2-qwen3-235b.png" alt="Qwen3-235B 雷达图" width="47%">
  <img src="docs/figures/figure-2-qwen3-32b.png" alt="Qwen3-32B 雷达图" width="47%">
</p>
<p align="center"><em>不同模型和记忆配置下的 OP-Bench 分数（论文图 2）。</em></p>

### 研究问题

| 问题 | 结论 |
| --- | --- |
| **RQ1——过度个性化是否存在？** | 存在。所有评估的记忆增强设置相较于 <code>BASE</code> 都出现明显下降；更复杂的记忆机制通常下降更严重。 |
| **RQ2——过度个性化为何发生？** | 记忆可能被过度检索，并获得不成比例的注意力。记忆与用户问题之间的平均注意力比超过 2 倍，从而可能导致记忆劫持和回答坍缩。 |
| **RQ3——过度个性化能否缓解？** | 只能部分缓解。后处理最多使 OP-Bench 提升 **+20.0%**，但部分配置会使 LoCoMo 个性化能力下降最多 **−8.5%**。没有一种评估方法能够弥合其与无记忆上界之间的差距。 |
| **RQ4——时间开销是多少？** | 补充性时间分析显示，记忆检索平均约为 **38–44 毫秒**，后处理约为 **0.97–4.13 秒**，模型回答生成约为 **2.71–3.41 秒**。 |

### 记忆检索是重要问题来源

即使问题与历史主题完全无关，也可能触发记忆检索。诱导型问题
可能与存储记忆具有较高的表面相似度，因此仅依赖检索本身并不能
有效避免过度个性化。

<p align="center">
  <img src="docs/figures/figure-5-retrieval.png" alt="RAG、MemOS、MemU 和 Mem0 在完全无关问题与诱导型问题上的检索相关性对比" width="760">
</p>
<p align="center"><em>无关性任务上的检索记忆相似度（论文图 5）。</em></p>

## 仓库内容

本开源仓库围绕可复现的基准运行路径组织，包含：

- 基准数据以及数据构建和评估提示词；
- 支持断点续作的基准构建器；
- 检索、生成、评分和后处理工具；
- 论文工作中使用的 <code>LDAgent</code> 和 <code>SimpleRAGAgent</code> 实现；
- 面向外部 MemOS 服务的 HTTP 适配器；
- 用于展示基准流程和实验结果的轻量级网站演示；
- 配置、评分、数据类型和网站演示测试。

仓库**不包含**上游 MemOS 服务、模型权重、私有部署脚本、
无关的实验代理或生成结果。原始研究工作区中的绘图脚本也未纳入；
README 使用的论文图保存在 [docs/figures/](docs/figures/)。

## 目录结构

~~~text
OPBench-public/
├── configs/
│   └── evaluation.example.yaml
├── data/
│   ├── locomo10.json
│   ├── locomo10_overpersonalized.json
│   └── README.md
├── docs/
│   └── figures/                  # 论文中选取的图片
├── prompts/
│   ├── benchmark/                # 任务构建模板
│   └── evaluation/               # 评分提示词和归档源文件
├── scripts/
│   ├── prepare_locomo_histories.py
│   ├── start_agent.py
│   ├── start_agents.py
│   └── ingest_memos.py
├── src/opbench/
│   ├── agents/                   # LDAgent 和 SimpleRAGAgent
│   ├── baselines/                # MemOS API 适配器
│   ├── benchmark_builder.py      # 构建或扩展基准任务
│   ├── evaluation.py             # 检索、生成、评分和指标
│   ├── postprocessing.py         # 上下文过滤和压缩接口
│   └── config.py                 # 基于环境变量的配置
├── tests/
├── web/                          # 本地全栈研究展示网站
├── .env.example
├── pyproject.toml
└── NOTICE.md
~~~

## 快速开始

### 1. 安装

需要 Python 3.10 或更高版本。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[agents]"
~~~

根据模板创建本地环境变量文件：

~~~bash
cp .env.example .env
~~~

在本地填写模型服务商相关变量。凭据和接口地址只能保存在
<code>.env</code> 或其他已被 Git 忽略的本地配置中；不要将它们写入
源代码、提交到 Git 的 YAML 文件或 Shell 脚本。

### 2. 启动本地网站演示

网站演示是自包含的，不需要模型凭据：

~~~bash
python web/backend/server.py --host 127.0.0.1 --port 8765
~~~

然后打开 <http://127.0.0.1:8765>。界面包含三个章节：
数据构建、代表性问答示例和实验结果。数据构建章节支持自动播放，
也支持手动使用上一步和下一步按钮切换。接口路由和实现说明请参阅
[web/README.md](web/README.md)。

### 使用 GitHub Pages 发布网站

网站演示也可以作为纯静态网站发布。当前端无法访问 Python 接口时，
会自动回退到仓库中提供的 JSON 数据。仓库中的 GitHub Actions 工作流
会在相关文件推送到 <code>main</code> 分支时构建 Pages 部署文件并发布。
请在 Settings → Pages 中将来源设置为 **GitHub Actions**，首次部署成功
后可访问 <https://yulinlp.github.io/OP-Bench/>。

### 3. 运行仓库中的代理

为每个 LoCoMo 用户准备一个历史文件：

~~~bash
python scripts/prepare_locomo_histories.py
~~~

启动两种公开记忆实现中的一种所对应的全部十个代理：

~~~bash
python scripts/start_agents.py --agent ldagent
# 或
python scripts/start_agents.py --agent simplerag
~~~

每个服务提供以下接口：

- <code>POST /v1/chat/completions</code>
- <code>GET /v1/models</code>
- <code>GET /health</code>

如需只启动一个进程，可以使用
<code>scripts/start_agent.py</code> 并提供历史文件和端口。

### 4. 运行 OP-Bench 评估

仓库中已提供论文实验使用的任务文件
<code>data/locomo10_overpersonalized.json</code>。对于进程内代理：

~~~bash
python -m opbench.cli generate --frame ldagent
python -m opbench.cli generate --frame simplerag
~~~

对于 MemOS，请先将历史记录写入已配置好的外部服务，然后执行检索、
生成和评分：

~~~bash
python scripts/ingest_memos.py --online
python -m opbench.cli search --frame memos-api-online
python -m opbench.cli generate --frame memos-api-online --postprocess none
python -m opbench.cli score --frame memos-api-online --responses results/responses/memos-api-online_gpt-4o-mini_none.json
python -m opbench.cli metrics --frame memos-api-online --judged results/judged/memos-api-online_gpt-4o-mini.json
~~~

查看所有命令和参数：

~~~bash
python -m opbench.cli --help
~~~

可用的上下文方法包括 <code>none</code>、<code>self_recheck</code>、
<code>reminder</code>、<code>self_critic</code>、
<code>few_shot_chain_of_thought</code>、<code>comorag</code> 和
<code>marag</code>。开源实现记录经过过滤的上下文，不记录隐藏的
思维链轨迹。生成的中间产物和结果默认写入被 Git 忽略的目录。

### 重建基准数据

构建器支持断点续作：如果输出文件已经存在，程序会对其规范化并
继续补全，不会静默覆盖原有文件。

~~~bash
python -m opbench.cli build-benchmark --source data/locomo10.json --output data/locomo10_overpersonalized.generated.json
~~~

构建任务所需的模型接口和凭据从本地环境变量中读取。

### 运行测试

~~~bash
python -m pip install -e ".[dev]"
pytest
~~~

## 复现说明

- <code>BASE</code> 是无记忆参考设置；<code>RAG</code>、<code>Mem0</code>、
  <code>MemU</code> 和 <code>MEMOS</code> 是论文中评估的记忆增强设置。
- 仓库中的提示词保留了基准使用的任务定义和评分定义。
- 重复性指标在显式进行 L2 归一化后计算余弦相似度。
- 每个问题的接口失败都会被记录，重试次数受到限制，结果文件以
  原子方式写入，因此长时间运行可以安全地恢复。
- 如果需要使用每段对话中的两个用户画像，请在本地配置中设置
  <code>use_both_personas: true</code>；默认行为与原始脚本一致，
  只使用每段对话中的第一个用户画像。

<a id="website-coming-soon"></a>

## 网站（建设中）

托管版 OP-Bench 网站正在制作中。在正式发布之前，可以使用
[web/](web/) 下的本地交互式演示查看数据构建、问答示例和实验结果。

## 许可证与数据

请参阅 [NOTICE.md](NOTICE.md) 中的来源和再分发说明。当前仓库尚未
声明软件许可证；在再分发代码前，请补充或确认合适的许可证。
LoCoMo 数据、模型权重和第三方服务仍受其原始许可条款约束。
本版本不包含模型权重或凭据。

## 引用

如果 OP-Bench 对你的研究有帮助，请引用：

~~~bibtex
@misc{hu2026opbench,
  title         = {OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents},
  author        = {Hu, Yulin and Long, Zimo and Guo, Jiahe and Sui, Xingyu and Fu, Xing and Zhao, Weixiang and Zhao, Yanyan and Qin, Bing},
  year          = {2026},
  eprint        = {2601.13722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.13722}
}
~~~

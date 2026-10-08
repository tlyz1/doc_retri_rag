# 掌柜智库 · 企业文档智能知识库（GraphRAG）

> 基于 **LangGraph** 的图谱增强 RAG（GraphRAG）系统：把 PDF/Markdown 产品文档自动变成可问答的知识库。
> 导入侧完成文档解析、图片语义提取、三级切片、混合向量化与知识图谱构建；查询侧用「**BGE-M3 混合向量 + HyDE 假设文档 + Neo4j 知识图谱**」三路召回，经 RRF 加权融合、BGE-Reranker 精排后生成答案，并通过 SSE 流式输出。

---

## ✨ 核心特性

**检索侧（GraphRAG）**

- **知识图谱检索路**：LLM 抽取问题实体 → 实体名向量对齐（Milvus）→ Neo4j 精确/模糊匹配种子节点 → 一跳关系双向扩展（种子节点权重 2.0、邻居节点 1.0）→ 反查并回填原文切片，图谱三元组同时注入答案 prompt
- **混合向量检索**：BGE-M3 同时产出稠密向量（语义理解）与稀疏向量（型号/参数精确匹配），Milvus WeightedRanker 融合两路结果
- **HyDE 检索**：由 LLM 生成假设性答案再做向量检索，弥补问题与文档表述之间的语义鸿沟
- **多路融合与精排**：RRF 按路加权融合（向量 1.0 / HyDE 1.0 / 图谱 0.7，不依赖分数绝对值），BGE-Reranker-Large 交叉编码器精排后按分数断崖动态截断
- **网络检索降级为兜底**：外部搜索默认不参与召回，仅当本地召回不足（RRF 后切片数低于阈值）时才由条件分支触发，避免通用网页内容污染答案

**导入侧**

- **复杂 PDF 解析**：MinerU 解析版式、表格与公式，图片资源上传 MinIO 并回写 Markdown 图片链接
- **图片语义提取**：视觉模型（Qwen3-VL）为图片生成上下文描述，把"图里的信息"纳入可检索内容
- **三级切片策略**：标题层级切分 + 递归字符切分 + 贪心合并，兼顾语义完整与切片大小
- **商品名识别**：为切片标注所属商品（item_name），作为后续检索的过滤维度
- **知识图谱构建**：LLM 抽取实体与关系，经实体标签白名单、关系类型白名单、实体名长度与 JSON 清洗后写入 Neo4j（MERGE 去重），实体名向量化入库供查询侧对齐；切片级图谱构建用线程池并发

**工程化**

- **LangGraph 双流程编排**：导入/查询两套状态图，支持条件分支路由与节点并发执行
- **商品名确认与多轮对话**：LLM 提取 → 向量对齐 → 评分对齐 → 分数差异过滤的多级校验；模糊输入返回候选让用户澄清；确认后回填历史会话的元数据
- **SSE 流式响应**：实时推送节点进度（progress）与答案增量（delta），前端用 EventSource 接收
- **RAG 效果评估**：自建评测集 + RAGAS 五项指标（忠实度 / 答案相关性 / 上下文精确率 / 上下文召回率 / 答案正确性）+ 逐题明细，支撑检索策略的量化迭代

---

## 🏗 系统架构

### 1. 文档导入流程（LangGraph）

```mermaid
flowchart LR
    A[上传 PDF/Markdown] --> B[PDF → Markdown<br/>MinerU 解析]
    B --> C[图片处理<br/>VLM 描述 + MinIO + 链接替换]
    C --> D[三级切片<br/>标题层级 + 递归字符 + 贪心合并]
    D --> E[商品名识别<br/>LLM 抽取 + 向量对齐]
    E --> F[混合向量化<br/>BGE-M3 dense + sparse]
    F --> G[(Milvus<br/>切片向量)]
    F --> H[知识图谱构建<br/>实体/关系抽取 + 白名单过滤]
    H --> I[(Neo4j<br/>实体 · 关系)]
    H --> J[(Milvus<br/>实体名向量)]
```

### 2. 知识查询流程（LangGraph）

```mermaid
flowchart TD
    Q[用户提问] --> N1[商品名确认<br/>LLM 提取 → 向量对齐 → 评分/差异过滤]
    N1 -->|商品名模糊| CLR[返回候选选项<br/>让用户澄清]
    N1 -->|已确认| MS[三路检索并行]
    MS --> V[向量检索<br/>BGE-M3 混合向量]
    MS --> HD[HyDE 检索<br/>假设文档 → 向量检索]
    MS --> KG[知识图谱检索<br/>实体对齐 → 种子节点 → 一跳扩展 → chunk 回填]
    V --> JOIN[Join 汇合]
    HD --> JOIN
    KG --> JOIN
    JOIN --> RRF[RRF 加权融合<br/>1.0 / 1.0 / 0.7]
    RRF -->|本地召回充足| RR[BGE-Reranker 精排<br/>+ 分数断崖截断]
    RRF -->|本地召回不足| WEB[网络检索兜底<br/>MCP 搜索]
    WEB --> RR
    RR --> ANS[答案生成<br/>检索上下文 + 图谱三元组 + 历史对话]
    ANS --> SSE[SSE 流式输出<br/>progress / delta / final]
```

### 3. 图谱检索链路（GraphRAG 关键实现）

| 阶段 | 实现 | 说明 |
|---|---|---|
| ① 实体抽取 | `_EntityExtractor`（LLM） | 从用户问题中抽取实体名，输出 JSON |
| ② 实体对齐 | `_EntityAligner`（Milvus） | 用实体名向量在实体集合中检索，相似度阈值 0.5，解决"用户说法 ≠ 文档说法" |
| ③ 图谱查询 | `_Neo4jGraphReader`（Cypher） | 精确匹配 + 模糊匹配（`toLower CONTAINS`）定位种子节点，再查一跳双向关系（排除 `MENTIONED_IN`） |
| ④ 权重与回填 | `_ChunkBackfiller` | 种子节点权重 2.0、一跳邻居 1.0，按权重聚合出 chunk_id 后回 Milvus 取回原文切片 |

---

## 🧰 技术栈

| 层次 | 技术 | 用途 |
|---|---|---|
| 工作流编排 | LangGraph / LangChain | 导入、查询双状态图，条件分支与并发 |
| 大语言模型 | Qwen3 系列（OpenAI 兼容网关） | 实体/商品名抽取、HyDE、答案生成 |
| 视觉模型 | Qwen3-VL | 文档图片语义描述 |
| 向量模型 | BGE-M3（稠密 + 稀疏） | 切片与查询的混合向量表示 |
| 重排序模型 | BGE-Reranker-Large | 交叉编码器精排 |
| 向量数据库 | Milvus（2.5+） | 混合检索、实体名对齐、切片回填 |
| 图数据库 | Neo4j | 实体-关系存储与一跳扩展查询 |
| 文档数据库 | MongoDB | 会话历史、任务结果 |
| 对象存储 | MinIO | 原始文档与图片资源 |
| PDF 解析 | MinerU | 复杂版式文档转 Markdown |
| Web 服务 | FastAPI + Uvicorn | 导入服务（8000）、查询服务（8001），SSE 流式接口 |
| 前端 | 原生 HTML + JavaScript（EventSource） | 文档上传页、问答页 |
| 效果评估 | RAGAS | 五项 RAG 指标 + 逐题明细 |

---

## 📂 目录结构

```
doc_retri_rag/
├── .env                          # 环境变量（已在 .gitignore 中，不入库）
├── knowledge/
│   ├── api/                      # 服务入口
│   │   ├── import_file_router.py # 导入服务（:8000）
│   │   └── query_router.py       # 查询服务（:8001）
│   ├── core/                     # 依赖装配、路径管理
│   ├── processor/
│   │   ├── import_process/       # 导入 LangGraph
│   │   │   ├── main_graph.py     # 图构建
│   │   │   ├── state.py          # 状态定义
│   │   │   ├── config.py         # 切片/模型/存储配置
│   │   │   └── nodes/            # entry / pdf_to_md / md_img / document_split /
│   │   │                         # item_name_recognition / bge_embedding /
│   │   │                         # import_milvus / knowledge_graph
│   │   └── query_process/        # 查询 LangGraph
│   │       ├── main_graph.py     # 图构建（三路检索 + 条件兜底）
│   │       ├── state.py          # QueryGraphState
│   │       ├── config.py         # RRF / Rerank / 兜底门控配置
│   │       └── nodes/            # item_name_confirm / vector_search / hyde_search /
│   │                             # kg_search / mcp_search / rrf / rerank / answer_output
│   ├── prompts/                  # LLM 提示词（导入侧、查询侧）
│   ├── schema/                   # Pydantic 请求/响应模型
│   ├── services/                 # 业务服务层（导入、查询、任务）
│   ├── tools/                    # Milvus / Neo4j / MongoDB / MinIO / LLM /
│   │                             # Embedding / Reranker / SSE / Markdown 工具
│   └── front/                    # import.html、chat.html
└── README.md
```

---

## 🚀 快速开始

### 1. 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.11（开发环境为 3.11.x） |
| CUDA（可选） | 显卡显存建议 ≥ 6GB；BGE-M3 与 Reranker 支持 fp16，CPU 亦可运行（设 `BGE_DEVICE=cpu`） |
| 依赖服务 | Milvus、Neo4j、MongoDB、MinIO（四个都必须可访问） |
| 外部 API | OpenAI 兼容的大模型网关（默认走 SiliconFlow）、百炼 MCP 搜索（仅在兜底分支使用） |

### 2. 启动依赖服务

| 服务 | 默认端口 | 说明 |
|---|---|---|
| Milvus | 19530 | 切片集合、实体名集合 |
| Neo4j | 7687 | bolt 协议 |
| MongoDB | 27017 | 会话历史与任务状态 |
| MinIO | 9000 | 文档与图片对象存储 |

### 3. 安装依赖

```bash
git clone https://github.com/tlyz1/doc_retri_rag.git
cd doc_retri_rag
python -m venv .venv && .venv\Scripts\activate     # Windows

# 核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # 无 GPU 可换成 cpu 版本
pip install fastapi uvicorn python-dotenv pymongo neo4j minio mcp \
            pymilvus[model] FlagEmbedding sentence-transformers \
            langgraph langchain-openai grandalf openai-agents "mineru[all]"
```

> 说明：仓库目前未包含完整的 `requirements.txt`，可先用上面的命令安装；如需一键复现，请参考 `knowledge/requirements_bak.txt`。

### 4. 配置环境变量

在项目根目录创建 `.env`（**不要提交到 Git**）：

```ini
# ===== LLM 网关 =====
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.siliconflow.cn/v1
LLM_DEFAULT_MODEL=Qwen/Qwen3-32B
VL_MODEL=Qwen/Qwen3-VL-8B-Instruct
ITEM_MODEL=Qwen/Qwen3-32B

# ===== 向量 / 重排模型 =====
BGE_M3_PATH=BAAI/bge-m3
BGE_DEVICE=cuda:0            # 可改 cpu
BGE_FP16=True
BGE_RERANKER_LARGE=your-local-path/bge-reranker-large
BGE_RERANKER_DEVICE=cuda:0
BGE_RERANKER_FP16=True

# ===== Milvus =====
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks
ITEM_NAME_COLLECTION=kb_item_names
ENTITY_NAME_COLLECTION=kb_entity_names

# ===== Neo4j =====
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# ===== MongoDB =====
MONGO_URL=mongodb://user:password@localhost:27017/?authSource=admin
MONGO_DB_NAME=kb001

# ===== MinIO =====
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET_NAME=knowledge-base

# ===== 兜底网络检索（可选）=====
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/EnhancedSearch/mcp
MCP_DASHSCOPE_API_KEY=your-dashscope-key
```

### 5. 启动服务

> 注意：以下命令都在**项目根目录**执行（保证 `knowledge` 包可导入、根目录 `.env` 可被加载）。

```bash
# 导入服务：http://localhost:8000/import
python -m knowledge.api.import_file_router

# 查询服务：http://localhost:8001/chat.html
python -m knowledge.api.query_router
```

### 6. 使用流程

1. 打开 `http://localhost:8000/import`，上传 PDF/Markdown → 页面轮询 `/status/{task_id}` 展示各节点进度；
2. 导入完成后打开 `http://localhost:8001/chat.html` 提问，页面通过 SSE 实时显示检索进度与答案增量。

---

## 🔌 API 说明

### 导入服务（:8000）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` `/import` | 文档上传页面 |
| POST | `/upload` | 上传文档（multipart：`file`、`overwrite`），返回 `task_id`、`doc_id` |
| GET | `/status/{task_id}` | 查询导入任务状态与节点进度 |

### 查询服务（:8001）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat.html` | 问答页面 |
| POST | `/query` | 提问（`query`、`session_id`、`is_stream`）；非流式直接返回答案，流式返回 `task_id` |
| GET | `/stream/{task_id}` | SSE 流：`progress`（节点进度）、`delta`（答案增量）、`final`（完整答案） |
| GET | `/history/{session_id}` | 查询会话历史（默认最近 50 条） |
| DELETE | `/history/{session_id}` | 清空会话历史 |

---

## 📊 效果评估（RAGAS）

**评估方法**

- 自建评测集：10 道问题覆盖说明书的安全手册、规格、技术指标、测量方法等章节，答案是文档原文口径
- 每条问题**真实调用查询流程**：`context` 取流程 state 中精排后的 `reranked_docs`，`answer` 取最终生成答案
- 评估与被评估系统使用**同一套模型**（同 LLM、同 BGE-M3 嵌入、同 Reranker），只对比检索策略的影响
- 指标：faithfulness、answer_relevancy、context_precision、context_recall、answer_correctness（RAGAS）

**三轮对照结果（同一评测集）**

| 指标 | 优化前 | 仅切片优化 | 切片 + 网页兜底门控 |
|---|---|---|---|
| faithfulness | 0.867 | 0.925 | **1.000** |
| answer_relevancy | 0.689 | 0.866 | 0.864 |
| context_precision | 0.569 | 0.534 | **0.634** |
| context_recall | 0.750 | 1.000 | 1.000 |
| answer_correctness | 0.722 | 0.786 | **0.840** |

**结论**

1. 细粒度切片解决"文档里有、检索却召不回来"：context_recall 0.75 → 1.00；
2. 把外部搜索降级为条件兜底，消除了网页通用内容污染：上下文中的网页切片占比 42% → 0，context_precision 0.534 → 0.634，faithfulness → 1.00；
3. context_precision 是序敏感指标、answer_correctness 计入表述差异，二者需与 context_recall、人工复核交叉解读。

---

## ⚙️ 主要检索配置（`knowledge/processor/query_process/config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_SEARCH_LIMIT` | 10 | 向量检索召回条数 |
| `HYDE_SEARCH_LIMIT` | 5 | HyDE 检索召回条数 |
| `RRF_K` / `RRF_KG_WEIGHT` / `RRF_MAX_RESULTS` | 60 / 0.7 / 10 | RRF 平滑参数、图谱路权重、融合后条数 |
| `RERANK_MAX_TOP_K` / `RERANK_MIN_TOP_K` | 15 / 8 | 精排后保留上下文的上下限 |
| `RERANK_GAP_RATIO` / `RERANK_GAP_ABS` | 0.5 / 1.0 | 断崖检测的相对/绝对阈值 |
| `WEB_FALLBACK_ENABLED` | 1 | 是否启用网络检索兜底 |
| `WEB_FALLBACK_MIN_LOCAL_CHUNKS` | 3 | 本地切片数低于该值才触发网络兜底 |
| `WEB_FALLBACK_MIN_LOCAL_SCORE` | 0 | 本地相似度兜底阈值（0 = 关闭；本地分数区分度不足时不建议开启） |
| `MAX_CONTEXT_CHARS` | 12000 | 送入答案生成的上下文字符预算 |

---

## ⚠️ 已知问题与 Roadmap

| 方向 | 现状 | 计划 |
|---|---|---|
| 断崖截断 | 在当前参数（`min_top_k=8`）下判断窗口被硬保底屏蔽，实测 10 题 0 次触发，等价于保留全部候选 | 扫描起点前移 + "正分 + 保底"策略，替代固定条数 |
| 清单型切片 | 《技术指标说明》《规格》等 20 行清单被整块切分，某一知识点提问时精排得分偏低（−0.8 ~ −3.5） | 按行拆分为独立切片，使每个知识点可独立召回 |
| 答案格式 | 个别答案尾部会带图片链接甚至未填充占位符 | 在答案 prompt 中约束输出格式（禁止图片链接/占位符） |
| 网络检索健壮性 | 兜底分支的网络异常会中断整条查询 | 增加 try/except 降级为"空召回" |
| 评测集规模 | 10 题，覆盖一个产品文档 | 扩充到每类知识点 2~3 题，并加入"库中无答案"的对照题 |
| 工程化 | `requirements.txt` 未纳入版本管理；完整评测脚本与调优记录待提交 | 补齐依赖清单与评估代码，接入 CI |

---

## 📄 License

本项目暂未指定开源许可证；如需用于商业场景，请先联系作者。

# 多模态 RAG 个人知识库问答系统

基于 **Python / FastAPI / ChromaDB** 从零构建的端到端 RAG（Retrieval-Augmented Generation）系统，支持 PDF、Markdown、图片等多模态文档的自动解析、混合索引与智能问答。

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/fastapi-0.115+-green" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/chromadb-0.5+-orange" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License"/>
</p>

---

## 架构总览

```
                            ┌──────────────────────────────┐
                            │     FastAPI / CLI 入口       │
                            └──────────────┬───────────────┘
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              POST /index            POST /chat            POST /eval
                    │                      │                      │
    ┌───────────────┼──────────────────────┼──────────────────────┼───────────────┐
    │               ▼                      ▼                      ▼               │
    │  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐     │
    │  │   离线索引管线        │  │   在线问答管线        │  │  效果评估     │     │
    │  │                      │  │                      │  │              │     │
    │  │  文档解析 (PDF/MD/IMG)│  │  ①  Query Rewriting  │  │  检索指标     │     │
    │  │     ↓                │  │  ②  Query Routing    │  │  MRR/Hit/    │     │
    │  │  结构化切分 (Chunk)   │  │  ③  三路召回          │  │  Prec/Recall/│     │
    │  │     ↓                │  │     · BM25 稀疏      │  │  NDCG        │     │
    │  │  Embedding 向量化     │  │     · Chroma 向量    │  │              │     │
    │  │     ↓                │  │     · 关键词匹配     │  │  生成指标     │     │
    │  │  BM25 + Chroma 双索引 │  │  ④  RRF 融合去重     │  │  Faithfulness│     │
    │  │     ↓                │  │  ⑤  gte-rerank 精排   │  │  Relevancy   │     │
    │  │  SQLite 元数据管理    │  │  ⑥  置信度检测/兜底   │  │  Context Prec│     │
    │  │                      │  │  ⑦  qwen-plus 生成    │  │              │     │
    │  └──────────────────────┘  └──────────────────────┘  └──────────────┘     │
    │                                                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐    │
    │  │                     工程化基础设施                                 │    │
    │  │  · 增量索引 (SHA256 Hash)  · 文件监听 (Watchdog)                   │    │
    │  │  · L1/L2/L3 三层缓存       · 可插拔组件工厂 (Component Registry)    │    │
    │  └──────────────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────────────┘
```

---

## 核心特性

### 离线索引
- **多格式解析**：PDF（PyMuPDF，支持标题层级检测、表格识别、图片提取）、Markdown（轻量 AST 解析，保留代码块/表格/图片引用）、图片（PaddleOCR 离线文字识别 + 可选 qwen-vl-plus 视觉理解）
- **语义感知切分**：有标题文档使用 LangChain MarkdownHeaderTextSplitter 按 h1-h4 层级切分，无标题文档使用 RecursiveCharacterTextSplitter 递进切分；chunk_size=512，overlap=128；每 chunk 绑定 source、page、section_title、doc_hash 等元数据
- **混合双索引**：BM25 稀疏索引（rank-bm25 + jieba 中文分词） + ChromaDB 稠密向量索引（阿里云 text-embedding-v3，HNSW，Cosine 距离），通过 SQLite 管理文档元数据与索引映射
- **图片处理**：PaddleOCR 本地离线提取图片中的文字参与索引；可选 qwen-vl-plus 多模态理解生成图片描述

### 在线问答（7 步管线）
| 步骤 | 模块 | 职责 |
|------|------|------|
| ① | **QueryRewriter** | LLM 驱动意图识别（knowledge_qa / document_search / chart_understanding / general）、关键词提取、查询变体生成（3-5 个）；LLM 调用失败自动降级为 jieba 规则方案 |
| ② | **QueryRouter** | 关键词规则快速预判 + LLM 细粒度分类，路由到 text / multimodal / external 三条分支 |
| ③ | **MultiRecall** | BM25 稀疏检索(Top20) + Chroma 向量检索(Top20) + 关键词匹配(Top10) 三路并行召回 |
| ④ | RRF 融合 | Reciprocal Rank Fusion 去重合并，公式 `score(d) = Σ 1/(k + rank_i(d))`，k=60 |
| ⑤ | **Reranker** | 阿里云 gte-rerank Cross-Encoder 对融合后候选进行语义精排，输出 TopK；API 失败自动回退 RRF 排序 |
| ⑥ | **ConfidenceChecker** | 以最高 RRF 分数为信号，低于阈值时触发查询改写二次检索，合并去重两次结果 |
| ⑦ | **Generator** | qwen-plus 生成带 `[来源编号] p.页码 §章节` 引用标注的结构化答案；支持 qwen-vl-plus 多模态图文联合理解 |

### 工程化能力
- **增量索引**：SHA256 哈希变更检测，仅处理新增/修改文件，自动清理已删除文档；BM25 删除后重建
- **文件监听**：Watchdog 监控 `data/docs/` 目录，文件变更自动触发增量索引（2 秒防抖）
- **三层缓存**：L1 答案级（24h TTL）/ L2 检索结果级（1h TTL）/ L3 Embedding 级（永久），SQLite 存储（可选 Redis）
- **可插拔组件**：ComponentRegistry 工厂模式，Embedding / LLM / Reranker / Chunker 均可按名替换
- **灵活配置**：YAML 配置文件 + `${ENV_VAR}` 环境变量模板注入，支持多环境切换

### 效果评估
- **检索指标**：MRR、Hit Rate@K、Precision@K、Recall@K、NDCG@K（手工实现，无框架依赖）
- **生成指标**：基于 LLM-as-Judge 的 Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确度）
- **消融实验**：支持分别关闭 Rerank、Query Rewriting、关键词召回来量化各模块贡献

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| LLM | 阿里云 qwen-plus | 中文生成能力强，OpenAI 兼容接口 |
| VLM | 阿里云 qwen-vl-plus | 视觉语言模型，图表/截图理解 |
| Embedding | text-embedding-v3 | 阿里云，1024 维，中文表现优秀 |
| Reranker | gte-rerank | 阿里云 Cross-Encoder 重排序 |
| 向量库 | ChromaDB (PersistentClient) | 嵌入式，HNSW 索引，Cosine 距离 |
| 稀疏检索 | rank-bm25 (BM25Okapi) | 无外部依赖，结合 jieba 中文分词 |
| PDF 解析 | PyMuPDF (fitz) | 文本块 + 字体信息 + 图片提取 |
| OCR | PaddleOCR | 离线中文识别，无 API 调用限制 |
| 文本切分 | LangChain text-splitters | MarkdownHeader + Recursive 双策略 |
| API 框架 | FastAPI + uvicorn | 异步高性能，自动 Swagger 文档 |
| 文件监听 | watchdog | 跨平台文件系统事件 |
| 缓存/元数据 | SQLite | 零配置嵌入式数据库 |
| 评估 | 手工实现 + LLM-as-Judge | 检索指标自实现，生成指标用 LLM 评分 |
| 配置 | YAML + python-dotenv | 环境变量模板注入 |

---

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key（[免费获取](https://dashscope.console.aliyun.com/)）

### 1. 安装

```bash
cd git_rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

> **注意：** PaddleOCR 首次运行会自动下载模型文件（约 100MB），需保持网络连接。

### 2. 配置

```bash
# 编辑 .env 或设置环境变量
export ALIYUN_API_KEY=sk-xxxxxxxxxxxxx
```

所有可调参数在 `config/default.yaml` 中，支持 `${ALIYUN_API_KEY}` 格式读取环境变量。

### 3. 准备文档

将 PDF、Markdown、图片文件放入 `data/docs/`：

```
data/docs/
├── ai-agent-deep-dive.pdf
├── engineering_notes.md
├── diagram.png
└── ...
```

支持格式：`.pdf` `.md` `.markdown` `.png` `.jpg` `.jpeg` `.bmp`

### 4. 构建索引

```bash
# 增量索引（默认，仅处理新增/变更文档）
python cli.py index --path ./data/docs

# 全量重建
python cli.py index --path ./data/docs --full
```

输出示例：
```
[增量] 解析文档: ai-agent-deep-dive-v2.pdf
[Index] 共 156 个chunks，开始构建索引...
[Index] 构建BM25索引...
[Index] 构建Chroma向量索引...
[完成] 索引构建完毕: {'scanned': 5, 'to_index': 3, 'unchanged': 2}
```

### 5. 交互式问答

```bash
python cli.py chat
```

```
>>> 什么是RAG系统？
[意图: knowledge_qa]
[路由: text]

RAG（检索增强生成）是一种结合信息检索与文本生成的AI架构...

--- 引用来源 ---
  [1] ai-agent-deep-dive-v2.pdf (p.3, §2.1 RAG概述)
  [2] engineering_notes.md (p.N/A, §基础知识)
[Token用量: {'prompt_tokens': 1234, 'completion_tokens': 256, 'total_tokens': 1490}]
```

### 6. 启动 API 服务

```bash
python cli.py serve --port 8000
# 浏览器打开 http://localhost:8000/docs 查看 Swagger 文档
```

---

## 使用方法

### CLI 命令速查

| 命令 | 用途 | 常用参数 |
|------|------|----------|
| `python cli.py index` | 构建/更新索引 | `--path` 文档目录, `--full` 全量重建 |
| `python cli.py chat` | 交互式问答 | `--top-k` 返回片段数, `--no-cache` 禁用缓存 |
| `python cli.py eval` | 效果评估 | `--dataset` 数据集路径, `--top-k` 片段数 |
| `python cli.py serve` | 启动 API 服务 | `--host`, `--port` |
| `python cli.py watch` | 文件监听 + 自动索引 | `--path` 监听目录 |

### API 接口

启动服务后访问 `http://localhost:8000/docs` 获取完整交互式文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/chat` | 问答接口 |
| POST | `/api/v1/index` | 构建/更新索引 |
| POST | `/api/v1/eval` | 运行效果评估 |

**问答请求示例：**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是混合检索？", "top_k": 10, "enable_cache": true}'
```

**响应：**

```json
{
  "query": "什么是混合检索？",
  "answer": "混合检索是指同时使用稀疏检索（如BM25）和稠密检索（如向量相似度）来提升召回质量的方法...",
  "references": [
    {
      "ref_id": 1,
      "source": "ai-agent-deep-dive-v2.pdf",
      "page": "12",
      "section": "混合检索策略",
      "content_preview": "混合检索结合了关键词匹配和语义理解两种范式的优势..."
    }
  ],
  "model": "qwen-plus",
  "was_retried": false,
  "token_usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
  "cached": false
}
```

### 文件监听

```bash
python cli.py watch --path ./data/docs
```

启动后持续监控文档目录，文件变更（新增/修改/删除）自动触发增量索引，无需手动执行 index 命令。

### 评估数据集

编辑 `data/eval_queries.json`：

```json
[
  {
    "query": "什么是大语言模型的幻觉现象？",
    "ground_truth": "幻觉是指模型生成的内容与事实不符...",
    "contexts": ["chunk_id_1", "chunk_id_2"],
    "category": "text"
  }
]
```

运行评估：

```bash
python cli.py eval --dataset ./data/eval_queries.json
```

输出：

```
检索评估结果:
  mrr                 : 0.7234
  hit_rate@5          : 0.8500
  precision@5         : 0.4600
  recall@5            : 0.5800
  ndcg@5              : 0.6120

生成评估结果:
  faithfulness        : 0.8210
  answer_relevancy    : 0.8920
  context_precision   : 0.7640
```

---

## 项目结构

```
git_rag/
├── cli.py                          # CLI 统一入口 (index/chat/eval/serve/watch)
├── requirements.txt                # Python 依赖清单
├── .env                            # 环境变量 (API Key)
├── .env.example                    # 环境变量模板
├── README.md
│
├── config/
│   └── default.yaml                # 全局配置 (模型/索引/检索/缓存/服务)
│
├── data/                           # 数据目录 (gitignored)
│   ├── docs/                       # 📂 待索引文档
│   ├── images/                     # PDF 提取的图片
│   ├── eval_queries.json          # 评估数据集
│   ├── chroma_db/                  # ChromaDB 向量持久化
│   ├── bm25_index.pkl             # BM25 序列化索引
│   ├── metadata.db                 # 文档元数据 (SQLite)
│   └── cache.db                    # 分层缓存 (SQLite)
│
└── src/
    ├── core/                       # ── 核心基础设施 ──
    │   ├── config.py               # 配置加载 (YAML + 环境变量模板注入)
    │   └── factory.py              # 组件注册表 (可插拔工厂模式)
    │
    ├── offline/                    # ── 离线索引管线 ──
    │   ├── parsers/
    │   │   ├── base.py             # BaseParser 抽象基类 + Document/ParsedElement
    │   │   ├── pdf_parser.py       # PDF 解析器 (PyMuPDF, 标题/表格/图片)
    │   │   ├── markdown_parser.py  # Markdown 解析器 (AST, 代码块/表格)
    │   │   └── image_parser.py     # 图片解析器 (PaddleOCR + 可选 VLM)
    │   ├── chunker.py              # 文档切分 (MarkdownHeader + Recursive 双策略)
    │   ├── embeddings.py           # 阿里云 text-embedding-v3 封装
    │   └── index_builder.py        # HybridIndexBuilder (BM25 + Chroma + SQLite)
    │
    ├── online/                     # ── 在线问答管线 ──
    │   ├── query_rewriter.py       # 查询改写 (意图识别 + 关键词 + 扩展变体)
    │   ├── query_router.py         # 问题路由 (text/multimodal/external)
    │   ├── multi_recall.py         # 三路召回 (BM25 + Vector + Keyword) + RRF 融合
    │   ├── reranker.py             # gte-rerank Cross-Encoder 重排序
    │   ├── confidence_check.py     # 置信度检测 + 二次检索兜底
    │   ├── context_assembler.py    # 上下文拼装 + 引用标注注入
    │   └── generator.py            # LLM 生成 (qwen-plus / qwen-vl-plus)
    │
    ├── engineering/                # ── 工程化 ──
    │   ├── incremental_index.py    # 增量索引管理器 (SHA256 变更检测)
    │   ├── file_watcher.py         # Watchdog 文件监听 (带防抖)
    │   └── cache_manager.py        # L1/L2/L3 三层缓存 (SQLite)
    │
    ├── evaluation/                 # ── 效果评估 ──
    │   ├── test_dataset.py         # 评估数据集加载与管理
    │   ├── metrics_retrieval.py    # 检索指标 (MRR/HR/Prec/Recall/NDCG@K)
    │   └── metrics_generation.py   # 生成质量评估 (LLM-as-Judge: 忠实度/相关性/精确度)
    │
    └── api/                        # ── API 服务 ──
        ├── main.py                 # FastAPI 应用入口 + 生命周期
        ├── routes.py               # 路由 + RAGService 服务封装
        └── schemas.py              # Pydantic 请求/响应模型
```

---

## 配置参数

`config/default.yaml` 中所有关键参数：

```yaml
# ── 阿里云模型 ──
aliyun:
  api_key: "${ALIYUN_API_KEY}"            # 从 .env 读取
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  llm_model: "qwen-plus"                  # 问答生成
  vlm_model: "qwen-vl-plus"              # 视觉理解
  embedding_model: "text-embedding-v3"    # 向量化
  rerank_model: "gte-rerank"             # 重排序

# ── 索引参数 ──
index:
  chunk_size: 512                         # 切分窗口大小
  chunk_overlap: 128                      # 切分重叠大小
  embedding_batch_size: 10               # Embedding API 批量大小

# ── 检索参数 ──
retrieval:
  top_k_bm25: 20                          # BM25 召回数
  top_k_vector: 20                        # 向量召回数
  top_k_keyword: 10                       # 关键词召回数
  top_k_final: 10                         # 最终输入 LLM 的片段数
  rrf_k: 60                               # RRF 融合常数
  confidence_threshold: 0.3               # 置信度阈值
  max_rewrite_attempts: 1                 # 二次检索最大尝试次数
  enable_rerank: true                     # 是否启用重排序

# ── 缓存 ──
cache:
  l1_answer_ttl: 86400                    # 答案缓存 (24小时)
  l2_retrieval_ttl: 3600                  # 检索结果缓存 (1小时)
  l3_embedding_ttl: 0                     # Embedding 缓存 (0=永久)

# ── API 服务 ──
api:
  host: "0.0.0.0"
  port: 8000
```

---

## 组件可插拔替换

通过 `src/core/factory.py` 的 ComponentRegistry，所有核心组件均可按名替换：

```python
from src.core.factory import register_embedder, register_llm, registry

# 注册自定义 Embedding 模型
@register_embedder("openai")
def create_openai_embedder():
    from openai import OpenAI
    client = OpenAI(api_key="...")
    return lambda texts: client.embeddings.create(
        model="text-embedding-3-small", input=texts
    ).data

# 注册本地 LLM
@register_llm("ollama")
def create_ollama_llm():
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:11434/v1")

# 使用
embedder = registry.get("embedder", "openai")
llm = registry.get("llm", "ollama")
```

---

## 调优指南

### Chunk 参数

| 场景 | chunk_size | chunk_overlap |
|------|------------|---------------|
| 精准问答 | 256 | 64 |
| 通用知识库（推荐） | 512 | 128 |
| 长文档摘要 | 1024 | 256 |

### 召回参数

- **rrf_k**：越小排名优先权越大（多样性降低），越大各路结果越均衡（噪声可能增加），推荐 40-80
- **confidence_threshold**：提高→更激进触发二次检索→召回率↑延迟↑；降低→依赖首次结果→速度↑可能漏召回

### Reranker 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 阿里云 gte-rerank（当前） | 中文效果好，API 调用简单 | 需网络，有成本 |
| 本地 BGE-Reranker-v2-m3 | 免费离线 | 需 GPU，部署复杂 |
| 无 Rerank | 零延迟零成本 | TopK 相关性明显下降 |

---

## 常见问题

<details>
<summary><b>Q: PyMuPDF 安装失败 / import fitz 报错</b></summary>

```bash
pip uninstall PyMuPDF -y && pip install PyMuPDF
```
</details>

<details>
<summary><b>Q: PaddleOCR 首次运行慢</b></summary>

首次运行会下载检测+识别模型（约 100MB），后续从缓存加载，速度正常。
</details>

<details>
<summary><b>Q: 索引加载失败</b></summary>

```bash
rm -rf data/chroma_db data/bm25_index.pkl data/metadata.db
python cli.py index --path ./data/docs --full
```
</details>

<details>
<summary><b>Q: 如何添加新文档格式？</b></summary>

1. 创建解析器继承 `src/offline/parsers/base.py` 的 `BaseParser`
2. 在 `src/core/factory.py` 注册：`@register_parser("docx")`
3. 在 CLI/API 的 `parse_file` 函数中添加扩展名路由
</details>

---

## License

MIT

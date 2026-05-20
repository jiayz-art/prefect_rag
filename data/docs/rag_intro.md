# RAG系统介绍

## 什么是RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的AI架构。它通过先检索相关文档片段，再基于这些片段生成回答，有效减少了大型语言模型的"幻觉"问题。

## RAG的核心组件

### 1. 文档解析器
负责将不同格式的文档（PDF、Markdown、Word等）解析为结构化文本。

### 2. 文本切分器（Chunker）
将长文档切分为适当大小的片段（chunk），通常为256-1024个token，保留一定的重叠以防止信息在边界处丢失。

### 3. 向量化模型（Embedding Model）
将文本片段转换为稠密向量，常用的有OpenAI的text-embedding-3、BGE系列等。

### 4. 向量数据库
存储和检索向量化后的文本片段，常用的有Chroma、Milvus、Pinecone等。

### 5. 检索器（Retriever）
根据用户查询从向量数据库中检索最相关的文档片段。

### 6. 生成器（Generator）
基于检索到的上下文信息，使用大语言模型生成最终回答。

## 混合检索

混合检索结合了两种检索策略：

1. **稀疏检索（BM25）**：基于关键词匹配，适合精确查找
2. **稠密检索（向量）**：基于语义相似度，适合理解用户意图

两者通过RRF（Reciprocal Rank Fusion）算法融合结果，通常能获得比单独使用任一种更好的效果。

## RRF算法

RRF的公式为：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中k为常数（通常取60），rank_i(d)是文档d在第i路召回中的排名。

## 评估指标

### 检索评估
- MRR（Mean Reciprocal Rank）：平均倒数排名
- Hit Rate@K：前K个结果中命中率
- NDCG@K：归一化折损累计增益

### 生成评估
- Faithfulness（忠实度）：答案是否忠于上下文
- Answer Relevancy（相关性）：答案与问题的关联度
- Context Precision（上下文精确度）：检索到的上下文质量

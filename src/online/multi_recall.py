"""多路召回模块 — BM25 + 向量 + 关键词三路召回 + RRF融合。"""
from typing import Dict, List, Tuple

from src.core.config import config
from src.offline.embeddings import AliyunEmbedder
from src.offline.index_builder import HybridIndexBuilder


class MultiRecall:
    """多路召回器 — BM25稀疏检索 + Chroma向量检索 + 关键词匹配，RRF融合。"""

    def __init__(self, index_builder: HybridIndexBuilder):
        self.index = index_builder
        self.embedder = AliyunEmbedder()
        self.top_k_bm25 = config.get("retrieval.top_k_bm25", 20)
        self.top_k_vector = config.get("retrieval.top_k_vector", 20)
        self.top_k_keyword = config.get("retrieval.top_k_keyword", 10)
        self.rrf_k = config.get("retrieval.rrf_k", 60)

    def recall(self, query: str, keywords: List[str] = None) -> List[Dict]:
        """执行多路召回并RRF融合去重。"""
        results_bm25 = self._recall_bm25(query)
        results_vector = self._recall_vector(query)
        results_keyword = self._recall_keyword(query, keywords or [])

        merged = self._rrf_merge([results_bm25, results_vector, results_keyword])
        return merged

    def _recall_bm25(self, query: str) -> List[Tuple[int, float]]:
        """BM25关键词召回。返回 [(chunk_index, score), ...]"""
        if not self.index._bm25_index:
            return []

        tokenized = self.index._tokenize(query)
        scores = self.index._bm25_index.get_scores(tokenized)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:self.top_k_bm25]

    def _recall_vector(self, query: str) -> List[Tuple[int, float]]:
        """Chroma向量召回。通过文本内容匹配找到对应的BM25 chunk索引。"""
        if not self.index._chroma_collection:
            return []

        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.index._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k_vector,
            )

            output = []
            if results and results.get("ids") and results["ids"][0]:
                docs = results.get("documents", [[]])[0]
                distances = results.get("distances", [[0]])[0]

                for i, doc_text in enumerate(docs):
                    distance = distances[i] if i < len(distances) else 0
                    score = 1.0 - distance if distance else 1.0

                    # 通过文本内容在BM25 chunk列表中查找对应索引
                    idx = self._find_chunk_index(doc_text)
                    if idx >= 0:
                        output.append((idx, score))
            return output
        except Exception as e:
            print(f"[MultiRecall] 向量召回失败: {e}")
            return []

    def _find_chunk_index(self, text: str) -> int:
        """在BM25 chunk列表中通过文本内容匹配找到索引。"""
        if not text or not self.index._bm25_chunks:
            return -1
        text_stripped = text.strip()
        # 精确匹配
        for i, chunk in enumerate(self.index._bm25_chunks):
            if chunk["content"].strip() == text_stripped:
                return i
        # 前缀匹配（前80字符）
        prefix = text_stripped[:80]
        for i, chunk in enumerate(self.index._bm25_chunks):
            if chunk["content"].strip()[:80] == prefix:
                return i
        # 子串包含匹配
        for i, chunk in enumerate(self.index._bm25_chunks):
            if text_stripped[:50] in chunk["content"]:
                return i
        return -1

    def _recall_keyword(self, query: str, keywords: List[str]) -> List[Tuple[int, float]]:
        """关键词精确/模糊匹配召回。"""
        if not self.index._bm25_chunks:
            return []

        search_terms = [query] + (keywords or [])
        scores_map: Dict[int, float] = {}

        for term in search_terms:
            term_lower = term.lower()
            for i, chunk in enumerate(self.index._bm25_chunks):
                content = chunk["content"].lower()
                if term_lower in content:
                    count = content.count(term_lower)
                    bonus = min(count * len(term) * 0.01, 2.0)
                    scores_map[i] = scores_map.get(i, 0) + 1.0 + bonus

        sorted_items = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:self.top_k_keyword]

    def _rrf_merge(self, result_lists: List[List[Tuple[int, float]]]) -> List[Dict]:
        """Reciprocal Rank Fusion 融合多路召回结果。"""
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict] = {}

        for results in result_lists:
            for rank, (chunk_idx, _) in enumerate(results):
                if chunk_idx < 0:
                    continue
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + rrf_score
                if chunk_idx not in chunk_map and chunk_idx < len(self.index._bm25_chunks):
                    chunk_map[chunk_idx] = self.index._bm25_chunks[chunk_idx]

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for chunk_idx, rrf_score in sorted_items:
            if chunk_idx in chunk_map:
                chunk = dict(chunk_map[chunk_idx])
                chunk["_rrf_score"] = round(rrf_score, 4)
                chunk["_chunk_index"] = chunk_idx
                merged.append(chunk)

        return merged

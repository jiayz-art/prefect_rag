"""检索效果评估指标 — MRR, Hit Rate, Precision@K, Recall@K, NDCG@K。"""
import math
from typing import Dict, List


class RetrievalMetrics:
    """检索评估指标计算器。"""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],    # 每个查询召回的文档ID列表
        relevant_docs: List[List[str]],      # 每个查询的真实相关文档ID列表
    ) -> Dict:
        """计算所有检索指标。

        Args:
            queries: 查询列表
            retrieved_docs: 每个查询召回的文档ID列表（按相关性排序）
            relevant_docs: 每个查询的真实相关文档ID列表
        """
        results = {
            "num_queries": len(queries),
            "mrr": self.mrr(retrieved_docs, relevant_docs),
        }

        for k in self.k_values:
            results.update({
                f"hit_rate@{k}": self.hit_rate_at_k(retrieved_docs, relevant_docs, k),
                f"precision@{k}": self.precision_at_k(retrieved_docs, relevant_docs, k),
                f"recall@{k}": self.recall_at_k(retrieved_docs, relevant_docs, k),
                f"ndcg@{k}": self.ndcg_at_k(retrieved_docs, relevant_docs, k),
            })

        return results

    def mrr(self, retrieved: List[List[str]], relevant: List[List[str]]) -> float:
        """Mean Reciprocal Rank。"""
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            for rank, doc_id in enumerate(ret):
                if doc_id in rel_set:
                    scores.append(1.0 / (rank + 1))
                    break
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def hit_rate_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Hit Rate@K — 前K个结果中至少命中一个相关文档的查询比例。"""
        hits = 0
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            if any(doc in rel_set for doc in ret[:k]):
                hits += 1
        return hits / len(retrieved) if retrieved else 0.0

    def precision_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Precision@K — 前K个结果中相关文档占比的平均值。"""
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            hits = sum(1 for doc in ret[:k] if doc in rel_set)
            scores.append(hits / k if k > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def recall_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """Recall@K — 前K个结果召回的相关文档占所有相关文档比例的平均值。"""
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            if not rel_set:
                continue
            hits = sum(1 for doc in ret[:k] if doc in rel_set)
            scores.append(hits / len(rel_set))
        return sum(scores) / len(scores) if scores else 0.0

    def ndcg_at_k(self, retrieved: List[List[str]], relevant: List[List[str]], k: int) -> float:
        """NDCG@K — 归一化折损累计增益。"""
        scores = []
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            dcg = 0.0
            for i, doc_id in enumerate(ret[:k]):
                if doc_id in rel_set:
                    dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

            # IDCG (理想排序)
            ideal_hits = min(len(rel_set), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

            scores.append(dcg / idcg if idcg > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

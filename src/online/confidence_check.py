"""置信度检测与二次检索模块。"""
from typing import Dict, List

from src.core.config import config


class ConfidenceChecker:
    """低置信度检测器 — 当首次召回分数不足时触发二次检索。"""

    def __init__(self, recall_fn, rewrite_fn):
        self.threshold = config.get("retrieval.confidence_threshold", 0.3)
        self.max_attempts = config.get("retrieval.max_rewrite_attempts", 1)
        self._recall = recall_fn
        self._rewrite = rewrite_fn

    def check_and_retry(
        self,
        query: str,
        first_results: List[Dict],
        keywords: List[str] = None,
    ) -> tuple:
        """检测置信度，必要时触发二次检索。"""
        confidence = self._compute_confidence(first_results)

        if confidence >= self.threshold or self.max_attempts <= 0:
            return first_results, False

        print(f"[ConfidenceCheck] 置信度不足 ({confidence:.3f} < {self.threshold})，触发二次检索...")

        # 二次检索：使用改写后的查询变体
        rewritten = self._rewrite(query)
        # RewrittenQuery 是 dataclass，用属性访问
        if hasattr(rewritten, 'expanded_queries') and rewritten.expanded_queries:
            retry_query = rewritten.expanded_queries[0]
        else:
            retry_query = query

        retry_results = self._recall(retry_query, keywords=None)

        # 合并两次结果（去重）
        seen_indices = set()
        merged = []
        for r in first_results + retry_results:
            idx = r.get("_chunk_index", id(r))
            if idx not in seen_indices:
                seen_indices.add(idx)
                merged.append(r)

        print(f"[ConfidenceCheck] 二次检索完成，合并后共 {len(merged)} 条结果")
        return merged, True

    def _compute_confidence(self, results: List[Dict]) -> float:
        """计算召回结果的置信度分数。"""
        if not results:
            return 0.0

        best_rrf = max(
            (r.get("_rrf_score", 0) for r in results),
            default=0,
        )

        normalized = min(best_rrf * 2.0, 1.0)
        return normalized

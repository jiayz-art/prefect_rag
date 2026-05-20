"""Rerank精排模块 — 阿里云 DashScope gte-rerank 原生API调用。"""
import json
from typing import Dict, List

import requests

from src.core.config import config


# DashScope Rerank API 地址
DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class Reranker:
    """重排序器 — 通过 HTTP 调用阿里云 DashScope gte-rerank。"""

    def __init__(self):
        self.enabled = config.get("retrieval.enable_rerank", True)
        self.top_k = config.get("retrieval.top_k_final", 10)
        self.api_key = config.get("aliyun.api_key")
        self.model = config.get("aliyun.rerank_model", "gte-rerank")

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """对候选文档重排序，返回TopK结果。"""
        if not self.enabled or not candidates:
            return candidates[:self.top_k]

        if len(candidates) <= self.top_k:
            return candidates

        try:
            documents = [c["content"] for c in candidates]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "top_n": self.top_k,
                    "return_documents": False,
                },
            }

            resp = requests.post(DASHSCOPE_RERANK_URL, headers=headers, json=body, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("output", {}).get("results", [])
                reranked = []
                for r in sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True):
                    idx = r.get("index", 0)
                    if idx < len(candidates):
                        c = dict(candidates[idx])
                        c["_rerank_score"] = round(r.get("relevance_score", 0), 4)
                        reranked.append(c)
                return reranked[:self.top_k] if reranked else self._fallback(candidates)

            return self._fallback(candidates)

        except Exception as e:
            print(f"[Reranker] API调用失败，回退到RRF排序: {e}")
            return self._fallback(candidates)

    def _fallback(self, candidates: List[Dict]) -> List[Dict]:
        """回退：按RRF分数降序排列。"""
        return sorted(
            candidates,
            key=lambda x: x.get("_rrf_score", 0),
            reverse=True,
        )[:self.top_k]

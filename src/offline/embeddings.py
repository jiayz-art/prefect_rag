"""Embedding向量化模块 — 阿里云 text-embedding-v3 封装。"""
from typing import List

from openai import OpenAI

from src.core.config import config


class AliyunEmbedder:
    """阿里云Embedding模型封装。"""

    def __init__(self):
        self._client = None
        self.model = config.get("aliyun.embedding_model", "text-embedding-v3")
        self.batch_size = config.get("index.embedding_batch_size", 10)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=config.get("aliyun.api_key"),
                base_url=config.get("aliyun.base_url"),
            )
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本，自动分批。"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """向量化单条查询。"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """调用API向量化一批文本。"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # 按输入顺序返回
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]

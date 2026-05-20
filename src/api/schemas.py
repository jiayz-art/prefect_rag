"""Pydantic请求/响应模型。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(default=10, description="返回的文档片段数量")
    enable_cache: bool = Field(default=True, description="是否使用缓存")


class Reference(BaseModel):
    ref_id: int
    source: str
    page: str
    section: str
    content_preview: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    references: List[Reference] = []
    model: str = ""
    was_retried: bool = False
    token_usage: Dict[str, int] = {}
    cached: bool = False


class IndexRequest(BaseModel):
    path: str = Field(default="./data/docs", description="文档目录路径")
    incremental: bool = Field(default=True, description="是否增量索引")


class IndexResponse(BaseModel):
    status: str
    summary: Dict[str, Any] = {}


class EvalRequest(BaseModel):
    dataset_path: str = Field(default="./data/eval_queries.json", description="评估数据集路径")
    top_k: int = Field(default=10)


class EvalResponse(BaseModel):
    retrieval_metrics: Dict[str, float] = {}
    generation_metrics: Dict[str, float] = {}
    num_samples: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    index_loaded: bool = False

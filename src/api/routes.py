"""FastAPI路由定义 — /chat, /index, /eval, /health。"""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    Reference,
)
from src.core.config import config
from src.engineering.cache_manager import CacheManager

router = APIRouter()

# 全局服务实例（由 main.py 初始化）
_rag_service: Optional["RAGService"] = None


def set_rag_service(service):
    global _rag_service
    _rag_service = service


class RAGService:
    """RAG服务封装 — 管理索引和检索管线的生命周期。"""

    def __init__(self):
        from src.offline.index_builder import HybridIndexBuilder
        from src.online.query_rewriter import QueryRewriter
        from src.online.query_router import QueryRouter
        from src.online.multi_recall import MultiRecall
        from src.online.reranker import Reranker
        from src.online.confidence_check import ConfidenceChecker
        from src.online.context_assembler import ContextAssembler
        from src.online.generator import Generator

        self.index = HybridIndexBuilder()
        self.rewriter = QueryRewriter()
        self.router = QueryRouter()
        self.reranker = Reranker()
        self.assembler = ContextAssembler()
        self.generator = Generator()
        self.cache = CacheManager()
        self.recall_engine: Optional[MultiRecall] = None
        self.confidence_checker: Optional[ConfidenceChecker] = None
        self._index_loaded = False

    def load_index(self) -> bool:
        """加载已有索引。"""
        bm25_ok = self.index.load_bm25()
        chroma_ok = self.index.load_chroma()
        if bm25_ok and chroma_ok:
            self.recall_engine = MultiRecall(self.index)
            self.confidence_checker = ConfidenceChecker(
                recall_fn=self.recall_engine.recall,
                rewrite_fn=self.rewriter.rewrite,
            )
            self._index_loaded = True
            print(f"[RAGService] 索引加载成功 (BM25: {len(self.index._bm25_chunks)} chunks)")
        return self._index_loaded

    @property
    def is_loaded(self) -> bool:
        return self._index_loaded

    def chat(self, query: str, top_k: int = 10, enable_cache: bool = True) -> dict:
        """执行完整RAG问答流程。"""
        was_cached = False

        # L1: 检查答案缓存
        if enable_cache:
            cached = self.cache.get("L1", query)
            if cached:
                cached["cached"] = True
                return cached

        # Step 1: 查询改写
        rewritten = self.rewriter.rewrite(query)

        # Step 2: 问题路由
        route = self.router.route(query, rewritten.intent)

        # Step 3: 多路召回
        if not self.recall_engine:
            raise HTTPException(status_code=503, detail="索引未加载，请先构建索引")
        results = self.recall_engine.recall(query, rewritten.keywords)

        # Step 4: Rerank
        results = self.reranker.rerank(query, results)

        # Step 5: 置信度检测
        results, was_retried = self.confidence_checker.check_and_retry(
            query, results, rewritten.keywords
        )

        # Step 6: 上下文拼装
        context_text, references = self.assembler.assemble(query, results, top_k)

        # Step 7: 生成
        result = self.generator.generate(query, context_text, references)

        response_data = {
            "query": query,
            "answer": result.answer,
            "references": [
                Reference(
                    ref_id=r["ref_id"],
                    source=r["source"],
                    page=str(r["page"]),
                    section=r["section"],
                    content_preview=r["content_preview"],
                ) for r in result.references
            ],
            "model": result.model,
            "was_retried": was_retried,
            "token_usage": result.token_usage,
            "cached": False,
        }

        # 写入缓存
        if enable_cache:
            self.cache.set("L1", query, response_data)

        return response_data

    def index_documents(self, doc_path: str, incremental: bool = True) -> dict:
        """索引文档。"""
        from src.engineering.incremental_index import IncrementalIndexManager

        idx_mgr = IncrementalIndexManager(self.index)
        idx_mgr.docs_dir = Path(doc_path) if Path(doc_path).is_absolute() else config.project_root / doc_path

        def parse_file(file_path: Path):
            from src.offline.parsers.pdf_parser import PDFParser
            from src.offline.parsers.markdown_parser import MarkdownParser
            from src.offline.parsers.image_parser import ImageParser

            ext = file_path.suffix.lower()
            if ext == ".pdf":
                return PDFParser().parse(file_path)
            elif ext in (".md", ".markdown"):
                return MarkdownParser().parse(file_path)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                return ImageParser().parse(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {ext}")

        if incremental:
            summary = idx_mgr.incremental_update(parse_file)
        else:
            idx_mgr.full_reindex(parse_file)

        # 重新加载索引
        self.load_index()

        return {"status": "success", "summary": summary}

    def run_evaluation(self, dataset_path: str, top_k: int = 10) -> dict:
        """运行评估。"""
        from src.evaluation.test_dataset import EvalDataset
        from src.evaluation.metrics_retrieval import RetrievalMetrics
        from src.evaluation.metrics_generation import GenerationMetrics

        dataset = EvalDataset(Path(dataset_path))
        ret_metrics = RetrievalMetrics()
        gen_metrics = GenerationMetrics()

        queries = []
        answers = []
        contexts_list = []
        retrieved_list = []
        relevant_list = []

        for sample in dataset:
            queries.append(sample.query)
            relevant_list.append(sample.contexts)

            # 执行检索
            results = self.recall_engine.recall(sample.query)
            retrieved_list.append([str(r.get("_chunk_index", "")) for r in results[:top_k]])

            # 执行生成
            context_text, refs = self.assembler.assemble(sample.query, results, top_k)
            contexts = [r.get("content_preview", "") for r in refs]
            contexts_list.append(contexts)

            result = self.generator.generate(sample.query, context_text, refs)
            answers.append(result.answer)

        retrieval_scores = ret_metrics.evaluate(queries, retrieved_list, relevant_list)
        gen_scores = gen_metrics.evaluate_batch(queries, answers, contexts_list)

        return {
            "retrieval_metrics": retrieval_scores,
            "generation_metrics": gen_scores,
            "num_samples": len(dataset),
        }


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        index_loaded=_rag_service.is_loaded if _rag_service else False,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _rag_service or not _rag_service.is_loaded:
        raise HTTPException(status_code=503, detail="索引未加载，请先 POST /index 构建索引")
    result = _rag_service.chat(req.query, req.top_k, req.enable_cache)
    return ChatResponse(**result)


@router.post("/index", response_model=IndexResponse)
async def index_docs(req: IndexRequest):
    if not _rag_service:
        raise HTTPException(status_code=500, detail="服务未初始化")
    result = _rag_service.index_documents(req.path, req.incremental)
    return IndexResponse(**result)


@router.post("/eval", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    if not _rag_service or not _rag_service.is_loaded:
        raise HTTPException(status_code=503, detail="索引未加载")
    result = _rag_service.run_evaluation(req.dataset_path, req.top_k)
    return EvalResponse(**result)

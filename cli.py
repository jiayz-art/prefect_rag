#!/usr/bin/env python
"""CLI统一入口 — 多模态RAG知识库问答系统。

用法:
    python cli.py chat              # 交互式问答
    python cli.py index [--path ./data/docs] [--incremental]  # 构建索引
    python cli.py eval [--dataset ./data/eval_queries.json]    # 运行评估
    python cli.py serve [--port 8000]                          # 启动API服务
    python cli.py watch [--path ./data/docs]                   # 文件监听+自动索引
"""
import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在sys.path中
sys.path.insert(0, str(Path(__file__).parent))


def cmd_index(args):
    """构建文档索引。"""
    from src.offline.index_builder import HybridIndexBuilder
    from src.offline.parsers.pdf_parser import PDFParser
    from src.offline.parsers.markdown_parser import MarkdownParser
    from src.offline.parsers.image_parser import ImageParser
    from src.engineering.incremental_index import IncrementalIndexManager

    doc_path = Path(args.path)
    if not doc_path.is_absolute():
        doc_path = Path.cwd() / doc_path

    if not doc_path.exists():
        print(f"[错误] 文档目录不存在: {doc_path}")
        sys.exit(1)

    print(f"[CLI] 文档目录: {doc_path}")
    print(f"[CLI] 增量模式: {args.incremental}")

    def parse_file(file_path: Path):
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return PDFParser().parse(file_path)
        elif ext in (".md", ".markdown"):
            return MarkdownParser().parse(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            return ImageParser().parse(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    index = HybridIndexBuilder()
    idx_mgr = IncrementalIndexManager(index)
    idx_mgr.docs_dir = doc_path

    if args.incremental:
        summary = idx_mgr.incremental_update(parse_file)
    else:
        summary = idx_mgr.full_reindex(parse_file)

    print(f"\n[完成] 索引构建完毕: {summary}")


def cmd_chat(args):
    """交互式问答。"""
    from src.offline.index_builder import HybridIndexBuilder
    from src.online.query_rewriter import QueryRewriter
    from src.online.query_router import QueryRouter
    from src.online.multi_recall import MultiRecall
    from src.online.reranker import Reranker
    from src.online.confidence_check import ConfidenceChecker
    from src.online.context_assembler import ContextAssembler
    from src.online.generator import Generator
    from src.engineering.cache_manager import CacheManager

    print("[CLI] 正在加载索引...")
    index = HybridIndexBuilder()

    if not index.load_bm25() or not index.load_chroma():
        print("[错误] 索引未构建！请先运行: python cli.py index --path ./data/docs")
        sys.exit(1)

    print(f"[CLI] 索引加载成功 (BM25: {len(index._bm25_chunks)} chunks)")

    rewriter = QueryRewriter()
    router = QueryRouter()
    recall = MultiRecall(index)
    reranker = Reranker()
    checker = ConfidenceChecker(recall_fn=recall.recall, rewrite_fn=rewriter.rewrite)
    assembler = ContextAssembler()
    generator = Generator()
    cache = CacheManager()

    print("\n" + "=" * 60)
    print("  多模态RAG知识库问答系统")
    print("  输入 'quit' 或 'exit' 退出，输入 'clear' 清屏")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if query.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # L1缓存
        if args.no_cache is False:
            cached = cache.get("L1", query)
            if cached:
                print(f"\n{cached['answer']}\n")
                print(f"[缓存命中 | 来源: {len(cached.get('references', []))} 个文档片段]")
                continue

        # Step 1: 查询改写
        rewritten = rewriter.rewrite(query)
        print(f"[意图: {rewritten.intent}]")

        # Step 2: 路由
        route = router.route(query, rewritten.intent)
        print(f"[路由: {route.route}]")

        # Step 3-7: 检索+重排+生成
        results = recall.recall(query, rewritten.keywords)
        results = reranker.rerank(query, results)
        results, was_retried = checker.check_and_retry(query, results, rewritten.keywords)
        context_text, refs = assembler.assemble(query, results, args.top_k)
        result = generator.generate(query, context_text, refs)

        print(f"\n{result.answer}\n")
        if result.references:
            print("--- 引用来源 ---")
            for r in result.references:
                print(f"  [{r['ref_id']}] {r['source']} (p.{r['page']}, §{r['section']})")
        if result.token_usage:
            print(f"\n[Token用量: {result.token_usage}]")
        if was_retried:
            print("[系统] 本次回答触发了二次检索")

        # 写缓存
        if not args.no_cache:
            cache.set("L1", query, {
                "query": query,
                "answer": result.answer,
                "references": result.references,
                "model": result.model,
                "was_retried": was_retried,
                "token_usage": result.token_usage,
            })


def cmd_eval(args):
    """运行效果测评。"""
    from src.offline.index_builder import HybridIndexBuilder
    from src.online.multi_recall import MultiRecall
    from src.online.context_assembler import ContextAssembler
    from src.online.generator import Generator
    from src.evaluation.test_dataset import EvalDataset
    from src.evaluation.metrics_retrieval import RetrievalMetrics
    from src.evaluation.metrics_generation import GenerationMetrics

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path

    print(f"[CLI] 评估数据集: {dataset_path}")

    print("[CLI] 加载索引...")
    index = HybridIndexBuilder()
    if not index.load_bm25() or not index.load_chroma():
        print("[错误] 索引未构建！")
        sys.exit(1)

    recall = MultiRecall(index)
    assembler = ContextAssembler()
    generator = Generator()

    dataset = EvalDataset(dataset_path)
    ret_metrics = RetrievalMetrics()
    gen_metrics = GenerationMetrics()

    print(f"[CLI] 开始评估 {len(dataset)} 条样本...\n")

    queries = []
    answers = []
    contexts_list = []
    retrieved_list = []
    relevant_list = []

    for i, sample in enumerate(dataset):
        print(f"  [{i + 1}/{len(dataset)}] {sample.query[:60]}...")
        queries.append(sample.query)
        relevant_list.append(sample.contexts)

        results = recall.recall(sample.query)
        retrieved_list.append([str(r.get("_chunk_index", "")) for r in results[:args.top_k]])

        context_text, refs = assembler.assemble(sample.query, results, args.top_k)
        contexts = [r.get("content_preview", "") for r in refs]
        contexts_list.append(contexts)

        result = generator.generate(sample.query, context_text, refs)
        answers.append(result.answer)

    # 计算指标
    ret_scores = ret_metrics.evaluate(queries, retrieved_list, relevant_list)
    gen_scores = gen_metrics.evaluate_batch(queries, answers, contexts_list)

    print("\n" + "=" * 50)
    print("  检索评估结果")
    print("=" * 50)
    for k, v in ret_scores.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    print("\n" + "=" * 50)
    print("  生成评估结果")
    print("=" * 50)
    for k, v in gen_scores.items():
        print(f"  {k:20s}: {v:.4f}")


def cmd_serve(args):
    """启动FastAPI服务。"""
    import uvicorn
    from src.core.config import config
    host = args.host or config.get("api.host", "0.0.0.0")
    port = args.port or config.get("api.port", 8000)
    print(f"[CLI] 启动API服务 http://{host}:{port}")
    print(f"[CLI] API文档 http://{host}:{port}/docs")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


def cmd_watch(args):
    """启动文件监听 + 自动增量索引。"""
    from src.engineering.file_watcher import FileWatcher
    from src.offline.index_builder import HybridIndexBuilder
    from src.engineering.incremental_index import IncrementalIndexManager

    doc_path = Path(args.path)
    if not doc_path.is_absolute():
        doc_path = Path.cwd() / doc_path

    index = HybridIndexBuilder()
    index.load_bm25()
    index.load_chroma()

    idx_mgr = IncrementalIndexManager(index)
    idx_mgr.docs_dir = doc_path

    from src.offline.parsers.pdf_parser import PDFParser
    from src.offline.parsers.markdown_parser import MarkdownParser
    from src.offline.parsers.image_parser import ImageParser

    def parse_file(file_path: Path):
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return PDFParser().parse(file_path)
        elif ext in (".md", ".markdown"):
            return MarkdownParser().parse(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            return ImageParser().parse(file_path)

    def on_change(changed_paths):
        print(f"[Watch] 检测到变更，执行增量索引...")
        idx_mgr.incremental_update(parse_file)

    watcher = FileWatcher(str(doc_path), on_change)
    print(f"[CLI] 开始监听 {doc_path} ...")
    watcher.run_forever()


def main():
    parser = argparse.ArgumentParser(
        description="多模态RAG知识库问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py index --path ./data/docs              # 增量索引
  python cli.py index --path ./data/docs --full       # 全量重建
  python cli.py chat                                  # 交互式问答
  python cli.py eval --dataset ./data/eval_queries.json  # 评估
  python cli.py serve --port 8000                     # 启动API服务
  python cli.py watch --path ./data/docs              # 文件监听
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # index
    p_index = subparsers.add_parser("index", help="构建/更新文档索引")
    p_index.add_argument("--path", default="./data/docs", help="文档目录路径")
    p_index.add_argument("--full", dest="incremental", action="store_false", default=True,
                          help="全量重建索引（默认为增量模式）")
    p_index.set_defaults(incremental=True)

    # chat
    p_chat = subparsers.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--top-k", type=int, default=10, help="返回文档片段数")
    p_chat.add_argument("--no-cache", action="store_true", help="禁用缓存")

    # eval
    p_eval = subparsers.add_parser("eval", help="运行效果测评")
    p_eval.add_argument("--dataset", default="./data/eval_queries.json", help="评估数据集路径")
    p_eval.add_argument("--top-k", type=int, default=10)

    # serve
    p_serve = subparsers.add_parser("serve", help="启动FastAPI服务")
    p_serve.add_argument("--host", default=None, help="绑定地址")
    p_serve.add_argument("--port", type=int, default=None, help="绑定端口")

    # watch
    p_watch = subparsers.add_parser("watch", help="文件监听+自动索引")
    p_watch.add_argument("--path", default="./data/docs", help="监听的文档目录")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "watch":
        cmd_watch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

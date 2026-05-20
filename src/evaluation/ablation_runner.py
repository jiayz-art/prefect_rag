"""消融对比测试运行器 — 对比不同切分策略、召回参数、重排模型的量化分析。"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.core.config import config
from src.evaluation.metrics_retrieval import RetrievalMetrics
from src.evaluation.metrics_generation import GenerationMetrics


class AblationRunner:
    """消融实验运行器 — 对比不同策略组合的效果。"""

    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.results: List[Dict] = []

    def run_ablation(
        self,
        experiment_name: str,
        config_overrides: Dict,
        run_fn: Callable[[Dict], Dict],   # 接收config返回{eval_data}的函数
    ):
        """运行一组消融实验。"""
        print(f"\n{'='*60}")
        print(f"[Ablation] {experiment_name}")
        print(f"[Ablation] 配置: {config_overrides}")
        print(f"{'='*60}")

        try:
            eval_data = run_fn(config_overrides)

            result = {
                "experiment": experiment_name,
                "config": config_overrides,
                "timestamp": datetime.now().isoformat(),
                "metrics": eval_data,
            }
            self.results.append(result)

            # 实时打印结果
            print(f"\n[结果] {experiment_name}:")
            for k, v in eval_data.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        except Exception as e:
            print(f"[Ablation] 实验失败: {e}")
            self.results.append({
                "experiment": experiment_name,
                "config": config_overrides,
                "error": str(e),
            })

    def compare_chunk_strategies(
        self,
        documents: List,
        build_index_fn: Callable,
        eval_queries: List,
        eval_relevant_docs: List[List[str]],
    ):
        """对比不同切分策略。"""
        strategies = [
            {"chunk_size": 256, "chunk_overlap": 64},
            {"chunk_size": 512, "chunk_overlap": 128},
            {"chunk_size": 1024, "chunk_overlap": 256},
        ]

        for strat in strategies:
            name = f"chunk_size={strat['chunk_size']}_overlap={strat['chunk_overlap']}"
            self.run_ablation(
                experiment_name=name,
                config_overrides=strat,
                run_fn=lambda cfg: self._eval_retrieval(
                    build_index_fn, documents, eval_queries, eval_relevant_docs, cfg
                ),
            )

    def compare_recall_strategies(
        self,
        recall_fns: Dict[str, Callable],
        eval_queries: List,
        eval_relevant_docs: List[List[str]],
    ):
        """对比不同召回策略。"""
        for name, recall_fn in recall_fns.items():
            self.run_ablation(
                experiment_name=f"recall_{name}",
                config_overrides={"recall_method": name},
                run_fn=lambda cfg, fn=recall_fn: self._eval_single_recall(
                    fn, eval_queries, eval_relevant_docs
                ),
            )

    def compare_rerank(
        self,
        base_recall_fn: Callable,
        rerank_fns: Dict[str, Callable],
        eval_queries: List,
        eval_relevant_docs: List[List[str]],
    ):
        """对比不同Reranker。"""
        for name, rerank_fn in rerank_fns.items():
            self.run_ablation(
                experiment_name=f"rerank_{name}",
                config_overrides={"rerank_method": name},
                run_fn=lambda cfg, fn=rerank_fn: self._eval_with_rerank(
                    base_recall_fn, fn, eval_queries, eval_relevant_docs
                ),
            )

    def _eval_retrieval(
        self, build_fn, documents, queries, relevant_docs, config_overrides
    ) -> Dict:
        """评估检索效果。"""
        index = build_fn(documents, config_overrides)
        retrieved = []
        for q in queries:
            results = index.search(q, top_k=10)
            retrieved.append([str(r.get("_chunk_index", "")) for r in results])
        return self.retrieval_metrics.evaluate(queries, retrieved, relevant_docs)

    def _eval_single_recall(self, recall_fn, queries, relevant_docs) -> Dict:
        """评估单个召回策略。"""
        retrieved = []
        for q in queries:
            results = recall_fn(q)
            retrieved.append([str(r.get("_chunk_index", "")) for r in results])
        return self.retrieval_metrics.evaluate(queries, retrieved, relevant_docs)

    def _eval_with_rerank(self, recall_fn, rerank_fn, queries, relevant_docs) -> Dict:
        retrieved = []
        for q in queries:
            recalled = recall_fn(q)
            reranked = rerank_fn(q, recalled)
            retrieved.append([str(r.get("_chunk_index", "")) for r in reranked])
        return self.retrieval_metrics.evaluate(queries, retrieved, relevant_docs)

    def generate_report(self, output_path: str = None) -> str:
        """生成Markdown格式的消融实验报告。"""
        if not output_path:
            output_path = str(
                config.project_root / "reports" / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

        lines = [
            "# 消融实验报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n实验数量: {len(self.results)}",
            "\n---\n",
        ]

        for r in self.results:
            lines.append(f"## {r['experiment']}")
            lines.append(f"\n配置: `{json.dumps(r.get('config', {}), ensure_ascii=False)}`")

            if "error" in r:
                lines.append(f"\n> 错误: {r['error']}")
            else:
                lines.append("\n| 指标 | 值 |")
                lines.append("|------|-----|")
                for k, v in r.get("metrics", {}).items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {v:.4f} |")
                    else:
                        lines.append(f"| {k} | {v} |")

            lines.append("\n---\n")

        report = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n[报告] 已保存至 {output_path}")
        return report

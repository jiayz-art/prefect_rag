"""评估数据集加载与管理。"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalSample:
    """单条评估样本。"""
    query: str
    ground_truth: str
    contexts: List[str] = field(default_factory=list)
    category: str = "text"  # text, multimodal, cross_doc, out_of_scope


class EvalDataset:
    """评估数据集管理器。"""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.samples: List[EvalSample] = []
        if dataset_path:
            self.load(dataset_path)

    def load(self, path: Path):
        """从JSON文件加载评估数据集。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            self.samples.append(EvalSample(
                query=item.get("query", ""),
                ground_truth=item.get("ground_truth", ""),
                contexts=item.get("contexts", []),
                category=item.get("category", "text"),
            ))

        print(f"[EvalDataset] 加载 {len(self.samples)} 条评估样本")

    def add_sample(self, sample: EvalSample):
        self.samples.append(sample)

    def save(self, path: Path):
        """保存评估数据集。"""
        data = []
        for s in self.samples:
            data.append({
                "query": s.query,
                "ground_truth": s.ground_truth,
                "contexts": s.contexts,
                "category": s.category,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def filter_by_category(self, category: str) -> List[EvalSample]:
        return [s for s in self.samples if s.category == category]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

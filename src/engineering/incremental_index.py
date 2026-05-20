"""增量索引模块 — 基于SHA256 Hash的增量索引与热更新。"""
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.core.config import config
from src.offline.index_builder import HybridIndexBuilder
from src.offline.parsers.base import Document


class IncrementalIndexManager:
    """增量索引管理器 — 文档变更检测 + 增量更新。"""

    def __init__(self, index_builder: HybridIndexBuilder):
        self.index = index_builder
        self.docs_dir = config.project_root / "data" / "docs"

    def scan_documents(self, extensions: List[str] = None) -> List[Path]:
        """扫描docs目录下所有支持的文档。"""
        if extensions is None:
            extensions = config.get("documents.supported_extensions", [".pdf", ".md", ".png", ".jpg"])
        files = []
        for ext in extensions:
            files.extend(self.docs_dir.glob(f"**/*{ext}"))
            files.extend(self.docs_dir.glob(f"**/*{ext.upper()}"))
        return sorted(set(files))

    def detect_changes(self) -> Tuple[List[Path], List[Path], List[str]]:
        """检测文档变更，返回 (新增/修改列表, 删除列表, 未变更列表)。

        Returns:
            (to_index, to_remove, unchanged): 需索引、需删除、未变更的文件
        """
        current_files = self.scan_documents()
        current_hashes = {}
        for f in current_files:
            current_hashes[str(f)] = self._compute_hash(f)

        indexed = self.index.get_indexed_docs()

        to_index = []
        unchanged = []

        for f in current_files:
            f_str = str(f)
            if f_str not in indexed:
                to_index.append(f)       # 新文件
            elif current_hashes[f_str] != indexed[f_str]:
                to_index.append(f)       # 内容已修改
            else:
                unchanged.append(f_str)

        to_remove = [path for path in indexed if path not in current_hashes]

        return to_index, to_remove, unchanged

    def incremental_update(self, parser_factory) -> Dict:
        """执行增量索引更新。

        Args:
            parser_factory: 文档解析器工厂函数，签名为 (file_path) -> Document

        Returns:
            更新摘要字典
        """
        to_index, to_remove, unchanged = self.detect_changes()

        summary = {
            "scanned": len(to_index) + len(unchanged),
            "to_index": len(to_index),
            "to_remove": len(to_remove),
            "unchanged": len(unchanged),
            "details": {
                "new_or_modified": [str(f) for f in to_index],
                "deleted": to_remove,
            },
        }

        # 处理修改：先清除旧数据再重新索引
        modified_hashes = set()
        indexed_docs = self.index.get_indexed_docs()
        for f in to_index:
            f_str = str(f)
            if f_str in indexed_docs:
                # 文件已存在但内容变了 → 先删旧数据
                self.index.remove_doc_meta(f_str)
                self.index.remove_chroma_by_source(f_str)
                print(f"[增量] 清除旧索引(已修改): {f_str}")

        # 处理删除：同时清理元数据 + Chroma向量 + BM25
        if to_remove:
            for path in to_remove:
                self.index.remove_doc_meta(path)
                self.index.remove_chroma_by_source(path)
                print(f"[增量] 移除已删除文档: {path}")
            self._rebuild_bm25_after_removal(to_remove)

        # 处理新增/修改：增量追加到已有索引
        if to_index:
            documents = []
            for f in to_index:
                try:
                    doc = parser_factory(f)
                    documents.append(doc)
                    print(f"[增量] 解析文档: {f.name}")
                except Exception as e:
                    print(f"[增量] 解析失败 {f.name}: {e}")

            if documents:
                # incremental=True: BM25合并新旧chunk, Chroma追加写入
                self.index.build_all(documents, incremental=True)
                print(f"[增量] 已索引 {len(documents)} 个文档")

        return summary

    def _rebuild_bm25_after_removal(self, removed_paths: List[str]):
        """删除文档后重建BM25索引（BM25不支持直接删除，需重建）。"""
        removed_set = set(removed_paths)
        remaining_chunks = [
            c for c in self.index._bm25_chunks
            if c["metadata"]["source"] not in removed_set
        ]
        if remaining_chunks:
            self.index.build_bm25(remaining_chunks)
            self.index.save_bm25()
            print(f"[增量] BM25索引重建完成，保留 {len(remaining_chunks)} chunks")
        else:
            self.index._bm25_index = None
            self.index._bm25_chunks = []
            print("[增量] 所有文档已删除，BM25索引已清空")

    def _compute_hash(self, file_path: Path) -> str:
        """计算文件SHA256哈希。"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def full_reindex(self, parser_factory, extensions: List[str] = None):
        """全量重建索引。"""
        files = self.scan_documents(extensions)
        documents = []
        for f in files:
            try:
                doc = parser_factory(f)
                documents.append(doc)
            except Exception as e:
                print(f"[全量] 解析失败 {f.name}: {e}")
        if documents:
            self.index.build_all(documents)
            print(f"[全量] 索引重建完成: {len(files)} 文件 -> {len(documents)} 文档")

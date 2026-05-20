"""混合索引构建器 — BM25稀疏索引 + Chroma稠密向量索引 + SQLite元数据管理。"""
import pickle
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.core.config import config
from src.offline.chunker import DocumentChunker
from src.offline.embeddings import AliyunEmbedder
from src.offline.parsers.base import Document


class HybridIndexBuilder:
    """混合索引构建器 — BM25 + Chroma 双索引。"""

    def __init__(self):
        self.persist_dir = str(config.project_root / config.get("index.persist_dir", "./data/chroma_db"))
        self.bm25_path = str(config.project_root / config.get("index.bm25_index_path", "./data/bm25_index.pkl"))
        self.metadata_db = str(config.project_root / config.get("index.metadata_db", "./data/metadata.db"))

        self.embedder = AliyunEmbedder()
        self.chunker = DocumentChunker()

        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Dict] = []
        self._chroma_client: Optional[chromadb.Client] = None
        self._chroma_collection = None

    # ===== BM25 索引 =====

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 + 英文分词混合。"""
        import jieba
        tokens = []
        # 对中文使用jieba分词
        chinese_pattern = re.compile(r"[一-鿿]+")
        parts = chinese_pattern.split(text)
        chinese_parts = chinese_pattern.findall(text)

        for i, part in enumerate(parts):
            if part.strip():
                tokens.extend(part.lower().split())
            if i < len(chinese_parts):
                tokens.extend(jieba.lcut(chinese_parts[i]))
        return tokens

    def build_bm25(self, chunks: List[Dict]):
        """构建BM25索引。"""
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_chunks = chunks

    def save_bm25(self):
        """持久化BM25索引到磁盘。"""
        data = {
            "index": self._bm25_index,
            "chunks": self._bm25_chunks,
        }
        Path(self.bm25_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(data, f)

    def load_bm25(self) -> bool:
        """从磁盘加载BM25索引。"""
        try:
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_index = data["index"]
            self._bm25_chunks = data["chunks"]
            return True
        except FileNotFoundError:
            return False

    # ===== Chroma 向量索引 =====

    def build_chroma(self, chunks: List[Dict], rebuild: bool = False):
        """构建/更新Chroma向量索引。

        Args:
            chunks: 待写入的chunk列表
            rebuild: True=删除旧库全量重建, False=增量追加到已有collection
        """
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        # 使用UUID避免增量追加时ID冲突
        ids = [f"chunk_{uuid.uuid4().hex[:12]}" for _ in chunks]

        # 批量生成向量
        print(f"[Chroma] 正在向量化 {len(texts)} 个chunks...")
        embeddings = self.embedder.embed_texts(texts)

        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        if rebuild:
            # 全量重建：删除旧collection
            try:
                self._chroma_client.delete_collection("knowledge_base")
            except Exception:
                pass
            self._chroma_collection = self._chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            # 增量模式：获取或创建collection
            try:
                self._chroma_collection = self._chroma_client.get_collection("knowledge_base")
            except Exception:
                self._chroma_collection = self._chroma_client.create_collection(
                    name="knowledge_base",
                    metadata={"hnsw:space": "cosine"},
                )

        # 分批写入
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Chroma写入"):
            batch_end = min(i + batch_size, len(texts))
            self._chroma_collection.add(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )

    def remove_chroma_by_source(self, file_path: str):
        """从Chroma中删除指定文档的所有chunk。"""
        if not self._chroma_collection:
            try:
                self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self._chroma_collection = self._chroma_client.get_collection("knowledge_base")
            except Exception:
                return
        try:
            results = self._chroma_collection.get(
                where={"source": file_path},
                include=["metadatas"],
            )
            if results and results["ids"]:
                self._chroma_collection.delete(ids=results["ids"])
                print(f"[Chroma] 已删除 {len(results['ids'])} 条来自 {file_path} 的向量")
        except Exception as e:
            print(f"[Chroma] 删除失败: {e}")

    def load_chroma(self) -> bool:
        """加载已有Chroma索引。"""
        try:
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self._chroma_collection = self._chroma_client.get_collection("knowledge_base")
            return True
        except Exception:
            return False

    # ===== 元数据管理 (SQLite) =====

    def _get_meta_conn(self) -> sqlite3.Connection:
        Path(self.metadata_db).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.metadata_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                indexed_at TEXT NOT NULL,
                file_type TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        conn.commit()
        return conn

    def upsert_doc_meta(self, file_path: str, file_hash: str, chunk_count: int, file_type: str):
        conn = self._get_meta_conn()
        conn.execute(
            """INSERT OR REPLACE INTO documents (file_path, file_hash, chunk_count, indexed_at, file_type, status)
               VALUES (?, ?, ?, datetime('now'), ?, 'active')""",
            (str(file_path), file_hash, chunk_count, file_type),
        )
        conn.commit()
        conn.close()

    def get_indexed_docs(self) -> Dict[str, str]:
        """返回 {file_path: file_hash} 映射。"""
        conn = self._get_meta_conn()
        rows = conn.execute("SELECT file_path, file_hash FROM documents WHERE status='active'").fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def remove_doc_meta(self, file_path: str):
        conn = self._get_meta_conn()
        conn.execute("UPDATE documents SET status='deleted' WHERE file_path=?", (str(file_path),))
        conn.commit()
        conn.close()

    # ===== 全量构建入口 =====

    def build_all(self, documents: List[Document], incremental: bool = False):
        """全量或增量构建BM25 + Chroma索引。

        Args:
            documents: 文档列表
            incremental: True=增量追加到已有索引, False=全量重建
        """
        all_chunks = []
        for doc in tqdm(documents, desc="文档切分"):
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            self.upsert_doc_meta(
                str(doc.file_path),
                chunks[0]["metadata"]["doc_hash"] if chunks else "",
                len(chunks),
                doc.file_type,
            )

        if not all_chunks:
            print("[Index] 没有可索引的内容")
            return

        print(f"[Index] 共 {len(all_chunks)} 个chunks，开始构建索引...")

        print("[Index] 构建BM25索引...")
        if incremental and self._bm25_index is not None:
            # 增量模式：合并新旧chunk
            existing_chunks = list(self._bm25_chunks)
            self.build_bm25(existing_chunks + all_chunks)
        else:
            self.build_bm25(all_chunks)
        self.save_bm25()

        print("[Index] 构建Chroma向量索引...")
        self.build_chroma(all_chunks, rebuild=not incremental)

        print(f"[Index] 索引构建完成: BM25({len(self._bm25_chunks)}) + Chroma")

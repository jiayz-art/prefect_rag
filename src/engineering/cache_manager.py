"""分层缓存管理 — L1答案缓存 + L2检索结果缓存 + L3 Embedding缓存。"""
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import config


class CacheManager:
    """三层缓存管理器。

    L1: 答案缓存 — query MD5 → {answer, sources, timestamp}
    L2: 检索结果缓存 — query MD5 → {recalled_docs, rerank_scores, timestamp}
    L3: Embedding缓存 — text MD5 → embedding_vector
    """

    def __init__(self):
        backend = config.get("cache.cache_backend", "sqlite")
        self.db_path = config.project_root / "data" / "cache.db"

        self.l1_ttl = config.get("cache.l1_answer_ttl", 86400)
        self.l2_ttl = config.get("cache.l2_retrieval_ttl", 3600)
        self.l3_ttl = config.get("cache.l3_embedding_ttl", 0)  # 0 = 永久

        self._init_db()

    def _init_db(self):
        """初始化缓存数据库。"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                cache_type TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON cache(cache_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
        conn.commit()
        conn.close()

    def _make_key(self, prefix: str, content: str) -> str:
        return f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"

    def _get_conn(self):
        return sqlite3.connect(str(self.db_path))

    def get(self, cache_type: str, key_content: str) -> Optional[Any]:
        """从缓存获取值。"""
        key = self._make_key(cache_type, key_content)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value, created_at, ttl FROM cache WHERE cache_key=?",
            (key,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        value_str, created_at, ttl = row
        # 检查是否过期（ttl=0表示永不过期）
        if ttl > 0 and (time.time() - created_at) > ttl:
            self.delete(cache_type, key_content)
            return None

        return json.loads(value_str)

    def set(self, cache_type: str, key_content: str, value: Any, ttl: int = None):
        """写入缓存。"""
        key = self._make_key(cache_type, key_content)
        if ttl is None:
            ttl_map = {"L1": self.l1_ttl, "L2": self.l2_ttl, "L3": self.l3_ttl}
            ttl = ttl_map.get(cache_type, 3600)

        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, cache_type, value, created_at, ttl) VALUES (?, ?, ?, ?, ?)",
            (key, cache_type, json.dumps(value, ensure_ascii=False), time.time(), ttl),
        )
        conn.commit()
        conn.close()

    def delete(self, cache_type: str, key_content: str):
        """删除单条缓存。"""
        key = self._make_key(cache_type, key_content)
        conn = self._get_conn()
        conn.execute("DELETE FROM cache WHERE cache_key=?", (key,))
        conn.commit()
        conn.close()

    def clear_type(self, cache_type: str):
        """清除某一层全部缓存。"""
        conn = self._get_conn()
        conn.execute("DELETE FROM cache WHERE cache_type=?", (cache_type,))
        conn.commit()
        conn.close()

    def clear_expired(self) -> int:
        """清理过期缓存，返回清理条数。"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM cache WHERE ttl > 0 AND (created_at + ttl) < ?",
            (time.time(),),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def get_stats(self) -> Dict:
        """获取缓存统计信息。"""
        conn = self._get_conn()
        stats = {}
        for cache_type in ["L1", "L2", "L3"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE cache_type=?", (cache_type,)
            ).fetchone()[0]
            stats[cache_type] = count
        conn.close()
        return stats

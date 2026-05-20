"""文本切分与Chunk优化 — 元数据绑定、文本清洗。"""
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.core.config import config
from src.offline.parsers.base import Document, ParsedElement


class ChunkMetadata:
    """Chunk元数据字段定义。"""
    source: str          # 来源文件路径
    page: int            # 页码
    section_title: str   # 章节标题
    chunk_index: int     # chunk序号
    doc_hash: str        # 文档哈希
    element_types: str   # 包含的元素类型
    created_at: str      # 创建时间


class DocumentChunker:
    """文档切分器 — 支持Recursive + Markdown Header两种策略。"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or config.get("index.chunk_size", 512)
        self.chunk_overlap = chunk_overlap or config.get("index.chunk_overlap", 128)

        self._headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

    def chunk_document(self, document: Document) -> List[Dict]:
        """将解析后的文档切分为带元数据的chunk列表。"""
        doc_hash = self._compute_hash(document.file_path)

        # 按元素类型分组处理
        text_content = self._build_structured_text(document)
        if not text_content.strip():
            return []

        # 主切分策略：MarkdownHeaderSplitter for structured docs
        if self._has_markdown_headers(text_content):
            chunks = self._chunk_by_headers(text_content)
        else:
            chunks = self._chunk_recursive(text_content)

        # 绑定元数据
        return self._enrich_chunks(chunks, document, doc_hash)

    def _build_structured_text(self, document: Document) -> str:
        """将解析元素重组成结构化文本，保留层级信息。"""
        sections = []
        current_section = ""

        for elem in document.elements:
            if elem.element_type == "heading":
                if elem.heading_level <= 2:
                    current_section = elem.content
                prefix = "#" * elem.heading_level
                sections.append(f"\n{prefix} {elem.content}\n")
            elif elem.element_type in ("text", "table", "code"):
                prefix = f"[{elem.element_type}]\n" if elem.element_type != "text" else ""
                sections.append(prefix + elem.content)
            elif elem.element_type == "image":
                img_path = elem.metadata.get("image_path", "")
                sections.append(f"[image: {img_path}]")

        return "\n\n".join(sections)

    def _has_markdown_headers(self, text: str) -> bool:
        """检查文本是否包含Markdown标题标记。"""
        return bool(re.search(r"^#{1,4}\s+\S", text, re.MULTILINE))

    def _chunk_by_headers(self, text: str) -> List[str]:
        """基于Markdown标题层级切分。"""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on,
            strip_headers=False,
        )
        md_chunks = splitter.split_text(text)

        # 对大chunk再做递归切分
        result = []
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " "],
        )
        for chunk in md_chunks:
            if len(chunk.page_content) > self.chunk_size:
                sub_chunks = recursive_splitter.split_text(chunk.page_content)
                result.extend(sub_chunks)
            else:
                result.append(chunk.page_content)
        return result

    def _chunk_recursive(self, text: str) -> List[str]:
        """递归字符切分（无标题层级的文档）。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        return splitter.split_text(text)

    def _enrich_chunks(self, chunks: List[str], document: Document, doc_hash: str) -> List[Dict]:
        """为每个chunk绑定元数据并做文本清洗。"""
        enriched = []
        for i, chunk_text in enumerate(chunks):
            cleaned = self._clean_text(chunk_text)
            if not cleaned or len(cleaned) < 10:
                continue

            enriched.append({
                "content": cleaned,
                "metadata": {
                    "source": str(document.file_path),
                    "source_name": document.file_path.name,
                    "page": self._guess_page(cleaned, document),
                    "section_title": self._guess_section(cleaned, document) or document.metadata.get("title", ""),
                    "chunk_index": i,
                    "doc_hash": doc_hash,
                    "element_types": "text",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            })
        return enriched

    def _clean_text(self, text: str) -> str:
        """文本清洗：去噪、合并空白。"""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        return text

    def _compute_hash(self, file_path) -> str:
        """计算文件SHA256哈希。"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _guess_page(self, chunk_text: str, document: Document) -> int:
        """从chunk内容推断所在页码。"""
        for elem in document.elements:
            if elem.content and elem.content[:50] in chunk_text:
                return elem.page
        return 0

    def _guess_section(self, chunk_text: str, document: Document) -> Optional[str]:
        """从chunk内容推断所属章节。"""
        for elem in document.elements:
            if elem.element_type == "heading" and elem.heading_level <= 2:
                if elem.content in chunk_text:
                    return elem.content
        for elem in document.elements:
            if elem.section_title and elem.content and elem.content[:30] in chunk_text:
                return elem.section_title
        return None

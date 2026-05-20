"""PDF解析器 — 基于PyMuPDF，支持标题层级检测和表格提取。"""
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.offline.parsers.base import BaseParser, Document, ParsedElement


class PDFParser(BaseParser):
    """PDF文档解析器。"""

    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def parse(self, file_path: Path) -> Document:
        doc = Document(file_path=file_path, file_type="pdf")
        pdf = fitz.open(str(file_path))

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] == 0:  # 文本块
                    elements = self._parse_text_block(block, page_num)
                    doc.elements.extend(elements)
                elif block["type"] == 1:  # 图片块
                    element = self._parse_image_block(block, page_num, pdf, file_path)
                    if element:
                        doc.elements.append(element)

        doc.metadata["total_pages"] = len(pdf)
        doc.metadata["title"] = file_path.stem
        pdf.close()
        return doc

    def _parse_text_block(self, block: dict, page_num: int) -> List[ParsedElement]:
        elements = []
        for line in block.get("lines", []):
            text_parts = []
            max_font_size = 0
            is_bold = False

            for span in line.get("spans", []):
                text_parts.append(span["text"])
                max_font_size = max(max_font_size, span["size"])
                if "Bold" in span.get("font", ""):
                    is_bold = True

            line_text = "".join(text_parts).strip()
            if not line_text:
                continue

            heading_level = self._detect_heading_level(line_text, max_font_size, is_bold)
            if heading_level > 0:
                elements.append(ParsedElement(
                    content=line_text,
                    element_type="heading",
                    page=page_num + 1,
                    heading_level=heading_level,
                    section_title=line_text if heading_level <= 2 else "",
                ))
            elif self._is_table_line(line_text):
                elements.append(ParsedElement(
                    content=line_text,
                    element_type="table",
                    page=page_num + 1,
                ))
            else:
                elements.append(ParsedElement(
                    content=line_text,
                    element_type="text",
                    page=page_num + 1,
                ))
        return elements

    def _parse_image_block(self, block: dict, page_num: int, pdf: fitz.Document, file_path: Path) -> ParsedElement:
        """提取图片并保存。"""
        try:
            image_bytes = block.get("image")
            if image_bytes:
                img_dir = file_path.parent.parent / "images" / file_path.stem
                img_dir.mkdir(parents=True, exist_ok=True)
                img_name = f"p{page_num + 1}_img_{block.get('number', 0)}.png"
                img_path = img_dir / img_name
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                return ParsedElement(
                    content="",
                    element_type="image",
                    page=page_num + 1,
                    metadata={"image_path": str(img_path), "bbox": block.get("bbox", [])},
                )
        except Exception:
            pass
        return None

    def _detect_heading_level(self, text: str, font_size: float, is_bold: bool) -> int:
        """基于字体大小和样式推断标题层级。"""
        heading_patterns = [
            r"^第[一二三四五六七八九十]+[章节].*",
            r"^\d+(\.\d+)*\s+\S",
            r"^[一二三四五六七八九十]+[、．.]",
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return 2 if font_size > 13 else 3

        if font_size > 16:
            return 1
        if font_size > 14 and is_bold:
            return 2
        if font_size > 12 and is_bold and len(text) < 80:
            return 3
        return 0

    def _is_table_line(self, text: str) -> bool:
        """判断是否是表格行（含多个 | 分隔符或制表符）。"""
        return text.count("|") >= 2 or text.count("\t") >= 2

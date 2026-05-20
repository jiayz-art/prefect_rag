"""Markdown解析器 — 基于AST解析，保留标题层级、代码块、表格。"""
import re
from pathlib import Path
from typing import List

from src.offline.parsers.base import BaseParser, Document, ParsedElement


class MarkdownParser(BaseParser):
    """Markdown文档解析器。"""

    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: Path) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc = Document(file_path=file_path, file_type="markdown")
        lines = content.split("\n")
        current_section = ""
        code_block_buffer = []
        in_code_block = False
        table_buffer = []
        in_table = False

        for line in lines:
            # 代码块处理
            if line.strip().startswith("```"):
                if in_code_block:
                    code_block_buffer.append(line)
                    doc.elements.append(ParsedElement(
                        content="\n".join(code_block_buffer),
                        element_type="code",
                        section_title=current_section,
                    ))
                    code_block_buffer = []
                    in_code_block = False
                else:
                    in_code_block = True
                    code_block_buffer.append(line)
                continue

            if in_code_block:
                code_block_buffer.append(line)
                continue

            # 表格处理
            if "|" in line and line.strip().startswith("|"):
                if not in_table:
                    in_table = True
                table_buffer.append(line)
                continue
            elif in_table:
                doc.elements.append(ParsedElement(
                    content="\n".join(table_buffer),
                    element_type="table",
                    section_title=current_section,
                ))
                table_buffer = []
                in_table = False

            # 标题检测
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                if level <= 2:
                    current_section = title
                doc.elements.append(ParsedElement(
                    content=title,
                    element_type="heading",
                    heading_level=level,
                    section_title=current_section,
                ))
                continue

            # 图片引用
            img_match = re.match(r"!\[.*\]\((.+)\)", line.strip())
            if img_match:
                img_path = img_match.group(1)
                doc.elements.append(ParsedElement(
                    content="",
                    element_type="image",
                    metadata={"image_path": img_path},
                    section_title=current_section,
                ))
                continue

            # 普通文本
            line = line.strip()
            if line and not line.startswith("---") and not line.startswith(">"):
                doc.elements.append(ParsedElement(
                    content=line,
                    element_type="text",
                    section_title=current_section,
                ))

        # 处理末尾未闭合的表格
        if table_buffer:
            doc.elements.append(ParsedElement(
                content="\n".join(table_buffer),
                element_type="table",
                section_title=current_section,
            ))

        doc.metadata["title"] = file_path.stem
        return doc

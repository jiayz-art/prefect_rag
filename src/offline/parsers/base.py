"""文档解析器抽象基类。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ParsedElement:
    """解析后的文档元素。"""
    content: str
    element_type: str  # text, table, image, heading, code
    page: int = 0
    heading_level: int = 0
    section_title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """解析后的完整文档。"""
    file_path: Path
    file_type: str
    elements: List[ParsedElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(e.content for e in self.elements if e.element_type != "image")


class BaseParser(ABC):
    """文档解析器抽象基类。"""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """解析文档，返回Document对象。"""
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """返回支持的文件扩展名列表。"""
        ...

"""图片解析器 — PaddleOCR文字提取 + 阿里云VL API图表语义理解。"""
import base64
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from src.core.config import config
from src.offline.parsers.base import BaseParser, Document, ParsedElement


class ImageParser(BaseParser):
    """图片文档解析器 — OCR + VLM双阶段。"""

    def __init__(self):
        self._ocr = None
        self._vlm_client: Optional[OpenAI] = None

    @property
    def ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=config.get("documents.ocr_lang", "ch"))
        return self._ocr

    @property
    def vlm_client(self):
        if self._vlm_client is None:
            self._vlm_client = OpenAI(
                api_key=config.get("aliyun.api_key"),
                base_url=config.get("aliyun.base_url"),
            )
        return self._vlm_client

    def supported_extensions(self) -> List[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]

    def parse(self, file_path: Path) -> Document:
        doc = Document(file_path=file_path, file_type="image")

        # 阶段1: OCR文字提取
        ocr_text = self._run_ocr(file_path)
        if ocr_text:
            doc.elements.append(ParsedElement(
                content=ocr_text,
                element_type="text",
                metadata={"source": "ocr"},
            ))

        # 阶段2: VLM语义理解（图表/示意图）
        if config.get("documents.enable_vlm_for_images", True):
            vlm_desc = self._run_vlm(file_path)
            if vlm_desc:
                doc.elements.append(ParsedElement(
                    content=vlm_desc,
                    element_type="text",
                    metadata={"source": "vlm_description"},
                ))

        doc.metadata["title"] = file_path.stem
        doc.metadata["image_path"] = str(file_path)
        return doc

    def _run_ocr(self, file_path: Path) -> str:
        """PaddleOCR提取图片中的文字。"""
        try:
            result = self.ocr.ocr(str(file_path), cls=True)
            if not result or not result[0]:
                return ""

            lines = []
            for line_info in result[0]:
                text = line_info[1][0]
                if text and text.strip():
                    lines.append(text.strip())
            return "\n".join(lines)
        except Exception as e:
            print(f"[OCR] 识别失败 {file_path.name}: {e}")
            return ""

    def _run_vlm(self, file_path: Path) -> str:
        """调用阿里云VL API理解图片语义。"""
        try:
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = file_path.suffix.lower().replace(".", "")
            if ext == "jpg":
                ext = "jpeg"
            data_uri = f"data:image/{ext};base64,{image_data}"

            response = self.vlm_client.chat.completions.create(
                model=config.get("aliyun.vlm_model", "qwen-vl-plus"),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "text",
                            "text": (
                                "请详细描述这张图片的内容。如果是图表，请说明图表的类型、"
                                "展示的数据和关键结论；如果是示意图，请解释其含义和关键信息。"
                                "如果是文档截图，请提取其中的文字内容。"
                            ),
                        },
                    ],
                }],
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[VLM] 理解失败 {file_path.name}: {e}")
            return ""

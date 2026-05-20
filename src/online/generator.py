"""LLM生成模块 — 调用阿里云qwen-plus生成最终答案 + 引用溯源。"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import OpenAI

from src.core.config import config


@dataclass
class GenerationResult:
    """LLM生成结果。"""
    answer: str
    references: List[Dict] = field(default_factory=list)
    model: str = ""
    token_usage: Dict = field(default_factory=dict)
    was_retried: bool = False


class Generator:
    """答案生成器 — 基于阿里云LLM生成带引用的最终答案。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.get("aliyun.api_key"),
            base_url=config.get("aliyun.base_url"),
        )
        self.model = config.get("aliyun.llm_model", "qwen-plus")

    def generate(self, query: str, context_text: str, references: List[Dict]) -> GenerationResult:
        """基于检索上下文生成答案。

        Args:
            query: 用户原始问题
            context_text: 拼装好的上下文文本（含引用标注）
            references: 引用来源列表

        Returns:
            GenerationResult: 包含答案、引用和token用量
        """
        from src.online.context_assembler import ContextAssembler
        assembler = ContextAssembler()
        prompt = assembler.build_prompt(query, context_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            answer = response.choices[0].message.content or ""

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return GenerationResult(
                answer=answer,
                references=references,
                model=self.model,
                token_usage=usage,
            )

        except Exception as e:
            print(f"[Generator] LLM调用失败: {e}")
            return GenerationResult(
                answer=f"生成回答时出错: {str(e)}。请稍后重试。",
                references=references,
                model=self.model,
            )

    def generate_with_multimodal(
        self,
        query: str,
        context_text: str,
        image_paths: List[str],
        references: List[Dict],
    ) -> GenerationResult:
        """多模态生成 — 同时输入文本上下文和图片。"""
        from src.online.context_assembler import ContextAssembler
        assembler = ContextAssembler()

        content_parts = [
            {"type": "text", "text": assembler.build_prompt(query, context_text)},
        ]

        for img_path in image_paths:
            import base64
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            ext = img_path.split(".")[-1].lower().replace("jpg", "jpeg")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{image_data}"},
            })

        try:
            response = self.client.chat.completions.create(
                model=config.get("aliyun.vlm_model", "qwen-vl-plus"),
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.3,
                max_tokens=2000,
            )

            answer = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return GenerationResult(
                answer=answer,
                references=references,
                model=config.get("aliyun.vlm_model", "qwen-vl-plus"),
                token_usage=usage,
            )
        except Exception as e:
            print(f"[Generator] VLM调用失败: {e}")
            return self.generate(query, context_text, references)

"""查询改写模块 — 意图识别、关键词提取、查询扩展。"""
import json
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from src.core.config import config


@dataclass
class RewrittenQuery:
    """改写后的查询结果。"""
    original_query: str
    intent: str            # knowledge_qa, document_search, chart_understanding, general
    keywords: List[str] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)


INTENT_SYSTEM_PROMPT = """你是一个查询意图分析专家。分析用户问题的意图，提取关键词，并生成查询变体。

意图类别：
- knowledge_qa: 针对文档内容的知识问答
- document_search: 查找特定文档或段落
- chart_understanding: 需要理解图表/图片内容
- general: 通用问题

请以JSON格式返回，格式如下：
{
  "intent": "knowledge_qa",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "expanded_queries": ["改写后的查询1", "改写后的查询2", "改写后的查询3"]
}"""


class QueryRewriter:
    """查询改写器 — 使用LLM进行意图识别、关键词提取和查询扩展。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.get("aliyun.api_key"),
            base_url=config.get("aliyun.base_url"),
        )
        self.model = config.get("aliyun.llm_model", "qwen-plus")

    def rewrite(self, query: str) -> RewrittenQuery:
        """改写查询，返回扩展后的查询集合。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "{}"
            data = self._parse_json(content)
        except Exception as e:
            print(f"[QueryRewriter] LLM调用失败，使用fallback: {e}")
            data = self._fallback(query)

        return RewrittenQuery(
            original_query=query,
            intent=data.get("intent", "knowledge_qa"),
            keywords=data.get("keywords", []),
            expanded_queries=data.get("expanded_queries", [query]),
        )

    def _parse_json(self, content: str) -> dict:
        """从LLM回复中提取JSON。"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 代码块
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    def _fallback(self, query: str) -> dict:
        """降级方案：基于规则的简单处理。"""
        import jieba
        keywords = jieba.lcut(query)
        keywords = [w for w in keywords if len(w) > 1][:5]
        return {
            "intent": "knowledge_qa",
            "keywords": keywords,
            "expanded_queries": [query],
        }

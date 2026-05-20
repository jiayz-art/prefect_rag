"""问题路由模块 — 区分多模态/文本问题、知识库内/外问题。"""
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from src.core.config import config


@dataclass
class RouteDecision:
    """路由决策结果。"""
    route: str            # text | multimodal | external
    reason: str
    image_path: Optional[str] = None   # 多模态问题关联的图片路径


ROUTER_SYSTEM_PROMPT = """你是一个问题路由专家。根据用户问题判断应该走什么处理流程。

路由类型：
- text: 纯文本问答，可以从知识库文档中检索答案
- multimodal: 问题涉及图片/图表/截图理解，需要多模态处理
- external: 问题超出知识库范围，属于通用知识/闲聊/时事，知识库中不太可能有答案

判断标准：
1. 如果问题明确提到"图片"、"图表"、"这个图"、"如图"等，或者其他明显提及图片的表述 → multimodal
2. 如果问题是关于个人笔记/文档/论文内容的 → text
3. 如果问题是通用知识、时事新闻、编程问题等与个人知识库无关的 → external

请以简短JSON格式返回：{"route": "text", "reason": "简要说明"}"""


class QueryRouter:
    """查询路由器。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.get("aliyun.api_key"),
            base_url=config.get("aliyun.base_url"),
        )
        self.model = config.get("aliyun.llm_model", "qwen-plus")

    def route(self, query: str, intent: str = "") -> RouteDecision:
        """根据查询意图和内容进行路由。"""
        # 快速规则预判
        image_keywords = ["图片", "图表", "如图", "下图", "截图", "这张图", "这个图", "示意图"]
        if any(kw in query for kw in image_keywords):
            return RouteDecision(route="multimodal", reason="查询中包含图片/图表关键词")

        # 如果意图已由rewriter识别为chart_understanding
        if intent == "chart_understanding":
            return RouteDecision(route="multimodal", reason="意图识别为图表理解类问题")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = response.choices[0].message.content or "{}"

            import json
            import re
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                data = json.loads(match.group(0)) if match else {}

            return RouteDecision(
                route=data.get("route", "text"),
                reason=data.get("reason", ""),
            )
        except Exception as e:
            print(f"[QueryRouter] LLM调用失败: {e}")
            return RouteDecision(route="text", reason="路由判断降级为默认文本路由")

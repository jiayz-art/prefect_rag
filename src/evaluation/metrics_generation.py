"""生成质量评估模块 — 基于RAGAS框架的生成质量评估。"""
from typing import Dict, List, Optional

from openai import OpenAI

from src.core.config import config


class GenerationMetrics:
    """基于LLM的生成质量评估器。

    评估维度：忠实度、答案相关性、上下文精确度、上下文召回率。
    使用阿里云LLM进行judge评估（不依赖RAGAS底层依赖时的手动实现）。
    """

    EVAL_PROMPT_FAITHFULNESS = """你是一个评估专家。判断以下"答案"是否完全基于"上下文"中的信息。

如果答案中的所有陈述都能在上下文中找到依据，则评分为1.0。
如果答案中有一部分陈述在上下文中找不到依据（即"幻觉"），则评分应为0.0到0.9之间。

上下文: {context}
答案: {answer}

请只输出一个0到1之间的数字作为评分，格式: score: 0.X"""

    EVAL_PROMPT_RELEVANCY = """你是一个评估专家。判断以下"答案"与"问题"的相关程度。

如果答案完全切题、直接回应了问题，则评分为1.0。
如果答案偏离问题、答非所问，则评分应低于0.5。

问题: {query}
答案: {answer}

请只输出一个0到1之间的数字作为评分，格式: score: 0.X"""

    EVAL_PROMPT_CONTEXT_PRECISION = """你是一个评估专家。判断"上下文"中的信息与"问题"的相关程度。

如果上下文中的信息对回答问题很有帮助、高度相关，则评分为1.0。
如果上下文中包含大量无关信息，则评分应相应降低。

问题: {query}
上下文: {context}

请只输出一个0到1之间的数字作为评分，格式: score: 0.X"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.get("aliyun.api_key"),
            base_url=config.get("aliyun.base_url"),
        )
        self.model = config.get("aliyun.llm_model", "qwen-plus")

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """评估单条问答的生成质量。

        Returns:
            {faithfulness, relevancy, context_precision}
        """
        context_text = "\n---\n".join(contexts[:5])  # 限制长度

        faithfulness = self._judge(
            self.EVAL_PROMPT_FAITHFULNESS.format(context=context_text[:2000], answer=answer[:1000])
        )
        relevancy = self._judge(
            self.EVAL_PROMPT_RELEVANCY.format(query=query, answer=answer[:1000])
        )
        context_precision = self._judge(
            self.EVAL_PROMPT_CONTEXT_PRECISION.format(query=query, context=context_text[:2000])
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_precision,
        }

    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
    ) -> Dict[str, float]:
        """批量评估，返回各维度平均分。"""
        scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}

        for query, answer, contexts in zip(queries, answers, contexts_list):
            result = self.evaluate(query, answer, contexts)
            for key in scores:
                scores[key].append(result[key])

        return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}

    def _judge(self, prompt: str) -> float:
        """调用LLM judge评分。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50,
            )
            content = response.choices[0].message.content or "score: 0.0"
            # 提取数字
            import re
            match = re.search(r"(\d+\.?\d*)", content)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            print(f"[GenerationMetrics] Judge失败: {e}")
            return 0.0

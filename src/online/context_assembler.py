"""上下文拼装模块 — 拼接检索结果、注入引用标注、构造LLM prompt。"""
from typing import Dict, List


class ContextAssembler:
    """上下文拼装器 — 将检索结果组装为LLM可用的prompt上下文。"""

    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length

    def assemble(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        top_k: int = 10,
    ) -> tuple[str, List[Dict]]:
        """拼装上下文并生成引用标注。

        Args:
            query: 用户原始查询
            retrieved_chunks: 多路召回+重排后的chunk列表
            top_k: 最终使用的chunk数量

        Returns:
            (context_text, references): 上下文文本和引用来源列表
        """
        selected = retrieved_chunks[:top_k]

        context_parts = []
        references = []

        for i, chunk in enumerate(selected):
            meta = chunk.get("metadata", {})
            ref_id = i + 1

            source = meta.get("source_name", "未知来源")
            page = meta.get("page", "")
            section = meta.get("section_title", "")

            # 引用标注格式
            ref_label = f"[{ref_id}]"
            if page:
                ref_label += f" p.{page}"
            if section:
                ref_label += f" §{section}"

            context_parts.append(
                f"--- 文档片段 {ref_id} (来源: {source}{', p.' + str(page) if page else ''}) ---\n"
                f"{chunk['content']}\n"
            )

            references.append({
                "ref_id": ref_id,
                "source": source,
                "page": page if page else "N/A",
                "section": section or "N/A",
                "content_preview": chunk["content"][:200],
            })

            # 控制上下文长度
            total_len = sum(len(p) for p in context_parts)
            if total_len > self.max_context_length:
                context_parts.pop()
                references.pop()
                break

        context_text = "\n".join(context_parts)
        return context_text, references

    def build_prompt(self, query: str, context_text: str) -> str:
        """构建完整的LLM prompt。"""
        return f"""你是一个个人知识库助手，基于提供的文档片段回答问题。

## 回答规则
1. 优先使用文档片段中的信息回答问题
2. 如果文档片段不足以回答，请明确说明"根据现有资料无法确定"，并给出你的最佳理解
3. 回答中引用文档时，使用 [来源编号] 标注引用
4. 保持回答简洁清晰、结构分明

## 文档片段
{context_text}

## 用户问题
{query}

## 回答（含引用标注）"""

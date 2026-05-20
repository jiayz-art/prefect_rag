"""评估数据准备脚本 — 用于从文档生成评估问答对。"""
import json
from pathlib import Path
from typing import List, Dict


def generate_synthetic_queries(doc_content: str) -> List[Dict]:
    """基于文档内容生成合成评估问题（规则+模板方式）。

    实际使用时建议用LLM批量生成更高质量的问题。
    """
    queries = []

    # 基于标题生成"是什么"类问题
    import re
    headings = re.findall(r"^#{1,3}\s+(.+)", doc_content, re.MULTILINE)
    for h in headings[:5]:
        queries.append({
            "query": f"请解释什么是{h}？",
            "ground_truth": f"关于{h}的详细说明...（需人工补充）",
            "contexts": [],
            "category": "text",
        })

    # 基于关键词生成"如何做"类问题
    keywords = ["配置", "安装", "索引", "检索", "评估", "优化"]
    for kw in keywords:
        if kw in doc_content:
            queries.append({
                "query": f"如何进行{kw}？",
                "ground_truth": f"关于{kw}的步骤说明...（需人工补充）",
                "contexts": [],
                "category": "text",
            })

    return queries


def main():
    """从知识库文档生成初始评估数据。"""
    docs_dir = Path(__file__).parent.parent / "data" / "docs"

    all_queries = []
    if docs_dir.exists():
        for doc_path in docs_dir.glob("*.md"):
            content = doc_path.read_text(encoding="utf-8")
            queries = generate_synthetic_queries(content)
            all_queries.extend(queries)
            print(f"[{doc_path.name}] 生成 {len(queries)} 个评估问题")

    # 保存
    output_path = Path(__file__).parent.parent / "data" / "eval_queries_generated.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 共生成 {len(all_queries)} 个评估问题，保存至 {output_path}")
    print("[提示] 请手动检查并补充 ground_truth 内容以保证评估质量")


if __name__ == "__main__":
    main()

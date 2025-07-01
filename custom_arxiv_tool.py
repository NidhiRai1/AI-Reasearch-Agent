# custom_arxiv_tool.py

import arxiv
from langchain_core.tools import Tool
from pydantic import BaseModel

# ✅ Define input schema for tool calling compatibility
class ArxivInput(BaseModel):
    query: str

def fetch_arxiv_summary(query: str) -> str:
    """
    Fetches up to 3 recent papers from arXiv based on a query.
    Enhances output with dataset mentions, authors, publication date, and clickable source.
    """

    few_shot_example = """
📘 **Title**: Emergent Behaviors in LLM Agents  
✅ **Summary**: This paper explores generalization in transformer-based agents. It also introduces a synthetic planning benchmark dataset.  
👤 **Authors**: Jane Doe, John Smith  
📅 **Published**: 2024-09-21  
🔗 **Source**: [https://arxiv.org/abs/2409.12345](https://arxiv.org/abs/2409.12345)  
🧬 **[Mentions dataset]**
--------------------------------------------------
"""

    keyword_map = {
        "transformer": "transformer OR attention OR large language model",
        "llm": "large language model OR LLM",
        "ai agent": "AI agent OR autonomous agent OR intelligent agent",
        "reinforcement learning": "reinforcement learning OR RL",
        "natural language processing": "natural language processing OR NLP",
        "nlp": "natural language processing",
        "neural network": "neural network OR deep learning OR cs.NE OR cs.LG",
        "vision": "computer vision OR image recognition OR object detection OR cs.CV",
        "hci": "human-computer interaction OR user modeling OR cs.HC",
        "security": "AI security OR adversarial learning OR privacy OR cs.CR",
        "multi-agent": "multi-agent systems OR agent communication OR cs.MA",
        "robotics": "robotics OR robot agents OR cs.RO",
        "retrieval": "RAG OR information retrieval OR cs.IR",
        "generative ai": "generative AI OR diffusion model OR image synthesis OR cs.CV OR cs.LG",
    }

    for keyword, rewritten in keyword_map.items():
        if keyword in query.lower():
            query = rewritten
            break

    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    seen_titles = set()
    results = []

    for paper in search.results():
        title = paper.title.strip()
        if title in seen_titles:
            continue
        seen_titles.add(title)

        authors = ', '.join(a.name for a in paper.authors)
        summary = paper.summary.replace("\n", " ").strip()
        if len(summary) > 400:
            summary = summary[:400].rsplit(" ", 1)[0] + "..."

        dataset_hint = ""
        if "dataset" in summary.lower() or "corpus" in summary.lower():
            dataset_hint = "🧬 **[Mentions dataset]**"

        formatted = f"""📘 **Title**: {title}
✅ **Summary**: {summary}
👤 **Authors**: {authors}
📅 **Published**: {paper.published.date()}
🔗 **Source**: [{paper.entry_id}]({paper.entry_id})
{dataset_hint}
--------------------------------------------------"""
        results.append(formatted)

        if len(results) == 3:
            break

    if not results:
        return """📘 **Title**: No papers found  
✅ **Summary**: Try a broader query like "transformers", "AI agents", or "NLP".  
👤 **Authors**: N/A  
📅 **Published**: N/A  
🔗 **Source**: N/A  
--------------------------------------------------"""

    return few_shot_example + "\n\n" + "\n\n".join(results)


# ✅ Register the tool with proper input schema to fix Groq tool use error
custom_arxiv_search = Tool(
    name="Arxiv Search",
    description="Searches arXiv.org for recent AI/ML/NLP papers. Returns title, summary, authors, published date, and URL. Highlights if dataset is mentioned.",
    func=fetch_arxiv_summary,
    args_schema=ArxivInput  # 🔥 Allows Groq/LLMs to call this properly
)

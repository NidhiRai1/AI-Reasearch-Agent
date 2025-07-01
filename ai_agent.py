# ai_agent.py

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from custom_arxiv_tool import custom_arxiv_search
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from pdf_tool import generate_pdf_report

# ✅ Wrapper to ensure Tavily tool accepts only a single string argument
def tavily_wrapper(query: str) -> str:
    tavily_tool = TavilySearchResults(max_results=3)
    return tavily_tool.run(query)

# ✅ Register Web Search Tool with correct function signature
web_search_tool = Tool(
    name="Web Search",
    description="Search for relevant datasets or sources using web search (Hugging Face, GitHub, Kaggle, etc.).",
    func=tavily_wrapper
)

# ✅ Main AI agent function
def get_response_from_ai_agent(
    llm_id: str,
    query: list,
    allow_search: bool,
    allow_arxiv: bool,
    allow_pdf: bool,
    system_prompt: str,
    provider: str
) -> dict:
    """
    Executes a LangGraph ReAct agent that can optionally:
    - Search arXiv for papers
    - Search the web for datasets
    - Generate a PDF of the output
    """

    # ✅ Step 1: Load the language model
    provider = provider.lower()
    if provider == "groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "openai":
        llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError("Unsupported provider. Use 'Groq' or 'OpenAI'.")

    # ✅ Step 2: Select tools
    tools = []
    if allow_arxiv:
        tools.append(custom_arxiv_search)
    if allow_search:
        tools.append(web_search_tool)

    # ✅ Step 3: Create ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    # ✅ Step 4: Inject natural prompt (Groq-safe tool instruction)
    if allow_arxiv or allow_search:
        user_input = query[-1].replace("User:", "").strip()
        tool_instruction = (
            f"User: Search for recent open-source research papers about: \"{user_input}\". "
            f"Summarize key findings and include source links. If any datasets or benchmarks are mentioned "
            f"(e.g., on Hugging Face, GitHub, Kaggle), provide short descriptions and direct links."
        )
        query[-1] = tool_instruction

    # ✅ Step 5: Run the agent
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages", [])

    # ✅ Step 6: Get the assistant response
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    final_message = ai_messages[-1] if ai_messages else "No response from the agent."

    # ✅ Step 7: Optional PDF generation
    pdf_path = generate_pdf_report(final_message) if allow_pdf else None

    # ✅ Step 8: Logging (for debug)
    print("🧠 FINAL AGENT RESPONSE:")
    print(final_message)
    print("-" * 50)

    # ✅ Step 9: Return output
    return {
        "response": final_message.strip(),
        "pdf_path": pdf_path
    }

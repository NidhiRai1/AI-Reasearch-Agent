# Agentic AI Chatbot with Dataset Discovery

A multi-modal AI assistant built with **LangGraph ReAct Agents**, **FastAPI**, and **Streamlit**, capable of:

- Answering text + image-based queries
- Fetching **research papers** using ArXiv
- Extracting **datasets & benchmarks** via Tavily Web Search
- Providing **metadata**: authors, publication date, source links
- Returning a **downloadable PDF summary**
- Supporting LLMs via **Groq** and **OpenAI**

---

## Features Completed

| Feature | Description |
|--------|-------------|
| ✅ LangGraph Agent Setup | Uses `create_react_agent` with tool calling (ArXiv, Web Search) |
| ✅ ArXiv Tool (Fixed `__arg1` error) | Added `args_schema` for compatibility with Groq models |
| ✅ Tavily Web Search Tool | Pulls dataset links, benchmark info from HuggingFace, GitHub, etc. |
| ✅ PDF Report Generation | Final agent response is optionally saved as a clean, downloadable PDF |
| ✅ Image + OCR Integration | Extract text from uploaded images via `pytesseract` |
| ✅ FastAPI Backend | `/chat`, `/chat_with_image`, `/chat_with_image_text` routes |
| ✅ Streamlit Frontend | Image upload, config UI, response viewer, PDF download |
| ✅ Session Memory | Short-term memory using Python `deque` for multi-turn context |
| ✅ FAQ Matching | Optional fallback for static questions via fuzzy matching |
| ✅ Logging + Debugging | Logs `FINAL AGENT RESPONSE` and handles errors gracefully |

---

## Stack Used

| Layer | Tech |
|-------|------|
| LLMs | Groq (`llama3`, `mixtral`), OpenAI (`gpt-4o-mini`) |
| Agent Framework | LangGraph + ReAct |
| Web Search Tool | Tavily API |
| Paper Search Tool | Arxiv + schema fix |
| PDF | ReportLab |
| OCR | pytesseract |
| Backend | FastAPI |
| Frontend | Streamlit |

---

---

## Requirements

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt

TAVILY_API_KEY=your_tavily_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

python backend.py
# Runs on http://127.0.0.1:9999

streamlit run frontend.py

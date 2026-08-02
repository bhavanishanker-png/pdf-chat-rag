# PDF RAG Chat — Corrective RAG with LangGraph

A production-grade Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and chat with them using AI. Built with **LangGraph**, **LangChain**, **ChromaDB**, and **Groq's free LLM API**.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) grounds an LLM's answers in your own documents instead of its training data. This prevents hallucinations and makes answers citable.

This app goes further with **Corrective RAG (CRAG)** — a self-correcting pipeline that grades its own retrieved documents, rewrites bad queries, and checks its own answers for hallucinations before returning them.

---

## Features

| Feature | Description |
|---------|-------------|
| Corrective RAG graph | LangGraph pipeline that self-corrects — grades docs, rewrites queries, checks hallucinations |
| Hybrid search | BM25 keyword search + semantic vector search combined via Reciprocal Rank Fusion |
| Document grading | LLM filters irrelevant retrieved chunks before generating |
| Query rewriting | If retrieved docs are poor, the query is automatically rewritten and retried |
| Hallucination check | Generated answer is verified to be grounded in the retrieved context |
| Usefulness check | Answer is verified to actually resolve the question |
| Reasoning trace | Every step the graph took is shown in a collapsible expander |
| Source citations | Retrieved chunks shown with page number and source file |
| Chunk size tuning | Sidebar sliders for chunk size, overlap, and top-K at runtime |
| Clear DB | One-click button to wipe the vector store |
| Download chat | Export the full conversation as `.txt` |

---

## Architecture

```
┌──────────────────────┐        HTTP        ┌───────────────────────────────────────┐
│   Streamlit UI       │ ◄────────────────► │         FastAPI Backend                │
│   (frontend.py)      │                    │         (backend.py)                   │
│                      │                    │                                        │
│  • PDF uploader      │                    │  POST /ingest  → extract, chunk,       │
│  • Chunk sliders     │                    │                  embed, store           │
│  • Chat input        │                    │  POST /ask     → LangGraph CRAG        │
│  • Reasoning trace   │                    │  POST /clear   → wipe ChromaDB         │
│  • Source citations  │                    │  GET  /        → health check          │
│  • Download chat     │                    └───────────────┬───────────────────────┘
└──────────────────────┘                                    │
                                                            │
                              ┌─────────────────────────────┼─────────────────────────┐
                              │                             │                         │
                    ┌─────────▼──────┐          ┌──────────▼──────┐       ┌──────────▼──────┐
                    │   ChromaDB     │          │   Groq API       │       │   PyMuPDF       │
                    │ (vector store) │          │ llama-3.1-8b     │       │ (PDF parsing)   │
                    │  local on disk │          │  (free LLM)      │       │                 │
                    └───────────────┘          └────────────────┘        └────────────────┘
```

---

## LangGraph Pipeline (CRAG)

The core of the app is a self-correcting graph that runs on every question:

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │retrieve │  BM25 + semantic hybrid search (EnsembleRetriever)
                    └────┬────┘
                         │
               ┌─────────▼──────────┐
               │  grade_documents   │  LLM scores each chunk for relevance
               └─────────┬──────────┘
                         │
          ┌──────────────┼──────────────────┐
          │ has docs     │ no docs           │ no docs, max retries
          │              │ retry < 2         │
          │        ┌─────▼──────┐           │
          │        │rewrite_query│           │
          │        └─────┬──────┘           │
          │              │                  │
          │         (loop back to retrieve) │
          │                                 │
          │                          ┌──────▼──────────┐
          │                          │ no_docs_response │ → END
          │                          └─────────────────┘
          │
     ┌────▼─────┐
     │ generate │  Groq llama-3.1-8b answers from context
     └────┬─────┘
          │
  ┌───────▼────────┐
  │ check_quality  │  hallucination check + usefulness check
  └───────┬────────┘
          │
     ┌────┴──────────────────┐
     │ ok                    │ not ok, retry < 2
     │                       │
   ┌─▼──┐              (loop back to generate)
   │ END│
   └────┘
```

**Why CRAG?** A basic RAG pipeline blindly uses whatever it retrieves. CRAG grades the retrieved documents, discards irrelevant ones, rewrites the query if needed, generates an answer, and then verifies the answer isn't hallucinated — all automatically.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Graph orchestration | LangGraph | CRAG pipeline with cycles and conditional edges |
| LLM chains | LangChain | Graders, rewriter, RAG chain |
| LLM | Groq — `llama-3.1-8b-instant` | Fast, free inference |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local, free, 384-dim vectors |
| Vector store | ChromaDB (persistent) | Store and search embeddings |
| Keyword search | rank-bm25 via LangChain | BM25 for exact term matching |
| Hybrid retrieval | LangChain EnsembleRetriever | RRF fusion of BM25 + semantic |
| PDF parsing | PyMuPDF (fitz) | Page-level text extraction |
| Text splitting | langchain-text-splitters | RecursiveCharacterTextSplitter |
| Backend | FastAPI + Uvicorn | REST API |
| Frontend | Streamlit | Chat UI |

---

## Project Structure

```
RAG_Tutorial/
├── step3/
│   ├── backend.py      # FastAPI + LangGraph CRAG pipeline
│   └── frontend.py     # Streamlit chat UI with reasoning trace
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
├── pyrightconfig.json  # Type checker config
├── chroma_db/          # Vector store (auto-created, not committed)
└── .venv/              # Virtual environment (not committed)
```

---

## Setup

### 1. Clone and enter the project

```bash
cd RAG_Tutorial
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). This is a one-time download.

### 4. Add your Groq API key

```
GROQ_API_KEY=your-groq-api-key-here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys). No credit card required.

---

## Running the App

**Terminal 1 — Backend:**

```bash
source .venv/bin/activate
python step3/backend.py
```

**Terminal 2 — Frontend:**

```bash
source .venv/bin/activate
streamlit run step3/frontend.py
```

Streamlit opens `http://localhost:8501` automatically.

---

## Usage

1. **Adjust settings** — set chunk size, overlap, and top-K in the sidebar before ingesting
2. **Upload a PDF** — pick a file and click "Ingest PDF"
3. **Ask a question** — type in the chat box
4. **Inspect the reasoning** — expand "Reasoning trace" to see every step the graph took
5. **View sources** — each source shows the page number and chunk text
6. **Download** — click "Download Chat (.txt)" to save the conversation
7. **Start over** — click "Clear DB" to wipe all stored chunks

---

## API Endpoints

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| GET | `/` | — | Health check — DB status and chunk count |
| POST | `/ingest` | `chunk_size`, `chunk_overlap` (query) | Upload and ingest a PDF |
| POST | `/ask` | `{ query, top_k }` (JSON body) | Run CRAG pipeline, return answer + sources + steps |
| POST | `/clear` | — | Wipe ChromaDB collection |

---

## Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CHUNK_SIZE` | 1000 | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `DEFAULT_TOP_K` | 5 | Chunks retrieved per query |
| `MAX_RETRIES` | 2 | Max query rewrites / generation retries |
| `model` | `llama-3.1-8b-instant` | Groq model |

---

## Dependencies

```
fastapi                 # Backend framework
uvicorn                 # ASGI server
chromadb                # Vector database
groq                    # Groq LLM client (direct)
pymupdf                 # PDF text extraction
python-dotenv           # .env file loading
streamlit               # Frontend UI
requests                # HTTP client
rank-bm25               # BM25 keyword search
langchain               # LLM orchestration
langchain-core          # Base interfaces
langchain-groq          # Groq LLM integration
langchain-chroma        # ChromaDB integration
langchain-huggingface   # HuggingFace embeddings
langchain-community     # BM25Retriever
langchain-classic       # EnsembleRetriever
langchain-text-splitters # RecursiveCharacterTextSplitter
langgraph               # Graph-based pipeline orchestration
sentence-transformers   # Local embedding model runtime
```

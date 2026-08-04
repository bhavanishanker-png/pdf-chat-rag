---
title: PDF RAG Chat
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# PDF RAG Chat — Corrective RAG with LangGraph

A full-stack AI-powered document chat application that lets you upload PDF files and ask questions about their content. Built with **FastAPI**, **LangGraph**, **ChromaDB**, **Next.js**, and **Groq's free LLM API**.

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
| Reasoning trace | Every step the graph took is shown in a collapsible section |
| Source citations | Retrieved chunks shown with page number and source file |
| Chunk size tuning | Sidebar sliders for chunk size, overlap, and top-K at runtime |
| Clear DB | One-click button to wipe the vector store |
| Download chat | Export the full conversation as `.txt` |

---

## Architecture

```
┌──────────────────────────┐        HTTP        ┌───────────────────────────────────────┐
│   Next.js Frontend       │ ◄────────────────► │         FastAPI Backend                │
│   (frontend/)            │                    │         (step3/backend.py)             │
│                          │                    │                                        │
│  • Drag-and-drop upload  │                    │  POST /ingest  → extract, chunk,       │
│  • Chunk sliders         │                    │                  embed, store           │
│  • Chat interface        │                    │  POST /ask     → LangGraph CRAG        │
│  • Reasoning trace       │                    │  POST /clear   → wipe ChromaDB         │
│  • Source cards          │                    │  GET  /        → health check          │
│  • Download chat         │                    └───────────────┬───────────────────────┘
└──────────────────────────┘                                    │
                                                                │
                              ┌─────────────────────────────────┼──────────────────────┐
                              │                                 │                      │
                    ┌─────────▼──────┐          ┌──────────────▼──┐       ┌───────────▼─────┐
                    │   ChromaDB     │          │   Groq API       │       │   PyMuPDF       │
                    │ (vector store) │          │ llama-3.1-8b     │       │ (PDF parsing)   │
                    │  local on disk │          │  (free LLM)      │       │                 │
                    └───────────────┘          └─────────────────┘        └────────────────┘
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
| Embeddings | HuggingFace Inference API — `all-MiniLM-L6-v2` | Free, API-based, 384-dim vectors |
| Vector store | ChromaDB (persistent) | Store and search embeddings |
| Keyword search | rank-bm25 via LangChain | BM25 for exact term matching |
| Hybrid retrieval | LangChain EnsembleRetriever | RRF fusion of BM25 + semantic |
| PDF parsing | PyMuPDF (fitz) | Page-level text extraction |
| Text splitting | langchain-text-splitters | RecursiveCharacterTextSplitter |
| Backend | FastAPI + Uvicorn | REST API |
| Frontend | Next.js 16 + shadcn/ui + Tailwind CSS | Professional chat UI |

---

## Project Structure

```
RAG_Tutorial/
├── step3/
│   └── backend.py          # FastAPI + LangGraph CRAG pipeline
├── frontend/               # Next.js app
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── ChatInterface.tsx
│   │   ├── Sidebar.tsx
│   │   ├── SourceCard.tsx
│   │   ├── ReasoningTrace.tsx
│   │   └── ui/             # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts          # Backend API client
│   │   └── utils.ts
│   └── .env.local          # Frontend env vars
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── .env                    # API keys (not committed)
├── .gitignore
├── pyrightconfig.json      # Python type checker config
├── chroma_db/              # Vector store (auto-created, not committed)
└── .venv/                  # Virtual environment (not committed)
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

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend && npm install
```

### 5. Add your API keys

Create a `.env` file at the project root:

```
GROQ_API_KEY=your-groq-api-key-here
HF_TOKEN=your-huggingface-token-here
```

- Free Groq key: [console.groq.com/keys](https://console.groq.com/keys)
- Free HuggingFace token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — Read access is enough

---

## Running the App

**Terminal 1 — Backend:**

```bash
source .venv/bin/activate
python step3/backend.py
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

---

## Usage

1. **Adjust settings** — set chunk size, overlap, and top-K in the sidebar before ingesting
2. **Upload a PDF** — drag and drop or click to browse
3. **Ask a question** — type in the chat box and press Enter
4. **Inspect the reasoning** — expand the reasoning trace to see every step the graph took
5. **View sources** — each source card shows the page number and chunk text
6. **Download** — click Download to export the conversation as `.txt`
7. **Start over** — click "Clear Vector DB" to wipe all stored chunks

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
| `model` | `llama-3.1-8b-instant` | Groq LLM model |

---

## Dependencies

```
fastapi                 # Backend framework
uvicorn                 # ASGI server
chromadb                # Vector database
groq                    # Groq LLM client
pymupdf                 # PDF text extraction
python-dotenv           # .env file loading
requests                # HTTP client
rank-bm25               # BM25 keyword search
langchain               # LLM orchestration
langchain-core          # Base interfaces
langchain-groq          # Groq LLM integration
langchain-chroma        # ChromaDB integration
langchain-huggingface   # HuggingFace Inference API embeddings
langchain-community     # BM25Retriever
langchain-classic       # EnsembleRetriever
langchain-text-splitters # RecursiveCharacterTextSplitter
langgraph               # Graph-based pipeline orchestration
```

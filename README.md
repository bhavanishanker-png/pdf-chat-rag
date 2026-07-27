# RAG Tutorial — PDF Chat with Groq + ChromaDB

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and chat with them using AI. Built with FastAPI, Streamlit, ChromaDB, and Groq's free LLM API.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that grounds an LLM's answers in your own documents instead of relying on its training data alone. The flow is:

```
PDF → extract text → chunk → embed → store in vector DB
Question → BM25 + semantic search → retrieve top chunks → LLM → grounded answer
```

This prevents hallucinations and makes the AI's answers citable and verifiable.

---

## Features

| Feature | Description |
|---------|-------------|
| PDF ingestion | Upload any PDF — text is extracted, chunked, embedded, and stored |
| Hybrid search | Combines BM25 keyword search + semantic vector search via Reciprocal Rank Fusion |
| Grounded answers | Groq `llama-3.1-8b-instant` answers strictly from your document context |
| Source citations | Every answer shows which chunks of the PDF it came from |
| Highlighted sentence | The most relevant sentence in each source chunk is bolded |
| Confidence score | Each source shows a relevance % score with a visual progress bar |
| Chunk size tuning | Sidebar sliders to adjust chunk size, overlap, and top-K at runtime |
| Clear DB | One-click button to wipe the vector store and start fresh |
| Download chat | Export the full conversation as a `.txt` file |

---

## Architecture

```
┌─────────────────────┐        HTTP        ┌───────────────────────────────────┐
│   Streamlit UI      │ ◄────────────────► │        FastAPI Backend             │
│   (frontend.py)     │                    │        (backend.py)                │
│                     │                    │                                    │
│  • PDF uploader     │                    │  POST /ingest  → extract, chunk,   │
│  • Chunk sliders    │                    │                  embed, store       │
│  • Chat input       │                    │  POST /ask     → hybrid search,    │
│  • Source + score   │                    │                  prompt, answer    │
│  • Download chat    │                    │  POST /clear   → wipe ChromaDB     │
└─────────────────────┘                    │  GET  /        → health check      │
                                           └──────────┬────────────────────────┘
                                                      │
                              ┌───────────────────────┼──────────────────────┐
                              │                       │                      │
                    ┌─────────▼──────┐    ┌───────────▼────┐    ┌────────────▼────┐
                    │   ChromaDB     │    │   Groq API      │    │   PyMuPDF       │
                    │ (vector store) │    │ llama-3.1-8b    │    │ (PDF parsing)   │
                    │  local on disk │    │  (free LLM)     │    │                 │
                    └───────────────┘    └────────────────┘    └────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI + Uvicorn | REST API server |
| Frontend | Streamlit | Chat UI with sliders and download |
| Vector store | ChromaDB (persistent) | Store and search embeddings |
| Embeddings | ChromaDB DefaultEmbeddingFunction | Local, free, no API key needed |
| Keyword search | rank-bm25 | BM25 keyword matching for exact terms |
| LLM | Groq — `llama-3.1-8b-instant` | Fast, free inference |
| PDF parsing | PyMuPDF (fitz) | Extract text from PDFs |
| Environment | python-dotenv | Load API keys from `.env` |

---

## Project Structure

```
RAG_Tutorial/
├── step3/
│   ├── backend.py      # FastAPI app — ingest, hybrid search, ask, clear
│   └── frontend.py     # Streamlit chat UI with sliders and UX features
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
├── chroma_db/          # Vector store data (auto-created, not committed)
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

### 4. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys). No credit card required.

---

## Running the App

You need two terminals open at the same time.

**Terminal 1 — Backend:**

```bash
source .venv/bin/activate
python step3/backend.py
```

You should see:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Frontend:**

```bash
source .venv/bin/activate
streamlit run step3/frontend.py
```

Streamlit opens `http://localhost:8501` in your browser automatically.

---

## Usage

1. **Adjust settings** — use the sidebar sliders to set chunk size, overlap, and how many results to retrieve before ingesting.
2. **Upload a PDF** — pick a file and click "Ingest PDF". The backend extracts text, splits it into overlapping chunks, embeds them locally, and stores them in ChromaDB.
3. **Ask a question** — type in the chat box. The backend runs hybrid search (BM25 + semantic) and sends the top results as context to the LLM.
4. **Inspect sources** — each answer shows collapsible source cards with a relevance %, a progress bar, and the most relevant sentence bolded.
5. **Download the chat** — click "Download Chat (.txt)" in the sidebar to save the conversation.
6. **Start over** — click "Clear DB" to wipe all stored chunks and re-ingest.

---

## How Ingestion Works

```
PDF bytes
   │
   ▼ PyMuPDF
Full text string
   │
   ▼ chunk_text(chunk_size, overlap) — configurable via sliders
List of overlapping text chunks
   │
   ▼ ChromaDB DefaultEmbeddingFunction (all-MiniLM-L6-v2, runs locally)
384-dimensional embeddings
   │
   ▼ ChromaDB PersistentClient
Stored in ./chroma_db with metadata: {source, chunk_index, chunk_size}
```

## How Hybrid Search Works

```
User question
   │
   ├──────────────────────────────────┐
   │                                  │
   ▼ BM25Okapi (rank-bm25)            ▼ ChromaDB semantic search
Keyword scores over all chunks     Vector similarity scores (top-K×3)
   │                                  │
   └──────────┬───────────────────────┘
              │
              ▼ Reciprocal Rank Fusion (RRF, k=60)
Unified ranked list — exact matches + semantic matches combined
              │
              ▼ Top-K chunks sent as context
              │
              ▼ Groq API — llama-3.1-8b-instant
Answer grounded in your PDF, with source citations
```

**Why hybrid?** Semantic search is great for meaning but misses exact terms like names, numbers, and codes. BM25 catches those precisely. RRF merges both ranked lists without needing to tune weights.

---

## API Endpoints

| Method | Endpoint | Params | Description |
|--------|----------|--------|-------------|
| GET | `/` | — | Health check — DB status and chunk count |
| POST | `/ingest` | `chunk_size`, `chunk_overlap` (query params) | Upload and ingest a PDF |
| POST | `/ask` | `{ query, top_k }` (JSON body) | Ask a question, get answer + sources |
| POST | `/clear` | — | Wipe the ChromaDB collection |

---

## Configuration

Defaults in `step3/backend.py` (all overridable at runtime via the UI):

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CHUNK_SIZE` | 1000 | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | 5 | Number of chunks retrieved per query |
| `model` | `llama-3.1-8b-instant` | Groq model used for answers |

---

## Dependencies

```
fastapi          # Backend framework
uvicorn          # ASGI server
chromadb         # Vector database + local embeddings
groq             # Groq LLM client
pymupdf          # PDF text extraction
rank-bm25        # BM25 keyword search
python-dotenv    # .env file loading
streamlit        # Frontend UI
requests         # HTTP client for frontend → backend
```

Install all with:

```bash
pip install -r requirements.txt
```

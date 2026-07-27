# RAG Tutorial — Step 3: PDF Chat with Groq + ChromaDB

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and chat with them using AI. Built with FastAPI, Streamlit, ChromaDB, and Groq's free LLM API.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that grounds an LLM's answers in your own documents instead of relying on its training data alone. The flow is:

```
PDF → extract text → chunk → embed → store in vector DB
Question → embed → similarity search → retrieve top chunks → LLM → grounded answer
```

This prevents hallucinations and makes the AI's answers citable and verifiable.

---

## Architecture

```
┌─────────────────┐        HTTP        ┌──────────────────────────────┐
│  Streamlit UI   │ ◄────────────────► │      FastAPI Backend          │
│  (frontend.py)  │                    │      (backend.py)             │
│                 │                    │                               │
│  • PDF uploader │                    │  /ingest  → extract, chunk,   │
│  • Chat input   │                    │             embed, store       │
│  • Source view  │                    │  /ask     → retrieve, prompt, │
└─────────────────┘                    │             answer            │
                                       │  /        → health check      │
                                       └──────────┬───────────────────┘
                                                  │
                              ┌───────────────────┼───────────────────┐
                              │                   │                   │
                    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌────────▼────────┐
                    │   ChromaDB     │  │  Groq API       │  │  PyMuPDF        │
                    │  (vector store)│  │  llama-3.1-8b   │  │  (PDF parsing)  │
                    │  local on disk │  │  (free LLM)     │  │                 │
                    └───────────────┘  └────────────────┘  └─────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI + Uvicorn | REST API server |
| Frontend | Streamlit | Chat UI |
| Vector store | ChromaDB (persistent) | Store and search embeddings |
| Embeddings | ChromaDB DefaultEmbeddingFunction | Local, free, no API key needed |
| LLM | Groq — `llama-3.1-8b-instant` | Fast, free inference |
| PDF parsing | PyMuPDF (fitz) | Extract text from PDFs |
| Environment | python-dotenv | Load API keys from `.env` |

---

## Project Structure

```
RAG_Tutorial/
├── step3/
│   ├── backend.py      # FastAPI app (ingest, ask, health check)
│   └── frontend.py     # Streamlit chat UI
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

Streamlit will open `http://localhost:8501` in your browser automatically.

---

## Usage

1. **Upload a PDF** — use the sidebar file uploader, then click "Ingest PDF". The backend extracts text, splits it into chunks, embeds them locally, and stores them in ChromaDB.
2. **Ask a question** — type in the chat box. The backend finds the most relevant chunks via similarity search and sends them as context to the LLM.
3. **View sources** — expand the "Sources" section below each answer to see exactly which parts of the PDF the answer came from.

---

## How Ingestion Works

```
PDF bytes
   │
   ▼ PyMuPDF
Full text string
   │
   ▼ chunk_text() — 1000 chars per chunk, 200 char overlap
List of overlapping text chunks
   │
   ▼ ChromaDB DefaultEmbeddingFunction (all-MiniLM-L6-v2, runs locally)
384-dimensional embeddings
   │
   ▼ ChromaDB PersistentClient
Stored in ./chroma_db with metadata: {source, chunk_index}
```

## How Querying Works

```
User question
   │
   ▼ ChromaDB DefaultEmbeddingFunction
Query embedding
   │
   ▼ ChromaDB similarity search (top 5 chunks)
Relevant text chunks + metadata
   │
   ▼ Prompt: system message + context + question
   │
   ▼ Groq API — llama-3.1-8b-instant
Answer grounded in your PDF
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — returns DB status and chunk count |
| POST | `/ingest` | Upload a PDF file for ingestion |
| POST | `/ask` | Submit a question, get an answer with sources |

---

## Configuration

Key constants in `step3/backend.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K` | 5 | Number of chunks retrieved per query |
| `model` | `llama-3.1-8b-instant` | Groq model used for answers |

---

## Dependencies

```
fastapi          # Backend framework
uvicorn          # ASGI server
chromadb         # Vector database
groq             # Groq LLM client
pymupdf          # PDF text extraction
python-dotenv    # .env file loading
streamlit        # Frontend UI
requests         # HTTP client for frontend
```

Install all with:

```bash
pip install -r requirements.txt
```

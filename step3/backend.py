from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import fitz  # PyMuPDF
from groq import Groq
from rank_bm25 import BM25Okapi
import os
import re
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("pdf_chunks")

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += str(page.get_text("text"))
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def find_highlight(chunk: str, query: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    if not sentences:
        return chunk[:200]
    query_words = set(tokenize(query))
    best, best_score = sentences[0], -1.0
    for sentence in sentences:
        words = set(tokenize(sentence))
        if not words:
            continue
        score = len(query_words & words) / len(query_words | words)
        if score > best_score:
            best_score, best = score, sentence
    return best


def distance_to_pct(distance: float) -> int:
    return max(0, round((1 - distance / 2) * 100))


def hybrid_search(
    query: str, top_k: int
) -> tuple[list[str], list[dict], list[float]]:
    """Combine BM25 keyword search and semantic search using Reciprocal Rank Fusion."""
    all_data = collection.get(include=["documents", "metadatas"])  # type: ignore[list-item]
    all_docs = all_data["documents"] or []
    all_metas = all_data["metadatas"] or []
    all_ids = all_data["ids"] or []

    if not all_docs:
        return [], [], []

    n_candidates = min(top_k * 3, len(all_docs))

    # --- BM25 ---
    bm25 = BM25Okapi([tokenize(d) for d in all_docs])
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_top_idx = bm25_scores.argsort()[::-1][:n_candidates]
    bm25_ranks: dict[str, int] = {all_ids[i]: rank for rank, i in enumerate(bm25_top_idx)}

    # --- Semantic ---
    sem = collection.query(
        query_texts=[query],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    sem_ids: list[str] = (sem["ids"] or [[]])[0]  # type: ignore[index]
    sem_distances: list[float] = (sem["distances"] or [[]])[0]
    sem_ranks: dict[str, int] = {doc_id: rank for rank, doc_id in enumerate(sem_ids)}
    sem_dist_map: dict[str, float] = dict(zip(sem_ids, sem_distances))

    # --- Reciprocal Rank Fusion ---
    k = 60
    all_candidate_ids = set(bm25_ranks) | set(sem_ranks)
    rrf_scores: dict[str, float] = {
        doc_id: (1 / (k + bm25_ranks[doc_id] + 1) if doc_id in bm25_ranks else 0)
        + (1 / (k + sem_ranks[doc_id] + 1) if doc_id in sem_ranks else 0)
        for doc_id in all_candidate_ids
    }
    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    id_to_data = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(all_ids, all_docs, all_metas)
    }
    result_docs, result_metas, result_dists = [], [], []
    for doc_id in top_ids:
        doc, meta = id_to_data[doc_id]
        result_docs.append(doc)
        result_metas.append(meta)
        result_dists.append(sem_dist_map.get(doc_id, 1.0))

    return result_docs, result_metas, result_dists


class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K


@app.get("/")
def health_check():
    count = collection.count()
    return {"status": "ok", "chunks_in_db": count}


@app.post("/clear")
def clear_db():
    global collection
    chroma_client.delete_collection("pdf_chunks")
    collection = chroma_client.get_or_create_collection("pdf_chunks")
    return {"message": "Database cleared successfully"}


@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Query(default=DEFAULT_CHUNK_OVERLAP),
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas: list[dict[str, str | int | float | bool | None]] = [
        {"source": file.filename, "chunk_index": i, "chunk_size": chunk_size}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,  # type: ignore[arg-type]
    )

    return {
        "message": f"Ingested {len(chunks)} chunks from '{file.filename}'",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    docs, metadatas, distances = hybrid_search(request.query, request.top_k)

    if not docs:
        return {"answer": "No relevant context found. Please ingest a PDF first.", "sources": []}

    context = "\n\n".join(
        f"[{m['source']}, chunk {m['chunk_index']}]: {d}"
        for d, m in zip(docs, metadatas)
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based solely on "
                    "the provided context. If the answer is not in the context, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {request.query}",
            },
        ],
    )

    answer = response.choices[0].message.content
    sources = [
        {
            "source": m["source"],
            "chunk_index": m["chunk_index"],
            "text": d[:300] + ("..." if len(d) > 300 else ""),
            "highlight": find_highlight(d, request.query),
            "similarity_pct": distance_to_pct(dist),
        }
        for d, m, dist in zip(docs, metadatas, distances)
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

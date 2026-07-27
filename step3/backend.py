from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import fitz  # PyMuPDF
from groq import Groq
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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += str(page.get_text("text"))
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def find_highlight(chunk: str, query: str) -> str:
    """Return the sentence in chunk with the highest word overlap with the query."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    if not sentences:
        return chunk[:200]
    query_words = set(re.findall(r"\w+", query.lower()))
    best, best_score = sentences[0], -1.0
    for sentence in sentences:
        words = set(re.findall(r"\w+", sentence.lower()))
        if not words:
            continue
        score = len(query_words & words) / len(query_words | words)
        if score > best_score:
            best_score, best = score, sentence
    return best


def distance_to_pct(distance: float) -> int:
    """Convert ChromaDB L2 distance to a 0–100 relevance percentage."""
    return max(0, round((1 - distance / 2) * 100))


class QueryRequest(BaseModel):
    query: str


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
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas: list[dict[str, str | int | float | bool | None]] = [
        {"source": file.filename, "chunk_index": i} for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,  # type: ignore[arg-type]
    )

    return {"message": f"Ingested {len(chunks)} chunks from '{file.filename}'"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    results = collection.query(
        query_texts=[request.query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    distances = (results["distances"] or [[]])[0]

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

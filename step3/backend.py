"""
Corrective RAG (CRAG) backend using LangGraph.

Graph flow:
  retrieve → grade_documents
    ├─ relevant docs found  ──────────────────────→ generate → check_quality
    │                                                              ├─ ok         → END
    │                                                              └─ retry < 2  → generate
    └─ no relevant docs, retry < 2  → rewrite_query → retrieve
    └─ no relevant docs, retry >= 2 → no_docs_response → END
"""

from __future__ import annotations

import os
import uuid
from typing import List, Literal, TypedDict

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ── Embeddings + Vector Store ─────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="pdf_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
MAX_RETRIES = 2

# ── Structured graders ────────────────────────────────────────────────────────
class BinaryGrade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' or 'no'")

_doc_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Given a retrieved document and a user question, "
     "output 'yes' if the document is relevant to the question, 'no' otherwise."),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])
doc_grader = _doc_grader_prompt | llm.with_structured_output(BinaryGrade)

_hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hallucination grader. Output 'yes' if the answer is fully grounded "
     "in the provided facts, 'no' if it contains unsupported claims."),
    ("human", "Facts:\n{documents}\n\nAnswer:\n{answer}"),
])
hallucination_grader = _hallucination_prompt | llm.with_structured_output(BinaryGrade)

_answer_grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an answer grader. Output 'yes' if the answer actually resolves "
     "the question, 'no' if it does not."),
    ("human", "Question: {question}\n\nAnswer: {answer}"),
])
answer_grader = _answer_grade_prompt | llm.with_structured_output(BinaryGrade)

# ── Query rewriter ────────────────────────────────────────────────────────────
_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query optimizer. Rewrite the question to be clearer and more specific "
     "for document retrieval. Return only the rewritten question."),
    ("human", "Question: {question}"),
])
query_rewriter = _rewrite_prompt | llm | StrOutputParser()

# ── RAG chain ─────────────────────────────────────────────────────────────────
_rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the provided context. "
     "If the answer is not in the context, say so clearly. Be concise and accurate."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
rag_chain = _rag_prompt | llm | StrOutputParser()

# ── Graph State ───────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    top_k: int
    documents: List[Document]
    answer: str
    steps: List[str]
    retry_count: int
    generation_ok: bool


# ── Helpers ───────────────────────────────────────────────────────────────────
def _all_langchain_docs() -> List[Document]:
    col = vectorstore._collection  # type: ignore[union-attr]
    result = col.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"] or [], result["metadatas"] or [])
    ]


def _ensemble_retriever(top_k: int) -> EnsembleRetriever | None:
    all_docs = _all_langchain_docs()
    if not all_docs:
        return None
    bm25 = BM25Retriever.from_documents(all_docs, k=top_k)
    semantic = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.4, 0.6],
    )


def _docs_as_text(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ── Graph nodes ───────────────────────────────────────────────────────────────
def retrieve(state: GraphState) -> GraphState:
    retriever = _ensemble_retriever(state["top_k"])
    if retriever is None:
        return {**state, "documents": [],
                "steps": state["steps"] + ["Retrieval: database is empty"]}
    docs = retriever.invoke(state["query"])
    return {
        **state,
        "documents": docs,
        "steps": state["steps"] + [f"Retrieved {len(docs)} chunks (BM25 + semantic hybrid)"],
    }


def grade_documents(state: GraphState) -> GraphState:
    filtered: List[Document] = []
    for doc in state["documents"]:
        score: BinaryGrade = doc_grader.invoke({  # type: ignore[assignment]
            "document": doc.page_content[:1000],
            "question": state["query"],
        })
        if score.binary_score == "yes":
            filtered.append(doc)
    kept, total = len(filtered), len(state["documents"])
    return {
        **state,
        "documents": filtered,
        "steps": state["steps"] + [f"Document grading: {kept}/{total} relevant"],
    }


def rewrite_query(state: GraphState) -> GraphState:
    new_q = query_rewriter.invoke({"question": state["query"]})
    return {
        **state,
        "query": new_q,
        "retry_count": state["retry_count"] + 1,
        "steps": state["steps"] + [f'Query rewritten → "{new_q}"'],
    }


def no_docs_response(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "I could not find relevant information in the document to answer your question.",
        "steps": state["steps"] + ["No relevant documents found after retries — giving up"],
    }


def generate(state: GraphState) -> GraphState:
    answer = rag_chain.invoke({
        "context": _docs_as_text(state["documents"]),
        "question": state["query"],
    })
    return {
        **state,
        "answer": answer,
        "retry_count": state["retry_count"] + 1,
        "steps": state["steps"] + ["Generated answer from context"],
    }


def check_quality(state: GraphState) -> GraphState:
    docs_text = _docs_as_text(state["documents"])

    h_score: BinaryGrade = hallucination_grader.invoke({  # type: ignore[assignment]
        "documents": docs_text,
        "answer": state["answer"],
    })
    grounded = h_score.binary_score == "yes"

    if grounded:
        a_score: BinaryGrade = answer_grader.invoke({  # type: ignore[assignment]
            "question": state["query"],
            "answer": state["answer"],
        })
        useful = a_score.binary_score == "yes"
        ok = grounded and useful
        label = "grounded + useful" if ok else "grounded but incomplete"
    else:
        ok = False
        label = "hallucination detected — will retry" if state["retry_count"] < MAX_RETRIES else "hallucination detected — max retries reached"

    return {
        **state,
        "generation_ok": ok,
        "steps": state["steps"] + [f"Quality check: {label}"],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────
def after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "no_docs_response"]:
    if state["documents"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return "no_docs_response"


def after_quality_check(state: GraphState) -> Literal["end", "generate"]:
    if state["generation_ok"]:
        return "end"
    if state["retry_count"] < MAX_RETRIES:
        return "generate"
    return "end"


# ── Build + compile graph ─────────────────────────────────────────────────────
def _build_graph():
    g = StateGraph(GraphState)  # type: ignore[arg-type]

    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("no_docs_response", no_docs_response)
    g.add_node("generate", generate)
    g.add_node("check_quality", check_quality)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "no_docs_response": "no_docs_response"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("no_docs_response", END)
    g.add_edge("generate", "check_quality")
    g.add_conditional_edges(
        "check_quality", after_quality_check,
        {"end": END, "generate": "generate"},
    )

    return g.compile()


rag_graph = _build_graph()


# ── FastAPI request schemas ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    count = vectorstore._collection.count()  # type: ignore[union-attr]
    return {"status": "ok", "chunks_in_db": count}


@app.post("/clear")
def clear_db():
    global vectorstore
    import chromadb as _chromadb
    client = _chromadb.PersistentClient(path="./chroma_db")
    client.delete_collection("pdf_chunks")
    vectorstore = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages: List[Document] = []
    for i in range(len(doc)):  # type: ignore[arg-type]
        page = doc[i]
        text = str(page.get_text("text")).strip()
        if text:
            pages.append(Document(
                page_content=text,
                metadata={"source": file.filename, "page": i},
            ))

    if not pages:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    vectorstore.add_documents(chunks, ids=[str(uuid.uuid4()) for _ in chunks])

    return {
        "message": f"Ingested {len(chunks)} chunks from '{file.filename}'",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    initial_state = GraphState(
        query=request.query,
        top_k=request.top_k,
        documents=[],
        answer="",
        steps=[],
        retry_count=0,
        generation_ok=False,
    )

    final_state = rag_graph.invoke(initial_state)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
        }
        for doc in final_state["documents"]
    ]

    return {
        "answer": final_state["answer"],
        "sources": sources,
        "steps": final_state["steps"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

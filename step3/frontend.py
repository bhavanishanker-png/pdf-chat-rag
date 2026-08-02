import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

STEP_ICONS = {
    "Retrieved": "🔍",
    "Document grading": "📋",
    "Query rewritten": "✏️",
    "Generated": "💬",
    "Quality check": "✅",
    "No relevant": "⚠️",
    "Retrieval:": "📂",
}


def step_icon(step: str) -> str:
    for key, icon in STEP_ICONS.items():
        if step.startswith(key):
            return icon
    return "▸"


def render_reasoning(steps: list) -> None:
    if not steps:
        return
    with st.expander("Reasoning trace", expanded=False):
        for i, step in enumerate(steps):
            icon = step_icon(step)
            st.markdown(f"`{i + 1}.` {icon} {step}")


def render_sources(sources: list) -> None:
    for i, src in enumerate(sources):
        label = f"Source {i + 1} — {src.get('source', '?')}  page {src.get('page', '?')}"
        with st.expander(label):
            pct = src.get("similarity_pct", src.get("relevance_pct", 0))
            if pct:
                st.caption(f"Relevance: {pct}%")
                st.progress(pct / 100)
            chunk_text = src.get("text", "")
            highlight = src.get("highlight", "").strip()
            if highlight and highlight in chunk_text:
                st.markdown(chunk_text.replace(highlight, f"**{highlight}**", 1))
            else:
                st.markdown(chunk_text)


def format_chat_as_text(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "You" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content']}")
        if msg.get("steps"):
            lines.append("  Reasoning:")
            for s in msg["steps"]:
                lines.append(f"    - {s}")
        if msg.get("sources"):
            lines.append("  Sources:")
            for src in msg["sources"]:
                lines.append(f"    - {src.get('source')} page {src.get('page')}")
        lines.append("")
    return "\n".join(lines)


st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("PDF RAG Chat")
st.caption("Powered by LangGraph · Corrective RAG · Hybrid BM25 + Semantic Search · Groq")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.subheader("Chunking settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 3000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 200, 50)
    top_k = st.slider("Top-K results", 1, 10, 5)

    if uploaded_file is not None:
        if st.button("Ingest PDF", type="primary"):
            with st.spinner("Ingesting..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["message"])
                        st.caption(f"chunk_size={data['chunk_size']}, overlap={data['chunk_overlap']}")
                    else:
                        st.error(response.json().get("detail", "Unknown error"))
                except requests.exceptions.ConnectionError:
                    st.error("Backend not reachable. Is it running on port 8000?")

    st.divider()
    st.subheader("Database")
    try:
        health = requests.get(f"{BACKEND_URL}/", timeout=2).json()
        st.metric("Chunks stored", health.get("chunks_in_db", 0))
    except Exception:
        st.warning("Backend not reachable")

    if st.button("Clear DB", type="secondary"):
        try:
            res = requests.post(f"{BACKEND_URL}/clear", timeout=5)
            st.success("Database cleared") if res.status_code == 200 else st.error("Failed")
        except requests.exceptions.ConnectionError:
            st.error("Backend not reachable")

    st.divider()
    if st.session_state.get("messages"):
        st.download_button(
            label="Download Chat (.txt)",
            data=format_chat_as_text(st.session_state["messages"]),
            file_name="chat_history.txt",
            mime="text/plain",
        )

# ── Chat history init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing messages ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_reasoning(message.get("steps", []))
            if message.get("sources"):
                render_sources(message["sources"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"query": prompt, "top_k": top_k},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    steps = data.get("steps", [])

                    st.write(answer)
                    render_reasoning(steps)
                    if sources:
                        render_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "steps": steps,
                    })
                else:
                    msg = f"Backend error {response.status_code}: {response.text}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "steps": []})
            except requests.exceptions.ConnectionError:
                msg = "Could not reach the backend. Is it running on port 8000?"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "steps": []})

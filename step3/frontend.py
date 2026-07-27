import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"


def render_sources(sources: list) -> None:
    for i, src in enumerate(sources):
        with st.expander(f"Source {i + 1} — {src['source']} (chunk {src['chunk_index']})"):
            # Confidence score
            pct = src.get("similarity_pct", 0)
            st.caption(f"Relevance: {pct}%")
            st.progress(pct / 100)

            # Highlighted sentence inside the chunk
            chunk_text = src.get("text", "")
            highlight = src.get("highlight", "")
            if highlight and highlight.strip() in chunk_text:
                marked = chunk_text.replace(
                    highlight.strip(),
                    f"**{highlight.strip()}**",
                    1,
                )
                st.markdown(marked)
            else:
                st.markdown(chunk_text)


def format_chat_as_text(messages: list) -> str:
    lines = []
    for msg in messages:
        role = "You" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content']}")
        if msg.get("sources"):
            lines.append("  Sources:")
            for src in msg["sources"]:
                lines.append(
                    f"    - {src['source']} chunk {src['chunk_index']} "
                    f"(relevance {src.get('similarity_pct', '?')}%)"
                )
        lines.append("")
    return "\n".join(lines)


st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("PDF RAG Chat")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Ingest PDF"):
            with st.spinner("Ingesting..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    )
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the backend. Is it running on port 8000?")

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
            if res.status_code == 200:
                st.success("Database cleared")
            else:
                st.error("Failed to clear database")
        except requests.exceptions.ConnectionError:
            st.error("Could not reach the backend")

    st.divider()
    if st.session_state.get("messages"):
        chat_text = format_chat_as_text(st.session_state["messages"])
        st.download_button(
            label="Download Chat (.txt)",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )

# --- Chat history init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render existing messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"query": prompt},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.write(answer)
                    if sources:
                        render_sources(sources)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    error_msg = f"Backend error {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            except requests.exceptions.ConnectionError:
                error_msg = "Could not reach the backend. Is it running on port 8000?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("PDF RAG Chat")

# --- Sidebar: PDF upload and DB stats ---
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
    st.subheader("Database status")
    try:
        health = requests.get(f"{BACKEND_URL}/", timeout=2).json()
        st.metric("Chunks stored", health.get("chunks_in_db", 0))
    except Exception:
        st.warning("Backend not reachable")

# --- Chat history init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render existing messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** — chunk {src['chunk_index']}")
                    st.text(src["text"])

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
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"**{src['source']}** — chunk {src['chunk_index']}")
                                st.text(src["text"])

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

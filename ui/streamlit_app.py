"""Streamlit front-end for the RAG platform (talks to the FastAPI backend)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise RAG Platform", page_icon="📚", layout="wide")


def api(method: str, path: str, **kw):
    headers = {"X-API-Key": st.session_state.get("api_key", "")}
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=120, **kw)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Is the backend running?")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json() if r.content else None


# ----------------------------------------------------------------------------- login
with st.sidebar:
    st.title("📚 RAG Platform")
    st.text_input("API key", type="password", key="api_key",
                  help="Sent as the X-API-Key header. Dev default: dev-admin-key")
if not st.session_state.get("api_key"):
    st.info("Enter your API key in the sidebar to begin.")
    st.stop()
try:
    me = api("GET", "/me")
    collections = api("GET", "/collections")
except RuntimeError as e:
    st.error(f"Login failed - {e}")
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as **{me['username']}** ({me['role']})")
    if collections:
        names = {f"{c['name']} (#{c['id']})": c["id"] for c in collections}
        label = st.selectbox("Document collection", list(names))
        cid = names[label]
    else:
        cid = None
        st.warning("No collections yet - create one in the Collections tab.")
    top_k = st.slider("Sources per answer (top-k)", 1, 10, 5)
    mode = st.selectbox("Retrieval mode", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    st.divider()
    st.subheader("Conversation history")
    if cid and st.button("➕ New chat"):
        st.session_state.pop("conversation_id", None)
        st.session_state["messages"] = []
    if cid:
        for conv in api("GET", "/conversations", params={"collection_id": cid})[:15]:
            if st.button(conv["title"][:34] or "(untitled)", key=f"conv{conv['id']}"):
                data = api("GET", f"/conversations/{conv['id']}")
                st.session_state["conversation_id"] = conv["id"]
                st.session_state["messages"] = data["messages"]

tab_chat, tab_docs, tab_cols, tab_usage = st.tabs(["💬 Chat", "📄 Documents", "🗂 Collections", "💰 Usage"])

# ----------------------------------------------------------------------------- chat
with tab_chat:
    if cid is None:
        st.stop()
    st.session_state.setdefault("messages", [])
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for c in m.get("citations", []):
                st.caption(f"[{c['n']}] {c['filename']} - page/section {c['page']}")
    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and answering..."):
                try:
                    r = api("POST", "/chat", json={"collection_id": cid, "question": question, "top_k": top_k,
                                                   "mode": mode, "conversation_id": st.session_state.get("conversation_id")})
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["conversation_id"] = r["conversation_id"]
            st.markdown(r["answer"])
            if r["injection_redactions"]:
                st.warning(f"{r['injection_redactions']} source passage(s) contained instruction-like text and were redacted.")
            for c in r["citations"]:
                with st.expander(f"[{c['n']}] {c['filename']} - page/section {c['page']}"):
                    try:
                        prev = api("GET", f"/documents/{c['document_id']}/preview", params={"page": c["page"]})
                        st.caption(f"Document preview - page {prev['page']} of {prev['total_pages']}"
                                   + (f" - {prev['heading']}" if prev.get("heading") else ""))
                        st.write(prev["text"])
                    except RuntimeError:
                        st.write(c["snippet"])
            u = r["usage"]
            st.caption(f"Model: {u['model']} - tokens in/out: {u['input_tokens']}/{u['output_tokens']} - cost: ${u['cost_usd']:.6f}")
        st.session_state["messages"] += [{"role": "user", "content": question, "citations": []},
                                         {"role": "assistant", "content": r["answer"], "citations": r["citations"]}]

# ----------------------------------------------------------------------------- documents
with tab_docs:
    if cid is None:
        st.stop()
    up = st.file_uploader("Upload documents (PDF, DOCX, TXT, MD, PNG/JPG scans)", accept_multiple_files=True,
                          type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"])
    if up and st.button("Ingest uploaded files", type="primary"):
        files = [("files", (f.name, f.getvalue())) for f in up]
        with st.spinner("Extracting, chunking and embedding..."):
            try:
                for res in api("POST", f"/collections/{cid}/documents", files=files):
                    note = "duplicate - skipped" if res.get("duplicate") else res.get("status")
                    st.write(f"{res['filename']}: **{note}**" + (f" - {res['error']}" if res.get("error") else ""))
            except RuntimeError as e:
                st.error(str(e))
    st.subheader("Indexed documents")
    for d in api("GET", f"/collections/{cid}/documents"):
        c1, c2, c3, c4 = st.columns([5, 2, 1, 1])
        flag = " ⚠️ injection-like text" if d["flagged_chunks"] else ""
        c1.write(f"**{d['filename']}** - {d['pages']} pages/sections, {d['chunks']} chunks"
                 f"{' (OCR)' if d['ocr_used'] else ''}{flag}")
        c2.write(d["status"])
        if c3.button("Re-index", key=f"re{d['id']}"):
            api("POST", f"/documents/{d['id']}/reindex")
            st.rerun()
        if c4.button("Delete", key=f"del{d['id']}"):
            api("DELETE", f"/documents/{d['id']}")
            st.rerun()
        with st.expander("Preview"):
            page = st.number_input("Page / section", 1, max(d["pages"], 1), 1, key=f"pg{d['id']}")
            try:
                st.write(api("GET", f"/documents/{d['id']}/preview", params={"page": int(page)})["text"])
            except RuntimeError as e:
                st.caption(str(e))

# ----------------------------------------------------------------------------- collections
with tab_cols:
    new = st.text_input("New collection name")
    if st.button("Create collection") and new:
        api("POST", "/collections", json={"name": new})
        st.rerun()
    if cid is not None:
        st.subheader("Sharing & deletion")
        who = st.text_input("Share the selected collection with username")
        if st.button("Share") and who:
            try:
                api("POST", f"/collections/{cid}/share", json={"username": who})
                st.success(f"Shared with {who}")
            except RuntimeError as e:
                st.error(str(e))
        if st.button("Delete selected collection (irreversible)"):
            api("DELETE", f"/collections/{cid}")
            st.rerun()
    if me["role"] == "admin":
        st.subheader("Create user (admin)")
        nu = st.text_input("Username")
        role = st.selectbox("Role", ["user", "admin"])
        if st.button("Create user") and nu:
            try:
                u = api("POST", "/users", json={"username": nu, "role": role})
                st.success("User created. Copy the API key now - it is shown only once:")
                st.code(u["api_key"])
            except RuntimeError as e:
                st.error(str(e))

# ----------------------------------------------------------------------------- usage
with tab_usage:
    scope = "all" if me["role"] == "admin" and st.toggle("Show all users", value=False) else "me"
    data = api("GET", "/usage", params={"scope": scope})
    a, b, c = st.columns(3)
    a.metric("Requests", data["total_requests"])
    b.metric("Tokens (in / out)", f"{data['total_input_tokens']:,} / {data['total_output_tokens']:,}")
    c.metric("Spend (USD)", f"${data['total_cost_usd']:.4f}")
    st.dataframe(data["by_model"])

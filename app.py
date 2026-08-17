# ==========================================
# SQLITE WORKAROUND FOR STREAMLIT CLOUD
# (This must be at the very top!)
# ==========================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import uuid

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Document | Document Companion", page_icon="📄", layout="centered")

# ==========================================
# 2. WARM WELLNESS THEME (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #F6F1E9;
        --bg-secondary: #EFE8DA;
        --bg-elevated: #FBF8F2;
        --border-subtle: #E2D8C4;
        --text-primary: #3A332A;
        --text-secondary: #766B5C;
        --text-muted: #A69C8A;
        --accent: #B9713D;
        --accent-soft: #F0DFC9;
        --accent-hover: #A05F30;
        --success: #7C8F6E;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    footer, #MainMenu, header { visibility: hidden; }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    .brand-mark { display: flex; align-items: center; gap: 10px; margin-bottom: 22px; }
    .brand-mark-icon {
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, #D9A867, var(--accent));
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; box-shadow: 0 4px 12px rgba(185,113,61,0.25);
    }
    .brand-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 19px; color: var(--text-primary); }

    .side-section-label {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--text-muted); margin: 22px 0 10px 0;
    }

    section[data-testid="stSidebar"] input[type="password"],
    section[data-testid="stSidebar"] input[type="text"] {
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background-color: #E9EFE4; border: 1px solid #D4E0CB; color: var(--success);
        font-size: 13px; font-weight: 500; padding: 8px 12px; border-radius: 10px;
        width: 100%; box-sizing: border-box;
    }
    .status-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); }

    .file-chip {
        display: flex; align-items: center; gap: 8px;
        background-color: var(--bg-elevated); border: 1px solid var(--border-subtle);
        border-radius: 10px; padding: 10px 12px; font-size: 13.5px;
        color: var(--text-secondary); margin-top: 4px;
    }

    section[data-testid="stSidebar"] .stButton button {
        background-color: var(--bg-elevated); color: var(--text-primary);
        border: 1px solid var(--border-subtle); border-radius: 10px;
        font-weight: 500; font-size: 14px; padding: 9px 0;
        transition: all 0.15s ease; text-align: left;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: var(--accent); color: var(--accent-hover); background-color: var(--accent-soft);
    }

    section[data-testid="stSidebar"] .history-active button {
        border-color: var(--accent) !important;
        background-color: var(--accent-soft) !important;
        color: var(--accent-hover) !important;
        font-weight: 600;
    }

    hr, section[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; }

    .app-header {
        text-align: center; padding: 40px 0 30px 0;
        border-bottom: 1px solid var(--border-subtle); margin-bottom: 30px;
    }
    .app-header-eyebrow {
        font-size: 12px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 10px;
    }
    .app-header-title { font-family: 'Fraunces', serif; font-size: 40px; font-weight: 600; color: var(--text-primary); margin: 0; }
    .app-header-sub { font-size: 15px; color: var(--text-secondary); margin-top: 10px; font-weight: 400; }

    [data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; padding: 4px 0 !important; }

    .msg-card {
        background-color: var(--bg-elevated); border: 1px solid var(--border-subtle);
        border-radius: 16px; padding: 16px 19px; font-size: 15px; line-height: 1.7;
        color: var(--text-primary); box-shadow: 0 2px 10px rgba(58,51,42,0.05);
    }

    .source-card {
        background-color: var(--bg-secondary); border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent); border-radius: 8px; padding: 10px 13px;
        font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; line-height: 1.55;
    }

    .streamlit-expanderHeader {
        background-color: transparent !important; color: var(--text-muted) !important;
        font-size: 13px !important; font-weight: 500 !important;
    }
    .streamlit-expanderContent { background-color: transparent !important; border: none !important; }

    .stChatInputContainer { padding-bottom: 24px !important; }
    [data-testid="stChatInput"] {
        border-radius: 18px !important;
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: 0 6px 20px rgba(58,51,42,0.08) !important;
    }
    [data-testid="stChatInput"] textarea { color: var(--text-primary) !important; font-size: 15.5px !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
    [data-testid="stChatInputSubmitButton"] { background-color: var(--accent) !important; border-radius: 10px !important; }
    [data-testid="stChatInput"] button svg { color: var(--text-secondary) !important; }

    div[data-testid="stAlert"] { border-radius: 10px !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Embeddings & ChromaDB ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="pdf_knowledge_base")

embedder = load_embedder()
collection = load_chroma()

# ==========================================
# GEMINI API CALL
# ==========================================
def ask_gemini(question, context_chunks, api_key):
    client = genai.Client(api_key=api_key)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful, knowledgeable AI assistant.
Answer the user's question using ONLY the provided document context below.
Always provide a full, natural-sounding answer consisting of 2 to 4 sentences.
Do not give one-word answers. If the answer isn't in the context, politely state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# ==========================================
# SESSION / HISTORY STATE
# ==========================================
def new_session():
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "chat_history": [],
        "processed_filename": None,
    }

if "sessions" not in st.session_state:
    st.session_state.sessions = [new_session()]
    st.session_state.active_session_id = st.session_state.sessions[0]["id"]

def get_active_session():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_session_id:
            return s
    return st.session_state.sessions[0]

active = get_active_session()

# --- Process PDF into the vector store ---
def chunk_text(text, chunk_size=800, chunk_overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks

def process_pdf(file):
    with st.spinner("Processing document..."):
        reader = PyPDF2.PdfReader(file)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(full_text)

        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])

        ids = [f"{file.name}_chunk_{i}" for i in range(len(chunks))]
        collection.add(ids=ids, embeddings=embedder.encode(chunks).tolist(), documents=chunks)
        active["processed_filename"] = file.name

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-mark-icon">📄</div>
        <div class="brand-title">Document</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-section-label">API Configuration</div>', unsafe_allow_html=True)

    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        st.markdown('<div class="status-pill"><span class="dot"></span> API key loaded securely</div>', unsafe_allow_html=True)
    else:
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your API key", label_visibility="collapsed")
        st.caption("Get a free key from [Google AI Studio](https://aistudio.google.com/)")

    if active["processed_filename"]:
        st.markdown('<div class="side-section-label">Document</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="file-chip">📄&nbsp; {active["processed_filename"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.sessions.insert(0, new_session())
        st.session_state.active_session_id = st.session_state.sessions[0]["id"]
        st.rerun()

    st.markdown('<div class="side-section-label">History</div>', unsafe_allow_html=True)
    if len(st.session_state.sessions) == 0:
        st.caption("No previous chats yet.")
    for s in st.session_state.sessions:
        label = s["title"] if s["title"] else "New chat"
        is_active = s["id"] == active["id"]
        wrapper_class = "history-active" if is_active else ""
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.button(f"💬  {label}", key=f"hist_{s['id']}", use_container_width=True):
            st.session_state.active_session_id = s["id"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="app-header">
    <div class="app-header-eyebrow">Document Companion</div>
    <p class="app-header-title">Ask, and read with ease</p>
    <p class="app-header-sub">Warm, grounded answers — sourced directly from your document.</p>
</div>
""", unsafe_allow_html=True)

def render_user_bubble(text):
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 18px;">
        <div style="background-color: #F0DFC9; color: #4A3A24; border: 1px solid #E2C79E; border-radius: 18px 18px 4px 18px; padding: 12px 18px; max-width: 78%; font-size: 15px; line-height: 1.6;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_assistant_bubble(text, sources=None):
    st.markdown(f"""
    <div style="display: flex; margin-bottom: 8px;">
        <div class="msg-card" style="max-width: 88%;">{text}</div>
    </div>
    """, unsafe_allow_html=True)
    if sources:
        with st.expander("View sources"):
            for chunk in sources:
                st.markdown(f'<div class="source-card">{chunk}</div>', unsafe_allow_html=True)

for message in active["chat_history"]:
    if message["role"] == "user":
        render_user_bubble(message["content"])
    else:
        render_assistant_bubble(message["content"], message.get("sources"))

# ==========================================
# CHAT INPUT (text + inline PDF attach button)
# ==========================================
chat_submission = st.chat_input(
    "Ask a question about your document...",
    accept_file=True,
    file_type=["pdf"],
)

if chat_submission:
    if chat_submission["files"]:
        attached_file = chat_submission["files"][0]
        process_pdf(attached_file)
        st.toast(f"📄 {attached_file.name} uploaded and ready", icon="✅")
        st.rerun()

    user_question = chat_submission.text

    if user_question:
        if not gemini_key:
            st.error("Please enter your Gemini API Key in the sidebar or save it in Streamlit Secrets!")
        elif not active["processed_filename"]:
            st.error("Please upload a PDF first — tap the ＋ icon in the chat bar!")
        else:
            if active["title"] == "New chat":
                active["title"] = user_question[:40] + ("…" if len(user_question) > 40 else "")

            render_user_bubble(user_question)

            try:
                retrieved_chunks = collection.query(
                    query_embeddings=embedder.encode([user_question]).tolist(),
                    n_results=3
                )['documents'][0]

                with st.spinner("Generating answer..."):
                    answer = ask_gemini(user_question, retrieved_chunks, gemini_key)

                render_assistant_bubble(answer, retrieved_chunks)

                active["chat_history"].append({"role": "user", "content": user_question})
                active["chat_history"].append({"role": "assistant", "content": answer, "sources": retrieved_chunks})

            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")

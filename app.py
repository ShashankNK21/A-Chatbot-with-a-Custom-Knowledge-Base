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

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Docurion | Document Intelligence", page_icon="📘", layout="centered")

# ==========================================
# 2. PROFESSIONAL THEME (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

    :root {
        --bg-primary: #0B0D12;
        --bg-secondary: #12151C;
        --bg-elevated: #171B24;
        --border-subtle: #232833;
        --text-primary: #EDEFF3;
        --text-secondary: #9BA3B0;
        --text-muted: #626A78;
        --accent: #4C7CF0;
        --accent-soft: #1B2740;
        --accent-hover: #6690FF;
        --success: #34C77B;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    footer, #MainMenu, header { visibility: hidden; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #10131B 0%, #0B0D12 55%);
        color: var(--text-primary);
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .brand-mark-icon {
        width: 34px; height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--accent), #8A5CF6);
        display: flex; align-items: center; justify-content: center;
        font-size: 17px;
        box-shadow: 0 4px 14px rgba(76,124,240,0.35);
    }
    .brand-title {
        font-weight: 700; font-size: 18px; color: var(--text-primary); letter-spacing: -0.01em;
    }
    .brand-subtitle {
        font-size: 12.5px; color: var(--text-muted); margin-bottom: 22px; margin-left: 44px; margin-top: -4px;
    }

    .side-section-label {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--text-muted); margin: 22px 0 10px 0;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-elevated) !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 12px !important;
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
        background-color: rgba(52,199,123,0.1);
        border: 1px solid rgba(52,199,123,0.25);
        color: var(--success);
        font-size: 13px; font-weight: 500;
        padding: 7px 12px; border-radius: 8px;
        width: 100%; box-sizing: border-box;
    }
    .status-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); }

    .file-chip {
        display: flex; align-items: center; gap: 8px;
        background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    section[data-testid="stSidebar"] .stButton button {
        background-color: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        font-weight: 500;
        font-size: 14px;
        padding: 9px 0;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: var(--accent);
        color: var(--accent-hover);
    }

    hr, section[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; }

    /* ---------- MAIN HEADER ---------- */
    .app-header {
        text-align: center;
        padding: 38px 0 28px 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 30px;
    }
    .app-header-eyebrow {
        font-size: 12px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--accent-hover); margin-bottom: 10px;
    }
    .app-header-title {
        font-family: 'Source Serif 4', serif;
        font-size: 38px; font-weight: 600; color: var(--text-primary);
        margin: 0; letter-spacing: -0.01em;
    }
    .app-header-sub {
        font-size: 15px; color: var(--text-secondary); margin-top: 10px; font-weight: 400;
    }

    /* ---------- EMPTY STATE ---------- */
    .empty-state {
        text-align: center;
        padding: 50px 20px 30px 20px;
    }
    .empty-state-icon {
        width: 56px; height: 56px; border-radius: 16px;
        background: linear-gradient(135deg, var(--accent-soft), rgba(138,92,246,0.12));
        border: 1px solid var(--border-subtle);
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; margin: 0 auto 20px auto;
    }
    .empty-state-title {
        font-size: 21px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 14.5px; color: var(--text-muted); max-width: 380px; margin: 0 auto; line-height: 1.6;
    }

    /* ---------- CHAT ---------- */
    [data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; padding: 4px 0 !important; }

    .msg-card {
        background-color: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 18px;
        font-size: 15px;
        line-height: 1.65;
        color: var(--text-primary);
    }

    .source-card {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 10px 13px;
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 8px;
        line-height: 1.55;
    }

    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: var(--text-muted) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background-color: transparent !important;
        border: none !important;
    }

    /* ---------- CHAT INPUT ---------- */
    .stChatInputContainer { padding-bottom: 24px !important; }
    .stChatInputContainer > div {
        border-radius: 16px !important;
        background-color: var(--bg-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important;
        padding: 6px 8px;
    }
    .stChatInputContainer textarea {
        color: var(--text-primary) !important;
        font-size: 15.5px !important;
    }
    .stChatInputContainer textarea::placeholder { color: var(--text-muted) !important; }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        font-size: 14px !important;
    }
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
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-mark-icon">📘</div>
        <div class="brand-title">Docurion</div>
    </div>
    <div class="brand-subtitle">Document Intelligence Workspace</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-section-label">API Configuration</div>', unsafe_allow_html=True)

    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        st.markdown('<div class="status-pill"><span class="dot"></span> API key loaded securely</div>', unsafe_allow_html=True)
    else:
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your API key", label_visibility="collapsed")
        st.caption("Get a free key from [Google AI Studio](https://aistudio.google.com/)")

    st.markdown('<div class="side-section-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f'<div class="file-chip">📄&nbsp; {uploaded_file.name}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Process PDF ---
def chunk_text(text, chunk_size=800, chunk_overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks

if uploaded_file is not None:
    if "processed_filename" not in st.session_state or st.session_state.processed_filename != uploaded_file.name:
        with st.spinner("Processing document..."):
            reader = PyPDF2.PdfReader(uploaded_file)
            full_text = "".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(full_text)

            existing = collection.get()
            if existing['ids']: collection.delete(ids=existing['ids'])

            ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
            collection.add(ids=ids, embeddings=embedder.encode(chunks).tolist(), documents=chunks)
            st.session_state.processed_filename = uploaded_file.name

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="app-header">
    <div class="app-header-eyebrow">Document Intelligence</div>
    <p class="app-header-title">Ask anything about your document</p>
    <p class="app-header-sub">Grounded answers, sourced directly from the PDF you upload.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EMPTY STATE
# ==========================================
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">✦</div>
        <div class="empty-state-title">Ready when you are</div>
        <p class="empty-state-text">Upload a PDF in the sidebar, then ask a question below.
        Every answer is drawn strictly from your document, with sources you can verify.</p>
    </div>
    """, unsafe_allow_html=True)

def render_user_bubble(text):
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 18px;">
        <div style="background-color: #1B2740; color: #DCE4FF; border: 1px solid #29365C; border-radius: 16px 16px 4px 16px; padding: 12px 18px; max-width: 78%; font-size: 15px; line-height: 1.55;">
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

# Render Chat History
for message in st.session_state.chat_history:
    if message["role"] == "user":
        render_user_bubble(message["content"])
    else:
        render_assistant_bubble(message["content"], message.get("sources"))

# Chat Input
if user_question := st.chat_input("Ask a question about your document..."):
    if not gemini_key:
        st.error("Please enter your Gemini API Key in the sidebar or save it in Streamlit Secrets!")
    elif not uploaded_file:
        st.error("Please upload a PDF first!")
    else:
        render_user_bubble(user_question)

        retrieved_chunks = collection.query(
            query_embeddings=embedder.encode([user_question]).tolist(),
            n_results=3
        )['documents'][0]

        with st.spinner("Generating answer..."):
            answer = ask_gemini(user_question, retrieved_chunks, gemini_key)

        render_assistant_bubble(answer, retrieved_chunks)

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})

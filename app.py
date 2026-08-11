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
# 1. PAGE CONFIG & GEMINI STYLING
# ==========================================
st.set_page_config(page_title="Document AI", page_icon="✨", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&display=swap');
    
    header, footer { visibility: hidden; }
    .stApp { background-color: #131314; }
    .stChatInputContainer { padding-bottom: 20px !important; }
    .stChatInputContainer > div {
        border-radius: 32px !important;
        background-color: #1E1F20 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        padding: 4px 10px;
    }
    .stChatInputContainer textarea { color: #E3E3E3 !important; font-size: 16px; }
    [data-testid="stChatMessage"] { background-color: transparent !important; border: none !important; }
    .gemini-greeting {
        font-family: 'Google Sans', sans-serif !important;
        font-size: 56px; font-weight: 500;
        background: -webkit-linear-gradient(74deg, #4285F4 0, #9B72CB 9%, #D96570 20%, #D96570 24%, #9B72CB 35%, #4285F4 44%, #9B72CB 50%, #D96570 56%, #131314 75%, #131314 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: -15px; line-height: 1.2;
    }
    .gemini-subtext { 
        font-family: 'Google Sans', sans-serif !important; 
        font-size: 56px; font-weight: 500; color: #444746; line-height: 1.2; margin-top: 0; 
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
    st.markdown("### ✨ Setup")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    st.caption("Get a free key from [Google AI Studio](https://aistudio.google.com/)")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.info(f"📄 `{uploaded_file.name}`")
        
    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
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
# MAIN CHAT UI
# ==========================================
if len(st.session_state.chat_history) == 0:
    st.markdown('<p class="gemini-greeting">Hello,</p>', unsafe_allow_html=True)
    st.markdown('<p class="gemini-subtext">How can I help with your PDF?</p>', unsafe_allow_html=True)

def render_user_bubble(text):
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
        <div style="background-color: #282A2C; color: #E3E3E3; border-radius: 20px; padding: 12px 20px; max-width: 75%; font-size: 16px;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.chat_history:
    if message["role"] == "user":
        render_user_bubble(message["content"])
    else:
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for chunk in message["sources"]:
                        st.markdown(f"*{chunk}*")

# Chat Input
if user_question := st.chat_input("Ask a question about your document..."):
    if not gemini_key:
        st.error("Please enter your Gemini API Key in the sidebar first!")
    elif not uploaded_file:
        st.error("Please upload a PDF first!")
    else:
        render_user_bubble(user_question)
        
        retrieved_chunks = collection.query(
            query_embeddings=embedder.encode([user_question]).tolist(),
            n_results=3
        )['documents'][0]

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Generating..."):
                answer = ask_gemini(user_question, retrieved_chunks, gemini_key)
            st.markdown(answer)
            with st.expander("Sources"):
                for chunk in retrieved_chunks:
                    st.markdown(f"*{chunk}*")

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": retrieved_chunks})

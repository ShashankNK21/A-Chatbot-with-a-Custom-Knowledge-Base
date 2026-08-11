# ==========================================
# SQLITE WORKAROUND FOR STREAMLIT CLOUD
# (Must be at the absolute top!)
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
        model="gemini-2.5-flash",
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
    st.markdown('Hello,', unsafe_allow_html=True)
    st.markdown('How can I help with your PDF?', unsafe_allow_html=True)

def render_user_bubble(text):
    st.markdown(f"""
    
        
            {text}
        
    
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

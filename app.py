# ==========================================
# SQLITE WORKAROUND FOR STREAMLIT CLOUD
# (This must be at the very top!)
# ==========================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ==========================================
# REST OF YOUR IMPORTS
# ==========================================
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
# ... (the rest of your code continues normally below this)
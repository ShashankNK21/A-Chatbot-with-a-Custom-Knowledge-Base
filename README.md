# ✨ Document AI: Custom Knowledge Base Chatbot

A sleek, mobile-responsive web application that allows users to upload PDF documents and chat with them using Google's **Gemini 3.6 Flash** model. 

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline. It extracts text from PDFs, generates semantic embeddings, stores them in a local vector database, and retrieves the most relevant context to answer user queries accurately.

## 🚀 Key Features

* **Context-Aware Chat:** Upload any PDF and ask natural language questions about its contents.
* **Semantic Search:** Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to understand the meaning of your queries, not just basic keyword matching.
* **Local Vector Database:** Integrates `ChromaDB` for fast, efficient, and private document retrieval without relying on external cloud vector stores.
* **Custom UI:** Features a custom dark-mode interface inspired by Google Gemini, complete with gradient text, hidden default footers, and seamless chat bubbles.
* **Mobile Responsive:** Fully optimized for smartphones and tablets with adaptive text sizing and a collapsible sidebar.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Integration:** [Google GenAI SDK](https://ai.google.dev/) (Gemini 3.6 Flash)
* **Embeddings:** [SentenceTransformers](https://sbert.net/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Document Processing:** [PyPDF2](https://pypdf2.readthedocs.io/)

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/a-chatbot-with-a-custom-knowledge-base.git](https://github.com/your-username/a-chatbot-with-a-custom-knowledge-base.git)
   cd a-chatbot-with-a-custom-knowledge-base

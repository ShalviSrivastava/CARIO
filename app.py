import streamlit as st
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ---- Page config ----
st.set_page_config(page_title="CARIO - Counselor AI", layout="wide")
st.title("CARIO: Counselor AI for Interpretable Risk Outcomes")
st.caption("Ask questions about student dropout risk, grounded in SHAP-explained model predictions.")

# ---- Load resources (cached so it only loads once) ----
@st.cache_resource
def load_resources():
    with open("shap_explanations.json", "r") as f:
        all_explanations = json.load(f)

    with open("embeddings_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    index = faiss.read_index("faiss_index.bin")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    return all_explanations, metadata, index, embedder, best_model

all_explanations, metadata, index, embedder, best_model = load_resources()

# ---- Groq LLM setup ----
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0.2)

# ---- RAG function ----
def rag_query(user_query, k=5):
    query_embedding = embedder.encode([user_query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    retrieved_context = "\n\n".join([all_explanations[str(idx)] for idx in indices[0]])

    prompt = f"""You are an academic counseling assistant helping advisors understand student dropout risk.
Use ONLY the student data below to answer the question. Cite specific student IDs when relevant.
If the data doesn't contain enough information to answer, say so honestly.

STUDENT DATA:
{retrieved_context}

QUESTION: {user_query}

ANSWER:"""

    response = llm.invoke(prompt)
    return response.content, indices[0]

# ---- UI ----
tab1, tab2 = st.tabs(["💬 Ask CARIO", "📋 Browse Student Profiles"])

with tab1:
    user_query = st.text_input("Ask a question about student dropout risk:",
                                placeholder="e.g. Which students need immediate intervention?")
    if st.button("Ask") and user_query:
        with st.spinner("Retrieving and analyzing..."):
            answer, retrieved_ids = rag_query(user_query)
        st.markdown("### Answer")
        st.write(answer)
        st.caption(f"Based on students: {list(retrieved_ids)}")

with tab2:
    student_id = st.number_input("Enter Student ID (0-648):", min_value=0, max_value=648, step=1)
    if st.button("View Profile"):
        st.text(all_explanations[str(student_id)])
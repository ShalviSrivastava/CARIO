import streamlit as st
import json
import pickle
import re
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

    return all_explanations, metadata, index, embedder

all_explanations, metadata, index, embedder = load_resources()

# ---- Extract risk probability from explanation text ----
@st.cache_data
def get_risk_scores(_all_explanations):
    risk_scores = {}
    for sid, text in _all_explanations.items():
        match = re.search(r'\(([\d.]+)\)', text)
        risk_scores[sid] = float(match.group(1)) if match else 0.0
    return risk_scores

risk_scores = get_risk_scores(all_explanations)

# ---- Groq LLM setup ----
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0.2)

# ---- RAG function ----
def rag_query(user_query, k=5, filter_high_risk=False):
    if filter_high_risk:
        high_risk_ids = [sid for sid, score in risk_scores.items() if score >= 0.6]
        candidate_texts = [all_explanations[sid] for sid in high_risk_ids]
        candidate_embeddings = embedder.encode(candidate_texts).astype('float32')
        temp_index = faiss.IndexFlatL2(candidate_embeddings.shape[1])
        temp_index.add(candidate_embeddings)

        query_embedding = embedder.encode([user_query]).astype('float32')
        distances, local_indices = temp_index.search(query_embedding, min(k, len(high_risk_ids)))
        result_ids = [high_risk_ids[i] for i in local_indices[0]]
    else:
        query_embedding = embedder.encode([user_query]).astype('float32')
        distances, indices = index.search(query_embedding, k)
        result_ids = [str(idx) for idx in indices[0]]

    retrieved_context = "\n\n".join([all_explanations[sid] for sid in result_ids])

    prompt = f"""You are an academic counseling assistant helping advisors understand student dropout risk.
Use ONLY the student data below to answer the question. Cite specific student IDs when relevant.
If the data doesn't contain enough information to answer, say so honestly.

STUDENT DATA:
{retrieved_context}

QUESTION: {user_query}

ANSWER:"""

    response = llm.invoke(prompt)
    return response.content, result_ids

# ---- UI ----
tab1, tab2 = st.tabs(["💬 Ask CARIO", "📋 Browse Student Profiles"])

with tab1:
    user_query = st.text_input("Ask a question about student dropout risk:",
                                placeholder="e.g. Which students need immediate intervention?")
    high_risk_toggle = st.checkbox("Search only among high-risk students", value=True)
    if st.button("Ask") and user_query:
        with st.spinner("Retrieving and analyzing..."):
            answer, retrieved_ids = rag_query(user_query, filter_high_risk=high_risk_toggle)
        st.markdown("### Answer")
        st.write(answer)
        st.caption(f"Based on students: {retrieved_ids}")

with tab2:
    student_id = st.number_input("Enter Student ID (0-648):", min_value=0, max_value=648, step=1)
    if st.button("View Profile"):
        st.text(all_explanations[str(student_id)])

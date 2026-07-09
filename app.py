import streamlit as st
import json
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ---- Page config ----
st.set_page_config(page_title="CARIO - Counselor AI", layout="wide", page_icon="🎓")

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .risk-moderate {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .risk-low {
        background-color: #d1fae5;
        border-left: 4px solid #059669;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .answer-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin-top: 16px;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="main-header">🎓 CARIO</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Counselor AI for Interpretable Risk Outcomes — SHAP-explained, RAG-grounded dropout risk insights</p>', unsafe_allow_html=True)

# ---- Load resources ----
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

@st.cache_data
def get_risk_scores(_all_explanations):
    risk_scores = {}
    for sid, text in _all_explanations.items():
        match = re.search(r'\(([\d.]+)\)', text)
        risk_scores[sid] = float(match.group(1)) if match else 0.0
    return risk_scores

risk_scores = get_risk_scores(all_explanations)

groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b", temperature=0.2)

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

def risk_badge_class(score):
    if score >= 0.6:
        return "risk-high", "🔴 HIGH RISK"
    elif score >= 0.3:
        return "risk-moderate", "🟡 MODERATE RISK"
    else:
        return "risk-low", "🟢 LOW RISK"

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")
    total = len(all_explanations)
    high_risk_count = sum(1 for s in risk_scores.values() if s >= 0.6)
    moderate_count = sum(1 for s in risk_scores.values() if 0.3 <= s < 0.6)
    low_count = total - high_risk_count - moderate_count

    st.metric("Total Students", total)
    st.metric("🔴 High Risk", high_risk_count)
    st.metric("🟡 Moderate Risk", moderate_count)
    st.metric("🟢 Low Risk", low_count)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("CARIO uses a Random Forest model with SHAP explainability and a RAG conversational layer to help counselors interpret dropout risk without needing to read raw model outputs.")

# ---- Main tabs ----
tab1, tab2 = st.tabs(["💬  Ask CARIO", "📋  Browse Student Profiles"])

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("Ask a question about student dropout risk:",
                                    placeholder="e.g. Which students need immediate intervention?",
                                    label_visibility="collapsed")
    with col2:
        high_risk_toggle = st.checkbox("High-risk only", value=True)

    ask_clicked = st.button("Ask CARIO", use_container_width=False)

    if ask_clicked and user_query:
        with st.spinner("Retrieving student data and generating grounded answer..."):
            answer, retrieved_ids = rag_query(user_query, filter_high_risk=high_risk_toggle)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        st.caption(f"📌 Based on students: {', '.join(map(str, retrieved_ids))}")
    elif ask_clicked:
        st.warning("Please enter a question first.")

with tab2:
    col1, col2 = st.columns([1, 3])
    with col1:
        student_id = st.number_input("Student ID (0–648):", min_value=0, max_value=648, step=1)
        view_clicked = st.button("View Profile")

    if view_clicked:
        text = all_explanations[str(student_id)]
        score = risk_scores.get(str(student_id), 0.0)
        badge_class, badge_label = risk_badge_class(score)

        with col2:
            st.markdown(f'<div class="{badge_class}"><b>{badge_label}</b> — Probability: {score:.2f}</div>', unsafe_allow_html=True)
            st.text(text)

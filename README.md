# CARIO — Counselor AI for Interpretable Risk Outcomes

**Live App:** https://ftfow8a2rappcqzjhqffifx.streamlit.app/

An end-to-end explainable AI system for student dropout risk assessment, combining ensemble machine learning, SHAP-based interpretability, and a Retrieval-Augmented Generation (RAG) conversational interface — built as part of an undergraduate research project under Dr. Abhishek Gaur, IGDTUW, targeting IEEE/ACM publication.

## Overview

Predictive dropout models are often accurate but opaque — counselors and advisors can't act on a raw probability score they don't understand. CARIO addresses this by pairing a validated ensemble ML pipeline with a natural-language explanation layer, letting non-technical staff query student risk profiles conversationally instead of reading raw model outputs.

## Architecture

**1. Ensemble ML Pipeline**
- Dataset: 649 student records, 34 attributes (UCI Student Performance dataset)
- Preprocessing: IQR-based outlier handling, label encoding, correlation filtering
- Class imbalance handled via SMOTE (applied to training data only)
- Models compared: Decision Tree, Random Forest, XGBoost — tuned via `RandomizedSearchCV` with stratified 5-fold cross-validation
- **Final model: Random Forest** — selected after comparing test-set generalization across models (96.9% test accuracy, 0.97 weighted F1), validated for absence of data leakage and duplicate-row contamination

**2. SHAP Explainability Layer**
- `SHAP TreeExplainer` applied to the trained Random Forest
- Local Shapley values computed for all 649 students
- Converted into structured, natural-language explanations (risk level, probability, top risk factors, protective factors)

**3. RAG Conversational System**
- Explanation texts embedded via `sentence-transformers` (all-MiniLM-L6-v2)
- Indexed in FAISS for semantic retrieval
- Risk-aware filtering ensures high-risk queries surface genuinely high-risk students
- LangChain-orchestrated prompt grounding, generated via LLaMA (Groq API)

**4. Streamlit Application**
- Conversational query interface ("Ask CARIO")
- Individual student profile browser
- Deployed publicly on Streamlit Cloud

## Tech Stack
Python · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · sentence-transformers · FAISS · LangChain · Groq (LLaMA) · Streamlit

## Status
Components 1–3 (ML pipeline, SHAP layer, RAG system) and deployment are complete. The planned controlled human-subject user study (N=15–20 counselors, comparing no-explanation vs. SHAP-only vs. RAG conditions) is the next phase of this research.

## Author
Shalvi Srivastava, B.Tech ECE, IGDTUW — Research Intern under Dr. Abhishek Gaur

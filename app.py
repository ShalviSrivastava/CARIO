def rag_query(user_query, k=5, filter_high_risk=False):
    if filter_high_risk:
        dropout_probs = best_model.predict_proba(
            np.array([list(all_explanations.keys())]).reshape(-1, 1)
        )  # placeholder, will fix below

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

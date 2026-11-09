import pandas as pd
import os
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POLICY_DIR = "data/policies"

# Load policy documents
documents = []

for filename in sorted(os.listdir(POLICY_DIR)):
    if filename.endswith(".txt"):
        path = os.path.join(POLICY_DIR, filename)

        with open(path, "r", encoding="utf-8") as file:
            documents.append({
                "source": filename,
                "text": file.read()
            })

print("POLICY DOCUMENTS LOADED:", len(documents))


# Split documents into chunks
def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


chunks = []

for document in documents:
    document_chunks = chunk_text(document["text"])

    for chunk in document_chunks:
        chunks.append({
            "source": document["source"],
            "text": chunk
        })


print("TOTAL POLICY CHUNKS:", len(chunks))


# Create TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words="english"
)

chunk_texts = [
    chunk["text"] for chunk in chunks
]

tfidf_matrix = vectorizer.fit_transform(chunk_texts)

print("TF-IDF MATRIX SHAPE:", tfidf_matrix.shape)


# Retrieve relevant policy chunks
def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    if scores.max() < 0.10:
        return []

    top_indices = np.argsort(scores)[::-1][:k]
    results = []

    for index in top_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })

    return results


from mock_llm import chat


def answer(question):
    results = retrieve(question, k=3)

    if not results:
        return {
            "answer": (
                "I could not find relevant information "
                "in the policy documents. Please contact "
                "human support."
            ),
            "sources": []
        }

    context = "\n".join(
        f"[{r['source']}] {r['text']}"
        for r in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. "
                "Answer using only the supplied policy context. "
                "Say you could not find the information otherwise. "
                "Do not invent policy rules."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": results
    }


def answer_policy(query):
    return answer(query)["answer"]


# RAG RETRIEVAL EVALUATION

def evaluate_retrieval(overlap=10):

    evaluation_chunks = []

    for document in documents:
        document_chunks = chunk_text(
            document["text"],
            chunk_size=40,
            overlap=overlap
        )

        for chunk in document_chunks:
            evaluation_chunks.append({
                "source": document["source"],
                "text": chunk
            })

    evaluation_vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    texts = [
        chunk["text"] for chunk in evaluation_chunks
    ]

    matrix = evaluation_vectorizer.fit_transform(texts)

    evaluation_data = pd.read_csv(
        "data/raw/policy_eval.csv"
    )

    document_hits = 0
    answer_fact_hits = 0
    fallback_hits = 0
    answerable_count = 0
    unanswerable_count = 0

    for _, row in evaluation_data.iterrows():

        query = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = str(row["expected_fact"])

        query_vector = evaluation_vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            matrix
        ).flatten()

        if scores.max() < 0.10:
            retrieved = []
        else:
            top_indices = np.argsort(scores)[::-1][:3]

            retrieved = [
                evaluation_chunks[i]
                for i in top_indices
            ]

        if not retrieved:
            generated_answer = (
                "I could not find relevant information "
                "in the policy documents. Please contact "
                "human support."
            )
        else:
            context = "\n".join(
                f"[{chunk['source']}] {chunk['text']}"
                for chunk in retrieved
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a POLICY ASSISTANT. "
                        "Answer using only the supplied policy context. "
                        "Say you could not find the information otherwise. "
                        "Do not invent policy rules."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n"
                        f"QUESTION:\n{query}"
                    )
                }
            ]

            response = chat(messages)
            generated_answer = response.choices[0].message.content

        if expected_doc == "NONE":
            unanswerable_count += 1

            if generated_answer.startswith("I could not find"):
                fallback_hits += 1

        else:
            answerable_count += 1

            retrieved_sources = [
                chunk["source"] for chunk in retrieved
            ]

            if expected_doc in retrieved_sources:
                document_hits += 1

            if expected_fact in generated_answer:
                answer_fact_hits += 1

    print("\nRETRIEVAL EVALUATION")
    print("Chunk overlap:", overlap)
    print("Total questions:", len(evaluation_data))
    print("Answerable questions:", answerable_count)
    print("Unanswerable questions:", unanswerable_count)

    print(
        "Expected document retrieved:",
        document_hits,
        "/",
        answerable_count
    )

    print(
        "Expected fact present in generated answer:",
        answer_fact_hits,
        "/",
        answerable_count
    )

    print(
        "Correct unanswerable fallbacks:",
        fallback_hits,
        "/",
        unanswerable_count
    )


print("\nEVALUATION WITH 10-WORD OVERLAP")
evaluate_retrieval(overlap=10)

print("\nEVALUATION WITH ZERO OVERLAP")
evaluate_retrieval(overlap=0)
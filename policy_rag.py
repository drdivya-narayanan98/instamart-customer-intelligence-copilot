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


def answer_policy(query):
    results = retrieve(query, k=3)

    if not results:
        return (
            "I could not find relevant information "
            "in the policy documents. Please contact "
            "human support."
        )

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
                "Do not invent policy rules."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{query}"
            )
        }
    ]

    response = chat(messages)

    return response.choices[0].message.content
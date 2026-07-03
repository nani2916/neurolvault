import os
import logging
import joblib
import faiss
import numpy as np
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "rag_system" / "vector_store"


def load_vector_store():
    """Load saved FAISS index and documents"""
    logger.info("Loading vector store...")
    index = faiss.read_index(str(VECTOR_STORE_DIR / "loan_index.faiss"))
    documents = joblib.load(VECTOR_STORE_DIR / "documents.pkl")
    encoder = joblib.load(VECTOR_STORE_DIR / "encoder.pkl")
    logger.info(f"Vector store loaded: {index.ntotal} documents")
    return index, documents, encoder


def search_relevant_docs(query, index, encoder, documents, top_k=3):
    """Search for relevant loan documents"""
    query_vector = encoder.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    results = [documents[idx]['text'] for idx in indices[0]]
    return results


def ask_neurolvault(question):
    """
    Main RAG function:
    1. Search relevant loan documents
    2. Send context + question to Groq LLM
    3. Return intelligent answer
    """
    logger.info(f"Question: {question}")

    # Load vector store
    index, documents, encoder = load_vector_store()

    # Step 1: Retrieve relevant documents
    relevant_docs = search_relevant_docs(
        question, index, encoder, documents
    )

    context = "\n\n".join(relevant_docs)

    # Step 2: Generate answer with Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""You are NeuralVault AI, an intelligent financial risk assistant 
for a bank. Use the following loan data context to answer the question.

CONTEXT FROM LOAN DATABASE:
{context}

QUESTION: {question}

Provide a clear, professional answer based on the context. 
If the context doesn't contain enough information, say so clearly."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are NeuralVault AI, a financial risk intelligence assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    logger.info("Answer generated successfully")
    return answer


if __name__ == "__main__":
    questions = [
        "Show me high risk customers with low credit scores",
        "What is the typical loan amount for defaulted loans?",
        "Which region has the most loan defaults?"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask_neurolvault(q)}")
        print("=" * 60)

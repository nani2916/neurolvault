import logging
import joblib
import faiss
import numpy as np
from pathlib import Path
from groq import Groq
import os
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
    """Load FAISS index, documents and encoder"""
    index = faiss.read_index(str(VECTOR_STORE_DIR / "loan_index.faiss"))
    documents = joblib.load(VECTOR_STORE_DIR / "documents.pkl")
    encoder = joblib.load(VECTOR_STORE_DIR / "encoder.pkl")
    return index, documents, encoder


def search_context(query, index, encoder, documents, top_k=4):
    """Search vector store for relevant context"""
    query_vector = encoder.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    context = "\n\n".join([documents[idx]['text'] for idx in indices[0]])
    return context


def ask_advisor(question):
    """
    Main advisor function.
    Banker types question → RAG finds context → Groq answers
    This is what runs inside Salesforce LWC chatbot
    """
    logger.info(f"Banker question: {question}")

    index, documents, encoder = load_vector_store()
    context = search_context(question, index, encoder, documents)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are NeuralVault AI Advisor, an intelligent 
assistant for bank loan officers. Answer questions clearly 
and professionally using the provided loan data context. 
Always base answers on the data provided."""
            },
            {
                "role": "user",
                "content": f"""Context from loan database:
{context}

Banker's question: {question}

Provide a clear, professional answer."""
            }
        ],
        temperature=0.3,
        max_tokens=400
    )

    answer = response.choices[0].message.content
    logger.info("Answer generated")
    return answer


if __name__ == "__main__":
    questions = [
        "Which customers from the south region are at risk of defaulting?",
        "What is the average credit score of customers who defaulted?",
        "Should I approve a loan for a customer with credit score 550 and income $3000?",
        "What factors most commonly lead to loan defaults in our portfolio?"
    ]

    print("\n" + "="*60)
    print("NEUROLVAULT AI ADVISOR — BANKER Q&A SESSION")
    print("="*60)

    for q in questions:
        print(f"\nBANKER: {q}")
        answer = ask_advisor(q)
        print(f"NEUROLVAULT AI: {answer}")
        print("-"*60) 

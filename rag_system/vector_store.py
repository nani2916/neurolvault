import logging
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "rag_system" / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)


def build_vector_store(documents):
    """
    Convert documents to vectors and store in FAISS.
    
    sentence-transformers converts text to 384-dimensional vectors
    FAISS indexes these vectors for fast similarity search
    """
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract text from documents
    texts = [doc['text'] for doc in documents]

    logger.info(f"Encoding {len(texts)} documents into vectors...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    logger.info(f"Embedding shape: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Save everything
    faiss.write_index(index, str(VECTOR_STORE_DIR / "loan_index.faiss"))
    joblib.dump(documents, VECTOR_STORE_DIR / "documents.pkl")
    joblib.dump(model, VECTOR_STORE_DIR / "encoder.pkl")

    logger.info("Vector store saved successfully")
    return index, model, documents


def search_vector_store(query, index, model, documents, top_k=3):
    """
    Search vector store for most relevant documents.
    Converts query to vector then finds nearest neighbors.
    """
    query_vector = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'document': documents[idx]['text'],
            'metadata': documents[idx]['metadata'],
            'distance': distances[0][i]
        })

    return results


if __name__ == "__main__":
    from document_loader import load_documents
    docs = load_documents()
    index, model, documents = build_vector_store(docs) 

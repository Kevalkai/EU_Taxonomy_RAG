# utils/faiss_manager.py

import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.npy"

def save_faiss_index(index, embeddings):
    """Save FAISS index and embeddings to disk."""
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)


def load_faiss_index():
    """Load FAISS index and embeddings from disk."""
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
            index = faiss.read_index(INDEX_FILE)
            embeddings = np.load(EMBEDDINGS_FILE)
            return index, embeddings
    except Exception as e:
        print(f"Error loading FAISS index or embeddings: {e}")
    return None, None

def build_faiss_index(documents, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Builds a FAISS index from document embeddings."""
    # Load the embedding model
    model = SentenceTransformer(embedding_model_name)

    # Compute embeddings
    embeddings = model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype='float32')  # FAISS needs float32

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    index.add(embeddings)

    # Save the index and embeddings
    save_faiss_index(index, embeddings)
    return index, embeddings, model

def query_faiss(index, query, model, documents, top_k=5):
    """Queries the FAISS index and returns the top-k documents."""
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype='float32')

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    return retrieved_docs

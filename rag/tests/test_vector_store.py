import os
from rag.vector_store import FAISSVectorStore


def test_vectorize_chunks():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunk_dir = os.path.join(base_dir, 'mini_corpus', 'chunks')
    vector_store_path = os.path.join(base_dir, 'mini_corpus', 'vector_store.faiss')

    # Create an instance of VectorStore
    vector_store = FAISSVectorStore(chunk_dir, vector_store_path)

    # Perform vectorization
    try:
        chunk_files = vector_store.vectorize_chunks()

        # Check if vector_store_path was created
        assert os.path.exists(vector_store_path), "Vector store file was not created."
        assert len(chunk_files) > 0, "No chunks were vectorized."

    finally:
        # Clean up: delete the vector store file
        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)

import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from rag.config import RAG_CONFIG  # Updated import path

class FAISSVectorStore:
    def __init__(self, chunk_dir, vector_store_path):
        self.chunk_dir = chunk_dir
        self.vector_store_path = vector_store_path
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=RAG_CONFIG["openai_api_key"])

    def vectorize_chunks(self):
        """
        Vectorize all text chunks and store them in the FAISS index.
        """
        # Initialize an empty list to hold all embeddings and the chunk-to-file map
        embeddings = []
        chunk_files = []

        # Iterate through all chunk files in the chunk directory
        for filename in os.listdir(self.chunk_dir):
            if filename.endswith('.txt'):
                chunk_path = os.path.join(self.chunk_dir, filename)
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                    # Convert the chunk text to a vector using OpenAI embeddings
                    chunk_vector = self.embeddings_model.embed_text(chunk_text)
                    embeddings.append(chunk_vector)
                    chunk_files.append(chunk_path)

        # Convert embeddings list to a numpy array
        embeddings_matrix = np.array(embeddings).astype('float32')

        # Build the FAISS index
        index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
        index.add(embeddings_matrix)

        # Save the index and chunk files map
        faiss.write_index(index, self.vector_store_path)
        return chunk_files

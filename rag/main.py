# rag/main.py
from fastapi import FastAPI, HTTPException
import os
from utils import extract_text_from_pdf
from chunker import VanillaChunker
from vector_store import FAISSVectorStore

app = FastAPI()


@app.get("/extract_text")
def extract_text(pdf_filename: str):
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, 'mini_corpus', pdf_filename)
    output_dir = os.path.join(base_dir, 'mini_corpus', 'extracted')

    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found.")

    # Extract text from the PDF
    try:
        text_file_path = extract_text_from_pdf(pdf_path, output_dir)
        return {"message": "Text extracted successfully.", "text_file": text_file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vanilla_chunk_text")
def chunk_text():
    # Path to the extracted text file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'mini_corpus', 'extracted', 'rehabilitation_guideline.txt')

    # Ensure the extracted text file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Extracted text file not found. Run /extract_text first.")

    # Perform chunking
    try:
        chunker = VanillaChunker(input_file)
        chunk_files = chunker.chunk()
        return {"message": "Text chunked successfully.", "chunk_files": chunk_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faiss_vectorize_chunks")
def vectorize_chunks():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chunk_dir = os.path.join(base_dir, 'mini_corpus', 'chunks')
    vector_store_path = os.path.join(base_dir, 'mini_corpus', 'vector_store.faiss')

    # Perform vectorization
    try:
        vector_store = FAISSVectorStore(chunk_dir, vector_store_path)
        chunk_files = vector_store.vectorize_chunks()
        return {"message": "Chunks vectorized successfully.", "chunk_files": chunk_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate_response")
def generate_response(query: str):
    # Placeholder for RAG logic
    response = f"Processed query: {query}"
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

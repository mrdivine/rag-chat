`./run_tests_and_build.sh`
```
#!/bin/bash

# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)

echo "Installing RAG module requirements..."
pip install -r rag/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install RAG module requirements."
    exit 1
fi

echo "Installing Chat module requirements..."
pip install -r chat/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install Chat module requirements."
    exit 1
fi

# Install pytest in case it's not included in either requirements
echo "Installing pytest..."
pip install pytest
if [ $? -ne 0 ]; then
    echo "Failed to install pytest."
    exit 1
fi

echo "Running tests for RAG module..."
python3 -m pytest rag/tests
if [ $? -ne 0 ]; then
    echo "RAG module tests failed. Fix them before building."
    exit 1
fi

echo "Running tests for Chat module..."
python3 -m pytest chat/tests
if [ $? -ne 0 ]; then
    echo "Chat module tests failed. Fix them before building."
    exit 1
fi

echo "All tests passed. Building the Docker containers..."
docker-compose up --build
```

`./chat/config.py`
```
# Placeholder for configuration settings
CHAT_CONFIG = {
    "api_key": "your-api-key"
}
```

`./chat/requirements.txt`
```
streamlit
openai
```

`./chat/Dockerfile`
```
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies for Chat
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Chat module into the container
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
```

`./chat/tests/test_chat_utils.py`
```
# Test placeholder for chat_utils.py
def test_chat_utils():
    # Placeholder test
    assert True, "Chat utils test is not implemented yet."
```

`./chat/tests/test_app.py`
```
# Test placeholder for app.py
def test_app():
    # Placeholder test
    assert True, "App test is not implemented yet."
```

`./chat/__init__.py`
```
[Skipped: Non-text or too large]
```

`./chat/app.py`
```
# chat/app.py
import streamlit as st
import requests


def main():
    st.title("RAG Chat Interface")

    user_input = st.text_input("Enter your query:")
    if st.button("Submit"):
        response = requests.get("http://rag-service:8000/generate_response", params={"query": user_input})
        if response.status_code == 200:
            result = response.json()
            st.write("RAG Response:", result["response"])
        else:
            st.write("Error in RAG service")


if __name__ == "__main__":
    main()
```

`./chat/chat_utils.py`
```
# Placeholder for chat utility functions
def process_input():
    print("Processing chat input...")
```

`./print_code_base.sh`
```
#!/bin/bash

# Check if code_base.md exists in the current directory and delete it
if [[ -f "code_base.md" ]]; then
    echo "Removing existing code_base.md..."
    rm "code_base.md"
fi

# Function to print file content in markdown format
print_file_content() {
    local file=$1
    # Only process text files smaller than 1MB
    if [[ $(file --mime-type -b "$file") == text/* ]] && [[ $(stat -f%z "$file") -lt 1048576 ]]; then
        {
            printf "\`%s\`\n" "$file"
            printf '```\n'
            cat "$file"
            printf '```\n\n'
        }
    else
        {
            printf "\`%s\`\n" "$file"
            printf '```\n[Skipped: Non-text or too large]\n```\n\n'
        }
    fi
}

# Main function to recursively print files, excluding specified patterns
print_code_base() {
    local directory=${1:-.}  # Default to current directory if no argument is given

    # Patterns to exclude (add more patterns to this array as needed)
    local exclude_patterns=("*/mini_corpus/*" "*/.*" "*/__pycache__/*" "*/README.md" "./code_base.md")

    # Construct the find command with all exclude patterns and avoid symlinks
    local find_command="find \"$directory\" -maxdepth 10 -type f"
    for pattern in "${exclude_patterns[@]}"; do
        find_command+=" ! -path \"$pattern\""
    done

    # Output file
    local output_file="code_base.md"

    # Execute the find command and loop through files
    eval $find_command | while IFS= read -r file; do
        # Check if the file actually exists
        if [[ -f "$file" ]]; then
            # Append content directly to the output file
            print_file_content "$file" >> "$output_file"
        fi
    done

    echo "Code base saved to $output_file."
}

# Execute the function with the first argument or without arguments
print_code_base "$1"
```

`./rag/config.py`
```
import os

RAG_CONFIG = {
    "retriever_model": "example-retriever-model",
    "generator_model": "example-generator-model",
    "openai_api_key": os.getenv("OPENAI_API_KEY")  # Retrieve API key from environment
}

if RAG_CONFIG["openai_api_key"] is None:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")
```

`./rag/requirements.txt`
```
langchain
openai
numpy
faiss-cpu
scikit-learn
fastapi
uvicorn
pdfplumber
langchain-community
langchain-openai
```

`./rag/chunker.py`
```
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class VanillaChunker:
    def __init__(self, input_file, chunk_size=1000, overlap=100):
        # Set the output directory to 'mini_corpus/chunks'
        self.input_file = input_file
        self.output_dir = os.path.join(os.path.dirname(input_file), '..', 'chunks')
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self):
        """
        Chunk the input text file into smaller pieces based on a fixed number of characters.
        Returns a list of file paths to the chunk files.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Read the entire text
        with open(self.input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Use LangChain's text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.overlap
        )
        chunks = text_splitter.split_text(text)

        # Write each chunk to a separate file
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file_path = os.path.join(self.output_dir, f'chunk_{i}.txt')
            with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
            chunk_files.append(chunk_file_path)

        return chunk_files
```

`./rag/vector_store.py`
```
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
```

`./rag/Dockerfile`
```
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies for RAG
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RAG module into the container
COPY . .

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Run the main RAG script
CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`./rag/retriever.py`
```
# Placeholder for the retriever component
def retrieve():
    print("Retrieving documents...")```

`./rag/tests/test_utils.py`
```
import os
from rag.utils import extract_text_from_pdf


def test_extract_text_from_pdf():
    # Dynamically determine the correct path to the mini_corpus directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels
    pdf_path = os.path.join(base_dir, 'mini_corpus', 'RehabGuidelines_en_CP_OT.pdf')
    output_dir = os.path.join(base_dir, 'tests', 'util_test_output')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract text from the PDF
    text_file_path = extract_text_from_pdf(pdf_path, output_dir)

    try:
        # Check if the text file was created
        assert os.path.exists(text_file_path), "Text file was not created."

        # Check if the text file has content
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0, "Extracted text is empty."

    finally:
        # Clean up: delete the generated text file and output directory
        if os.path.exists(text_file_path):
            os.remove(text_file_path)
        # Remove the directory if it's empty
        if os.path.isdir(output_dir):
            os.rmdir(output_dir)
```

`./rag/tests/test_chunker.py`
```
import os
from rag.chunker import VanillaChunker


def test_vanilla_chunker():
    # Path to the extracted text file and output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'mini_corpus', 'extracted', 'rehabilitation_guideline.txt')
    output_dir = os.path.join(base_dir, 'mini_corpus', 'chunks')

    # Create dummy extracted text if it doesn't exist
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("This is some test content. " * 100)  # Dummy content

    # Create an instance of VanillaChunker
    chunker = VanillaChunker(input_file, chunk_size=100, overlap=20)

    # Perform chunking
    chunk_files = chunker.chunk()

    try:
        # Check if the chunks are created
        assert len(chunk_files) > 0, "No chunks were created."

        # Check the contents of the first chunk
        with open(chunk_files[0], 'r', encoding='utf-8') as f:
            chunk_content = f.read()
            assert len(chunk_content) > 0, "First chunk is empty."

    finally:
        # Clean up: delete the generated chunk files and output directory
        for chunk_path in chunk_files:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        # Remove the directory if it's empty
        if os.path.isdir(output_dir):
            os.rmdir(output_dir)
```

`./rag/tests/test_generator.py`
```
# Test placeholder for generator.py
def test_generator():
    # Placeholder test
    assert True, "Generator test is not implemented yet."
```

`./rag/tests/test_main.py`
```
# Test placeholder for main.py
def test_main():
    # Placeholder test
    assert True, "Main function test is not implemented yet."
```

`./rag/tests/test_vector_store.py`
```
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
```

`./rag/tests/test_retriever.py`
```
# Test placeholder for retriever.py
def test_retrieve():
    # Placeholder test
    assert True, "Retriever test is not implemented yet."
```

`./rag/__init__.py`
```
[Skipped: Non-text or too large]
```

`./rag/generator.py`
```
# Placeholder for the generator component
def generate():
    print("Generating response...")```

`./rag/utils.py`
```
import pdfplumber
import os


def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a PDF and saves it as a text file in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'

    # Save the extracted text to a file
    text_file_path = os.path.join(output_dir, 'rehabilitation_guideline.txt')
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return text_file_path

# Example usage:
# extract_text_from_pdf('mini_corpus/Rehabilitation_Guideline.pdf', 'mini_corpus')```

`./rag/main.py`
```
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
```

`./main_objective.md`
```
Main Objective:

To build a Retrieval-Augmented Generation (RAG) system that can answer questions based on the “Rehabilitation Guideline for the Management of Children with Cerebral Palsy” PDF, using LangChain and OpenAI’s GPT-4 API. The RAG system should consist of:

	1.	Extracting the text from the PDF.
	2.	Chunking the extracted text into manageable pieces.
	3.	Vectorizing these chunks for efficient retrieval.
	4.	Implementing a retrieval mechanism.
	5.	Generating responses using a language model.
	6.	Integrating the system into a simple API with monitoring and testing.

Open List of Things to Do:

	1.	Vectorization of Chunks:
	•	Implement the logic in vector_store.py to vectorize the text chunks using FAISS.
	•	Store the vectors in a vector store for retrieval.
	2.	Implement the Retriever:
	•	Use the vectorized chunks to implement a retriever in retriever.py.
	•	Ensure the retriever can pull relevant chunks for a given query based on cosine similarity or other distance metrics.
	3.	Integrate OpenAI GPT-4 API:
	•	Use the OpenAI GPT-4 API to generate responses using the retrieved chunks.
	•	Modify generator.py to interact with the API and incorporate the context provided by the retriever.
	4.	Testing of Each Component:
	•	Write unit tests for vector_store.py to ensure vectorization is accurate and efficient.
	•	Test retriever.py to verify that the most relevant chunks are retrieved for sample queries.
	•	Write integration tests to confirm that the RAG system works end-to-end.
	5.	API Enhancements and Integration:
	•	Extend main.py to integrate vectorization, retrieval, and generation into the API.
	•	Add endpoints for querying the RAG system and generating responses.
	•	Implement logging and monitoring to capture usage metrics and potential issues.
	6.	Monitoring and Testing in Production:
	•	Set up basic logging to monitor queries, retrieval results, and generated responses.
	•	Implement anomaly detection for inputs and outputs to identify outliers or retrieval failures.
	•	Deploy the system in a manner that allows for ongoing maintenance and updates.
	7.	Documentation and Cleanup:
	•	Document each part of the system, including how to run, test, and maintain the RAG system.
	•	Clean up the codebase, remove any unnecessary files, and ensure directory structures are optimized for production.

Optional Enhancements (Time Permitting):

	•	Implement more sophisticated chunking strategies based on content structure.
	•	Enhance the retriever using more advanced methods like dense passage retrieval (DPR).
	•	Improve response generation by adding fine-tuning or prompt engineering to tailor responses more precisely to user queries.```

`./docker-compose.yml`
```
services:
  rag:
    build: ./rag
    container_name: rag-service
    command: python main.py
    ports:
      - "8000:8000"
    volumes:
      - ./rag/mini_corpus:/app/mini_corpus  # Maps local directory to container
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # Pass the API key to the container
  chat:
    build: ./chat
    container_name: chat-service
    command: streamlit run app.py
    volumes:
      - ./chat:/chat-app
    ports:
      - "8501:8501"  # Streamlit default port
```


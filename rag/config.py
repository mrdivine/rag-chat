import os

RAG_CONFIG = {
    "retriever_model": "example-retriever-model",
    "generator_model": "example-generator-model",
    "openai_api_key": os.getenv("OPENAI_API_KEY")  # Retrieve API key from environment
}

if RAG_CONFIG["openai_api_key"] is None:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

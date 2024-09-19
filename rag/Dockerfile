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

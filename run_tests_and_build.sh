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

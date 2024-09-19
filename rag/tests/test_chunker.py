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

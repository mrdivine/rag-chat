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

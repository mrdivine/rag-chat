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

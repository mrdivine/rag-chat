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
# extract_text_from_pdf('mini_corpus/Rehabilitation_Guideline.pdf', 'mini_corpus')
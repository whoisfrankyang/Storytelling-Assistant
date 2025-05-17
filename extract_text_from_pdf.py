import PyPDF2
import os
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def process_pdf_folder(folder_path: str, output_folder: str = "data") -> List[str]:
    """
    Process all PDFs in a folder and convert them to text files.
    Returns list of created text file paths.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    created_files = []
    
    # Process each PDF file
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(folder_path, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_path)
        
        if text_content:
            # Write to text file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            created_files.append(txt_path)
            print(f"✓ Converted {filename} to {txt_filename}")
        else:
            print(f"✗ Failed to convert {filename}")
    
    return created_files

if __name__ == "__main__":
    # Example usage
    pdf_folder = "arxiv_papers"  # folder containing PDFs
    output_folder = "data"  # folder for output text files
    process_pdf_folder(pdf_folder, output_folder)
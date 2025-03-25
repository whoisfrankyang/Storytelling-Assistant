import os
import re
from typing import Dict, Optional, Tuple

def extract_front_matter(content: str) -> Tuple[str, str]:
    """
    Extract the front matter (everything before abstract) and the abstract from the text.
    Returns a tuple of (front_matter, abstract)
    """
    # Common patterns for abstract headers
    abstract_patterns = [
        r'\nAbstract[\n\s]*',
        r'\nABSTRACT[\n\s]*',
        r'\n[0-9]+\.?\s*Abstract[\n\s]*',
        r'\nAbstract[:.]\s*',
        r'\n[0-9]+\.?\s*ABSTRACT[\n\s]*'
    ]
    
    # Try to find the abstract section
    abstract_start = None
    abstract_pattern_used = None
    
    for pattern in abstract_patterns:
        match = re.search(pattern, content)
        if match:
            abstract_start = match.start()
            abstract_pattern_used = pattern
            break
    
    if abstract_start is None:
        return content, ""  # Return entire content if no abstract found
    
    # Extract front matter (everything before abstract)
    front_matter = content[:abstract_start].strip()
    
    # Extract abstract
    abstract_text = ""
    if abstract_pattern_used:
        # Get the text after the abstract header
        post_abstract = content[abstract_start:]
        abstract_match = re.split(abstract_pattern_used, post_abstract, maxsplit=1)
        if len(abstract_match) > 1:
            # Try to find where abstract ends (next section or reasonable length)
            abstract_text = abstract_match[1]
            
            # Look for common section headers that might come after abstract
            section_patterns = [
                r'\n[0-9]+\.?\s*Introduction\s*\n',
                r'\nIntroduction\s*\n',
                r'\n[0-9]+\.?\s*Background\s*\n',
                r'\n[0-9]+\.?\s*Related Work\s*\n',
                r'\nKeywords[:.]',
                r'\n[0-9]+\.\s' 
                r'\n[0-9]+\.\s'
            ]
            
            # Find the earliest occurrence of any section pattern
            min_pos = len(abstract_text)
            for pattern in section_patterns:
                match = re.search(pattern, abstract_text)
                if match and match.start() < min_pos:
                    min_pos = match.start()
            
            abstract_text = abstract_text[:min_pos].strip()
    
    return front_matter, abstract_text

def process_document(file_path: str) -> Dict[str, str]:
    """Process a single document and extract its front matter and abstract."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        front_matter, abstract = extract_front_matter(content)
        
        return {
            "filename": os.path.basename(file_path),
            "front_matter": front_matter,
            "abstract": abstract
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_document_folder(input_folder: str, output_folder: str = "front_matter") -> None:
    """
    Process all text documents in a folder and save their front matter to separate files.
    Also creates a summary JSON file with all extracted data.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    all_documents = []
    
    for filename in os.listdir(input_folder):
        if not filename.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_folder, filename)
        result = process_document(input_path)
        
        if result:
            # Create individual front matter file
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}_front_matter.txt")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== FRONT MATTER ===\n\n")
                f.write(result["front_matter"])
                f.write("\n\n=== ABSTRACT ===\n\n")
                f.write(result["abstract"])
            
            all_documents.append(result)
            print(f"✓ Extracted front matter from {filename}")
        else:
            print(f"✗ Failed to process {filename}")
    
    # Save summary to JSON
    import json
    summary_path = os.path.join(output_folder, "front_matter_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(all_documents)} documents. Results saved to {output_folder}/")

if __name__ == "__main__":
    # Example usage
    input_folder = "datas/documents"  # folder containing text files
    output_folder = "datas/front_matter"  # folder for output files
    process_document_folder(input_folder, output_folder) 
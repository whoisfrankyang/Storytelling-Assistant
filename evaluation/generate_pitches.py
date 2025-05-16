import os
from word_embedding import build_vector_database
from ragcot import RAGSystem

def ensure_directories():
    """Create necessary directories for output files."""
    modes = ["general", "investor", "conference"]
    for mode in modes:
        os.makedirs(f"generated_pitch/{mode}", exist_ok=True)

def read_evaluation_samples(folder_path: str = "evaluation/benchmark_set"):
    """Read all text files from the evaluation folder."""
    samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                samples.append((filename, content))
    return samples

def generate_and_save_pitches():
    """Generate different types of pitches for each sample and save to files."""
    # Initialize RAG system
    rag = RAGSystem()
    
    # Read evaluation samples
    samples = read_evaluation_samples()
    
    # Generate pitches for each sample
    for filename, content in samples:
        base_name = os.path.splitext(filename)[0]
        
        # Generate and save general version
        general_version = rag.generate_storytelling_output(
            user_abstract=content,
            mode="general",
            k=3
        )
        with open(f"generated_pitch/general/{base_name}_general.txt", 'w', encoding='utf-8') as f:
            f.write(general_version)
        
        # Generate and save investor version
        investor_version = rag.generate_storytelling_output(
            user_abstract=content,
            mode="investor",
            k=3
        )
        with open(f"generated_pitch/investor/{base_name}_investor.txt", 'w', encoding='utf-8') as f:
            f.write(investor_version)
        
        # Generate and save conference version
        conference_version = rag.generate_storytelling_output(
            user_abstract=content,
            mode="conference",
            k=3
        )
        with open(f"generated_pitch/conference/{base_name}_conference.txt", 'w', encoding='utf-8') as f:
            f.write(conference_version)
        
        print(f"✓ Generated pitches for {filename}")

def main():
    # Create necessary directories
    ensure_directories()
    
    # Generate and save pitches
    print("Generating pitches for evaluation samples...")
    generate_and_save_pitches()
    print("\nAll pitches have been generated and saved!")

if __name__ == "__main__":
    main() 
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ensure_directories():
    """Create necessary directories for output files."""
    modes = ["general", "investor", "conference"]
    for mode in modes:
        os.makedirs(f"base_pitch/{mode}", exist_ok=True)

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

def get_base_prompt(mode: str, abstract: str) -> str:
    """Generate the base prompt for each mode."""
    prompts = {
        "general": f"""Rewrite this academic abstract in simple, non-technical language that anyone can understand. 
        Focus on the main idea and why it matters to everyday people.
        The final generated abstract should be within 100 words.
        
        Abstract: {abstract}""",
        
        "investor": f"""Rewrite this academic abstract as a compelling pitch for potential investors. 
        Focus on the market potential, innovation, and business value.
        The final generated abstract should be within 100 words.
        Abstract: {abstract}""",
        
        "conference": f"""Rewrite this academic abstract in a more formal, technical style suitable for a research conference. 
        Emphasize the scientific contributions and technical details.
        The final generated abstract should be within 100 words.
        Abstract: {abstract}"""
    }
    return prompts[mode]

def generate_base_pitch(abstract: str, mode: str) -> str:
    """Generate a pitch using the base GPT model."""
    prompt = get_base_prompt(mode, abstract)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites academic text for different audiences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip()

def generate_and_save_base_pitches():
    """Generate base pitches for each sample and save to files."""
    # Read evaluation samples
    samples = read_evaluation_samples()
    
    # Generate pitches for each sample
    for filename, content in samples:
        base_name = os.path.splitext(filename)[0]
        
        # Generate and save each version
        for mode in ["general", "investor", "conference"]:
            pitch = generate_base_pitch(content, mode)
            output_path = f"base_pitch/{mode}/{base_name}_{mode}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pitch)
            
            print(f"✓ Generated {mode} pitch for {filename}")

def main():
    # Create necessary directories
    ensure_directories()
    
    # Generate and save pitches
    print("Generating base pitches for evaluation samples...")
    generate_and_save_base_pitches()
    print("\nAll base pitches have been generated and saved!")

if __name__ == "__main__":
    main() 
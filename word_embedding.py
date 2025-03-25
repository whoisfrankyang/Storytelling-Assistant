import os
import json
import numpy as np
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VEC_PATH = "db/vectors.npy"
DOC_PATH = "db/documents.json"

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")  # encoding for text-embedding-ada-002
    return len(encoding.encode(text))

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def build_vector_database(folder_path: str, output_dir: str, max_tokens: int = 2000):
    os.makedirs(output_dir, exist_ok=True)

    vectors = []
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check token length
        num_tokens = count_tokens(content)
        if num_tokens > max_tokens:
            print(f"Skipping {filename}: {num_tokens} tokens (exceeds {max_tokens} limit)")
            continue

        # Skip if already saved (based on content hash)
        doc_entry = {"filename": filename, "content": content}
        if doc_entry in documents:
            print(f"Skipping cached: {filename}")
            continue

        print(f"Embedding: {filename} ({num_tokens} tokens)")
        embedding = get_embedding(content)

        vectors.append(embedding)
        documents.append(doc_entry)

    np.save(os.path.join(output_dir, "vectors.npy"), np.array(vectors))
    with open(os.path.join(output_dir, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print("Vector DB saved.")


if __name__ == "__main__":
    build_vector_database(folder_path="datas/front_matter", output_dir="datas/db")

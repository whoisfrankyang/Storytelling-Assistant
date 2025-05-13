from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants for vector database paths
VEC_PATH = "datas/db/vectors.npy"
DOC_PATH = "datas/db/documents.json"

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for input text."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# ========== RAG Storytelling System ==========
class RAGSystem:
    def __init__(self, vec_path: str = VEC_PATH, doc_path: str = DOC_PATH):
        """Initialize RAG system with pre-computed embeddings."""
        self.documents = []          # Raw text documents
        self.embeddings = []         # Corresponding embeddings
        
        # Load pre-computed embeddings if they exist
        if os.path.exists(vec_path) and os.path.exists(doc_path):
            self.load_vector_database(vec_path, doc_path)
    
    def load_vector_database(self, vec_path: str, doc_path: str):
        """Load pre-computed embeddings and documents from vector database."""
        # Load embeddings
        self.embeddings = np.load(vec_path).tolist()
        
        # Load documents
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
            self.documents = [doc["content"] for doc in doc_data]
    
    def add_document(self, text: str):
        """Add a new document to the knowledge base (with embedding computation)."""
        embedding = get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def load_documents_from_folder(self, folder_path: str):
        """Load all .txt files from a folder into the knowledge base.
        Note: This should primarily be used for adding new documents.
        For bulk loading, use word_embedding.py to build the vector database first."""
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    self.add_document(file.read())
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[str]:
        """Return top-k most similar documents to the query."""
        query_embedding = get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

    def extract_main_points(self, document: str) -> str:
        """Extract the main points from a document using GPT."""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract the key contribution or main point from the given text in one concise sentence."},
                {"role": "user", "content": document}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def format_context(self, relevant_docs: List[str]) -> str:
        """Format the context by extracting and summarizing main points."""
        # Extract main points from each document
        main_points = []
        for doc in relevant_docs:
            point = self.extract_main_points(doc)
            if point and not any(p.lower() == point.lower() for p in main_points):  # Avoid duplicates
                main_points.append(point)
        
        # Format the context summary
        if main_points:
            context_summary = "Here are some recent trends and related work in the field:\n\n"
            context_summary += "\n".join(f"• {point}" for point in main_points)
        else:
            context_summary = "No directly relevant prior work found in the database."
        
        return context_summary

    def create_prompt(self, context: str, user_abstract: str, mode: str = "general") -> str:
        """Construct a goal-specific prompt."""
        if mode == "investor":
            instruction = (
                # """You are a professional pitch writer helping technical founders tell compelling stories to investors.
                # Structure the pitch as:
                # 1. Problem & Market Opportunity: What's the pain point and market size?
                # 2. Solution & Innovation: What makes your approach unique?
                # 3. Technical Edge: Why is your solution hard to replicate?
                # 4. Impact & Business Potential: What's the broader impact and go-to-market strategy?
                
                # Keep it concise, engaging, and focused on value proposition."""
                """You are a professional pitch writer helping technical founders tell compelling stories to investors.
                Let's work through the structure together, making sure to address the following. 
                1. First, outline the problem and opportunity within the market. Make sure to answer questions about why the problem exists, what the problem means for people who have it, how many people have the problem, and how badly they are affected by it.
                2. Next, explain your solution and why it's innovative. How are you solving the problem and what about that solution is novel? What makes your solution better than competing solutions?
                3. Then, outline what gives your solution a technical edge. What makes your solution difficult to copy? How will you remain ahead of the game?
                4. Finally, outline the impact and the business potential. Call back to the problem and its size and impact, as well as explaining how your solution could be monetized.
                Once all of these points are considered, combine the 4 steps together clearly  and concisely in a project pitch."""
            )
        elif mode == "conference":
            instruction = (
                # """Rewrite the project description into a strong academic abstract that would appeal to top-tier ML/robotics conferences.
                # Structure it as:
                # 1. Problem Statement & Motivation
                # 2. Technical Challenges & Current Limitations
                # 3. Key Innovation & Methodology
                # 4. Results & Impact
                
                # Emphasize technical novelty and scientific contributions."""
                """Rewrite the project description into a strong academic abstract that would appeal to top-tier ML/robotics conferences.
                In order to do this, let's make sure to address the following:
                1. First, what's the problem or question you are trying to solve? Why are you trying to solve it? Why does it need to be solved more generally? How do you know that it hasn't been solved already?
                2. Next, How is your work innovative? What was your methodology when trying to solve this problem or solution? What makes it a unique new addition to the field?
                3. Then, what were the results of your study? How do they affect the current  state of the field you're  working in? What conclusions can be drawn?
                4. Finally, consider limitations and possible future work. What were some technical challenges you faced? What were the constraints you had to work with? How would you improve the study if you were to do it again in the future? What other questions did the results raise that you might explore further in the future?
                Once all these points have been  considered, combine them all together in a clear, concise, and succinct abstract, 1 or 2 paragraphs long."""
            )
        else:  # general
            instruction = (
                # """Rewrite the project description to make it engaging and accessible to a broad technical audience.
                # Focus on:
                # 1. The core problem and why it matters
                # 2. The key innovation in simple terms
                # 3. The potential impact
                
                # Use clear language while maintaining technical accuracy."""
                """Rewrite this project description to make it more engaging and accessible to a broader audience.
                For structure, address these points:
                1. First, What is the core problem that this project address. How many people are affected by this problem? How deeply does it affect them?
                2. Next, explain the key innovation that the project is centered around. How does this address the problem? What sets it apart from other potential solutions?
                3. Finally, explain the impact of the solution. How much of the original problem would be solved? For how many people and to what degree would it be solved?
                Once all these points have been considered, combine them all together in a clear and concise manner, with the final result being 1 or 2 paragraphs."""
            )
        
        return f"""Background Context:
                ---------------------
                {context}
                ---------------------

                Project Description:
                {user_abstract}

                Task:
                {instruction}
                """

    def generate_storytelling_output(self, user_abstract: str, mode: str = "general", k: int = 5) -> str:
        """Main function to create a storytelling-style revision of the input abstract."""
        # Get relevant documents and format context
        relevant_docs = self.retrieve_relevant_docs(user_abstract, k)
        formatted_context = self.format_context(relevant_docs)
        
        # Generate the enhanced version
        prompt = self.create_prompt(formatted_context, user_abstract, mode)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a storytelling assistant that enhances technical abstracts for specific audiences while maintaining technical accuracy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )

        return response.choices[0].message.content

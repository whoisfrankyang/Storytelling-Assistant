import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
import re

# Download required NLTK data
required_nltk_data = ['punkt', 'punkt_tab']
for data in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}')
    except LookupError:
        print(f"Downloading {data}...")
        nltk.download(data)

class PitchEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary components."""
        self.metrics = {}
    
    def calculate_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate various readability scores."""
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "linsear_write_formula": textstat.linsear_write_formula(text),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text)
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str, get_embedding_func) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        emb1 = get_embedding_func(text1)
        emb2 = get_embedding_func(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def calculate_text_statistics(self, text: str) -> Dict[str, int]:
        """Calculate basic text statistics."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / len(words) if words else 0
        }
    
    def calculate_repetition_metrics(self, text: str) -> Dict[str, float]:
        """Calculate metrics related to repetition and diversity."""
        words = word_tokenize(text.lower())
        word_freq = Counter(words)
        
        # Calculate repetition metrics
        total_words = len(words)
        unique_words = len(word_freq)
        
        # Calculate word repetition rate
        repetition_rate = sum(count - 1 for count in word_freq.values()) / total_words if total_words > 0 else 0
        
        # Calculate most common word frequency
        most_common_freq = word_freq.most_common(1)[0][1] / total_words if total_words > 0 else 0
        
        return {
            "repetition_rate": repetition_rate,
            "most_common_word_frequency": most_common_freq,
            "unique_word_ratio": unique_words / total_words if total_words > 0 else 0
        }
    
    def evaluate_pitch(self, original_text: str, generated_pitch: str, get_embedding_func) -> Dict:
        """Evaluate a generated pitch against the original text."""
        # Calculate all metrics
        readability = self.calculate_readability_scores(generated_pitch)
        similarity = self.calculate_semantic_similarity(original_text, generated_pitch, get_embedding_func)
        text_stats = self.calculate_text_statistics(generated_pitch)
        repetition = self.calculate_repetition_metrics(generated_pitch)
        
        # Combine all metrics
        evaluation = {
            "readability": readability,
            "semantic_similarity": float(similarity),  # Convert numpy float to Python float
            "text_statistics": text_stats,
            "repetition_metrics": repetition
        }
        
        return evaluation
    
    def evaluate_pitches_batch(self, original_texts: List[str], generated_pitches: List[str], 
                             get_embedding_func) -> Dict:
        """Evaluate multiple pitches and calculate aggregate statistics."""
        evaluations = []
        
        for orig, pitch in zip(original_texts, generated_pitches):
            eval_result = self.evaluate_pitch(orig, pitch, get_embedding_func)
            evaluations.append(eval_result)
        
        # Calculate aggregate statistics
        aggregate = {
            "readability": {
                metric: np.mean([e["readability"][metric] for e in evaluations])
                for metric in evaluations[0]["readability"].keys()
            },
            "semantic_similarity": np.mean([e["semantic_similarity"] for e in evaluations]),
            "text_statistics": {
                metric: np.mean([e["text_statistics"][metric] for e in evaluations])
                for metric in evaluations[0]["text_statistics"].keys()
            },
            "repetition_metrics": {
                metric: np.mean([e["repetition_metrics"][metric] for e in evaluations])
                for metric in evaluations[0]["repetition_metrics"].keys()
            }
        }
        
        return {
            "individual_evaluations": evaluations,
            "aggregate_statistics": aggregate
        }

def print_evaluation_results(evaluation: Dict):
    """Print evaluation results in a readable format."""
    print("\n=== Evaluation Results ===")
    
    print("\nReadability Scores:")
    for metric, score in evaluation["readability"].items():
        print(f"{metric.replace('_', ' ').title()}: {score:.2f}")
    
    print(f"\nSemantic Similarity: {evaluation['semantic_similarity']:.3f}")
    
    print("\nText Statistics:")
    for metric, value in evaluation["text_statistics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
    
    print("\nRepetition Metrics:")
    for metric, value in evaluation["repetition_metrics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

def main():
    from ragcot import get_embedding
    
    evaluator = PitchEvaluator()
    
    # Define the modes and their corresponding paths
    modes = {
        'base_pitch_conference': 'evaluation/base_pitch/conference',
        'base_pitch_general': 'evaluation/base_pitch/general',
        'base_pitch_investor': 'evaluation/base_pitch/investor',
        'generated_pitch_conference': 'evaluation/generated_pitch/conference',
        'generated_pitch_general': 'evaluation/generated_pitch/general',
        'generated_pitch_investor': 'evaluation/generated_pitch/investor'
    }
    
    # Dictionary to store results for each mode
    mode_results = {}
    
    # Process each mode
    for mode_name, pitch_dir in modes.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {mode_name}")
        print(f"{'='*50}")
        
        # Get all text files in the pitch directory
        pitch_files = [f for f in os.listdir(pitch_dir) if f.endswith('.txt')]
        
        if not pitch_files:
            print(f"No files found in {pitch_dir}")
            continue
        
        # List to store evaluations for this mode
        mode_evaluations = []
        
        # Process each file
        for pitch_file in pitch_files:
            # Get corresponding original text file
            original_file = pitch_file.replace('_general.txt', '.txt').replace('_conference.txt', '.txt').replace('_investor.txt', '.txt')
            original_path = os.path.join('evaluation/benchmark_set', original_file)
            pitch_path = os.path.join(pitch_dir, pitch_file)
            
            try:
                # Read the files
                with open(original_path, 'r') as f:
                    original_text = f.read()
                with open(pitch_path, 'r') as f:
                    generated_pitch = f.read()
                
                # Evaluate the pitch
                evaluation = evaluator.evaluate_pitch(original_text, generated_pitch, get_embedding)
                mode_evaluations.append(evaluation)
                
                print(f"\nProcessed: {pitch_file}")
                
            except FileNotFoundError as e:
                print(f"Error processing {pitch_file}: {str(e)}")
                continue
        
        if mode_evaluations:
            # Calculate average metrics for this mode
            avg_evaluation = {
                "readability": {
                    metric: np.mean([e["readability"][metric] for e in mode_evaluations])
                    for metric in mode_evaluations[0]["readability"].keys()
                },
                "semantic_similarity": np.mean([e["semantic_similarity"] for e in mode_evaluations]),
                "text_statistics": {
                    metric: np.mean([e["text_statistics"][metric] for e in mode_evaluations])
                    for metric in mode_evaluations[0]["text_statistics"].keys()
                },
                "repetition_metrics": {
                    metric: np.mean([e["repetition_metrics"][metric] for e in mode_evaluations])
                    for metric in mode_evaluations[0]["repetition_metrics"].keys()
                }
            }
            
            mode_results[mode_name] = avg_evaluation
            
            # Print results for this mode
            print(f"\nAverage metrics for {mode_name}:")
            print_evaluation_results(avg_evaluation)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON OF ALL MODES")
    print("="*80)
    
    # Print readability scores comparison
    print("\nReadability Scores:")
    print(f"{'Mode':<30} {'Flesch Reading Ease':>20} {'Flesch-Kincaid Grade':>20}")
    print("-" * 70)
    for mode, results in mode_results.items():
        print(f"{mode:<30} {results['readability']['flesch_reading_ease']:>20.2f} {results['readability']['flesch_kincaid_grade']:>20.2f}")
    
    # Print semantic similarity comparison
    print("\nSemantic Similarity:")
    print(f"{'Mode':<30} {'Similarity Score':>20}")
    print("-" * 50)
    for mode, results in mode_results.items():
        print(f"{mode:<30} {results['semantic_similarity']:>20.3f}")
    
    # Print text statistics comparison
    print("\nText Statistics:")
    print(f"{'Mode':<30} {'Word Count':>15} {'Sentence Count':>15} {'Lexical Diversity':>15}")
    print("-" * 75)
    for mode, results in mode_results.items():
        print(f"{mode:<30} {results['text_statistics']['word_count']:>15.1f} "
              f"{results['text_statistics']['sentence_count']:>15.1f} "
              f"{results['text_statistics']['lexical_diversity']:>15.3f}")

if __name__ == "__main__":
    main()

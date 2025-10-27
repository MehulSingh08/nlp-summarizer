# backend/model.py
# ===========================================
# Text Summarization Models
# - Extractive: TextRank Algorithm (Graph-based)
# - Abstractive: T5 Model (Transformer-based)
# ===========================================

import os
import re
import pickle
import logging
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deep Learning libraries for T5
try:
    import torch
    from transformers import (
        T5Tokenizer, 
        T5ForConditionalGeneration,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    from datasets import Dataset
    import pandas as pd
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("   Warning: transformers library not installed. Abstractive summarization will not be available.")
    print("   Install with: pip install transformers torch datasets")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# ============================================
# UTILITY FUNCTIONS
# ============================================

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by tokenizing into sentences and cleaning.
    
    Args:
        text: Input text string
        
    Returns:
        List of cleaned sentences
    """
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            # Remove extra whitespace
            sent = ' '.join(sent.split())
            
            # Keep sentences with at least 3 words
            if len(sent.split()) >= 3:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return []


def get_original_sentences(text: str) -> List[str]:
    """Get original sentences without cleaning (for display)."""
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error getting original sentences: {str(e)}")
        return [text]


# ============================================
# EXTRACTIVE SUMMARIZATION (TextRank)
# ============================================

def calculate_sentence_similarity(sent1: str, sent2: str, stop_words: set) -> float:
    """
    Calculate cosine similarity between two sentences.
    
    Args:
        sent1: First sentence
        sent2: Second sentence
        stop_words: Set of stopwords to filter
        
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize and filter stopwords
    words1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stop_words and w.isalnum()]
    words2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stop_words and w.isalnum()]
    
    # Get unique words
    all_words = list(set(words1 + words2))
    
    if not all_words:
        return 0.0
    
    # Create word vectors
    vector1 = [1 if w in words1 else 0 for w in all_words]
    vector2 = [1 if w in words2 else 0 for w in all_words]
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sum(a * a for a in vector1) ** 0.5
    magnitude2 = sum(b * b for b in vector2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def build_similarity_matrix(sentences: List[str], stop_words: set) -> np.ndarray:
    """
    Build similarity matrix for sentences.
    
    Args:
        sentences: List of sentences
        stop_words: Set of stopwords
        
    Returns:
        Similarity matrix
    """
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = calculate_sentence_similarity(sentences[i], sentences[j], stop_words)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix


def run_textrank(sentences: List[str], 
                 original_sentences: List[str], 
                 num_sentences: int = 3,
                 threshold: float = 0.1) -> Dict:
    """
    Run TextRank algorithm for extractive summarization.
    
    Args:
        sentences: Preprocessed sentences
        original_sentences: Original sentences for display
        num_sentences: Number of sentences to extract
        threshold: Minimum similarity threshold for edges
        
    Returns:
        Dictionary with summary, scores, and graph data
    """
    try:
        if not sentences:
            return {
                'summary': '',
                'sentence_scores': {},
                'graph_data': {'nodes': [], 'edges': [], 'stats': {}}
            }
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Build similarity matrix
        similarity_matrix = build_similarity_matrix(sentences, stop_words)
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank scores
        scores = nx.pagerank(graph, max_iter=100, tol=1e-6)
        
        # Rank sentences
        ranked_sentences = sorted(
            ((scores[i], i, sent) for i, sent in enumerate(original_sentences[:len(sentences)])),
            reverse=True
        )
        
        # Select top sentences
        num_sentences = min(num_sentences, len(ranked_sentences))
        top_sentences_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])
        
        # Create summary maintaining original order
        summary_sentences = [original_sentences[idx] for idx in top_sentences_indices]
        summary = ' '.join(summary_sentences)
        
        # Prepare sentence scores for response
        sentence_scores = {
            sent: scores[i] for i, sent in enumerate(original_sentences[:len(sentences)])
        }
        
        # Prepare graph data for visualization
        graph_data = prepare_graph_data(
            similarity_matrix, 
            original_sentences[:len(sentences)], 
            scores,
            threshold
        )
        
        return {
            'summary': summary,
            'sentence_scores': sentence_scores,
            'graph_data': graph_data
        }
    
    except Exception as e:
        logger.error(f"Error in TextRank: {str(e)}")
        return {
            'summary': '',
            'sentence_scores': {},
            'graph_data': {'nodes': [], 'edges': [], 'stats': {}},
            'error': str(e)
        }


def prepare_graph_data(similarity_matrix: np.ndarray, 
                       sentences: List[str], 
                       scores: Dict,
                       threshold: float = 0.1) -> Dict:
    """
    Prepare graph data for visualization.
    
    Args:
        similarity_matrix: Sentence similarity matrix
        sentences: List of sentences
        scores: TextRank scores
        threshold: Minimum similarity for edge creation
        
    Returns:
        Dictionary with nodes and edges
    """
    nodes = []
    edges = []
    
    # Create nodes
    for i, sent in enumerate(sentences):
        nodes.append({
            'id': i,
            'label': sent[:50] + "..." if len(sent) > 50 else sent,
            'title': sent,
            'score': scores.get(i, 0)
        })
    
    # Create edges
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                edges.append({
                    'from': i,
                    'to': j,
                    'width': similarity * 5,  # Scale for visualization
                    'label': f"{similarity:.2f}"
                })
    
    # Graph statistics
    stats = {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'avg_score': np.mean(list(scores.values())) if scores else 0,
        'max_score': max(scores.values()) if scores else 0
    }
    
    return {
        'nodes': nodes,
        'edges': edges,
        'stats': stats
    }


# ============================================
# EXTRACTIVE MODEL CLASS (Wrapper)
# ============================================

class CustomSummarizerModel:
    """
    Wrapper for the extractive TextRank-based summarization.
    This class maintains compatibility with the existing API.
    """
    
    def __init__(self):
        self.trained = True  # TextRank doesn't need training
        logger.info("  Extractive TextRank model initialized")
    
    def enhanced_summarize(self, text: str, num_sentences: int = 3) -> Dict:
        """
        Generate summary using TextRank (for backward compatibility).
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Dictionary with summary and metadata
        """
        cleaned_sentences = preprocess_text(text)
        original_sentences = get_original_sentences(text)
        
        if not cleaned_sentences:
            return {'error': 'Unable to process text'}
        
        result = run_textrank(
            sentences=cleaned_sentences,
            original_sentences=original_sentences,
            num_sentences=num_sentences
        )
        
        return {
            'summary': result['summary'],
            'sentence_scores': result['sentence_scores'],
            'method': 'extractive_textrank'
        }
    
    def load_model(self) -> bool:
        """Load model (no-op for TextRank)."""
        return True
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'trained': True,
            'model_type': 'TextRank (Extractive)',
            'method': 'graph_based',
            'parameters': 'PageRank algorithm with cosine similarity'
        }
    
    def train_on_arxiv_dataset(self, csv_path: str, max_samples: int = 5000) -> Dict:
        """
        Placeholder for training (TextRank doesn't need training).
        """
        return {
            'success': True,
            'message': 'TextRank is an unsupervised algorithm and does not require training.',
            'model_type': 'extractive'
        }


# ============================================
# ABSTRACTIVE SUMMARIZATION (T5)
# ============================================

class AbstractiveSummarizerModel:
    """
    T5-based abstractive text summarization model.
    Can be fine-tuned on domain-specific data (arXiv educational texts).
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the T5 model.
        
        Args:
            model_name: Hugging Face model name (default: t5-small)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("  Transformers library not available!")
            self.trained = False
            return
        
        self.model_name = model_name
        self.model_dir = "models/abstractive_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize tokenizer immediately for training
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            logger.info(f"Tokenizer loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        self.model = None
        self.trained = False
        
        logger.info(f"Abstractive model initialized (device: {self.device})")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a pre-trained or fine-tuned T5 model.
        
        Args:
            model_path: Path to saved model (if None, loads base model)
            
        Returns:
            True if successful
        """
        try:
            if model_path is None:
                # Check if fine-tuned model exists
                if os.path.exists(self.model_dir):
                    model_path = self.model_dir
                    logger.info(f"Loading fine-tuned model from {model_path}")
                else:
                    model_path = self.model_name
                    logger.info(f"Loading base model: {model_path}")
            
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
                
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.trained = True
            logger.info(f" Model loaded successfully on {self.device}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.trained = False
            return False
    
    def generate_summary(self, 
                        text: str, 
                        max_length: int = 150, 
                        min_length: int = 40,
                        num_beams: int = 4,
                        length_penalty: float = 2.0) -> str:
        """
        Generate abstractive summary using T5.
        
        Args:
            text: Input text
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            num_beams: Beam search width
            length_penalty: Length penalty for beam search
            
        Returns:
            Generated summary text
        """
        if not self.trained:
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return "Error: Model not available"
        
        try:
            # Prepare input
            input_text = "summarize: " + text
            
            # Tokenize
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=True
                )
            
            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error: {str(e)}"
    
    def train_on_arxiv_dataset(self, 
                               csv_path: str, 
                               max_samples: int = 1000,
                               num_epochs: int = 3,
                               batch_size: int = 4,
                               learning_rate: float = 5e-5) -> Dict:
        """
        Fine-tune T5 model on arXiv dataset.
        
        Args:
            csv_path: Path to arXiv CSV file
            max_samples: Maximum number of samples to use
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Training results dictionary
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                'success': False,
                'error': 'Transformers library not installed'
            }
        
        try:
            logger.info("=" * 60)
            logger.info("Starting T5 Model Training on arXiv Dataset")
            logger.info("=" * 60)
            
            # Load dataset
            logger.info(f"Loading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Check required columns
            if 'summaries' not in df.columns or 'abstracts' not in df.columns:
                return {
                    'success': False,
                    'error': 'CSV must have "abstracts" and "summaries" columns'
                }
            
            # Sample data
            df = df.head(max_samples)
            logger.info(f"Using {len(df)} samples for training")
            
            # Prepare data
            train_data = []
            for _, row in df.iterrows():
                if pd.notna(row['abstracts']) and pd.notna(row['summaries']):
                    train_data.append({
                        'input_text': 'summarize: ' + str(row['abstracts']),
                        'target_text': str(row['summaries'])
                    })
            
            if len(train_data) == 0:
                return {
                    'success': False,
                    'error': 'No valid training samples found'
                }
            
            logger.info(f"  Prepared {len(train_data)} valid training samples")
            
            # Create dataset
            dataset = Dataset.from_list(train_data)
            
            # Tokenize dataset
            logger.info("Tokenizing dataset...")
            
            # Ensure tokenizer is loaded
            if self.tokenizer is None:
                logger.info("Loading tokenizer...")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            
            def tokenize_function(examples):
                model_inputs = self.tokenizer(
                    examples['input_text'],
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )
                
                # Use as_target_tokenizer context for labels
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        examples['target_text'],
                        max_length=150,
                        truncation=True,
                        padding='max_length'
                    )
                
                model_inputs['labels'] = labels['input_ids']
                return model_inputs
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Split train/validation
            split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            
            logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
            
            # Load model
            logger.info(f"Loading base model: {self.model_name}")
            if not self.trained:
                self.load_model()
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.model_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f"{self.model_dir}/logs",
                logging_steps=50,
                evaluation_strategy="epoch",  # Fixed: was eval_strategy
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to="none",  # Disable wandb, tensorboard etc.
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("=" * 60)
            logger.info("Starting Training...")
            logger.info("=" * 60)
            
            train_result = trainer.train()
            
            # Save model
            logger.info(f"Saving fine-tuned model to: {self.model_dir}")
            trainer.save_model(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            
            logger.info("=" * 60)
            logger.info("  Training Complete!")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'train_loss': train_result.training_loss,
                'epochs': num_epochs,
                'samples': len(train_data),
                'model_saved': self.model_dir
            }
        
        except Exception as e:
            logger.error(f"  Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'trained': self.trained,
            'model_type': 'T5 (Abstractive)',
            'model_name': self.model_name,
            'device': self.device,
            'method': 'transformer_based',
            'saved_model': self.model_dir if os.path.exists(self.model_dir) else None
        }


# ============================================
# MAIN EXECUTION (For Training)
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 KG-Summarizer: Model Training")
    print("=" * 60)
    print()
    
    # Configuration
    ARXIV_CSV_PATH = "../data/arxiv_data.csv"
    MAX_SAMPLES = 1000  # Adjust based on your needs
    NUM_EPOCHS = 3
    
    # Check if dataset exists
    if not os.path.exists(ARXIV_CSV_PATH):
        print(f"  Dataset not found: {ARXIV_CSV_PATH}")
        print(f"   Please ensure arxiv_data.csv is in the data/ folder")
        print()
        print("   Required CSV format:")
        print("   - Column 'abstracts': Full text to summarize")
        print("   - Column 'summaries': Reference summaries")
        exit(1)
    
    print(f"Dataset found: {ARXIV_CSV_PATH}")
    print()
    
    # Initialize models
    print("🔧 Initializing models...")
    extractive_model = CustomSummarizerModel()
    abstractive_model = AbstractiveSummarizerModel(model_name="t5-small")
    print()
    
    # Train Abstractive Model
    print("=" * 60)
    print("Training Abstractive Model (T5)")
    print("=" * 60)
    print(f"Configuration:")
    print(f" - Samples: {MAX_SAMPLES}")
    print(f" - Epochs: {NUM_EPOCHS}")
    print(f" - Model: t5-small")
    print()
    
    input("Press Enter to start training (or Ctrl+C to cancel)...")
    print()
    
    result = abstractive_model.train_on_arxiv_dataset(
        csv_path=ARXIV_CSV_PATH,
        max_samples=MAX_SAMPLES,
        num_epochs=NUM_EPOCHS,
        batch_size=4,
        learning_rate=5e-5
    )
    
    print()
    print("=" * 60)
    print("Training Results")
    print("=" * 60)
    
    if result['success']:
        print(" Training Successful!")
        print(f"   - Loss: {result.get('train_loss', 'N/A')}")
        print(f"   - Epochs: {result.get('epochs', 'N/A')}")
        print(f"   - Samples: {result.get('samples', 'N/A')}")
        print(f"   - Model saved to: {result.get('model_saved', 'N/A')}")
    else:
        print("  Training Failed!")
        print(f" Error: {result.get('error', 'Unknown error')}")
    
    print()
    print("=" * 60)
    print("Model training complete!")
    print("You can now start the backend server with: python app.py")
    print("=" * 60)
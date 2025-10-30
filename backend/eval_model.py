# evaluate_models.py
# =========================================================================
# Comprehensive Model Evaluation & Visualization
# Compares custom models with baselines using multiple metrics and plots
# =========================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, f1_score
)

# ROUGE metrics for summarization
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")

# BERT Score for semantic similarity
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not installed. Install with: pip install bert-score")

# Word Cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud not installed. Install with: pip install wordcloud")

# Transformers for models
try:
    import torch
    from transformers import (
        T5Tokenizer, 
        T5ForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")

# Import your custom models
from model import (
    CustomSummarizerModel, 
    AbstractiveSummarizerModel,
    preprocess_text,
    run_textrank
)

# Download NLTK data
for package in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =========================================================================
# BASELINE MODELS
# =========================================================================

class BaselineExtractiveModel:
    """Simple first-N sentences baseline for extractive summarization."""
    
    def __init__(self):
        self.name = "Baseline: First-N Sentences"
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Extract first N sentences."""
        sentences = sent_tokenize(text)
        summary_sentences = sentences[:num_sentences]
        return ' '.join(summary_sentences)


class BaselineTFIDFModel:
    """TF-IDF based extractive summarization baseline."""
    
    def __init__(self):
        self.name = "Baseline: TF-IDF"
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Extract sentences with highest TF-IDF scores."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Maintain order
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
        except:
            return ' '.join(sentences[:num_sentences])


class BaselineAbstractivModel:
    """Baseline T5-small without fine-tuning for abstractive summarization."""
    
    def __init__(self):
        self.name = "Baseline: T5-small (No fine-tuning)"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Generate summary using base T5."""
        if not self.model:
            return "T5 not available"
        
        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=40,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


# =========================================================================
# EVALUATION METRICS
# =========================================================================

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict:
    """Calculate ROUGE scores for summaries."""
    if not ROUGE_AVAILABLE:
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }


def calculate_bert_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BERTScore for semantic similarity."""
    if not BERTSCORE_AVAILABLE:
        return 0.0
    
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
    return F1.mean().item()


def calculate_classification_metrics(predictions: List[str], 
                                     references: List[str],
                                     all_sentences: List[List[str]]) -> Dict:
    """
    Calculate classification metrics by treating sentence selection as binary classification.
    For each sentence: 1 if selected in summary, 0 otherwise.
    """
    y_true_all = []
    y_pred_all = []
    y_scores_all = []
    
    for pred, ref, sentences in zip(predictions, references, all_sentences):
        pred_sents = set(sent_tokenize(pred))
        ref_sents = set(sent_tokenize(ref))
        
        for sent in sentences:
            # True label: 1 if in reference, 0 otherwise
            y_true = 1 if any(sent.strip() in ref_s for ref_s in ref_sents) else 0
            
            # Predicted label: 1 if in prediction, 0 otherwise
            y_pred = 1 if any(sent.strip() in pred_s for pred_s in pred_sents) else 0
            
            # Score: simple overlap ratio for ROC curve
            score = max([len(set(sent.split()) & set(pred_s.split())) / len(set(sent.split()) | set(pred_s.split())) 
                        if len(set(sent.split()) | set(pred_s.split())) > 0 else 0
                        for pred_s in pred_sents] + [0])
            
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            y_scores_all.append(score)
    
    return {
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all),
        'y_scores': np.array(y_scores_all)
    }


# =========================================================================
# VISUALIZATION FUNCTIONS
# =========================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: str):
    """Plot confusion matrix for sentence selection."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Selected', 'Selected'],
                yticklabels=['Not Selected', 'Selected'])
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                   model_name: str, save_path: str):
    """Plot ROC curve for sentence selection."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray,
                                model_name: str, save_path: str):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_word_cloud(texts: List[str], model_name: str, save_path: str):
    """Generate word cloud from summaries."""
    if not WORDCLOUD_AVAILABLE:
        print(f"Skipping word cloud for {model_name} (wordcloud not installed)")
        return
    
    combined_text = ' '.join(texts)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(combined_text) 
             if w.isalnum() and w.lower() not in stop_words]
    
    text_for_cloud = ' '.join(words)
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text_for_cloud)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rouge_comparison(results_dict: Dict, save_path: str):
    """Compare ROUGE scores across models."""
    models = list(results_dict.keys())
    rouge1_scores = [results_dict[m]['rouge']['rouge1'] for m in models]
    rouge2_scores = [results_dict[m]['rouge']['rouge2'] for m in models]
    rougeL_scores = [results_dict[m]['rouge']['rougeL'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, rouge1_scores, width, label='ROUGE-1', color='skyblue')
    ax.bar(x, rouge2_scores, width, label='ROUGE-2', color='lightcoral')
    ax.bar(x + width, rougeL_scores, width, label='ROUGE-L', color='lightgreen')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('ROUGE Score', fontsize=12)
    ax.set_title('ROUGE Score Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_length_distribution(summaries_dict: Dict, save_path: str):
    """Compare summary length distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name, summaries in summaries_dict.items():
        lengths = [len(word_tokenize(s)) for s in summaries]
        ax.hist(lengths, alpha=0.5, label=model_name, bins=20)
    
    ax.set_xlabel('Summary Length (words)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Summary Length Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_heatmap(text: str, summary: str, model, tokenizer, save_path: str):
    """
    Visualize attention weights from T5 model.
    Shows which input tokens the model attended to when generating summary.
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping attention visualization (transformers not available)")
        return
    
    try:
        # Prepare input
        input_text = "summarize: " + text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate with attention output
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                num_beams=1,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Get attention weights from last layer
        # Note: T5 attention structure is complex, we'll visualize encoder self-attention
        if hasattr(model, 'encoder'):
            encoder_outputs = model.encoder(inputs, output_attentions=True)
            attentions = encoder_outputs.attentions[-1]  # Last layer
            
            # Average over heads and batch
            attention_avg = attentions[0].mean(dim=0).cpu().numpy()
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs[0])
            tokens = [t.replace('▁', '') for t in tokens][:30]  # First 30 tokens
            
            # Plot
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_avg[:30, :30],  # First 30x30
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'}
            )
            plt.title('Self-Attention Heatmap (Encoder Last Layer)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Key Tokens', fontsize=12)
            plt.ylabel('Query Tokens', fontsize=12)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
    
    except Exception as e:
        print(f"Could not generate attention visualization: {e}")


# =========================================================================
# MAIN EVALUATION PIPELINE
# =========================================================================

def run_evaluation(csv_path: str, num_samples: int = 100, output_dir: str = "evaluation_results"):
    """
    Run comprehensive evaluation of all models.
    
    Args:
        csv_path: Path to arXiv dataset CSV
        num_samples: Number of samples to evaluate
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("MODEL EVALUATION & VISUALIZATION")
    print("=" * 70)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    if 'abstracts' not in df.columns or 'summaries' not in df.columns:
        print("Error: CSV must have 'abstracts' and 'summaries' columns")
        return
    
    df_sample = df.head(num_samples)
    texts = df_sample['abstracts'].tolist()
    references = df_sample['summaries'].tolist()
    
    print(f"Loaded {len(texts)} samples")
    print()
    
    print("Initializing models...")
    print()
    
    models = {
        'Custom Extractive (TextRank)': CustomSummarizerModel(),
        'Baseline: First-N': BaselineExtractiveModel(),
        'Baseline: TF-IDF': BaselineTFIDFModel(),
    }
    
    if TRANSFORMERS_AVAILABLE:
        print("Loading Custom Abstractive (Fine-tuned T5)...")
        custom_abstract = AbstractiveSummarizerModel()
        if custom_abstract.load_model():
            models['Custom Abstractive (Fine-tuned T5)'] = custom_abstract
        
        print("Loading Baseline Abstractive (Base T5)...")
        models['Baseline: T5-small'] = BaselineAbstractivModel()
    
    print(f"Initialized {len(models)} models")
    print()
    
    print("=" * 70)
    print("Generating summaries...")
    print("=" * 70)
    
    summaries_dict = {}
    all_sentences_list = []
    
    for model_name, model in models.items():
        print(f"   {model_name}...")
        summaries = []
        
        for text in texts:
            try:
                if 'Abstractive' in model_name or 'T5' in model_name:
                    summary = model.generate_summary(text) if hasattr(model, 'generate_summary') else model.summarize(text)
                elif hasattr(model, 'enhanced_summarize'):
                    result = model.enhanced_summarize(text, num_sentences=3)
                    summary = result.get('summary', '')
                else:
                    summary = model.summarize(text, num_sentences=3)
                
                summaries.append(summary if summary else "")
            except Exception as e:
                print(f"      Warning: Error with {model_name}: {e}")
                summaries.append("")
        
        summaries_dict[model_name] = summaries
    
    for text in texts:
        all_sentences_list.append(sent_tokenize(text))
    
    print("Summaries generated")
    print()
    
    print("=" * 70)
    print("Calculating metrics...")
    print("=" * 70)
    
    results = {}
    
    for model_name, summaries in summaries_dict.items():
        print(f"   {model_name}...")
        
        rouge_scores = calculate_rouge_scores(summaries, references) if ROUGE_AVAILABLE else {}
        bert_score_val = calculate_bert_score(summaries, references) if BERTSCORE_AVAILABLE else 0.0
        class_metrics = calculate_classification_metrics(summaries, references, all_sentences_list)
        
        results[model_name] = {
            'rouge': rouge_scores,
            'bert_score': bert_score_val,
            'classification': class_metrics
        }
    
    print("Metrics calculated")
    print()
    
    print("=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    print()
    
    print("Confusion Matrices...")
    for model_name, result in results.items():
        safe_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        plot_confusion_matrix(
            result['classification']['y_true'],
            result['classification']['y_pred'],
            model_name,
            f"{output_dir}/confusion_matrix_{safe_name}.png"
        )
    print()
    
    print("ROC Curves...")
    for model_name, result in results.items():
        safe_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        plot_roc_curve(
            result['classification']['y_true'],
            result['classification']['y_scores'],
            model_name,
            f"{output_dir}/roc_curve_{safe_name}.png"
        )
    print()
    
    print("Precision-Recall Curves...")
    for model_name, result in results.items():
        safe_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        plot_precision_recall_curve(
            result['classification']['y_true'],
            result['classification']['y_scores'],
            model_name,
            f"{output_dir}/pr_curve_{safe_name}.png"
        )
    print()
    
    print("Word Clouds...")
    for model_name, summaries in summaries_dict.items():
        safe_name = model_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        plot_word_cloud(
            summaries,
            model_name,
            f"{output_dir}/wordcloud_{safe_name}.png"
        )
    print()
    
    if ROUGE_AVAILABLE:
        print("ROUGE Score Comparison...")
        plot_rouge_comparison(results, f"{output_dir}/rouge_comparison.png")
        print()
    
    print("Summary Length Distribution...")
    plot_length_distribution(summaries_dict, f"{output_dir}/length_distribution.png")
    print()
    
    if TRANSFORMERS_AVAILABLE and 'Custom Abstractive (Fine-tuned T5)' in models:
        print("Attention Heatmap (sample)...")
        model = models['Custom Abstractive (Fine-tuned T5)']
        if model.model and model.tokenizer:
            plot_attention_heatmap(
                texts[0],
                summaries_dict['Custom Abstractive (Fine-tuned T5)'][0],
                model.model,
                model.tokenizer,
                f"{output_dir}/attention_heatmap.png"
        )
    print()
    
    print("=" * 70)
    print("Saving results summary...")
    print("=" * 70)
    
    summary_file = f"{output_dir}/evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 70 + "\n")
            
            if result['rouge']:
                f.write(f"ROUGE-1: {result['rouge']['rouge1']:.4f}\n")
                f.write(f"ROUGE-2: {result['rouge']['rouge2']:.4f}\n")
                f.write(f"ROUGE-L: {result['rouge']['rougeL']:.4f}\n")
            
            if result['bert_score']:
                f.write(f"BERTScore: {result['bert_score']:.4f}\n")
            
            # Classification metrics
            y_true = result['classification']['y_true']
            y_pred = result['classification']['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
            
            f.write("\n")
    
    print(f"Saved: {summary_file}")
    print()
    
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    if ROUGE_AVAILABLE:
        print("ROUGE Scores:")
        print("-" * 70)
        for model_name, result in results.items():
            if result['rouge']:
                print(f"{model_name:40s} | R1: {result['rouge']['rouge1']:.4f} | "
                      f"R2: {result['rouge']['rouge2']:.4f} | RL: {result['rouge']['rougeL']:.4f}")
        print()
    
    if BERTSCORE_AVAILABLE:
        print("BERTScore:")
        print("-" * 70)
        for model_name, result in results.items():
            print(f"{model_name:40s} | {result['bert_score']:.4f}")
        print()
    
    print("=" * 70)
    print(f"Evaluation complete! Results saved to: {output_dir}/")
    print("=" * 70)


# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate summarization models')
    parser.add_argument('--csv', type=str, default='../data/arxiv_data.csv',
                       help='Path to arXiv dataset CSV')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.csv):
        print(f"Error: Dataset not found at {args.csv}")
        print(f"   Please ensure arxiv_data.csv is available")
        exit(1)
    
    # Run evaluation
    run_evaluation(args.csv, args.samples, args.output)
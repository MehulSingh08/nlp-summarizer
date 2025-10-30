# backend/app.py
# ===========================================
# Flask API for Text Summarization
# - Extractive: TextRank
# - Abstractive: T5
# ===========================================

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import custom modules
from model import (
    CustomSummarizerModel,          # Extractive model
    AbstractiveSummarizerModel,     # Abstractive model  
    run_textrank,
    preprocess_text,
    get_original_sentences
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize both models
extractive_model = CustomSummarizerModel()
abstractive_model = AbstractiveSummarizerModel(model_name="t5-small")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# UTILITY FUNCTIONS

def enhance_graph_visualization(graph_data, sentence_scores, original_sentences):
    """
    Enhance graph data for better visualization with additional properties.
    """
    if not graph_data or not graph_data.get('nodes'):
        return graph_data
    
    try:
        enhanced_nodes = []
        enhanced_edges = []
        
        # Process nodes with additional visualization data
        for node in graph_data.get('nodes', []):
            node_id = node['id']
            sentence_text = node.get('label', '')
            
            # Get importance score
            importance_score = sentence_scores.get(sentence_text, 0)
            
            # Determine if this sentence is in the summary (high importance)
            is_summary_sentence = importance_score > 0.1  # Adjust threshold as needed
            
            enhanced_node = {
                'id': node_id,
                'label': sentence_text,
                'title': f"Sentence {node_id + 1}: {sentence_text[:100]}...",  # Tooltip
                'size': max(10, min(50, importance_score * 100)),  # Size based on importance
                'color': {
                    'background': '#ff6b6b' if is_summary_sentence else '#4ecdc4',
                    'border': '#2c3e50',
                    'highlight': {
                        'background': '#ff5252' if is_summary_sentence else '#26a69a',
                        'border': '#1a252f'
                    }
                },
                'font': {
                    'color': '#2c3e50',
                    'size': 12 if is_summary_sentence else 10
                },
                'importance_score': importance_score,
                'is_summary': is_summary_sentence,
                'word_count': len(sentence_text.split()),
                'position_in_text': node_id
            }
            enhanced_nodes.append(enhanced_node)
        
        # Process edges with enhanced visualization
        for edge in graph_data.get('edges', []):
            similarity = edge.get('width', 1) / 5  # Normalize back to 0-1
            
            enhanced_edge = {
                'from': edge['from'],
                'to': edge['to'],
                'width': max(1, min(8, similarity * 10)),  # Visual width
                'color': {
                    'color': f'rgba(52, 152, 219, {min(1, similarity * 2)})',  # Transparency based on similarity
                    'highlight': 'rgba(41, 128, 185, 0.8)'
                },
                'similarity': similarity,
                'title': f"Similarity: {similarity:.3f}",  # Tooltip
                'smooth': {
                    'type': 'continuous',
                    'forceDirection': 'none',
                    'roundness': 0.3
                }
            }
            enhanced_edges.append(enhanced_edge)
        
        # Calculate additional statistics
        stats = graph_data.get('stats', {})
        enhanced_stats = {
            **stats,
            'summary_nodes': sum(1 for node in enhanced_nodes if node['is_summary']),
            'avg_importance': sum(node['importance_score'] for node in enhanced_nodes) / len(enhanced_nodes) if enhanced_nodes else 0,
            'max_importance': max(node['importance_score'] for node in enhanced_nodes) if enhanced_nodes else 0,
            'connection_density': len(enhanced_edges) / (len(enhanced_nodes) * (len(enhanced_nodes) - 1) / 2) if len(enhanced_nodes) > 1 else 0
        }
        
        return {
            'nodes': enhanced_nodes,
            'edges': enhanced_edges,
            'stats': enhanced_stats,
            'layout_options': {
                'physics': {
                    'enabled': True,
                    'stabilization': {'iterations': 200},
                    'barnesHut': {
                        'gravitationalConstant': -2000,
                        'centralGravity': 0.3,
                        'springLength': 95,
                        'springConstant': 0.04,
                        'damping': 0.09
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error enhancing graph visualization: {str(e)}")
        return graph_data  # Return original data if enhancement fails


# API ENDPOINTS

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Text Summarizer API is running',
        'models': {
            'extractive': extractive_model.trained,
            'abstractive': abstractive_model.trained
        }
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Summarize text using either extractive (TextRank) or abstractive (T5) method.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400

        text = data['text']
        num_sentences = data.get('num_sentences', 3)
        method = data.get('method', 'extractive')  # NEW: 'extractive' or 'abstractive'
        
        # Legacy support for 'use_model' parameter
        if 'use_model' in data and data['use_model']:
            method = 'extractive'

        # Validate input
        if not isinstance(text, str) or len(text.strip()) < 50:
            return jsonify({'error': 'Text must be a non-empty string with at least 50 characters.'}), 400
        
        try:
            num_sentences = int(num_sentences)
            if not (1 <= num_sentences <= 10):
                num_sentences = 3
        except (ValueError, TypeError):
            num_sentences = 3

        logger.info(f"Received summarization request (method: {method}) for text of length: {len(text)}")

        # ABSTRACTIVE SUMMARIZATION (T5)
        if method == 'abstractive':
            logger.info("Using Abstractive Model (T5)")
            
            if not abstractive_model.trained:
                logger.info("Loading abstractive model...")
                if not abstractive_model.load_model():
                    return jsonify({
                        'success': False,
                        'error': 'Abstractive model not available. Please train the model first or use extractive method.'
                    }), 400
            
            # Generate abstractive summary
            summary = abstractive_model.generate_summary(
                text,
                max_length=num_sentences * 30,  # Approximate tokens per sentence
                min_length=num_sentences * 10,
                num_beams=4
            )
            
            if summary.startswith("Error:"):
                return jsonify({
                    'success': False,
                    'error': summary
                }), 500
            
            response = {
                'success': True,
                'summary': summary,
                'method': 'abstractive_t5',
                'metadata': {
                    'original_length': len(text),
                    'summary_length': len(summary),
                    'model': 't5-small',
                    'approach': 'generative'
                }
            }
            
            logger.info("Successfully generated abstractive summary")
            return jsonify(response)
        
        # ========================================
        # EXTRACTIVE SUMMARIZATION (TextRank)
        # ========================================
        else:  # method == 'extractive'
            logger.info("Using Extractive Model (TextRank)")
            
            cleaned_sentences = preprocess_text(text)
            original_sentences = get_original_sentences(text)
            
            if not cleaned_sentences:
                return jsonify({'error': 'Unable to extract meaningful sentences from the text'}), 400

            result = run_textrank(
                sentences=cleaned_sentences,
                original_sentences=original_sentences,
                num_sentences=num_sentences
            )
            
            # Enhance the graph data for better visualization
            enhanced_graph_data = enhance_graph_visualization(
                result['graph_data'], 
                result['sentence_scores'],
                original_sentences
            )
            
            response = {
                'success': True,
                'summary': result['summary'],
                'method': 'extractive_textrank',
                'graph_data': enhanced_graph_data,
                'sentence_scores': result['sentence_scores'],
                'metadata': {
                    'original_length': len(text),
                    'total_sentences': len(original_sentences),
                    'summary_sentences': len(result['summary'].split('. ')) - 1,
                    'processed_sentences': len(cleaned_sentences),
                    'approach': 'extractive'
                }
            }
            
            logger.info("Successfully generated extractive summary with graph")
            return jsonify(response)

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500


@app.route('/train-extractive-model', methods=['POST'])
def train_extractive_model():
    """
    Train the extractive model (TextRank doesn't need training - placeholder).
    """
    return jsonify({
        'success': True,
        'message': 'TextRank is an unsupervised algorithm and does not require training.',
        'model_type': 'extractive'
    })


@app.route('/train-abstractive-model', methods=['POST'])
def train_abstractive_model():
    """
    Train the abstractive T5 model on the arXiv dataset.
    """
    try:
        data = request.get_json()
        if not data:
            data = {}
        
        csv_path = data.get('csv_path', 'data/arxiv_data.csv')
        max_samples = data.get('max_samples', 1000)
        num_epochs = data.get('num_epochs', 3)

        if not os.path.exists(csv_path):
            return jsonify({
                'success': False,
                'error': f'Dataset file not found: {csv_path}'
            }), 400
        
        logger.info(f"Starting abstractive model training with {csv_path}")
        
        training_results = abstractive_model.train_on_arxiv_dataset(
            csv_path=csv_path,
            max_samples=max_samples,
            num_epochs=num_epochs,
            batch_size=4,
            learning_rate=5e-5
        )
        
        if training_results.get('success'):
            return jsonify({
                'success': True,
                'message': 'Abstractive model trained successfully!',
                'training_stats': training_results
            })
        else:
            return jsonify({
                'success': False,
                'error': training_results.get('error', 'Training failed')
            }), 500

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Training error: {str(e)}'
        }), 500


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about both models."""
    try:
        extractive_info = extractive_model.get_model_info()
        abstractive_info = abstractive_model.get_model_info()
        
        return jsonify({
            'success': True,
            'models': {
                'extractive': extractive_info,
                'abstractive': abstractive_info
            },
            'message': 'Model information retrieved successfully'
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/sample-text', methods=['GET'])
def get_sample_text():
    """Provide sample text for testing."""
    sample = """Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to automatically improve their performance on a specific task through experience. Unlike traditional programming, where explicit instructions are provided for every possible scenario, machine learning systems learn patterns from data and make predictions or decisions based on that learning. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models that can make predictions on new, unseen data. Common applications include image classification, spam detection, and medical diagnosis. Unsupervised learning, on the other hand, works with unlabeled data to discover hidden patterns or structures within the dataset. Reinforcement learning is inspired by behavioral psychology and involves an agent learning to make decisions by interacting with an environment. Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to model and understand complex patterns in data."""
    
    return jsonify({'sample_text': sample.strip()})


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Text Summarizer API Server")
    logger.info("=" * 60)
    
    # Load extractive model (always available)
    logger.info("Extractive Model: Ready (TextRank)")
    
    # Try to load abstractive model
    logger.info("Checking Abstractive Model...")
    if abstractive_model.load_model():
        logger.info("Abstractive Model: Loaded successfully!")
    else:
        logger.info("Abstractive Model: Not found")
        logger.info("   Run 'python model.py' to train the model first")
    
    logger.info("=" * 60)
    logger.info("API Server running at: http://127.0.0.1:5000")
    logger.info("=" * 60)
    print()
    
    app.run(debug=True, host='127.0.0.1', port=5000)
# frontend/app.py
# Streamlit Frontend for Text Summarizer
# Supports both Extractive (TextRank) and Abstractive (T5) Summarization

import streamlit as st
import requests
import json
import pandas as pd
import time
import streamlit.components.v1 as components
import os


# Page configuration
st.set_page_config(
    page_title="Educational Text Summarizer",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #333333;
    }
    .extractive-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 1rem 0;
        color: #333333;
    }
    .abstractive-box {
        background-color: #fff4e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
        color: #333333;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .graph-container {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        margin: 1rem 0;
    }
    .legend-item {
        display: inline-block;
        margin: 5px 10px;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    .summary-node {
        background-color: #ff6b6b;
        color: white;
    }
    .regular-node {
        background-color: #4ecdc4;
        color: white;
    }
    .method-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        margin: 5px;
    }
    .badge-extractive {
        background-color: #4ecdc4;
        color: white;
    }
    .badge-abstractive {
        background-color: #ff6b6b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Backend API endpoint
API_BASE_URL = "http://127.0.0.1:5000"


# API communication functions

def check_api_health():
    """Check if the backend API is running and responsive."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_sample_text():
    """Fetch sample text from API or return fallback text."""
    try:
        response = requests.get(f"{API_BASE_URL}/sample-text", timeout=10)
        if response.status_code == 200:
            return response.json().get('sample_text', '')
    except requests.exceptions.RequestException:
        pass
    
    # Fallback sample text if API is unavailable
    return """
    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to automatically improve their performance on a specific task through experience. Unlike traditional programming, where explicit instructions are provided for every possible scenario, machine learning systems learn patterns from data and make predictions or decisions based on that learning.

    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models that can make predictions on new, unseen data. Common applications include image classification, spam detection, and medical diagnosis. The algorithm learns from input-output pairs and generalizes to make predictions on new inputs.

    Unsupervised learning, on the other hand, works with unlabeled data to discover hidden patterns or structures within the dataset. Clustering algorithms, dimensionality reduction techniques, and association rule learning are typical examples of unsupervised learning methods. These techniques are valuable for data exploration, customer segmentation, and anomaly detection.

    Reinforcement learning is inspired by behavioral psychology and involves an agent learning to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties and learns to maximize cumulative reward over time. This approach has been particularly successful in game playing, robotics, and autonomous systems.

    Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to model and understand complex patterns in data. These deep neural networks have achieved remarkable success in areas such as computer vision, natural language processing, and speech recognition. The availability of large datasets and powerful computational resources has fueled the recent advances in deep learning.
    """


def get_model_info():
    """Retrieve current model configuration and status from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'error': 'Could not get model info'}
    
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Connection error: {str(e)}'}


def summarize_text(text, num_sentences=3, method='extractive'):
    """
    Send text to backend for summarization.
    
    Args:
        text: Input text to summarize
        num_sentences: Number of sentences for extractive method
        method: 'extractive' or 'abstractive'
    
    Returns:
        dict: Response containing summary and metadata
    """
    try:
        payload = {
            'text': text,
            'num_sentences': num_sentences,
            'method': method
        }
        
        response = requests.post(
            f"{API_BASE_URL}/summarize",
            json=payload,
            timeout=60  # Longer timeout for T5 model processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                return {'success': False, 'error': error_data.get('error', 'Unknown error')}
            except:
                return {'success': False, 'error': f'Server returned status {response.status_code}'}
            
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout. The text might be too long to process.'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Connection error: {str(e)}'}
    except json.JSONDecodeError:
        return {'success': False, 'error': 'Invalid response from server'}


# Visualization functions

def create_graph_html(graph_data):
    """
    Generate interactive network graph HTML using vis.js library.
    
    Args:
        graph_data: Dictionary containing nodes and edges
    
    Returns:
        str: HTML content with embedded graph or None if invalid data
    """
    if not graph_data or not graph_data.get('nodes'):
        return None
    
    nodes_data = json.dumps(graph_data['nodes'])
    edges_data = json.dumps(graph_data['edges'])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            #mynetworkid {{
                width: 100%;
                height: 600px;
                border: 1px solid lightgray;
                background-color: #fafafa;
            }}
            .legend {{
                padding: 15px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 15px;
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 20px;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="legend">
            <span class="legend-item" style="background-color: #ff6b6b; color: white;">Summary Sentences</span>
            <span class="legend-item" style="background-color: #4ecdc4; color: white;">Other Sentences</span>
            <span style="margin-left: 20px; color: #555;">Edge thickness = Similarity strength</span>
        </div>
        <div id="mynetworkid"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({nodes_data});
            var edges = new vis.DataSet({edges_data});
            
            var container = document.getElementById('mynetworkid');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            
            var options = {{
                nodes: {{
                    shape: 'dot',
                    font: {{
                        size: 12,
                        color: '#2c3e50'
                    }},
                    borderWidth: 2,
                    shadow: true
                }},
                edges: {{
                    smooth: {{
                        type: 'continuous'
                    }},
                    shadow: true
                }},
                physics: {{
                    enabled: true,
                    stabilization: {{
                        iterations: 200
                    }},
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Log selected node information on click
            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    console.log("Selected sentence:", node.label);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content


def display_interactive_graph(graph_data):
    """Render interactive knowledge graph with statistics."""
    st.subheader("Knowledge Graph Visualization")
    st.markdown("""
    This interactive graph shows the relationships between sentences in the text. 
    **Red nodes** are selected for the summary, **blue nodes** are other sentences.
    Edge thickness indicates similarity strength between sentences.
    """)
    
    html_content = create_graph_html(graph_data)
    if html_content:
        components.html(html_content, height=700, scrolling=False)
        
        # Display graph statistics if available
        stats = graph_data.get('stats', {})
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sentences", stats.get('num_nodes', 0))
            with col2:
                st.metric("Connections", stats.get('num_edges', 0))
            with col3:
                st.metric("Avg Importance", f"{stats.get('avg_score', 0):.3f}")
            with col4:
                st.metric("Max Importance", f"{stats.get('max_score', 0):.3f}")
    else:
        st.warning("Could not generate graph visualization.")


def display_simple_graph(graph_data):
    """Display graph data as tabular information."""
    st.subheader("Graph Data Details")
    
    tab1, tab2 = st.tabs(["Nodes (Sentences)", "Edges (Connections)"])
    
    with tab1:
        nodes = graph_data.get('nodes', [])
        if nodes:
            nodes_df = pd.DataFrame(nodes)
            st.dataframe(nodes_df, use_container_width=True)
        else:
            st.info("No node data available.")
    
    with tab2:
        edges = graph_data.get('edges', [])
        if edges:
            edges_df = pd.DataFrame(edges)
            st.dataframe(edges_df, use_container_width=True)
        else:
            st.info("No edge data available.")


# Main application

def main():
    # Application header
    st.markdown('<h1 class="main-header">Educational Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Powered by Knowledge Graphs & Deep Learning
        </p>
        <div>
            <span class="method-badge badge-extractive">Extractive (TextRank)</span>
            <span class="method-badge badge-abstractive">Abstractive (T5)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Verify backend API connection
    api_status = check_api_health()
    
    if api_status:
        st.success("Backend API is connected")
        
        # Display model information
        model_info = get_model_info()
        if model_info.get('success'):
            with st.expander("Model Information"):
                models = model_info.get('models', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Extractive Model")
                    extractive = models.get('extractive', {})
                    st.write(f"**Type:** {extractive.get('model_type', 'N/A')}")
                    st.write(f"**Method:** {extractive.get('method', 'N/A')}")
                    st.write(f"**Status:** {'Ready' if extractive.get('trained') else 'Not available'}")
                
                with col2:
                    st.markdown("### Abstractive Model")
                    abstractive = models.get('abstractive', {})
                    st.write(f"**Type:** {abstractive.get('model_type', 'N/A')}")
                    st.write(f"**Model:** {abstractive.get('model_name', 'N/A')}")
                    st.write(f"**Device:** {abstractive.get('device', 'N/A')}")
                    st.write(f"**Status:** {'Ready' if abstractive.get('trained') else 'Not trained'}")
                    
                    if not abstractive.get('trained'):
                        st.warning("Abstractive model not trained. Run `python model.py` to train it.")
    else:
        st.error("Backend API is not available. Please start the backend server first.")
        st.code("cd backend && python app.py", language="bash")
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Control number of sentences for extractive summarization
        num_sentences = st.slider(
            "Number of sentences (Extractive)",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of sentences to extract for extractive summarization"
        )
        
        st.markdown("---")
        
        # Load sample text into input
        if st.button("Load Sample Text"):
            st.session_state.input_text = get_sample_text().strip()
            st.rerun()
        
        st.markdown("---")
        
        # Visualization options for extractive method
        st.subheader("Visualization Options")
        st.caption("(Only for Extractive Method)")
        show_interactive_graph = st.checkbox("Interactive Graph", value=True)
        show_simple_graph = st.checkbox("Simple Graph Data", value=False)
        
        st.markdown("---")
        
        # Application information
        with st.expander("About"):
            st.markdown("""
            **Educational Text Summarizer**
            
            This application provides two summarization methods:
            
            **Extractive (TextRank)**
            - Selects important sentences from the original text
            - Graph-based algorithm using PageRank
            - Shows knowledge graph visualization
            
            **Abstractive (T5)**
            - Generates new summary text
            - Fine-tuned on educational texts (arXiv)
            - Uses transformer deep learning
            
            **How to use:**
            1. Enter or paste your text
            2. Choose summarization method
            3. View results and visualizations
            """)
    
    # Text input section
    st.header("Input Text")
    
    input_text = st.text_area(
        "Enter the text you want to summarize:",
        height=250,
        key="input_text",
        placeholder="Paste your text here (minimum 50 characters)..."
    )
    
    # Display character count and validation
    if input_text:
        char_count = len(input_text.strip())
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Character count: {char_count}")
        with col2:
            if char_count >= 50:
                st.caption("Ready to summarize")
            else:
                st.caption(f"Need {50 - char_count} more characters")
    
    # Summarization method selection
    st.markdown("### Choose Summarization Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        extractive_button = st.button(
            "Extractive Summary",
            type="primary",
            use_container_width=True,
            help="Extract important sentences using TextRank algorithm + Knowledge Graph"
        )
    
    with col2:
        abstractive_button = st.button(
            "Abstractive Summary",
            type="primary",
            use_container_width=True,
            help="Generate new summary using T5 transformer model"
        )
    
    # Handle extractive summarization request
    if extractive_button:
        if not input_text or len(input_text.strip()) < 50:
            st.markdown('<div class="error-box">Please enter at least 50 characters of text to summarize.</div>', unsafe_allow_html=True)
        elif not api_status:
            st.markdown('<div class="error-box">Backend API is not available. Please start the backend server first.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running Extractive Summarization (TextRank)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Visual feedback during processing
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("Preprocessing text...")
                    elif i < 40:
                        status_text.text("Extracting sentences...")
                    elif i < 60:
                        status_text.text("Running TextRank algorithm...")
                    elif i < 80:
                        status_text.text("Building knowledge graph...")
                    else:
                        status_text.text("Enhancing visualization...")
                    time.sleep(0.02)
                
                progress_bar.empty()
                status_text.empty()
                
                result = summarize_text(input_text, num_sentences, method='extractive')
                
                if result.get('success'):
                    st.session_state.result = result
                    st.session_state.show_interactive_graph = show_interactive_graph
                    st.session_state.show_simple_graph = show_simple_graph
                    st.session_state.method_used = 'extractive'
                    st.success("Extractive summarization complete!")
                    st.rerun()
                else:
                    error_msg = result.get('error', 'Unknown error occurred')
                    st.markdown(f'<div class="error-box">Error: {error_msg}</div>', unsafe_allow_html=True)
    
    # Handle abstractive summarization request
    if abstractive_button:
        if not input_text or len(input_text.strip()) < 50:
            st.markdown('<div class="error-box">Please enter at least 50 characters of text to summarize.</div>', unsafe_allow_html=True)
        elif not api_status:
            st.markdown('<div class="error-box">Backend API is not available. Please start the backend server first.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running Abstractive Summarization (T5)... This may take a moment..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Visual feedback during processing
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Loading T5 model...")
                    elif i < 60:
                        status_text.text("Tokenizing text...")
                    else:
                        status_text.text("Generating summary...")
                    time.sleep(0.03)
                
                progress_bar.empty()
                status_text.empty()
                
                result = summarize_text(input_text, num_sentences, method='abstractive')
                
                if result.get('success'):
                    st.session_state.result = result
                    st.session_state.method_used = 'abstractive'
                    st.success("Abstractive summarization complete!")
                    st.rerun()
                else:
                    error_msg = result.get('error', 'Unknown error occurred')
                    st.markdown(f'<div class="error-box">Error: {error_msg}</div>', unsafe_allow_html=True)
    
    # Display summarization results
    if hasattr(st.session_state, 'result') and st.session_state.result.get('success'):
        result = st.session_state.result
        method_used = st.session_state.get('method_used', 'unknown')
        
        st.markdown("---")
        st.header("Results")
        
        # Display method used
        method_display = {
            'extractive': 'Extractive (TextRank)',
            'abstractive': 'Abstractive (T5)',
            'extractive_textrank': 'Extractive (TextRank)',
            'abstractive_t5': 'Abstractive (T5)'
        }
        
        result_method = result.get('method', method_used)
        st.info(f"**Method Used:** {method_display.get(result_method, result_method)}")
        
        # Display summary with appropriate styling
        st.subheader("Generated Summary")
        summary = result.get('summary', 'No summary available')
        
        if 'extractive' in result_method:
            st.markdown(f'<div class="extractive-box"><strong>Summary:</strong><br><br>{summary}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="abstractive-box"><strong>Summary:</strong><br><br>{summary}</div>', unsafe_allow_html=True)
        
        # Display metadata and statistics
        if 'metadata' in result:
            metadata = result['metadata']
            st.subheader("Statistics")
            
            if 'extractive' in result_method:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Text", f"{metadata.get('original_length', 0)} chars")
                with col2:
                    st.metric("Total Sentences", metadata.get('total_sentences', 0))
                with col3:
                    st.metric("Processed Sentences", metadata.get('processed_sentences', 0))
                with col4:
                    st.metric("Summary Sentences", metadata.get('summary_sentences', 0))
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Text", f"{metadata.get('original_length', 0)} chars")
                with col2:
                    st.metric("Summary Text", f"{metadata.get('summary_length', 0)} chars")
                with col3:
                    st.metric("Approach", metadata.get('approach', 'N/A'))
        
        # Show graph visualization for extractive method
        if 'extractive' in result_method and 'graph_data' in result:
            st.markdown("---")
            
            if st.session_state.get('show_interactive_graph', True):
                display_interactive_graph(result['graph_data'])
            
            if st.session_state.get('show_simple_graph', False):
                st.markdown("---")
                display_simple_graph(result['graph_data'])
            
            # Display sentence importance scores
            if 'sentence_scores' in result:
                st.markdown("---")
                st.subheader("Sentence Importance Scores")
                
                scores = result['sentence_scores']
                if scores:
                    # Truncate long sentences for display
                    scores_data = [
                        {'Sentence': k[:100] + "..." if len(k) > 100 else k, 'Importance Score': v}
                        for k, v in scores.items()
                    ]
                    df_scores = pd.DataFrame(scores_data)
                    df_scores = df_scores.sort_values('Importance Score', ascending=False)
                    
                    st.dataframe(df_scores, use_container_width=True)
                    st.bar_chart(df_scores.set_index('Sentence')['Importance Score'])
        
        # Provide download options
        st.markdown("---")
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method_name = method_display.get(result_method, result_method)
            summary_text = f"""
Text Summarizer Results
=======================

Method: {method_name}
Original Text Length: {metadata.get('original_length', 0)} characters

SUMMARY:
{summary}

Generated by Knowledge Graph Text Summarizer
            """
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"summary_{result_method}.txt",
                mime="text/plain"
            )
        
        with col2:
            if 'graph_data' in result:
                graph_json = json.dumps(result, indent=2)
                st.download_button(
                    label="Download Full Results (JSON)",
                    data=graph_json,
                    file_name=f"results_{result_method}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
        st.code(str(e))
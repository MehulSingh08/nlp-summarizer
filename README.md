# NLP Summarizer

An educational text summarization system using knowledge graphs and hybrid machine learning approaches for academic research papers.

## Overview

This project implements a sophisticated text summarization pipeline combining:
- **Extractive Summarization**: Enhanced TextRank algorithm with domain adaptation
- **Knowledge Graphs**: Semantic relationship extraction and visualization
- **Hybrid Learning**: TF-IDF vectorization with K-Means clustering
- **Flask Backend**: RESTful API for model serving
- **Streamlit Frontend**: Interactive UI for document analysis

The system processes academic abstracts using learned patterns from 5,000+ arXiv papers to generate coherent, contextually-aware summaries.

## Dataset

- **Source**: arXiv Academic Papers Dataset
- **Format**: CSV with abstracts and reference summaries
- **Features**: Research paper abstracts and corresponding summaries
- **File**: `data/arxiv_data.csv`

## Project Structure

```
kg-summarizer/
├── data/
│   └── arxiv_data.csv
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   └── requirements.txt
├── frontend/
│   ├── app1.py
│   └── requirements.txt
├── models/
│   └── abstractive_model/
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.11
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/MehulSingh08/nlp-summarizer.git
cd kg-summarizer

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Create data folder
mkdir data

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies (in new terminal)
cd frontend
pip install -r requirements.txt
```

## Model Setup

The trained T5 model weights (~240 MB) are not included in this repository.

### Training the Model

```bash
# Prepare dataset
# Place arxiv_data.csv in data/ folder with columns: 'abstracts', 'summaries'

# Train model (15-20 minutes on CPU)
cd backend
python model.py
```

The trained model will be saved to `backend/models/abstractive_model/`

**Note**: The `models/` and `data/` folders are excluded via `.gitignore` to keep the repository lightweight.

## Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

Backend runs on `http://127.0.0.1:5000`

### Start Frontend Interface

```bash
# In a new terminal
cd frontend
streamlit run app.py
```

Frontend runs on `http://localhost:8501`

## Usage

### Web Interface

1. Open `http://localhost:8501` in your browser
2. Upload or paste academic text
3. Click "Generate Summary"
4. View extractive summary with knowledge graph visualization

### API Endpoints

**Generate Summary**
```bash
POST http://127.0.0.1:5000/summarize
Content-Type: application/json

{
  "text": "Your academic abstract here..."
}
```

**Response**
```json
{
  "summary": "Generated summary text",
  "knowledge_graph": {...},
  "topic_cluster": 5
}
```

## Model Architecture

### Hybrid Unsupervised Learning System

**1. Domain-Specific TF-IDF Vectorizer**
- Learns word importance from 5,000+ arXiv abstracts
- Builds custom vocabulary weights for academic domain
- Output: Domain-aware word importance scores

**2. K-Means Topic Clustering**
- Discovers hidden topic patterns in research papers
- Groups similar papers into 20 topic clusters
- Output: Topic classification for new text

**3. Sentence Importance Pattern Learner**
- Analyzes sentence position and word patterns
- Identifies where key information appears in abstracts
- Output: Custom scoring weights for sentence ranking

**4. Enhanced TextRank Scorer**
- Combines all learned patterns for final scoring
- Uses trained components for domain-aware ranking
- Output: Context-aware extractive summaries

## Key Features

- **Domain Adaptation**: Trained on academic research papers
- **Knowledge Graph Generation**: Visual semantic relationships
- **Topic Classification**: Automatic research area detection
- **Hybrid Approach**: Combines unsupervised and graph-based methods
- **Interactive UI**: Streamlit-based dashboard
- **RESTful API**: Flask backend for easy integration

## Technical Stack

**Backend**
- Python 3.11
- Flask
- scikit-learn
- transformers (T5)
- pandas, numpy
- NetworkX

**Frontend**
- Streamlit
- Plotly
- Requests

## Git Workflow

### Before Work
```bash
git pull origin main
```

### During Work
```bash
git add .
git commit -m "Descriptive message"
```

### After Work
```bash
git pull origin main
git push origin main
```

### Adding Dependencies
```bash
pip install package-name
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add package-name dependency"
git push origin main
```

## Limitations

- Designed for academic text only
- Single-server deployment
- No real-time model updates
- Limited to English language
- Requires pre-trained model weights

## Future Enhancements

- Multi-document summarization
- Real-time knowledge graph updates
- Support for multiple languages
- Advanced graph neural networks
- Production-grade API authentication

## Acknowledgments

- Dataset: arXiv Academic Papers
- Research: TextRank algorithm by Mihalcea and Tarau
- Framework: Hugging Face Transformers
- Baseline Model: T5-base hugging face model

## License

This project is for educational purposes as part of an NLP course.

## Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This is a collaborative educational project. The `models/` and `data/` directories are not tracked in version control due to file size constraints.
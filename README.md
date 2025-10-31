# nlp-summarizer


Educational Text Summarization using Knowledge Graphs  
Team collaborative NLP course project.

---
## 1.  Quick Setup

```bash
# Clone repo
git clone https://github.com/MehulSingh08/kg-summarizer.git

cd kg-summarizer

# Create virtual environment
py -3.11 -m venv venv

# Activate
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Prerequisites

Python 3.11
pip package manager

# Installation

1.     Clone or download the project files
    Create the data folder:

mkdir data

2.  Set up the backend:

cd backend
pip install -r requirements.txt


3.  Set up the frontend (in a new terminal):

cd frontend
   pip install -r requirements.txt

#  Running the Application

1.  Start the backend server:

  cd backend
   python app.py

The API will be available at http://127.0.0.1:5000

2.  Start the frontend (in a new terminal):

   cd frontend
   streamlit run app.py
The web interface will open in your browser at http://localhost:8501

## 🤖 Model Setup

The trained T5 model weights (~240 MB) are not included in this repository.

### To train the model:
```bash
# 1. Prepare your dataset
# Place arxiv_data.csv in data/ folder with columns: 'abstracts', 'summaries'

# 2. Train the model (15-20 minutes on CPU)
cd backend
python model.py


The trained model will be saved to: `backend/models/abstractive_model/`

### Note
The `models/` and `data/` folders are excluded via `.gitignore` to keep the repository lightweight.


#  Updating / Adding Dependencies

pip install package-name
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add package-name dependency"
git push origin main

#  Git Workflow
Before work:

git pull origin main
While working:


git add .
git commit -m "Descriptive message"
After work:


git pull origin main  # check for conflicts
git push origin main



#  Project Strcuture

kg-summarizer/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── model.py                # TextRank algorithm implementation
│   ├── utils.py                # Text preprocessing utilities
│   └── requirements.txt        # Backend dependencies
│
├── frontend/
│   ├── app.py                  # Streamlit user interface
│   └── requirements.txt        # Frontend dependencies
│
├── data/
│   └── arxiv_data.csv          # Your dataset (create this folder)
│
├── .gitignore
└── README.md

Prominent Questions: 
Q1. What Model Are We Actually Training?
We're training a Hybrid Unsupervised Learning System with 4 components:
1. Domain-Specific TF-IDF Vectorizer

What it learns: Which words are important in academic papers
How: Analyzes 5,000+ arXiv abstracts to build vocabulary weights
Result: Custom word importance scores (not generic English)

2. K-Means Topic Clustering Model

What it learns: Hidden topic patterns in academic research
How: Groups similar papers into 20 topic clusters
Result: Can classify new text into learned research areas

3. Sentence Importance Pattern Learner

What it learns: Which sentence positions/words indicate importance
How: Analyzes where key information appears in abstracts
Result: Custom scoring weights for different sentence types

4. Enhanced TextRank Scorer

What it learns: How to combine all learned patterns
How: Uses trained components to score sentences better
Result: Domain-aware summarization



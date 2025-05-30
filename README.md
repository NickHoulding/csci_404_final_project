# Medical RAG Chatbot

A chatbot that leverages Retrieval-Augmented Generation (RAG) to support doctors by providing medically relevant insights based on symptom input. The system retrieves information from a curated PubMed-based knowledge base and generates context-aware suggestions using LLaMA 3.2:1B-Instruct and BioBERT.

---

## Disclaimer
This system does not provide medical diagnoses. All responses include a disclaimer and are intended only to assist licensed healthcare professionals.

---

## Features

- Retrieval of top most relevant medical information using BioBERT embeddings
- Contextual response generation via LLaMA 3.2:1B-Instruct
- Domain-specific insights for respiratory infections (e.g., pneumonia, influenza)
- Ethical safeguards including automated disclaimers in every output
- Simple Streamlit interface for interaction

---

## Technologies Used

- Python 3.10.12
- [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) via Hugging Face Transformers
- [LLaMA 3.2:1B-Instruct](https://ollama.com/library/llama3.2) via Ollama API
- Streamlit (for basic GUI)
- Hugging Face `evaluate` library for metric scoring

---

## Dataset

Filtered subset of the [PubMedQA dataset](https://huggingface.co/datasets/qiaojin/PubMedQA), focused on respiratory infection cases. Preprocessed and split into chunks for efficient retrieval.

---

## Evaluation Metrics

### Retrieval
- **Recall@3:** Measures whether at least one relevant document appears in the top K retrieved contexts. We will compute Recall@3 with a target score > 0.8.
- **Precision@3:** Evaluates the proportion of retrieved documents that are relevant in the top K documents. We will use Precision@3 to quantify contextual relevance.
- **Mean Reciprocal Rank (MRR):** Captures how highly ranked the first retrieved document is, averaged across queries. Higher MRR indicates better prioritization of useful context.

### Generation
- **BERTScore:** Computes semantic similarity using contextual embeddings from a pre-trained language model. This is ideal for evaluating whether the chatbot's responses capture the correct meaning and nuance of medical insights, even if phrased differently. For the purposes of this project, BERTScore will be used as the primary metric, as it captures semantic similarity rather than just lexical rigidity. ROUGE-L and BLEU are included for additional context, comparison, and completeness.
- **BLEU:** Measures n-gram overlap with a reference answer. This ensures that the chatbot’s responses include key medical terms and phrases drawn directly from authoritative sources.
- **ROUGE-L:** Captures longest common subsequence between generated and reference answers. It is particularly useful for verifying that the chatbot preserves the structure and sequence of clinically relevant information when summarizing evidence.

---

## Project Structure

```
├── .streamlit/             # Streamlit theme configuration and styling
├── data/                   # Datasets and dataset management scripts
├── docs/                   # Documentation files and images
├── eval/                   # Evaluation data and scripts
├── knowledge/              # Knowledge bases and population script
├── models/                 # Model storage
│   └── dmis-lab/biobert-v1.1/ # BioBERT embedding model
├── .env                    # Environment variables and configuration
├── .gitignore              # Git ignore rules
├── main.py                 # Application entry point
├── env.py                  # Environment variable handling
├── models.py               # Model interface for LLaMA
├── rag.py                  # RAG pipeline management
├── knowledge_base.py       # Knowledge base class definition 
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Streamlit app configuration
├── chat_gui.py             # Streamlit chat interface logic
└── utils.py                # Utility functions
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone git@github.com:NickHoulding/csci_404_final_project.git
cd csci_404_final_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download models and dataset
Install [Ollama](https://ollama.com/) and pull the LLaMA model:

```bash
# Pull the LLaMA model using Ollama
ollama pull llama3.2:1b-instruct-q4_0

# Download the BioBERT embedding model
python models/download_model.py

```

At this point, you should have all necessary dependencies 
and models to run using the existing knowledge base.

---

## How to Run

Launch the application using:

```bash
python3 main.py
```

This will start the Streamlit chat gui.

Type in a symptom-based query (e.g., "fever, shortness of breath"), and the system will:
1. Retrieve the most relevant medical contexts using BioBERT embeddings
2. Generate a contextual response using LLaMA 3.2-Instruct
3. Display the response with an appropriate medical disclaimer

---

## Authors
- Nicholas Houlding
- Parker Mina

Project for CSCI 404: Natural Language Processing - Spring 2025
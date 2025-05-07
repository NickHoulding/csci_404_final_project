# Medical RAG Chatbot

A chatbot that leverages Retrieval-Augmented Generation (RAG) to support doctors by providing medically relevant insights based on symptom input. The system retrieves information from a curated PubMed-based knowledge base and generates context-aware suggestions using LLaMA and BioBERT.

---

## Disclaimer
This system does not provide medical diagnoses. All responses include a disclaimer and are intended only to assist licensed healthcare professionals.

---

## Features

- Retrieval of top-3 relevant medical abstracts using FAISS and BioBERT embeddings
- Contextual response generation via LLaMA 3.2:1B-Instruct
- Domain-specific insights for respiratory infections (e.g., pneumonia, influenza)
- Ethical safeguards including automated disclaimers in every output
- Simple Streamlit interface for interaction

---

## Technologies Used

- Python 3.10.12
- [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) via Hugging Face Transformers
- [LLaMA 3.2:1B-Instruct](https://ollama.com/library/llama3.2) via Ollama
- FAISS database (GPU-accelerated cosine similarity search)
- Langchain (for RAG implementation)
- Streamlit (for basic GUI)
- Hugging Face `evaluate` library for metric scoring

---

## Dataset

Filtered subset of the [PubMedQA dataset](https://huggingface.co/datasets/qiaojin/PubMedQA), focused on respiratory infection cases. Preprocessed and split into ~500-token chunks for efficient retrieval.

---

## Evaluation Metrics

### Retrieval
- **Recall@3:** Measures whether at least one relevant document appears in the top K retrieved contexts. We will compute Recall@3 with a target score > 0.8.
- **Precision@3:** Evaluates the proportion of retrieved documents that are relevant in the top K documents. We will use Precision@3 to quantify contextual relevance.
- **Mean Reciprocal Rank (MRR):** Captures how highly ranked the first retrieved document is, averaged across queries. Higher MRR indicates better prioritization of useful context.

### Generation
- **BLEU:** Measures n-gram overlap with a reference answer. This ensures that the chatbot’s responses include key medical terms and phrases drawn directly from authoritative sources.
- **ROUGE-L:** Captures longest common subsequence between generated and reference answers. It is particularly useful for verifying that the chatbot preserves the structure and sequence of clinically relevant information when summarizing evidence.
- **BERTScore:** Computes semantic similarity using contextual embeddings from a pre-trained language model. This is ideal for evaluating whether the chatbot's responses capture the correct meaning and nuance of medical insights, even if phrased differently.

---

## Project Structure

```
├── chat_gui.py             # Streamlit chat interface logic
├── data/                   # Dataset directory
│   ├── download_dataset.py # Script to download the dataset
│   └── PubMedQA.csv        # Original dataset
├── docs/                   # Documentation files
├── eval.py                 # Evaluation metrics implementation
├── knowledge/              # Processed knowledge base
├── main.py                 # Application entry point
├── models.py               # Model interface
├── rag.py                  # RAG implementation
├── requirements.txt        # Python dependencies
└── streamlit_app.py        # Streamlit configuration
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

### 4. Download models
Install [Ollama](https://ollama.com/) and pull the LLaMA model:

```bash
ollama pull llama3.2:1b-instruct-q4_0
```

Make sure FAISS is GPU-enabled (install faiss-gpu) and BioBERT is loaded via HuggingFace.

## How to Run
```bash
python3 main.py
```

Type in a symptom-based query (e.g., "fever, shortness of breath"), and the chatbot will return a generated insight and disclaimer.

## Authors
- Nicholas Houlding
- Parker Mina

Project for CSCI 404: Natural Language Processing – Spring 2025
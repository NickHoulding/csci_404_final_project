# Medical RAG Chatbot for Respiratory Infection Insights

A command-line chatbot that leverages Retrieval-Augmented Generation (RAG) to support doctors by providing medically relevant insights based on symptom input. The system retrieves information from a curated PubMed-based knowledge base and generates context-aware suggestions using LLaMA and BioBERT.

---

## Features

- Retrieval of top-3 relevant medical abstracts using FAISS and BioBERT embeddings
- Contextual response generation via LLaMA 3.2:1B-Instruct
- Domain-specific insights for respiratory infections (e.g., pneumonia, influenza)
- Ethical safeguards including automated disclaimers in every output
- Simple Streamlit interface for interaction

---

## Technologies Used

- Python 3.10+
- [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) via Hugging Face Transformers
- [LLaMA 3.2:1B-Instruct](https://ollama.com/library/llama3.2) via Ollama
- FAISS (GPU-accelerated cosine similarity search)
- Langchain
- Streamlit (for basic GUI)
- Hugging Face `evaluate` for metric scoring

---

## Dataset

Filtered subset of the [PubMedQA dataset](https://huggingface.co/datasets/qiaojin/PubMedQA), focused on respiratory infection cases. Preprocessed and split into ~500-token chunks for efficient retrieval.

---

## Evaluation Metrics

### Retrieval
- **Recall@3**
- **Precision@3**
- **Mean Reciprocal Rank (MRR)**

### Generation
- **BLEU** – Checks overlap of key medical phrases
- **ROUGE-L** – Evaluates structured summary retention
- **BERTScore** – Measures semantic similarity between generated and reference outputs

---

## Project Structure

```
├── chat_gui.py           # Streamlit chat interface logic
├── data/                 # Dataset directory
│   ├── download_dataset.py
│   └── PubMedQA.csv
├── docs/                 # Documentation files
├── eval.py               # Evaluation metrics implementation
├── knowledge/            # Processed knowledge base
├── main.py               # Application entry point
├── models.py             # ML model interface
├── rag.py                # RAG implementation
├── requirements.txt      # Python dependencies
└── streamlit_app.py      # Streamlit configuration
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
cd medical-rag-chatbot
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
Install Ollama and pull the LLaMA model:

```bash
ollama run llama3.2
```

Make sure FAISS is GPU-enabled (install faiss-gpu) and BioBERT is loaded via Hugging Face.

## How to Run
```bash
streamlit run streamlit_app.py
```

Then open the local link in your browser. Type in a symptom-based query (e.g., "fever, shortness of breath"), and the chatbot will return a generated insight and disclaimer.

## Disclaimer
This system does not provide medical diagnoses. All responses include a disclaimer and are intended only to assist licensed healthcare professionals.

## Authors
Nicholas Houlding

Parker Mina

Project for CSCI 404: Natural Language Processing – Spring 2025
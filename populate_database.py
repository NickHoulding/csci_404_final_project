"""
Converts .csv data into embeddings and stores them in a vector database.

This script:    1. Iterates through the .csv data.
                2. Create embeddings for each row.
                3. Store the embeddings in the database.

NOTE:           The target .csv file must be located in the 'data'
                directory of this project.
"""

# Imports
from transformers import AutoTokenizer, AutoModel
from env import get_env_var
import pandas as pd
import numpy as np
import pickle
import faiss
import torch
import os

# Globals
CSV_FILENAME = "Respiratory_Small_PubMedQA.csv"
BATCH_SIZE = 64

def main():
    embedding_model_path = os.path.join(
        os.path.dirname(__file__), 
        'models', 
        get_env_var("EMBEDDING_MODEL")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    model = AutoModel.from_pretrained(embedding_model_path)
    model = model.to(device)
    model.eval()

    csv_path = os.path.join(
        os.path.dirname(__file__), 
        'data', 
        CSV_FILENAME
    )
    df = pd.read_csv(csv_path)
    contexts = df['context'].tolist()
    all_embeddings = []
    
    total_batches = (len(contexts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(contexts)} contexts in {total_batches} batches")

    for i in range(0, len(contexts), BATCH_SIZE):
        batch = contexts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches}...", end="\r")

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)
        
        del inputs, outputs
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\nFinalized all batches. Creating index...")
    final_embeddings = np.vstack(all_embeddings)

    # Create FAISS index
    d = final_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # If using GPU, convert the index to GPU
    if device.type == "cuda":
        gpu_resource = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        gpu_index.add(final_embeddings)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index.add(final_embeddings)

    # Save the index and contexts
    index_save_path = os.path.join(
        os.path.dirname(__file__), 
        'knowledge', 
        'index.faiss'
    )
    faiss.write_index(index, index_save_path)

    contexts_save_path = os.path.join(
        os.path.dirname(__file__), 
        'knowledge', 
        'contexts.pkl'
    )
    with open(contexts_save_path, 'wb') as f:
        pickle.dump(contexts, f)

if __name__ == "__main__":
    main()
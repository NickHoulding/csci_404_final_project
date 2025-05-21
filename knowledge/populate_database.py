"""
Converts .csv data into embeddings and populates a FAISS index.

This script:    1. Grabs all rows of an attribute from a .csv file.
                2. Create embeddings from the specified attribute.
                3. Stores the embeddings in the FAISS index.
                4. Saves the index and the corresponding texts.

Usage:          1. Install python dependencies.
                2. Install models and set env variables.
                3. Run: python populate_database.py --help
"""

# Imports
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import argparse
import pickle
import faiss
import torch
import sys
import os
import re

# Adds the parent directory to the system path so the env import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import get_env_var

def setup_args() -> argparse.Namespace:
    """
    Sets up the command line arguments for the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Populates a FAISS index with embeddings from a CSV file.")

    parser.add_argument("--infile", type=str, required=True,
                        help="(str) path to the input csv")
    parser.add_argument("--save_path", type=str, required=True,
                        help="(int) path to the output directory")
    parser.add_argument("--attr", type=str, required=True,
                        help="(str) the name of the attribute to filter on")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="(int) the batch size to use for processing")
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> bool:
    """
    Validates the command line arguments.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
    Returns:
        bool: True if the arguments are valid, 
            False otherwise.
    """
    # Validate input file
    if not os.path.isfile(args.infile):
        print(f"Arg Error: Input file: \"{args.infile}\" does not exist.")
        return False
    elif not re.match(r'.*\.csv$', args.infile):
        print(f"Arg Error: Input file: \"{args.infile}\" is not a .csv file.")
        return False
    
    # Validate save path
    if not os.path.isdir(args.save_path):
        print(f"Arg Error: Save path: \"{args.save_path}\" does not exist.")
        return False
    elif not os.access(args.save_path, os.W_OK):
        print(f"Arg Error: Save path: \"{args.save_path}\" is not writable.")
        return False

    # Validate attribute
    df = pd.read_csv(args.infile, nrows=0)
    if args.attr not in df.columns:
        print(f"Arg Error: Attribute: \"{args.attr}\" does not exist in the input file.")
        return False
    
    # Validate batch size
    if args.batch_size <= 0:
        print(f"Arg Error: Batch size: \"{args.batch_size}\" must be greater than 0.")
        return False
    
    return True

def main():
    args = setup_args()

    if not validate_args(args):
        print("Arg Error: Invalid arguments. Exiting...")
        sys.exit(1)

    embedding_model_path = os.path.join(os.path.dirname(__file__), 
                                        '..', 
                                        'models', 
                                        get_env_var("EMBEDDING_MODEL"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for embeddings generation")
    
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

    df = pd.read_csv(args.infile)
    texts = df[args.attr].tolist()
    all_embeddings = []
    
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(f"Processing {len(texts)} texts in {total_batches} batches")

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...", end="\r")

        inputs = tokenizer(batch,
                           padding=True,
                           truncation=True,
                           max_length=512,
                           return_tensors="pt",)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / torch.norm(embeddings, 
                                             dim=1, 
                                             keepdim=True)
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)
        
        del inputs, outputs
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\nFinalized all batches. Creating index...")
    final_embeddings = np.vstack(all_embeddings)

    # Create FAISS index
    d = final_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(final_embeddings)

    # Save the index and corresponding texts
    faiss.write_index(index, os.path.join(args.save_path, 'index.faiss'))
    with open(os.path.join(args.save_path, 'texts.pkl'), 'wb') as f:
        pickle.dump(texts, f)

if __name__ == "__main__":
    main()
"""
Converts .csv data into embeddings and populates a knowledgeBase object.

This script:    1. Grabs all rows of an attribute from a .csv file.
                2. Create embeddings from the specified attribute.
                3. Adds each row to a knowledgeBase object.
                4. Saves the knowledgeBase object.

Usage:          1. Install python dependencies.
                2. Install models and set env variables.
                3. Run: python populate_database.py --help
"""

# Imports
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import argparse
import pickle
import torch
import sys
import os
import re

# Adds the parent directory to the system path so the env import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import knowledgeBase
from env import get_env_var

def setup_args() -> argparse.Namespace:
    """
    Sets up the command line arguments for the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Populates a knowledgeBase with data from a CSV file.")

    parser.add_argument("--infile", type=str, required=True,
                        help="(str) path to the input csv")
    parser.add_argument("--save_path", type=str, required=True,
                        help="(int) path to the output directory")
    parser.add_argument("--attr", type=str, required=True,
                        help="(str) the name of the attribute to filter on")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="(int) the batch size to use for processing")
    
    # Parse the command line arguments
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
    # Create and validate cmd line args
    args = setup_args()

    if not validate_args(args):
        print("Arg Error: Invalid arguments. Exiting...")
        sys.exit(1)

    # Set up the embedding model and device
    embedding_model_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        get_env_var("EMBEDDING_MODEL"
    ))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for embeddings generation")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    model = AutoModel.from_pretrained(embedding_model_path)
    model = model.to(device)

    # Disable training-specific features
    model.eval()

    # Load and process the input CSV file
    df = pd.read_csv(args.infile)
    texts = df[args.attr].tolist()
    
    # Create a new knowledgeBase instance
    kb = knowledgeBase()
    
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(f"Processing {len(texts)} texts in {total_batches} batches")

    # Process texts in batches
    for i in range(0, len(texts), args.batch_size):
        batch_indices = range(i, min(i + args.batch_size, len(texts)))
        batch = [texts[j] for j in batch_indices]
        batch_pubids = df.iloc[batch_indices]['pubid'].tolist()
        batch_num = i // args.batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...", end="\r")

        # Tokenize the batch
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract the embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.cpu().numpy()
        
        # Add entries to the knowledge base using pubid as entry_id
        for j, (text, embedding, pubid) in enumerate(zip(batch, embeddings, batch_pubids)):
            kb.add_entry(pubid, text, embedding)
        
        # Clear memory
        del inputs, outputs
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Save the knowledge base object
    print(f"\nFinalized all batches. Saving knowledgeBase with {len(kb)} entries...")
    with open(os.path.join(args.save_path, 'knowledge_base.pkl'), 'wb') as f:
        pickle.dump(kb, f)
    
    print(f"knowledgeBase saved to {os.path.join(args.save_path, 'knowledge_base.pkl')}")

if __name__ == "__main__":
    main()
"""
Evaluation module for RAG (Retrieval-Augmented Generation) 
retrieval performance.

This module implements evaluation metrics to assess how well 
a RAG system retrieves relevant context chunks for a given 
query. The module uses cosine similarity between embeddings 
to determine whether a retrieved chunk is relevant to the 
gold answer, with a configurable threshold parameter.

Metrics implemented:
1. Recall@k: 
    Measures whether at least one relevant document appears 
    in the top-k retrieved results. Calculated as the percentage 
    of queries where at least one retrieved chunk exceeds the 
    similarity threshold with the gold answer. A higher recall 
    indicates the system's ability to find relevant information 
    within the top-k results.

2. Precision@k:
    Measures the proportion of retrieved documents that are 
    relevant out of all retrieved documents. Calculated as 
    the average ratio of retrieved chunks that exceed the 
    similarity threshold with the gold answer across all 
    queries. Higher precision indicates more accurate 
    retrieval with fewer irrelevant chunks.

3. Mean Reciprocal Rank (MRR):
    Evaluates how highly the first relevant document is ranked 
    in the results. Calculated as the average reciprocal of 
    the rank position of the first relevant chunk across all 
    queries. A higher MRR indicates that relevant information 
    appears earlier in the results list.
"""

# Imports
import pandas as pd
import numpy as np
import sys
import os

# Adds the parent directory to the system path so the rag import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import load_kb, get_embedding
from env import get_env_var

# Globals
kb = load_kb(os.path.join(
    os.path.dirname(__file__), 
    '..',
    'knowledge', 
    get_env_var('KNOWLEDGE_BASE')
))
embedding_cache = {}

def get_cached_embedding(text: str) -> np.ndarray:
    """
    Retrieves the embedding for a given text, using a cache 
    to avoid redundant computations.

    Args:
        text (str): The input text to get the embedding for.
    Returns:
        np.ndarray: The embedding vector for the input text.
    """
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = get_embedding(text)
        embedding_cache[text] = embedding
        return embedding

def compute_bin_recall(
        retrieved_chunks: list[dict], 
        gold_embedding: np.ndarray,
        thresh: float = 0.7
) -> int:
    """
    Computes binary recall for k retrieved chunks against 
    gold answer. (Tests if there is at least one relevant 
    chunk in the top-k results.)

    Args:
        retrieved_chunks (list): The retrieved chunks.
        gold_embedding (np.ndarray): The gold answer embedding.
    Returns:
        int: 1 if gold answer is found in any chunk, else 0.
    """
    # Check if any chunk exceeds the similarity threshold
    for chunk in retrieved_chunks:
        chunk_embedding = get_cached_embedding(chunk['text'])
        similarity = kb.compute_cosine_similarity(
            gold_embedding, 
            chunk_embedding
        )

        if similarity >= thresh:
            return 1
        
    return 0

def compute_precision(
        retrieved_chunks: list[dict], 
        gold_embedding: np.ndarray,
        thresh: float = 0.7
) -> float:
    """
    Computes precision for k retrieved chunks against gold answer.

    Args:
        retrieved_chunks (list): The retrieved chunks.
        gold_embedding (np.ndarray): The gold answer embedding.
    Returns:
        float: The number of matches over total number of chunks.
    """
    total_chunks = len(retrieved_chunks)
    matches = 0

    if total_chunks == 0:
        return 0.0

    # Count how many chunks exceed the similarity threshold
    for chunk in retrieved_chunks:
        chunk_embedding = get_cached_embedding(chunk['text'])
        similarity = kb.compute_cosine_similarity(
            gold_embedding, 
            chunk_embedding
        )

        if similarity >= thresh:
            matches += 1

    return matches / total_chunks

def compute_mrr(
        retrieved_chunks: list[dict], 
        gold_embedding: np.ndarray,
        thresh: float = 0.7
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for k retrieved chunks 
    against gold answer.

    Args:
        retrieved_chunks (list): The retrieved chunks.
        gold_embedding (np.ndarray): The gold answer embedding.
    Returns:
        float: The reciprocal of the rank of the first match, 
            or 0 if no match is found.
    """
    # Find the rank of the first chunk that exceeds the threshold
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_embedding = get_cached_embedding(chunk['text'])
        similarity = kb.compute_cosine_similarity(
            gold_embedding, 
            chunk_embedding
        )
        
        if similarity >= thresh:
            return 1.0 / rank
    
    return 0.0

def retrieval_eval_at_k(
        df: pd.DataFrame, 
        thresh: float = 0.7,
        k: int = 3
) -> tuple[list]:
    """
    Evaluates the retrieval performance of a RAG system using
    Recall, Precision, and Mean Reciprocal Rank (MRR) metrics.

    Args:
        df (pd.DataFrame): The input DataFrame containing 
            'context' and 'long_answer' columns.
        thresh (float): The threshold for cosine similarity to
            consider a match.
        k (int): The number of top results to consider 
            for evaluation.
    Returns:
        tuple: A tuple containing three lists:
            - Recall@k scores
            - Precision@k scores
            - MRR scores
    """
    recall_scores = []
    precision_scores = []
    mrr_scores = []

    # Calculate scores for each row in the eval DataFrame
    for question, long_answer in zip(df['question'], df['long_answer']):
        prompt_embedding = get_embedding(question)
        results = kb.search(q_embed=prompt_embedding, top_k=k)
        gold_embedding = get_embedding(long_answer)

        recall_scores.append(compute_bin_recall(
            results, 
            gold_embedding, 
            thresh
        ))
        precision_scores.append(compute_precision(
            results, 
            gold_embedding, 
            thresh
        ))
        mrr_scores.append(compute_mrr(
            results, 
            gold_embedding, 
            thresh
        ))

    return recall_scores, precision_scores, mrr_scores

# Entry Point
if __name__ == '__main__':
    # Load the evaluation data from a CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'eval.csv')
    df = pd.read_csv(csv_path)

    thresh = 0.75   # Cosine similarity threshold for relevance
    top_k = 3       # Top-k chunks to retrieve for evaluation

    # Calculate retrieval evaluation metrics
    recall, precision, mrr = retrieval_eval_at_k(
        df=df, 
        thresh=thresh, 
        k=top_k
    )

    # Calculate average scores
    recall_avg = sum(recall) / len(recall)
    precision_avg = sum(precision) / len(precision)
    mrr_avg = sum(mrr) / len(mrr)

    print(f"Retrieval Evaluation Results for {len(df)} queries:")
    print(f"Recall@{top_k}:\t{recall_avg:.4f}")
    print(f"Precision@{top_k}:\t{precision_avg:.4f}")
    print(f"MRR:\t\t{mrr_avg:.4f}")
"""
This script plots the retrieval evaluation metrics 
against varying cosine similarity thresholds.
"""

# Imports
from retrieval_eval import retrieval_eval_at_k
import matplotlib.pyplot as plt
import pandas as pd
import os

def calculate_metrics(df: pd.DataFrame) -> tuple:
    """
    Calculates retrieval evaluation metrics (Recall, Precision, 
    MRR) for varying cosine similarity thresholds.

    Args:
        df (pd.DataFrame): The DataFrame containing evaluation data.
    Returns:
        tuple: A tuple containing lists of thresholds, recall 
            scores, precision scores, and MRR scores.
    """
    recall_scores = []
    precision_scores = []
    mrr_scores = []
    thresholds = []
    thresh = 0.7
    
    # Iterate over thresholds from 0.7 to 0.85
    while thresh <= 0.85:
        # Round the threshold and calculate metrics
        thresholds.append(round(thresh, 2))
        recall, precision, mrr = retrieval_eval_at_k(
            df=df, 
            thresh=thresh,
            k=3
        )

        # Calculate and append average scores
        recall_avg = sum(recall) / len(recall)
        precision_avg = sum(precision) / len(precision)
        mrr_avg = sum(mrr) / len(mrr)
        recall_scores.append(recall_avg)
        precision_scores.append(precision_avg)
        mrr_scores.append(mrr_avg)

        thresh += 0.01

    return (
        thresholds, 
        recall_scores, 
        precision_scores, 
        mrr_scores
    )

def plot_and_save(
        thresholds: list[float], 
        recall_scores: list[float], 
        precision_scores: list[float], 
        mrr_scores: list[float]
):
    """
    Plots the retrieval evaluation metrics against cosine 
    similarity thresholds and saves the plot to a file.

    Args:
        thresholds (list[float]): List of cosine similarity 
            thresholds.
        recall_scores (list[float]): List of Recall@3 scores.
        precision_scores (list[float]): List of Precision@3 scores.
        mrr_scores (list[float]): List of MRR scores.
    Returns:
        None
    """
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        thresholds,
        recall_scores,
        label='Recall@3',
        marker='o'
    )
    plt.plot(
        thresholds,
        precision_scores,
        label='Precision@3',
        marker='o'
    )
    plt.plot(
        thresholds,
        mrr_scores,
        label='MRR',
        marker='o'
    )

    # Set plot title and labels
    plt.title('Retrieval Evaluation Metrics vs Cosine Similarity Threshold')
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Score')
    plt.xticks(thresholds)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot to a file and show it
    plt.savefig(os.path.join(
        os.path.dirname(__file__), 
        'retrieval_eval_scores_plot.png'
    ))
    plt.show()

def main():
    # Load the evaluation data from a CSV file
    csv_path = os.path.join(
        os.path.dirname(__file__), 
        'eval.csv'
    )
    df = pd.read_csv(csv_path)

    # Calculate metrics for varying thresholds
    results = calculate_metrics(df=df)

    # Extract results
    thresholds = results[0]
    recall_scores = results[1]
    precision_scores = results[2]
    mrr_scores = results[3]

    # Plot and save the results
    plot_and_save(
        thresholds=thresholds, 
        recall_scores=recall_scores, 
        precision_scores=precision_scores, 
        mrr_scores=mrr_scores
    )

# Entry Point
if __name__ == '__main__':
    main()
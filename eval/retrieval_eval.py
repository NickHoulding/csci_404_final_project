"""
Implementation of the RAG retrieval evaluation metrics.
"""
import evaluate
import pandas as pd
from rag import get_context_prompt, get_embedding, searck_kb

def retrieval_eval(predictions, references):
    df = pd.read_csv('eval.csv')
    for question, long_answer in zip(df['question'], df['long_answer']):
        prompt_embedding = get_embedding(question)
        results = searck_kb(prompt_embedding, top_k=3)
        print(compute_recall_at_k(results, long_answer))
        print(compute_precison_at_k(results, long_answer))
        print(compute_mrr(results, long_answer))

def compute_recall_at_k(retrieved_chunks, gold_answer):
    return int(any(gold_answer.lower() in chunk.lower() for chunk in retrieved_chunks))
def compute_precison_at_k(retrieved_chunks, gold_answer):
    matches = sum(gold_answer.lower() in chunk.lower() for chunk in retrieved_chunks)
    return matches / len(retrieved_chunks)
def compute_mrr(retrieved_chunks, gold_answer):
    for i, chunk in enumerate(retrieved_chunks):
        if gold_answer.lower() in chunk.lower():
            return 1 / (i + 1)
    return 0.0
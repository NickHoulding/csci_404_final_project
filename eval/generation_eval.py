"""
Evaluation module for assessing the generation quality of 
AI responses.

This module implements evaluation metrics to assess how 
well the AI system generates responses compared to reference 
answers. It uses standard NLP evaluation metrics to measure 
different aspects of text generation quality.

For the purposes of this project, BERTScore will be used 
as the primary metric, as it captures semantic similarity 
rather than just lexical rigidity. ROUGE-L and BLEU are 
included for additional context, comparison, and 
completeness.

Metrics implemented:
1. ROUGE-L: 
   Measures the longest common subsequence between the 
   generated text and reference text. It captures the 
   fluency and coverage of the generated text compared 
   to the reference. Higher ROUGE-L scores indicate 
   better semantic similarity and content coverage.

2. BLEU:
   Measures the precision of n-gram matches between the 
   generated text and reference text. It evaluates how 
   many word sequences in the generated text appear in 
   the reference text. Higher BLEU scores suggest better 
   lexical precision and translation quality.

3. BERTScore:
   Uses contextualized embeddings to measure semantic 
   similarity between generated and reference texts. 
   By computing token-level similarities using BERT 
   embeddings, it captures meaning beyond exact word 
   matches. Higher BERTScores indicate better semantic 
   alignment between generated and reference texts.
"""

# Imports
import pandas as pd
import evaluate
import sys
import os

# Adds the parent directory to the system path so these imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import get_context_prompt, load_kb, get_embedding
from models import query_model
from env import get_env_var

# Globals
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
kb = load_kb(os.path.join(
    os.path.dirname(__file__), 
    '..',
    'knowledge', 
    get_env_var('KNOWLEDGE_BASE')
))

def compute_rouge_l(
        prediction: str, 
        reference: str
) -> float:
    """
    Computes the ROUGE score for the generated text against 
    the reference text.

    Args:
        prediction (str): The generated text.
        reference (str): The reference text.
    Returns:
        dict: The ROUGE score.
    """
    return rouge.compute(
        predictions=[prediction], 
        references=[reference]
    )['rougeL']

def compute_bleu(
        prediction: str, 
        reference: str
) -> float:
    """
    Computes the BLEU score for the generated text against 
    the reference text.

    Args:
        prediction (str): The generated text.
        reference (str): The reference text.
    Returns:
        float: The BLEU score.
    """
    return bleu.compute(
        predictions=[prediction], 
        references=[[reference]]
    )['bleu']

def compute_bertscore(
        prediction: str, 
        reference: str
) -> float:
    """
    Computes the BERTScore for the generated text against 
    the reference text.

    Args:
        prediction (str): The generated text.
        reference (str): The reference text.
    Returns:
        dict: The F1 score of the BERTScore.
    """
    return bertscore.compute(
        predictions=[prediction], 
        references=[reference], 
        lang='en'
    )['f1'][0]

def generation_eval(df: pd.DataFrame) -> tuple[list]:
    """
    Evaluates the generation model using ROUGE, BLEU, and 
    BERTScore metrics.

    Args:
        df (pd.DataFrame): The input DataFrame containing 
            'context' and 'long_answer' columns.
    Returns:
        tuple: A tuple containing three lists:
            - ROUGE scores
            - BLEU scores
            - BERTScores
    """
    rougel_scores = []
    bleu_scores = []
    bertscore_scores = []

    # Calculate scores for each row in the eval DataFrame
    for question, long_answer in zip(df['question'], df['long_answer']):
        prompt_embedding = get_embedding(question)
        results = kb.search(q_embed=prompt_embedding, top_k=3)
        context_prompt = get_context_prompt(question, results)
        _, generated = query_model(context_prompt)

        bleu_scores.append(compute_bleu(
            generated, 
            long_answer
        ))
        rougel_scores.append(compute_rouge_l(
            generated, 
            long_answer
        ))
        bertscore_scores.append(compute_bertscore(
            generated, 
            long_answer
        ))

    return rougel_scores, bleu_scores, bertscore_scores

# Entry Point
if __name__ == '__main__':
    # Load the evaluation data from a CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'eval.csv')
    df = pd.read_csv(csv_path)

    # Calculate the evaluation metrics
    rougel, bleu, bertscore = generation_eval(df=df)

    # Calculate the average scores
    rougel_avg = sum(rougel) / len(df)
    bleu_avg = sum(bleu) / len(df)
    bert_score_avg = sum(bertscore) / len(df)

    print(f"Generation Evaluation Results for {len(df)} queries:")
    print(f"BERTScore:\t{bert_score_avg:.4f}")
    print(f"ROUGE-L:\t{rougel_avg:.4f}")
    print(f"BLEU:\t\t{bleu_avg:.4f}")
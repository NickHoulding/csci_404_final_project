"""
Implementation of the generation evaluation metrics.
"""

import evaluate
import pandas as pd
from rag import get_context_prompt, get_embedding, searck_kb
from models import query_model

def generation_eval(predictions, references):
    df = pd.read_csv('eval.csv')
    for question, long_answer in zip(df['question'], df['long_answer']):
        prompt_embedding = get_embedding(question)
        results = searck_kb(prompt_embedding, top_k=3)
        context_prompt = get_context_prompt(question, results)
        title, response = query_model(context_prompt)

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        bertscore = evaluate.load("bertscore")

        results_rouge = rouge.compute(predictions=question, references=long_answer)
        print(results_rouge)

        tokenized_preds = [pred.split() for pred in question]
        tokenized_refs = [[ref.split()] for ref in long_answer]

        results_bleu = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
        print(results_bleu)

        results_bertscore = bertscore.compute(predictions=question, references=long_answer, lang="en")
        print(results_bertscore)
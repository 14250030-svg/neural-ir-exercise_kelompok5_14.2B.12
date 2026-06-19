"""
FiRA Extractive Question Answering (Part 3) - Fixed Version
Dosen: Zico Pratama Putra
Kelompok: 5 [Mohamad Prastya - Wahid Diyono - Wan Berry Pranata]
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from tqdm import tqdm
import numpy as np
import string

class ExtractiveQA:
    def __init__(self, model_name="deepset/roberta-base-squad2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        print(f"✅ QA Model {model_name} loaded on {self.device}")

    def answer(self, question: str, context: str, max_answer_length=30) -> dict:
        """Extractive QA inference"""
        try:
            inputs = self.tokenizer(
                question, context,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores) + 1

            answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            score = float(torch.max(torch.softmax(start_scores, dim=1)) * 
                         torch.max(torch.softmax(end_scores, dim=1)))

            return {
                'answer': answer.strip(),
                'score': score,
                'start': int(start_idx),
                'end': int(end_idx)
            }
        except Exception as e:
            return {'answer': '', 'score': 0.0, 'start': 0, 'end': 0}

    def batch_answer(self, questions: list, contexts: list, batch_size=8):
        """Batch processing untuk banyak query"""
        results = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Answering QA"):
            batch_q = questions[i:i+batch_size]
            batch_c = contexts[i:i+batch_size]
            batch_results = [self.answer(q, c) for q, c in zip(batch_q, batch_c)]
            results.extend(batch_results)
        return results


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)
    if not pred or not gt:
        return 0.0
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = len(set(pred_tokens) & set(gt_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def evaluate_qa(predictions: list, gold_answers: list):
    em_scores = [compute_exact_match(pred['answer'], gold) for pred, gold in zip(predictions, gold_answers)]
    f1_scores = [compute_f1(pred['answer'], gold) for pred, gold in zip(predictions, gold_answers)]
    
    return {
        'Exact_Match': np.mean(em_scores) * 100,
        'F1_Score': np.mean(f1_scores) * 100,
        'Num_Queries': len(predictions)
    }


# ====================== MAIN ======================
if __name__ == "__main__":
    qa_system = ExtractiveQA()
    
    q = "How to make a good cappuccino?"
    ctx = "To make a perfect cappuccino, you need espresso, steamed milk, and milk foam..."
    result = qa_system.answer(q, ctx)
    print(f"Jawaban: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")